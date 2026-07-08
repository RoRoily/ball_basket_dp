# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an RGB image-conditioned diffusion policy from visual demonstrations."""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lowdim_diffusion import DDPMScheduler, resolve_torch_device, set_seed
from visual_diffusion import (
    VisualSequenceDataset,
    build_visual_model_from_config,
    visual_checkpoint_config_from_args,
)


def _default_output_path() -> str:
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join("runs", "visual_diffusion", run_name, "policy.pt")


def _save_checkpoint(
    output_path: str,
    model_state_dict: dict[str, torch.Tensor],
    normalizer: dict[str, torch.Tensor],
    config: dict,
    epoch: int,
    train_loss: float,
    val_loss: float | None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    best_metric = val_loss if val_loss is not None else train_loss
    checkpoint = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model_state_dict.items()},
        "normalizer": {key: value.detach().cpu() for key, value in normalizer.items()},
        "config": config,
        "epoch": epoch,
        "loss": best_metric,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, output_path)


class EMAModel:
    """Exponential moving average of model weights for more stable sampling."""

    def __init__(self, model: torch.nn.Module, decay: float):
        if decay <= 0.0 or decay >= 1.0:
            raise ValueError("EMA decay must be in (0, 1).")
        self.decay = decay
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        model_state = model.state_dict()
        for key, shadow_value in self.shadow.items():
            shadow_value.mul_(self.decay).add_(model_state[key].detach(), alpha=1.0 - self.decay)

    def state_dict(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        for key, shadow_value in self.shadow.items():
            state[key] = shadow_value.detach().clone()
        return state

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        backup = {key: value.detach().clone() for key, value in model.state_dict().items()}
        state = model.state_dict()
        for key, shadow_value in self.shadow.items():
            state[key].copy_(shadow_value)
        model.load_state_dict(state)
        return backup

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        model.load_state_dict(backup)


def _split_episode_indices(num_episodes: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Split complete episodes into train and validation sets."""
    all_indices = list(range(num_episodes))
    if val_ratio <= 0.0 or num_episodes < 2:
        return all_indices, []

    val_size = max(1, int(round(num_episodes * val_ratio)))
    val_size = min(val_size, num_episodes - 1)
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(num_episodes, generator=generator).tolist()
    val_indices = sorted(int(index) for index in shuffled[:val_size])
    train_indices = sorted(int(index) for index in shuffled[val_size:])
    return train_indices, val_indices


def _diffusion_loss(
    batch: dict[str, torch.Tensor],
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    device: torch.device,
    num_diffusion_steps: int,
    use_lowdim_obs: bool,
) -> tuple[torch.Tensor, int]:
    image_cond = batch["images"].to(device, non_blocking=True)
    obs_cond = batch["obs"].to(device, non_blocking=True) if use_lowdim_obs else None
    clean_actions = batch["actions"].to(device, non_blocking=True)
    noise = torch.randn_like(clean_actions)
    timesteps = torch.randint(
        low=0,
        high=num_diffusion_steps,
        size=(clean_actions.shape[0],),
        device=device,
        dtype=torch.long,
    )
    noisy_actions = scheduler.add_noise(clean_actions, noise, timesteps)
    noise_pred = model(noisy_actions, timesteps, image_cond, obs_cond)
    return F.mse_loss(noise_pred, noise), clean_actions.shape[0]


def _train_one_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_diffusion_steps: int,
    grad_clip: float,
    ema_model: EMAModel | None,
    use_lowdim_obs: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        loss, batch_size = _diffusion_loss(batch, model, scheduler, device, num_diffusion_steps, use_lowdim_obs)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema_model is not None:
            ema_model.update(model)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.inference_mode()
def _evaluate_loss(
    dataloader: DataLoader,
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    device: torch.device,
    num_diffusion_steps: int,
    use_lowdim_obs: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        loss, batch_size = _diffusion_loss(batch, model, scheduler, device, num_diffusion_steps, use_lowdim_obs)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a visual diffusion policy.")
    parser.add_argument("--dataset", type=str, required=True, help="HDF5 dataset from collect_visual_demos.py.")
    parser.add_argument("--output", type=str, default=None, help="Checkpoint path. Defaults to runs/visual_diffusion/...")
    parser.add_argument("--device", type=str, default="auto", help="Torch device, for example auto, cuda, cuda:0, cpu.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of episodes used for validation.")
    parser.add_argument("--lr", type=float, default=1.0e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1.0e-6, help="AdamW weight decay.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--cache_images", action="store_true", default=False, help="Load all uint8 images into RAM.")
    parser.add_argument("--obs_horizon", type=int, default=2, help="Number of visual/obs steps used as condition.")
    parser.add_argument("--pred_horizon", type=int, default=16, help="Number of future actions predicted by diffusion.")
    parser.add_argument("--action_horizon", type=int, default=8, help="Number of sampled actions executed before replanning.")
    parser.add_argument("--image_feature_dim", type=int, default=128, help="CNN feature dimension per image.")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Denoising MLP hidden dimension.")
    parser.add_argument("--num_layers", type=int, default=4, help="Denoising MLP hidden layers.")
    parser.add_argument("--time_embed_dim", type=int, default=64, help="Diffusion timestep embedding dimension.")
    parser.add_argument("--num_diffusion_steps", type=int, default=100, help="DDPM diffusion steps.")
    parser.add_argument("--beta_start", type=float, default=1.0e-4, help="DDPM beta schedule start.")
    parser.add_argument("--beta_end", type=float, default=2.0e-2, help="DDPM beta schedule end.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--save_every", type=int, default=25, help="Save an intermediate checkpoint every N epochs.")
    parser.add_argument("--log_csv", type=str, default=None, help="CSV path for train/validation loss.")
    parser.add_argument("--ema_decay", type=float, default=0.995, help="EMA decay for checkpoint weights. 0 disables EMA.")
    parser.add_argument("--image_only", action="store_true", default=False, help="Do not condition on low-dimensional obs.")
    args = parser.parse_args()
    args.use_lowdim_obs = not args.image_only

    if args.action_horizon > args.pred_horizon:
        raise ValueError("action_horizon must be <= pred_horizon.")
    if args.val_ratio < 0.0 or args.val_ratio >= 1.0:
        raise ValueError("val_ratio must be in [0, 1).")
    if args.ema_decay < 0.0 or args.ema_decay >= 1.0:
        raise ValueError("ema_decay must be in [0, 1).")

    set_seed(args.seed)
    device = resolve_torch_device(args.device)
    output_path = args.output or _default_output_path()

    source_dataset = VisualSequenceDataset(
        dataset_path=args.dataset,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        cache_images=False,
    )
    train_episode_indices, val_episode_indices = _split_episode_indices(
        source_dataset.num_episodes, args.val_ratio, args.seed
    )
    train_dataset = VisualSequenceDataset(
        dataset_path=args.dataset,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        episode_indices=train_episode_indices,
        cache_images=args.cache_images,
    )
    val_dataset = None
    if val_episode_indices:
        val_dataset = VisualSequenceDataset(
            dataset_path=args.dataset,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            episode_indices=val_episode_indices,
            normalizer=train_dataset.normalizer,
            cache_images=args.cache_images,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

    config = visual_checkpoint_config_from_args(args, source_dataset)
    config["train_episode_indices"] = train_episode_indices
    config["val_episode_indices"] = val_episode_indices
    config["ema_decay"] = float(args.ema_decay)
    model = build_visual_model_from_config(config).to(device)
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema_model = EMAModel(model, args.ema_decay) if args.ema_decay > 0.0 else None

    print(f"[INFO]: dataset={os.path.abspath(args.dataset)}")
    print(
        f"[INFO]: episodes={source_dataset.num_episodes}, windows={len(source_dataset)}, "
        f"image_shape={source_dataset.image_shape}, obs_dim={source_dataset.obs_dim}, "
        f"action_dim={source_dataset.action_dim}"
    )
    print(
        f"[INFO]: train_episodes={len(train_episode_indices)}, val_episodes={len(val_episode_indices)}, "
        f"train_windows={len(train_dataset)}, val_windows={len(val_dataset) if val_dataset is not None else 0}"
    )
    print(f"[INFO]: device={device}, output={os.path.abspath(output_path)}")
    print(f"[INFO]: use_lowdim_obs={args.use_lowdim_obs}, ema_decay={args.ema_decay}")

    best_loss = float("inf")
    best_path = os.path.join(os.path.dirname(output_path), "best.pt")
    log_csv = args.log_csv or os.path.join(os.path.dirname(output_path), "metrics.csv")
    os.makedirs(os.path.dirname(os.path.abspath(log_csv)), exist_ok=True)
    with open(log_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["epoch", "train_loss", "val_loss", "best_metric"])
        writer.writeheader()

    train_loss = float("inf")
    val_loss = None
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(
            train_loader,
            model,
            scheduler,
            optimizer,
            device,
            args.num_diffusion_steps,
            args.grad_clip,
            ema_model,
            args.use_lowdim_obs,
        )
        val_loss = None
        eval_backup = None
        if ema_model is not None:
            eval_backup = ema_model.apply_to(model)
        if val_loader is not None:
            val_loss = _evaluate_loss(
                val_loader, model, scheduler, device, args.num_diffusion_steps, args.use_lowdim_obs
            )
        if ema_model is not None and eval_backup is not None:
            ema_model.restore(model, eval_backup)
        best_metric = val_loss if val_loss is not None else train_loss
        val_text = f", val_loss={val_loss:.6f}" if val_loss is not None else ""
        print(f"[INFO]: epoch {epoch:04d}/{args.epochs:04d} train_loss={train_loss:.6f}{val_text}")

        with open(log_csv, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["epoch", "train_loss", "val_loss", "best_metric"])
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": "" if val_loss is None else val_loss,
                    "best_metric": best_metric,
                }
            )

        if best_metric < best_loss:
            best_loss = best_metric
            model_state = ema_model.state_dict(model) if ema_model is not None else model.state_dict()
            _save_checkpoint(best_path, model_state, train_dataset.normalizer, config, epoch, train_loss, val_loss)

        if args.save_every > 0 and epoch % args.save_every == 0:
            intermediate_path = os.path.join(os.path.dirname(output_path), f"epoch_{epoch:04d}.pt")
            model_state = ema_model.state_dict(model) if ema_model is not None else model.state_dict()
            _save_checkpoint(intermediate_path, model_state, train_dataset.normalizer, config, epoch, train_loss, val_loss)

    model_state = ema_model.state_dict(model) if ema_model is not None else model.state_dict()
    _save_checkpoint(output_path, model_state, train_dataset.normalizer, config, args.epochs, train_loss, val_loss)
    print(f"[INFO]: saved final checkpoint: {os.path.abspath(output_path)}")
    print(f"[INFO]: saved best checkpoint: {os.path.abspath(best_path)}")
    print(f"[INFO]: saved metrics CSV: {os.path.abspath(log_csv)}")
    source_dataset.close()
    train_dataset.close()
    if val_dataset is not None:
        val_dataset.close()


if __name__ == "__main__":
    main()
