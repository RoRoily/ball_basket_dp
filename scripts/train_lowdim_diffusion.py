# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a small low-dimensional diffusion policy from collected demonstrations."""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lowdim_diffusion import (
    DDPMScheduler,
    LowDimSequenceDataset,
    build_model_from_config,
    checkpoint_config_from_args,
    resolve_torch_device,
    set_seed,
)


def _default_output_path() -> str:
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join("runs", "lowdim_diffusion", run_name, "policy.pt")


def _save_checkpoint(
    output_path: str,
    model: torch.nn.Module,
    normalizer: dict[str, torch.Tensor],
    config: dict,
    epoch: int,
    loss: float,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    checkpoint = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "normalizer": {key: value.detach().cpu() for key, value in normalizer.items()},
        "config": config,
        "epoch": epoch,
        "loss": loss,
    }
    torch.save(checkpoint, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a low-dimensional diffusion policy.")
    parser.add_argument("--dataset", type=str, required=True, help="HDF5 dataset produced by scripts/collect_demos.py.")
    parser.add_argument("--output", type=str, default=None, help="Checkpoint path. Defaults to runs/lowdim_diffusion/...")
    parser.add_argument("--device", type=str, default="auto", help="Torch device, for example auto, cuda, cuda:0, cpu.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1.0e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1.0e-6, help="AdamW weight decay.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--obs_horizon", type=int, default=2, help="Number of observation steps used as condition.")
    parser.add_argument("--pred_horizon", type=int, default=16, help="Number of future actions predicted by diffusion.")
    parser.add_argument("--action_horizon", type=int, default=8, help="Number of sampled actions to execute before replanning.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Denoising MLP hidden dimension.")
    parser.add_argument("--num_layers", type=int, default=4, help="Denoising MLP hidden layers.")
    parser.add_argument("--time_embed_dim", type=int, default=64, help="Diffusion timestep embedding dimension.")
    parser.add_argument("--num_diffusion_steps", type=int, default=100, help="DDPM diffusion steps.")
    parser.add_argument("--beta_start", type=float, default=1.0e-4, help="DDPM beta schedule start.")
    parser.add_argument("--beta_end", type=float, default=2.0e-2, help="DDPM beta schedule end.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--save_every", type=int, default=25, help="Save an intermediate checkpoint every N epochs.")
    args = parser.parse_args()

    if args.action_horizon > args.pred_horizon:
        raise ValueError("action_horizon must be <= pred_horizon.")

    set_seed(args.seed)
    device = resolve_torch_device(args.device)
    output_path = args.output or _default_output_path()

    dataset = LowDimSequenceDataset(
        dataset_path=args.dataset,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    config = checkpoint_config_from_args(args, dataset.obs_dim, dataset.action_dim)
    model = build_model_from_config(config).to(device)
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"[INFO]: dataset={os.path.abspath(args.dataset)}")
    print(f"[INFO]: windows={len(dataset)}, obs_dim={dataset.obs_dim}, action_dim={dataset.action_dim}")
    print(f"[INFO]: device={device}, output={os.path.abspath(output_path)}")

    best_loss = float("inf")
    best_path = os.path.join(os.path.dirname(output_path), "best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            obs = batch["obs"].to(device, non_blocking=True)
            clean_actions = batch["actions"].to(device, non_blocking=True)
            noise = torch.randn_like(clean_actions)
            timesteps = torch.randint(
                low=0,
                high=args.num_diffusion_steps,
                size=(clean_actions.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_actions = scheduler.add_noise(clean_actions, noise, timesteps)
            noise_pred = model(noisy_actions, timesteps, obs)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch_size = clean_actions.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

        epoch_loss = total_loss / max(total_samples, 1)
        print(f"[INFO]: epoch {epoch:04d}/{args.epochs:04d} loss={epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            _save_checkpoint(best_path, model, dataset.normalizer, config, epoch, epoch_loss)

        if args.save_every > 0 and epoch % args.save_every == 0:
            intermediate_path = os.path.join(os.path.dirname(output_path), f"epoch_{epoch:04d}.pt")
            _save_checkpoint(intermediate_path, model, dataset.normalizer, config, epoch, epoch_loss)

    _save_checkpoint(output_path, model, dataset.normalizer, config, args.epochs, epoch_loss)
    print(f"[INFO]: saved final checkpoint: {os.path.abspath(output_path)}")
    print(f"[INFO]: saved best checkpoint: {os.path.abspath(best_path)}")


if __name__ == "__main__":
    main()
