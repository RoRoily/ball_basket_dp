# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained low-dimensional diffusion policy in Isaac Lab."""

from __future__ import annotations

import argparse
import json
import os
from collections import deque
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a low-dimensional diffusion policy.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--task", type=str, default="BallBasket-LowDim-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint produced by train_lowdim_diffusion.py.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to evaluate.")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of vectorized rollout batches to evaluate.")
parser.add_argument("--steps", type=int, default=430, help="Maximum environment steps.")
parser.add_argument("--action_clip", type=float, default=1.0, help="Clamp deployed actions to [-clip, clip].")
parser.add_argument("--sample_clip", type=float, default=2.0, help="Clamp normalized samples during DDPM reverse steps.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video.")
parser.add_argument("--video_dir", type=str, default="videos/lowdim_diffusion", help="Directory for recorded videos.")
parser.add_argument("--metrics_path", type=str, default=None, help="Path to write rollout metrics JSON.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import ball_basket_dp.tasks  # noqa: F401
from lowdim_diffusion import (  # noqa: E402
    DDPMScheduler,
    build_model_from_config,
    normalize_obs,
    normalizer_to_device,
    sample_action_sequences,
    unnormalize_actions,
)
from expert_policy import policy_obs  # noqa: E402


def _torch_load(path: str, map_location: torch.device) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _make_obs_history(obs_tensor: torch.Tensor, obs_horizon: int) -> deque[torch.Tensor]:
    history: deque[torch.Tensor] = deque(maxlen=obs_horizon)
    for _ in range(obs_horizon):
        history.append(obs_tensor.clone())
    return history


def _history_tensor(history: deque[torch.Tensor]) -> torch.Tensor:
    return torch.stack(tuple(history), dim=1)


def _default_metrics_path(run_name: str) -> str:
    return os.path.join("runs", "lowdim_diffusion_eval", run_name, "metrics.json")


def _json_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def main():
    if args_cli.num_episodes < 1:
        raise ValueError("num_episodes must be >= 1.")
    device = torch.device(args_cli.device)
    checkpoint = _torch_load(args_cli.checkpoint, map_location=torch.device("cpu"))
    config = checkpoint["config"]
    model = build_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalizer = normalizer_to_device(checkpoint["normalizer"], device)
    scheduler = DDPMScheduler(
        num_train_timesteps=int(config["num_diffusion_steps"]),
        beta_start=float(config["beta_start"]),
        beta_end=float(config["beta_end"]),
        device=device,
    )

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args_cli.video:
        video_folder = os.path.abspath(os.path.join(args_cli.video_dir, run_name))
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.steps,
            disable_logger=True,
        )
        print(f"[INFO]: Recording diffusion policy video to: {video_folder}")
        if args_cli.num_episodes > 1:
            print("[INFO]: Video records the first rollout batch; JSON metrics include all rollout batches.")

    print(f"[INFO]: checkpoint={os.path.abspath(args_cli.checkpoint)}")
    print(f"[INFO]: trained epoch={checkpoint.get('epoch', '<unknown>')} loss={checkpoint.get('loss', '<unknown>')}")
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    action_horizon = int(config["action_horizon"])
    action_dim = int(config["action_dim"])
    if env.action_space.shape[-1] != action_dim:
        raise RuntimeError(
            f"Checkpoint action_dim={action_dim} but environment action_dim={env.action_space.shape[-1]}."
        )

    terminated_count = 0
    truncated_count = 0
    success_count = 0
    episode_success_counts = []
    episode_truncated_counts = []
    episode_step_counts = []

    with torch.inference_mode():
        for episode_idx in range(args_cli.num_episodes):
            obs = env.reset()
            obs_tensor = policy_obs(obs).to(device)
            if obs_tensor.shape[-1] != int(config["obs_dim"]):
                raise RuntimeError(
                    f"Checkpoint obs_dim={config['obs_dim']} but environment returns obs_dim={obs_tensor.shape[-1]}."
                )
            obs_history = _make_obs_history(obs_tensor, int(config["obs_horizon"]))
            success_seen = torch.zeros((args_cli.num_envs,), dtype=torch.bool, device=device)
            truncated_seen = torch.zeros((args_cli.num_envs,), dtype=torch.bool, device=device)
            step_count = 0

            while step_count < args_cli.steps:
                obs_cond = normalize_obs(_history_tensor(obs_history), normalizer)
                sampled_actions = sample_action_sequences(
                    model,
                    scheduler,
                    obs_cond,
                    action_dim=action_dim,
                    clip_sample=args_cli.sample_clip,
                )
                action_sequence = unnormalize_actions(sampled_actions, normalizer)
                if args_cli.action_clip > 0.0:
                    action_sequence = torch.clamp(action_sequence, -args_cli.action_clip, args_cli.action_clip)

                for horizon_step in range(action_horizon):
                    if step_count >= args_cli.steps:
                        break
                    actions = action_sequence[:, horizon_step, :]
                    obs, _, terminated, truncated, _ = env.step(actions)
                    obs_tensor = policy_obs(obs).to(device)
                    obs_history.append(obs_tensor.clone())
                    terminated_bool = terminated.to(device=device, dtype=torch.bool)
                    truncated_bool = truncated.to(device=device, dtype=torch.bool)
                    success_seen |= terminated_bool
                    truncated_seen |= truncated_bool
                    terminated_count += int(terminated_bool.sum().item())
                    truncated_count += int(truncated_bool.sum().item())
                    step_count += 1

            episode_success = int(success_seen.sum().item())
            episode_truncated = int(truncated_seen.sum().item())
            success_count += episode_success
            episode_success_counts.append(episode_success)
            episode_truncated_counts.append(episode_truncated)
            episode_step_counts.append(step_count)
            print(
                f"[INFO]: episode {episode_idx + 1}/{args_cli.num_episodes} "
                f"success={episode_success}/{args_cli.num_envs} truncated={episode_truncated}/{args_cli.num_envs}"
            )

    total_rollouts = args_cli.num_episodes * args_cli.num_envs
    success_rate = success_count / max(total_rollouts, 1)
    metrics_path = args_cli.metrics_path or _default_metrics_path(run_name)
    os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
    metrics = {
        "task": args_cli.task,
        "checkpoint": os.path.abspath(args_cli.checkpoint),
        "trained_epoch": _json_scalar(checkpoint.get("epoch")),
        "checkpoint_loss": _json_scalar(checkpoint.get("loss")),
        "checkpoint_train_loss": _json_scalar(checkpoint.get("train_loss")),
        "checkpoint_val_loss": _json_scalar(checkpoint.get("val_loss")),
        "num_envs": args_cli.num_envs,
        "num_episodes": args_cli.num_episodes,
        "max_steps": args_cli.steps,
        "total_rollouts": total_rollouts,
        "success_count": success_count,
        "success_rate": success_rate,
        "terminated_count": terminated_count,
        "truncated_count": truncated_count,
        "episode_success_counts": episode_success_counts,
        "episode_truncated_counts": episode_truncated_counts,
        "episode_step_counts": episode_step_counts,
    }
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"[INFO]: Evaluated rollout batches: {args_cli.num_episodes}")
    print(f"[INFO]: Total rollouts: {total_rollouts}")
    print(f"[INFO]: Success count: {success_count}")
    print(f"[INFO]: Success rate: {success_rate:.3f}")
    print(f"[INFO]: Terminated count: {terminated_count}")
    print(f"[INFO]: Truncated count: {truncated_count}")
    print(f"[INFO]: Wrote metrics JSON to: {os.path.abspath(metrics_path)}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
