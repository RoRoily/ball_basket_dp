# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Finite-step smoke test for ball_basket_dp environments."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test an Isaac Lab environment for a fixed number of steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="BallBasket-LowDim-v0", help="Name of the task.")
parser.add_argument("--steps", type=int, default=100, help="Number of zero-action steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import ball_basket_dp.tasks  # noqa: F401


def _policy_obs_shape(obs) -> tuple[int, ...] | str:
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict) and "policy" in obs:
        return tuple(obs["policy"].shape)
    if hasattr(obs, "shape"):
        return tuple(obs.shape)
    return type(obs).__name__


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    obs = env.reset()
    print(f"[INFO]: Initial policy observation shape: {_policy_obs_shape(obs)}")

    total_terminated = 0
    total_truncated = 0
    reward_sum = 0.0

    with torch.inference_mode():
        for _ in range(args_cli.steps):
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            obs, reward, terminated, truncated, _ = env.step(actions)
            reward_sum += float(reward.mean().item())
            total_terminated += int(terminated.sum().item())
            total_truncated += int(truncated.sum().item())

    print(f"[INFO]: Final policy observation shape: {_policy_obs_shape(obs)}")
    print(f"[INFO]: Mean reward over {args_cli.steps} steps: {reward_sum / args_cli.steps:.4f}")
    print(f"[INFO]: Terminated count: {total_terminated}")
    print(f"[INFO]: Truncated count: {total_truncated}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
