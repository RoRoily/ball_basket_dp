# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a simple state-based Franka expert for the ball-basket task.

The expert is intentionally simple. It is meant for visual/debug validation and
as the first stepping stone toward demonstration collection, not as a robust
grasping solution yet.
"""

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted expert for BallBasket-LowDim-v0.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="BallBasket-LowDim-v0", help="Name of the task.")
parser.add_argument("--steps", type=int, default=420, help="Number of expert steps to run.")
parser.add_argument("--position_scale", type=float, default=0.04, help="Position scale used by the IK action term.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video.")
parser.add_argument("--video_dir", type=str, default="videos/scripted_expert", help="Directory for recorded videos.")
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

# Observation layout produced by BallBasket-LowDim-v0.
EE_POS_SLICE = slice(18, 21)
BALL_POS_SLICE = slice(25, 28)
BASKET_POS_SLICE = slice(31, 34)


def _policy_obs(obs) -> torch.Tensor:
    if isinstance(obs, tuple):
        obs = obs[0]
    return obs["policy"] if isinstance(obs, dict) else obs


def _stage(step: int) -> str:
    if step < 90:
        return "above_ball"
    if step < 150:
        return "descend"
    if step < 190:
        return "close"
    if step < 250:
        return "lift"
    if step < 340:
        return "above_basket"
    if step < 380:
        return "release"
    return "hold"


def _target_for_stage(stage: str, ball_pos: torch.Tensor, basket_pos: torch.Tensor) -> torch.Tensor:
    if stage == "above_ball":
        return ball_pos + torch.tensor((0.0, 0.0, 0.28), device=ball_pos.device)
    if stage in ("descend", "close"):
        return ball_pos + torch.tensor((0.0, 0.0, 0.09), device=ball_pos.device)
    if stage == "lift":
        return ball_pos + torch.tensor((0.0, 0.0, 0.34), device=ball_pos.device)
    if stage == "above_basket":
        return basket_pos + torch.tensor((0.0, 0.0, 0.34), device=basket_pos.device)
    if stage == "release":
        return basket_pos + torch.tensor((0.0, 0.0, 0.22), device=basket_pos.device)
    return basket_pos + torch.tensor((0.0, 0.0, 0.26), device=basket_pos.device)


def _expert_action(obs: torch.Tensor, step: int, position_scale: float) -> torch.Tensor:
    stage = _stage(step)
    ee_pos = obs[:, EE_POS_SLICE]
    ball_pos = obs[:, BALL_POS_SLICE]
    basket_pos = obs[:, BASKET_POS_SLICE]

    target = _target_for_stage(stage, ball_pos, basket_pos)
    arm_cmd = torch.clamp((target - ee_pos) / position_scale, min=-1.0, max=1.0)

    gripper_cmd = torch.ones((obs.shape[0], 1), device=obs.device)
    if stage in ("close", "lift", "above_basket"):
        gripper_cmd.fill_(-1.0)

    return torch.cat((arm_cmd, gripper_cmd), dim=-1)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_folder = os.path.abspath(os.path.join(args_cli.video_dir, run_name))
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.steps,
            disable_logger=True,
        )
        print(f"[INFO]: Recording scripted expert video to: {video_folder}")

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    obs = env.reset()
    terminated_count = 0
    truncated_count = 0

    with torch.inference_mode():
        for step in range(args_cli.steps):
            obs_tensor = _policy_obs(obs)
            actions = _expert_action(obs_tensor, step, args_cli.position_scale)
            obs, _, terminated, truncated, _ = env.step(actions)
            terminated_count += int(terminated.sum().item())
            truncated_count += int(truncated.sum().item())

    print(f"[INFO]: Expert steps: {args_cli.steps}")
    print(f"[INFO]: Terminated count: {terminated_count}")
    print(f"[INFO]: Truncated count: {truncated_count}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
