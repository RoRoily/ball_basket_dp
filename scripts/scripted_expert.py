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
parser.add_argument(
    "--mode",
    choices=["auto", "drop", "throw"],
    default="auto",
    help="Expert behavior. Auto drops when the basket is reachable and throws otherwise.",
)
parser.add_argument(
    "--reachable_xy_radius",
    type=float,
    default=0.68,
    help="Basket xy distance from robot base considered reachable for a direct drop.",
)
parser.add_argument(
    "--virtual_grasp",
    action="store_true",
    default=False,
    help="Teleport the ball with the end-effector while grasped, then release it. Useful before contact grasping is tuned.",
)
parser.add_argument("--throw_time", type=float, default=0.55, help="Ballistic flight time used to compute throw velocity.")
parser.add_argument("--throw_speed_scale", type=float, default=1.0, help="Scale factor on computed throw velocity.")
parser.add_argument("--grasp_distance", type=float, default=0.12, help="Maximum hand-ball distance for virtual attach.")
parser.add_argument("--grasp_offset_z", type=float, default=-0.075, help="Ball offset from the hand while virtually grasped.")
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
from expert_policy import (  # noqa: E402
    carry_ball_with_hand,
    expert_action,
    grasp_candidate_mask,
    plan_mode,
    policy_obs,
    release_throw_ball,
    stage_at_step,
)


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
    first_obs = policy_obs(obs)
    plan = plan_mode(first_obs, args_cli.mode, args_cli.reachable_xy_radius)
    print(f"[INFO]: Expert mode: {plan}")
    if args_cli.virtual_grasp:
        print("[INFO]: Conditional virtual grasp is enabled.")
    terminated_count = 0
    truncated_count = 0
    attach_count = 0
    throw_released = False
    grasped = torch.zeros((args_cli.num_envs,), dtype=torch.bool, device=first_obs.device)

    with torch.inference_mode():
        for step in range(args_cli.steps):
            obs_tensor = policy_obs(obs)
            stage = stage_at_step(step, plan)
            if args_cli.virtual_grasp:
                newly_grasped = grasp_candidate_mask(obs_tensor, stage, args_cli.grasp_distance) & ~grasped
                if bool(newly_grasped.any().item()):
                    attach_count += int(newly_grasped.sum().item())
                    grasped |= newly_grasped
                if stage in ("close", "lift", "above_basket", "windup", "swing"):
                    carry_ball_with_hand(env, obs_tensor, grasped, args_cli.grasp_offset_z)
            if args_cli.virtual_grasp and stage == "release_throw" and not throw_released:
                release_throw_ball(
                    env,
                    obs_tensor,
                    grasped,
                    args_cli.throw_time,
                    args_cli.throw_speed_scale,
                    args_cli.grasp_offset_z,
                )
                grasped[:] = False
                throw_released = True
            if args_cli.virtual_grasp and stage == "release_drop":
                grasped[:] = False

            actions = expert_action(obs_tensor, step, plan, args_cli.position_scale, args_cli.reachable_xy_radius)
            obs, _, terminated, truncated, _ = env.step(actions)
            terminated_count += int(terminated.sum().item())
            truncated_count += int(truncated.sum().item())

    print(f"[INFO]: Expert steps: {args_cli.steps}")
    print(f"[INFO]: Virtual attach count: {attach_count}")
    print(f"[INFO]: Terminated count: {terminated_count}")
    print(f"[INFO]: Truncated count: {truncated_count}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
