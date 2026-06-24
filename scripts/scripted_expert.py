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
EE_VEL_SLICE = slice(25, 28)
BALL_POS_SLICE = slice(28, 31)
BALL_VEL_SLICE = slice(31, 34)
BASKET_POS_SLICE = slice(34, 37)


def _policy_obs(obs) -> torch.Tensor:
    if isinstance(obs, tuple):
        obs = obs[0]
    return obs["policy"] if isinstance(obs, dict) else obs


def _basket_reachable(basket_pos: torch.Tensor, reachable_xy_radius: float) -> bool:
    return bool(torch.linalg.norm(basket_pos[0, :2]).item() <= reachable_xy_radius)


def _plan_mode(obs: torch.Tensor, requested_mode: str, reachable_xy_radius: float) -> str:
    if requested_mode != "auto":
        return requested_mode
    return "drop" if _basket_reachable(obs[:, BASKET_POS_SLICE], reachable_xy_radius) else "throw"


def _stage(step: int, plan: str) -> str:
    if step < 90:
        return "above_ball"
    if step < 150:
        return "descend"
    if step < 190:
        return "close"
    if step < 250:
        return "lift"
    if plan == "throw":
        if step < 310:
            return "windup"
        if step < 350:
            return "swing"
        if step < 370:
            return "release_throw"
    else:
        if step < 340:
            return "above_basket"
        if step < 380:
            return "release_drop"
    return "hold"


def _basket_direction(basket_pos: torch.Tensor) -> torch.Tensor:
    direction = basket_pos[:, :2].clone()
    norm = torch.linalg.norm(direction, dim=1, keepdim=True).clamp_min(1.0e-6)
    return direction / norm


def _throw_release_xy(basket_pos: torch.Tensor, reachable_xy_radius: float) -> torch.Tensor:
    direction = _basket_direction(basket_pos)
    release_radius = min(reachable_xy_radius, max(0.25, reachable_xy_radius - 0.04))
    return direction * release_radius


def _target_for_stage(
    stage: str, ball_pos: torch.Tensor, basket_pos: torch.Tensor, reachable_xy_radius: float
) -> torch.Tensor:
    if stage == "above_ball":
        return ball_pos + torch.tensor((0.0, 0.0, 0.28), device=ball_pos.device)
    if stage in ("descend", "close"):
        return ball_pos + torch.tensor((0.0, 0.0, 0.09), device=ball_pos.device)
    if stage == "lift":
        return ball_pos + torch.tensor((0.0, 0.0, 0.34), device=ball_pos.device)
    if stage == "windup":
        direction = _basket_direction(basket_pos)
        target = torch.zeros_like(ball_pos)
        target[:, :2] = _throw_release_xy(basket_pos, reachable_xy_radius) - 0.18 * direction
        target[:, 2] = 0.34
        return target
    if stage == "swing":
        target = torch.zeros_like(ball_pos)
        target[:, :2] = _throw_release_xy(basket_pos, reachable_xy_radius) + 0.10 * _basket_direction(basket_pos)
        target[:, 2] = 0.26
        return target
    if stage == "release_throw":
        target = torch.zeros_like(ball_pos)
        target[:, :2] = _throw_release_xy(basket_pos, reachable_xy_radius) + 0.13 * _basket_direction(basket_pos)
        target[:, 2] = 0.24
        return target
    if stage == "above_basket":
        return basket_pos + torch.tensor((0.0, 0.0, 0.34), device=basket_pos.device)
    if stage == "release_drop":
        return basket_pos + torch.tensor((0.0, 0.0, 0.22), device=basket_pos.device)
    return basket_pos + torch.tensor((0.0, 0.0, 0.26), device=basket_pos.device)


def _expert_action(obs: torch.Tensor, step: int, plan: str, position_scale: float, reachable_xy_radius: float) -> torch.Tensor:
    stage = _stage(step, plan)
    ee_pos = obs[:, EE_POS_SLICE]
    ball_pos = obs[:, BALL_POS_SLICE]
    basket_pos = obs[:, BASKET_POS_SLICE]

    target = _target_for_stage(stage, ball_pos, basket_pos, reachable_xy_radius)
    arm_cmd = torch.clamp((target - ee_pos) / position_scale, min=-1.0, max=1.0)

    gripper_cmd = torch.ones((obs.shape[0], 1), device=obs.device)
    if stage in ("close", "lift", "above_basket", "windup", "swing"):
        gripper_cmd.fill_(-1.0)

    return torch.cat((arm_cmd, gripper_cmd), dim=-1)


def _write_ball_state(env, pos_env: torch.Tensor, lin_vel: torch.Tensor) -> None:
    scene = env.unwrapped.scene
    ball = scene["ball"]
    pos_w = pos_env + scene.env_origins
    quat_w = torch.zeros((pos_w.shape[0], 4), device=pos_w.device)
    quat_w[:, 0] = 1.0
    ang_vel = torch.zeros_like(lin_vel)
    ball.write_root_pose_to_sim(torch.cat((pos_w, quat_w), dim=-1))
    ball.write_root_velocity_to_sim(torch.cat((lin_vel, ang_vel), dim=-1))


def _carry_ball_with_hand(env, obs: torch.Tensor) -> None:
    ee_pos = obs[:, EE_POS_SLICE]
    ee_vel = obs[:, EE_VEL_SLICE]
    grasp_offset = torch.tensor((0.0, 0.0, -0.075), device=obs.device).unsqueeze(0)
    _write_ball_state(env, ee_pos + grasp_offset, ee_vel)


def _release_throw_ball(env, obs: torch.Tensor, throw_time: float, throw_speed_scale: float) -> None:
    ee_pos = obs[:, EE_POS_SLICE]
    basket_pos = obs[:, BASKET_POS_SLICE]
    release_pos = ee_pos + torch.tensor((0.0, 0.0, -0.075), device=obs.device).unsqueeze(0)
    target_pos = basket_pos + torch.tensor((0.0, 0.0, 0.08), device=obs.device).unsqueeze(0)
    gravity = torch.tensor((0.0, 0.0, -9.81), device=obs.device).unsqueeze(0)
    velocity = (target_pos - release_pos - 0.5 * gravity * throw_time * throw_time) / throw_time
    _write_ball_state(env, release_pos, velocity * throw_speed_scale)


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
    first_obs = _policy_obs(obs)
    plan = _plan_mode(first_obs, args_cli.mode, args_cli.reachable_xy_radius)
    print(f"[INFO]: Expert mode: {plan}")
    if args_cli.virtual_grasp:
        print("[INFO]: Virtual grasp is enabled.")
    terminated_count = 0
    truncated_count = 0
    throw_released = False

    with torch.inference_mode():
        for step in range(args_cli.steps):
            obs_tensor = _policy_obs(obs)
            stage = _stage(step, plan)
            if args_cli.virtual_grasp and stage in ("close", "lift", "above_basket", "windup", "swing"):
                _carry_ball_with_hand(env, obs_tensor)
            if args_cli.virtual_grasp and stage == "release_throw" and not throw_released:
                _release_throw_ball(env, obs_tensor, args_cli.throw_time, args_cli.throw_speed_scale)
                throw_released = True

            actions = _expert_action(obs_tensor, step, plan, args_cli.position_scale, args_cli.reachable_xy_radius)
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
