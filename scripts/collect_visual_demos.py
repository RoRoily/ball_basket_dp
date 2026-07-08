# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect rendered RGB + low-dimensional scripted demonstrations into HDF5."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect visual BallBasket-LowDim-v0 demonstrations.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_demos", type=int, default=10, help="Number of episodes to collect.")
parser.add_argument("--task", type=str, default="BallBasket-LowDim-v0", help="Name of the task.")
parser.add_argument("--steps", type=int, default=430, help="Steps per episode.")
parser.add_argument("--image_size", type=int, default=96, help="Square RGB image size stored in the dataset.")
parser.add_argument("--position_scale", type=float, default=0.04, help="Position scale used by the IK action term.")
parser.add_argument("--mode", choices=["auto", "drop", "throw"], default="auto", help="Expert behavior.")
parser.add_argument("--reachable_xy_radius", type=float, default=0.68, help="Radius for direct drop reachability.")
parser.add_argument("--virtual_grasp", action="store_true", default=False, help="Use conditional virtual grasp.")
parser.add_argument("--grasp_distance", type=float, default=0.12, help="Maximum hand-ball distance for virtual attach.")
parser.add_argument("--grasp_offset_z", type=float, default=-0.075, help="Ball offset from the hand while grasped.")
parser.add_argument("--throw_time", type=float, default=0.55, help="Ballistic flight time for throw velocity.")
parser.add_argument("--throw_speed_scale", type=float, default=1.0, help="Scale factor on computed throw velocity.")
parser.add_argument("--keep_success_only", action="store_true", default=False, help="Only save successful episodes.")
parser.add_argument("--min_attach_count", type=int, default=0, help="Minimum virtual attach count to save an episode.")
parser.add_argument("--grasp_offset_x", type=float, default=0.0, help="End-effector x offset from ball center.")
parser.add_argument("--grasp_offset_y", type=float, default=0.0, help="End-effector y offset from ball center.")
parser.add_argument("--above_ball_z", type=float, default=0.28, help="Above-ball target z offset.")
parser.add_argument("--descend_z", type=float, default=0.09, help="Descend target z offset.")
parser.add_argument("--close_z", type=float, default=0.09, help="Close-gripper target z offset.")
parser.add_argument("--lift_z", type=float, default=0.34, help="Lift target z offset.")
parser.add_argument(
    "--max_final_distance",
    type=float,
    default=-1.0,
    help="Maximum final ball-to-basket distance. Negative disables this filter.",
)
parser.add_argument(
    "--max_demos_attempts",
    type=int,
    default=0,
    help="Maximum attempts. 0 chooses num_demos or num_demos * 10 when filters are active.",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output HDF5 path. Defaults to datasets/ball_basket_visual/<timestamp>.hdf5.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import ball_basket_dp.tasks  # noqa: F401
from expert_policy import (  # noqa: E402
    BALL_POS_SLICE,
    BASKET_POS_SLICE,
    ExpertGraspTuning,
    carry_ball_with_hand,
    expert_action,
    grasp_candidate_mask,
    plan_mode,
    policy_obs,
    release_throw_ball,
    stage_at_step,
)
from visual_diffusion import render_rgb_frame  # noqa: E402


def _default_output_path() -> str:
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join("datasets", "ball_basket_visual", f"{run_name}.hdf5")


def _has_quality_filters() -> bool:
    return args_cli.keep_success_only or args_cli.min_attach_count > 0 or args_cli.max_final_distance >= 0.0


def _max_attempts() -> int:
    if args_cli.max_demos_attempts > 0:
        return args_cli.max_demos_attempts
    return args_cli.num_demos * 10 if _has_quality_filters() else args_cli.num_demos


def _episode_quality(
    image_traj: list[np.ndarray],
    obs_traj: list[np.ndarray],
    action_traj: list[np.ndarray],
    success: bool,
    attach_count: int,
    final_distance: float,
) -> tuple[bool, list[str]]:
    reasons = []
    if not image_traj or not obs_traj or not action_traj:
        reasons.append("empty")
    else:
        if not np.isfinite(np.stack(obs_traj, axis=0)).all():
            reasons.append("nonfinite_obs")
        if not np.isfinite(np.stack(action_traj, axis=0)).all():
            reasons.append("nonfinite_actions")
        first_shape = image_traj[0].shape
        if any(image.shape != first_shape for image in image_traj):
            reasons.append("inconsistent_image_shape")
    if args_cli.keep_success_only and not success:
        reasons.append("not_success")
    if attach_count < args_cli.min_attach_count:
        reasons.append("low_attach_count")
    if args_cli.max_final_distance >= 0.0 and final_distance > args_cli.max_final_distance:
        reasons.append("far_final_distance")
    return len(reasons) == 0, reasons


def main():
    if args_cli.image_size <= 0:
        raise ValueError("image_size must be positive.")

    output_path = args_cli.output or _default_output_path()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    grasp_tuning = ExpertGraspTuning(
        grasp_offset_x=args_cli.grasp_offset_x,
        grasp_offset_y=args_cli.grasp_offset_y,
        above_ball_z=args_cli.above_ball_z,
        descend_z=args_cli.descend_z,
        close_z=args_cli.close_z,
        lift_z=args_cli.lift_z,
    )

    image_episodes = []
    obs_episodes = []
    action_episodes = []
    success_flags = []
    plan_names = []
    attach_counts = []
    episode_lengths = []
    final_distances = []
    final_ball_positions = []
    final_basket_positions = []
    attempt_indices = []
    rejected_count = 0
    reject_reasons: dict[str, int] = {}
    max_attempts = _max_attempts()

    with torch.inference_mode():
        for attempt_idx in range(max_attempts):
            if len(obs_episodes) >= args_cli.num_demos:
                break
            obs = env.reset()
            first_obs = policy_obs(obs)
            plan = plan_mode(first_obs, args_cli.mode, args_cli.reachable_xy_radius)
            grasped = torch.zeros((1,), dtype=torch.bool, device=first_obs.device)
            throw_released = False
            attach_count = 0

            image_traj = []
            obs_traj = []
            action_traj = []
            terminated_seen = False
            truncated_seen = False
            final_obs_tensor = first_obs

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

                image = render_rgb_frame(env, args_cli.image_size)
                actions = expert_action(
                    obs_tensor,
                    step,
                    plan,
                    args_cli.position_scale,
                    args_cli.reachable_xy_radius,
                    grasp_tuning,
                )
                image_traj.append(image)
                obs_traj.append(obs_tensor[0].detach().cpu().numpy().astype(np.float32))
                action_traj.append(actions[0].detach().cpu().numpy().astype(np.float32))
                obs, _, terminated, truncated, _ = env.step(actions)
                terminated_seen = terminated_seen or bool(terminated[0].item())
                truncated_seen = truncated_seen or bool(truncated[0].item())
                final_obs_tensor = policy_obs(obs)
                if bool(terminated[0].item()) or bool(truncated[0].item()):
                    break

            final_obs_np = final_obs_tensor[0].detach().cpu().numpy().astype(np.float32)
            final_ball_pos = final_obs_np[BALL_POS_SLICE]
            final_basket_pos = final_obs_np[BASKET_POS_SLICE]
            final_distance = float(np.linalg.norm(final_ball_pos - final_basket_pos))
            keep_episode, reasons = _episode_quality(
                image_traj,
                obs_traj,
                action_traj,
                terminated_seen,
                attach_count,
                final_distance,
            )
            if not keep_episode:
                rejected_count += 1
                for reason in reasons:
                    reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                print(
                    f"[INFO]: rejected attempt {attempt_idx + 1}/{max_attempts} plan={plan} "
                    f"steps={len(obs_traj)} success={terminated_seen} truncated={truncated_seen} "
                    f"attaches={attach_count} final_dist={final_distance:.4f} reasons={','.join(reasons)}"
                )
                continue

            image_episodes.append(np.stack(image_traj, axis=0).astype(np.uint8))
            obs_episodes.append(np.stack(obs_traj, axis=0))
            action_episodes.append(np.stack(action_traj, axis=0))
            success_flags.append(terminated_seen)
            plan_names.append(plan)
            attach_counts.append(attach_count)
            episode_lengths.append(len(obs_traj))
            final_distances.append(final_distance)
            final_ball_positions.append(final_ball_pos)
            final_basket_positions.append(final_basket_pos)
            attempt_indices.append(attempt_idx)
            print(
                f"[INFO]: saved visual demo {len(obs_episodes)}/{args_cli.num_demos} "
                f"attempt={attempt_idx + 1}/{max_attempts} plan={plan} steps={len(obs_traj)} "
                f"success={terminated_seen} truncated={truncated_seen} attaches={attach_count} "
                f"final_dist={final_distance:.4f}"
            )

    if not obs_episodes:
        env.close()
        raise RuntimeError(
            f"No demonstrations passed the filters after {max_attempts} attempts. "
            f"Rejected reasons: {reject_reasons}"
        )
    if len(obs_episodes) < args_cli.num_demos:
        print(
            f"[WARN]: Requested {args_cli.num_demos} demos but only collected {len(obs_episodes)} "
            f"after {max_attempts} attempts. Consider increasing --max_demos_attempts or relaxing filters."
        )

    episode_lengths = np.asarray(episode_lengths, dtype=np.int64)
    episode_ends = np.cumsum(episode_lengths)
    image_data = np.concatenate(image_episodes, axis=0)
    obs_data = np.concatenate(obs_episodes, axis=0)
    action_data = np.concatenate(action_episodes, axis=0)

    with h5py.File(output_path, "w") as h5_file:
        data_group = h5_file.create_group("data")
        data_group.create_dataset(
            "images",
            data=image_data,
            compression="gzip",
            compression_opts=4,
            chunks=(1, args_cli.image_size, args_cli.image_size, 3),
        )
        data_group.create_dataset("obs", data=obs_data, compression="gzip")
        data_group.create_dataset("actions", data=action_data, compression="gzip")
        data_group.create_dataset("success", data=np.asarray(success_flags, dtype=np.bool_))
        data_group.create_dataset("plan", data=np.asarray(plan_names, dtype=h5py.string_dtype()))
        data_group.create_dataset("attach_count", data=np.asarray(attach_counts, dtype=np.int64))
        data_group.create_dataset("episode_length", data=episode_lengths)
        data_group.create_dataset("final_ball_to_basket_distance", data=np.asarray(final_distances, dtype=np.float32))
        data_group.create_dataset("final_ball_pos", data=np.asarray(final_ball_positions, dtype=np.float32))
        data_group.create_dataset("final_basket_pos", data=np.asarray(final_basket_positions, dtype=np.float32))
        data_group.create_dataset("attempt_index", data=np.asarray(attempt_indices, dtype=np.int64))
        meta_group = h5_file.create_group("meta")
        meta_group.create_dataset("episode_ends", data=episode_ends)
        meta_group.attrs["task"] = args_cli.task
        meta_group.attrs["obs_dim"] = obs_data.shape[1]
        meta_group.attrs["action_dim"] = action_data.shape[1]
        meta_group.attrs["image_height"] = image_data.shape[1]
        meta_group.attrs["image_width"] = image_data.shape[2]
        meta_group.attrs["image_channels"] = image_data.shape[3]
        meta_group.attrs["requested_demos"] = args_cli.num_demos
        meta_group.attrs["attempts"] = len(attempt_indices) + rejected_count
        meta_group.attrs["rejected_count"] = rejected_count
        meta_group.attrs["keep_success_only"] = args_cli.keep_success_only
        meta_group.attrs["min_attach_count"] = args_cli.min_attach_count
        meta_group.attrs["max_final_distance"] = args_cli.max_final_distance
        meta_group.attrs["visual_input"] = True
        meta_group.attrs["reject_reasons_json"] = json.dumps(reject_reasons, sort_keys=True)

    print(f"[INFO]: Wrote {len(obs_episodes)} visual demos to: {os.path.abspath(output_path)}")
    print(
        f"[INFO]: Attempts={len(attempt_indices) + rejected_count}, "
        f"rejected={rejected_count}, reasons={reject_reasons}"
    )
    print(f"[INFO]: image shape={image_data.shape}, obs shape={obs_data.shape}, action shape={action_data.shape}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
