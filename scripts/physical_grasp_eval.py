# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate physical, no-teleport grasp attempts for BallBasket-LowDim-v0."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Evaluate physical contact grasp attempts without virtual ball attachment."
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--task", type=str, default="BallBasket-LowDim-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of vectorized environments.")
parser.add_argument("--trials", type=int, default=1, help="Reset batches per grasp configuration.")
parser.add_argument(
    "--steps", type=int, default=260, help="Steps per grasp attempt. 260 covers approach, close, and lift."
)
parser.add_argument("--position_scale", type=float, default=0.04, help="Position scale used by the IK action term.")
parser.add_argument("--reachable_xy_radius", type=float, default=0.68, help="Reachability radius passed to the expert.")
parser.add_argument("--above_ball_z", type=float, default=0.28, help="Above-ball target z offset.")
parser.add_argument(
    "--descend_zs", type=float, nargs="+", default=[0.08, 0.09, 0.10], help="Descend z offsets to test."
)
parser.add_argument(
    "--close_zs",
    type=float,
    nargs="+",
    default=None,
    help="Close z offsets to test. Defaults to descend_zs.",
)
parser.add_argument("--lift_zs", type=float, nargs="+", default=[0.34], help="Lift z offsets to test.")
parser.add_argument(
    "--xy_offsets",
    type=str,
    nargs="+",
    default=["0.0,0.0"],
    help="End-effector xy offsets from ball center, formatted as x,y.",
)
parser.add_argument("--min_lift_height", type=float, default=0.12, help="Ball max z required for lift success.")
parser.add_argument("--hold_height", type=float, default=0.10, help="Final ball z required for hold success.")
parser.add_argument("--max_hold_distance", type=float, default=0.18, help="Final hand-ball distance for hold success.")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for CSV/JSON metrics.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the first rollout.")
parser.add_argument("--video_dir", type=str, default="videos/physical_grasp", help="Directory for recorded videos.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import ball_basket_dp.tasks  # noqa: F401
from expert_policy import (  # noqa: E402
    BALL_POS_SLICE,
    EE_POS_SLICE,
    ExpertGraspTuning,
    expert_action,
    policy_obs,
    stage_at_step,
)


def _parse_xy_offsets(values: list[str]) -> list[tuple[float, float]]:
    offsets = []
    for value in values:
        parts = value.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid xy offset '{value}'. Expected format x,y.")
        offsets.append((float(parts[0]), float(parts[1])))
    return offsets


def _default_output_dir() -> str:
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join("runs", "physical_grasp", run_name)


def _metric_row(config_id: int, tuning: ExpertGraspTuning, metrics: dict) -> dict:
    return {
        "config_id": config_id,
        "grasp_offset_x": tuning.grasp_offset_x,
        "grasp_offset_y": tuning.grasp_offset_y,
        "above_ball_z": tuning.above_ball_z,
        "descend_z": tuning.descend_z,
        "close_z": tuning.close_z,
        "lift_z": tuning.lift_z,
        **metrics,
    }


def _evaluate_config(env, tuning: ExpertGraspTuning, config_id: int) -> dict:
    max_ball_z_values = []
    final_ball_z_values = []
    min_hand_ball_dist_values = []
    final_hand_ball_dist_values = []
    lift_success_count = 0
    hold_success_count = 0
    total_rollouts = 0

    with torch.inference_mode():
        for trial_idx in range(args_cli.trials):
            obs = env.reset()
            max_ball_z = torch.full((args_cli.num_envs,), -1.0e6, device=args_cli.device)
            min_hand_ball_dist = torch.full((args_cli.num_envs,), 1.0e6, device=args_cli.device)

            for step in range(args_cli.steps):
                obs_tensor = policy_obs(obs)
                ball_pos = obs_tensor[:, BALL_POS_SLICE]
                ee_pos = obs_tensor[:, EE_POS_SLICE]
                hand_ball_dist = torch.linalg.norm(ball_pos - ee_pos, dim=1)
                max_ball_z = torch.maximum(max_ball_z, ball_pos[:, 2])
                if stage_at_step(step, "drop") in ("descend", "close", "lift"):
                    min_hand_ball_dist = torch.minimum(min_hand_ball_dist, hand_ball_dist)

                actions = expert_action(
                    obs_tensor,
                    step,
                    "drop",
                    args_cli.position_scale,
                    args_cli.reachable_xy_radius,
                    tuning,
                )
                obs, _, _, _, _ = env.step(actions)

            final_obs = policy_obs(obs)
            final_ball_pos = final_obs[:, BALL_POS_SLICE]
            final_ee_pos = final_obs[:, EE_POS_SLICE]
            final_hand_ball_dist = torch.linalg.norm(final_ball_pos - final_ee_pos, dim=1)
            lifted = max_ball_z >= args_cli.min_lift_height
            held = torch.logical_and(
                final_ball_pos[:, 2] >= args_cli.hold_height,
                final_hand_ball_dist <= args_cli.max_hold_distance,
            )

            lift_success_count += int(lifted.sum().item())
            hold_success_count += int(held.sum().item())
            total_rollouts += args_cli.num_envs
            max_ball_z_values.extend(max_ball_z.detach().cpu().numpy().tolist())
            final_ball_z_values.extend(final_ball_pos[:, 2].detach().cpu().numpy().tolist())
            min_hand_ball_dist_values.extend(min_hand_ball_dist.detach().cpu().numpy().tolist())
            final_hand_ball_dist_values.extend(final_hand_ball_dist.detach().cpu().numpy().tolist())
            print(
                f"[INFO]: config={config_id} trial={trial_idx + 1}/{args_cli.trials} "
                f"lift={int(lifted.sum().item())}/{args_cli.num_envs} "
                f"hold={int(held.sum().item())}/{args_cli.num_envs}"
            )

    return {
        "total_rollouts": total_rollouts,
        "lift_success_count": lift_success_count,
        "lift_success_rate": lift_success_count / max(total_rollouts, 1),
        "hold_success_count": hold_success_count,
        "hold_success_rate": hold_success_count / max(total_rollouts, 1),
        "mean_max_ball_z": float(np.mean(max_ball_z_values)),
        "mean_final_ball_z": float(np.mean(final_ball_z_values)),
        "mean_min_hand_ball_dist": float(np.mean(min_hand_ball_dist_values)),
        "mean_final_hand_ball_dist": float(np.mean(final_hand_ball_dist_values)),
    }


def main():
    output_dir = os.path.abspath(args_cli.output_dir or _default_output_dir())
    os.makedirs(output_dir, exist_ok=True)

    close_zs = args_cli.close_zs if args_cli.close_zs is not None else args_cli.descend_zs
    xy_offsets = _parse_xy_offsets(args_cli.xy_offsets)
    tunings = []
    for offset_x, offset_y in xy_offsets:
        for descend_z in args_cli.descend_zs:
            for close_z in close_zs:
                for lift_z in args_cli.lift_zs:
                    tunings.append(
                        ExpertGraspTuning(
                            grasp_offset_x=offset_x,
                            grasp_offset_y=offset_y,
                            above_ball_z=args_cli.above_ball_z,
                            descend_z=descend_z,
                            close_z=close_z,
                            lift_z=lift_z,
                        )
                    )

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.abspath(args_cli.video_dir),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.steps,
            disable_logger=True,
        )

    rows = []
    for config_id, tuning in enumerate(tunings):
        print(f"[INFO]: Evaluating config {config_id + 1}/{len(tunings)}: {tuning}")
        metrics = _evaluate_config(env, tuning, config_id)
        row = _metric_row(config_id, tuning, metrics)
        rows.append(row)

    csv_path = os.path.join(output_dir, "physical_grasp_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_row = max(rows, key=lambda row: (row["hold_success_rate"], row["lift_success_rate"], row["mean_max_ball_z"]))
    summary = {
        "task": args_cli.task,
        "num_configs": len(tunings),
        "num_envs": args_cli.num_envs,
        "trials": args_cli.trials,
        "steps": args_cli.steps,
        "min_lift_height": args_cli.min_lift_height,
        "hold_height": args_cli.hold_height,
        "max_hold_distance": args_cli.max_hold_distance,
        "best_config": best_row,
        "all_configs": rows,
    }
    json_path = os.path.join(output_dir, "physical_grasp_summary.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)

    print(f"[INFO]: Wrote CSV: {csv_path}")
    print(f"[INFO]: Wrote summary: {json_path}")
    print(f"[INFO]: Best config: {best_row}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
