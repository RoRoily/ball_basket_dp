# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Filter BallBasket low-dimensional HDF5 demonstrations by quality metadata."""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np

from expert_policy import BALL_POS_SLICE, BASKET_POS_SLICE


def _decode_strings(values) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _episode_ranges(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    starts = np.concatenate(([0], episode_ends[:-1]))
    return [(int(start), int(end)) for start, end in zip(starts, episode_ends)]


def _optional_dataset(group: h5py.Group, name: str, default):
    return np.asarray(group[name]) if name in group else default


def _final_metadata(obs: np.ndarray, data_group: h5py.Group, episode_ranges: list[tuple[int, int]]) -> tuple:
    num_episodes = len(episode_ranges)
    final_ball_pos = _optional_dataset(data_group, "final_ball_pos", None)
    final_basket_pos = _optional_dataset(data_group, "final_basket_pos", None)
    final_distance = _optional_dataset(data_group, "final_ball_to_basket_distance", None)

    if final_ball_pos is None:
        final_ball_pos = np.stack([obs[end - 1, BALL_POS_SLICE] for _, end in episode_ranges], axis=0)
    if final_basket_pos is None:
        final_basket_pos = np.stack([obs[end - 1, BASKET_POS_SLICE] for _, end in episode_ranges], axis=0)
    if final_distance is None:
        final_distance = np.linalg.norm(final_ball_pos - final_basket_pos, axis=1)

    return (
        np.asarray(final_ball_pos, dtype=np.float32).reshape(num_episodes, 3),
        np.asarray(final_basket_pos, dtype=np.float32).reshape(num_episodes, 3),
        np.asarray(final_distance, dtype=np.float32).reshape(num_episodes),
    )


def _episode_keep(
    episode_index: int,
    obs_episode: np.ndarray,
    action_episode: np.ndarray,
    success: np.ndarray,
    plans: list[str],
    attach_count: np.ndarray,
    final_distance: np.ndarray,
    args,
) -> tuple[bool, list[str]]:
    reasons = []
    if not np.isfinite(obs_episode).all():
        reasons.append("nonfinite_obs")
    if not np.isfinite(action_episode).all():
        reasons.append("nonfinite_actions")
    if args.keep_success_only and not bool(success[episode_index]):
        reasons.append("not_success")
    if int(attach_count[episode_index]) < args.min_attach_count:
        reasons.append("low_attach_count")
    if args.max_final_distance >= 0.0 and float(final_distance[episode_index]) > args.max_final_distance:
        reasons.append("far_final_distance")
    if args.plans and plans[episode_index] not in args.plans:
        reasons.append("plan_filtered")
    return len(reasons) == 0, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter BallBasket low-dimensional demonstration datasets.")
    parser.add_argument("--input", type=str, required=True, help="Input HDF5 dataset.")
    parser.add_argument("--output", type=str, required=True, help="Output filtered HDF5 dataset.")
    parser.add_argument(
        "--keep_success_only", action="store_true", default=False, help="Keep only successful episodes."
    )
    parser.add_argument("--min_attach_count", type=int, default=0, help="Minimum attach count.")
    parser.add_argument(
        "--max_final_distance",
        type=float,
        default=-1.0,
        help="Maximum final ball-to-basket distance. Negative disables this filter.",
    )
    parser.add_argument("--plans", nargs="*", default=None, help="Optional allowed plans, for example drop throw.")
    parser.add_argument("--max_episodes", type=int, default=0, help="Maximum selected episodes to write. 0 keeps all.")
    args = parser.parse_args()

    with h5py.File(args.input, "r") as input_file:
        data_group = input_file["data"]
        obs = np.asarray(data_group["obs"], dtype=np.float32)
        actions = np.asarray(data_group["actions"], dtype=np.float32)
        success = np.asarray(data_group["success"], dtype=np.bool_)
        plans = _decode_strings(data_group["plan"][:])
        attach_count = np.asarray(data_group["attach_count"], dtype=np.int64)
        episode_ends = np.asarray(input_file["meta/episode_ends"], dtype=np.int64)
        task = input_file["meta"].attrs.get("task", "<unknown>")
        source_attempt_index = _optional_dataset(data_group, "attempt_index", np.arange(len(episode_ends)))

        ranges = _episode_ranges(episode_ends)
        final_ball_pos, final_basket_pos, final_distance = _final_metadata(obs, data_group, ranges)

    selected_indices = []
    reject_reasons: dict[str, int] = {}
    for episode_index, (start, end) in enumerate(ranges):
        keep, reasons = _episode_keep(
            episode_index,
            obs[start:end],
            actions[start:end],
            success,
            plans,
            attach_count,
            final_distance,
            args,
        )
        if keep:
            if args.max_episodes > 0 and len(selected_indices) >= args.max_episodes:
                reject_reasons["max_episodes"] = reject_reasons.get("max_episodes", 0) + 1
                continue
            selected_indices.append(episode_index)
        else:
            for reason in reasons:
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

    if not selected_indices:
        raise RuntimeError(f"No episodes passed filters. Rejected reasons: {reject_reasons}")

    selected_obs = []
    selected_actions = []
    selected_lengths = []
    for episode_index in selected_indices:
        start, end = ranges[episode_index]
        selected_obs.append(obs[start:end])
        selected_actions.append(actions[start:end])
        selected_lengths.append(end - start)

    obs_data = np.concatenate(selected_obs, axis=0)
    action_data = np.concatenate(selected_actions, axis=0)
    episode_lengths = np.asarray(selected_lengths, dtype=np.int64)
    new_episode_ends = np.cumsum(episode_lengths)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with h5py.File(args.output, "w") as output_file:
        data_group = output_file.create_group("data")
        data_group.create_dataset("obs", data=obs_data, compression="gzip")
        data_group.create_dataset("actions", data=action_data, compression="gzip")
        data_group.create_dataset("success", data=success[selected_indices])
        selected_plans = np.asarray([plans[index] for index in selected_indices], dtype=h5py.string_dtype())
        data_group.create_dataset("plan", data=selected_plans)
        data_group.create_dataset("attach_count", data=attach_count[selected_indices])
        data_group.create_dataset("episode_length", data=episode_lengths)
        data_group.create_dataset("final_ball_to_basket_distance", data=final_distance[selected_indices])
        data_group.create_dataset("final_ball_pos", data=final_ball_pos[selected_indices])
        data_group.create_dataset("final_basket_pos", data=final_basket_pos[selected_indices])
        data_group.create_dataset("source_episode_index", data=np.asarray(selected_indices, dtype=np.int64))
        data_group.create_dataset("source_attempt_index", data=source_attempt_index[selected_indices])

        meta_group = output_file.create_group("meta")
        meta_group.create_dataset("episode_ends", data=new_episode_ends)
        meta_group.attrs["task"] = task
        meta_group.attrs["obs_dim"] = obs_data.shape[1]
        meta_group.attrs["action_dim"] = action_data.shape[1]
        meta_group.attrs["source_dataset"] = os.path.abspath(args.input)
        meta_group.attrs["source_episodes"] = len(ranges)
        meta_group.attrs["selected_episodes"] = len(selected_indices)
        meta_group.attrs["rejected_count"] = len(ranges) - len(selected_indices)
        meta_group.attrs["keep_success_only"] = args.keep_success_only
        meta_group.attrs["min_attach_count"] = args.min_attach_count
        meta_group.attrs["max_final_distance"] = args.max_final_distance
        meta_group.attrs["plans"] = json.dumps(args.plans or [])
        meta_group.attrs["reject_reasons_json"] = json.dumps(reject_reasons, sort_keys=True)

    print(f"[INFO]: input={os.path.abspath(args.input)}")
    print(f"[INFO]: output={os.path.abspath(args.output)}")
    print(f"[INFO]: selected={len(selected_indices)}/{len(ranges)} episodes")
    print(f"[INFO]: rejected={len(ranges) - len(selected_indices)} reasons={reject_reasons}")
    print(f"[INFO]: obs shape={obs_data.shape}, action shape={action_data.shape}")


if __name__ == "__main__":
    main()
