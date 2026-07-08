# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect a collected BallBasket low-dimensional HDF5 dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter

import h5py
import numpy as np

from expert_policy import BALL_POS_SLICE, BASKET_POS_SLICE


def _preview(values: np.ndarray, limit: int = 6) -> str:
    values = np.asarray(values).reshape(-1)
    shown = ", ".join(f"{value:.4g}" for value in values[:limit])
    return shown + (", ..." if values.shape[0] > limit else "")


def _optional_dataset(group: h5py.Group, name: str, default=None):
    return np.asarray(group[name]) if name in group else default


def _final_metadata(
    obs: np.ndarray, data_group: h5py.Group, episode_starts: np.ndarray, episode_ends: np.ndarray
) -> tuple:
    final_ball_pos = _optional_dataset(data_group, "final_ball_pos")
    final_basket_pos = _optional_dataset(data_group, "final_basket_pos")
    final_distance = _optional_dataset(data_group, "final_ball_to_basket_distance")

    if final_ball_pos is None:
        final_ball_pos = np.stack([obs[end - 1, BALL_POS_SLICE] for end in episode_ends], axis=0)
    if final_basket_pos is None:
        final_basket_pos = np.stack([obs[end - 1, BASKET_POS_SLICE] for end in episode_ends], axis=0)
    if final_distance is None:
        final_distance = np.linalg.norm(final_ball_pos - final_basket_pos, axis=1)
    return final_ball_pos, final_basket_pos, final_distance


def _safe_json_attr(attrs: dict, key: str):
    value = attrs.get(key)
    if value is None:
        return {}
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _plan_success_report(plans: list[str], success: np.ndarray) -> dict[str, str]:
    report = {}
    for plan in sorted(set(plans)):
        mask = np.asarray([item == plan for item in plans], dtype=np.bool_)
        count = int(mask.sum())
        successes = int(success[mask].sum())
        report[plan] = f"{successes}/{count} ({successes / max(count, 1):.3f})"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a BallBasket-LowDim-v0 HDF5 demonstration dataset.")
    parser.add_argument("dataset", type=str, help="Path to the HDF5 dataset produced by scripts/collect_demos.py.")
    args = parser.parse_args()

    with h5py.File(args.dataset, "r") as h5_file:
        data_group = h5_file["data"]
        obs = np.asarray(data_group["obs"])
        actions = np.asarray(data_group["actions"])
        success = np.asarray(data_group["success"], dtype=np.bool_)
        episode_ends = np.asarray(h5_file["meta/episode_ends"], dtype=np.int64)
        plans = [plan.decode("utf-8") if isinstance(plan, bytes) else str(plan) for plan in data_group["plan"][:]]
        attach_count = np.asarray(data_group["attach_count"], dtype=np.int64)
        attrs = dict(h5_file["meta"].attrs)
        episode_starts = np.concatenate(([0], episode_ends[:-1]))
        final_ball_pos, final_basket_pos, final_distance = _final_metadata(
            obs, data_group, episode_starts, episode_ends
        )

    episode_lengths = episode_ends - episode_starts
    success_count = int(success.sum())
    success_rate = success_count / max(len(success), 1)
    attach_counter = dict(sorted(Counter(attach_count.tolist()).items()))

    print(f"Dataset: {args.dataset}")
    print(f"Task: {attrs.get('task', '<unknown>')}")
    print(f"Episodes: {len(episode_ends)}")
    print(f"Transitions: {obs.shape[0]}")
    print(f"Obs shape: {obs.shape}")
    print(f"Action shape: {actions.shape}")
    print(
        "Episode length min/mean/max: "
        f"{episode_lengths.min()} / {episode_lengths.mean():.1f} / {episode_lengths.max()}"
    )
    print(f"Success: {success_count} / {len(success)} ({success_rate:.3f})")
    print(f"Plans: {dict(Counter(plans))}")
    print(f"Plan success: {_plan_success_report(plans, success)}")
    print(f"Attach count min/mean/max: {attach_count.min()} / {attach_count.mean():.1f} / {attach_count.max()}")
    print(f"Attach count distribution: {attach_counter}")
    print(
        "Final ball-to-basket distance min/mean/max: "
        f"{final_distance.min():.4f} / {final_distance.mean():.4f} / {final_distance.max():.4f}"
    )
    print(f"Final ball pos mean: {_preview(final_ball_pos.mean(axis=0), limit=3)}")
    print(f"Final basket pos mean: {_preview(final_basket_pos.mean(axis=0), limit=3)}")
    print(f"Obs finite: {np.isfinite(obs).all()}")
    print(f"Action finite: {np.isfinite(actions).all()}")
    print(f"Obs mean preview: {_preview(obs.mean(axis=0))}")
    print(f"Obs std preview: {_preview(obs.std(axis=0))}")
    print(f"Action mean preview: {_preview(actions.mean(axis=0))}")
    print(f"Action std preview: {_preview(actions.std(axis=0))}")
    if "source_dataset" in attrs:
        print(f"Source dataset: {attrs['source_dataset']}")
    if "rejected_count" in attrs:
        print(f"Rejected count: {attrs['rejected_count']}")
        print(f"Reject reasons: {_safe_json_attr(attrs, 'reject_reasons_json')}")


if __name__ == "__main__":
    main()
