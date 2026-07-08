# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect a collected BallBasket low-dimensional HDF5 dataset."""

from __future__ import annotations

import argparse
from collections import Counter

import h5py
import numpy as np


def _preview(values: np.ndarray, limit: int = 6) -> str:
    values = np.asarray(values).reshape(-1)
    shown = ", ".join(f"{value:.4g}" for value in values[:limit])
    return shown + (", ..." if values.shape[0] > limit else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a BallBasket-LowDim-v0 HDF5 demonstration dataset.")
    parser.add_argument("dataset", type=str, help="Path to the HDF5 dataset produced by scripts/collect_demos.py.")
    args = parser.parse_args()

    with h5py.File(args.dataset, "r") as h5_file:
        obs = np.asarray(h5_file["data/obs"])
        actions = np.asarray(h5_file["data/actions"])
        success = np.asarray(h5_file["data/success"], dtype=np.bool_)
        episode_ends = np.asarray(h5_file["meta/episode_ends"], dtype=np.int64)
        plans = [plan.decode("utf-8") if isinstance(plan, bytes) else str(plan) for plan in h5_file["data/plan"][:]]
        attach_count = np.asarray(h5_file["data/attach_count"], dtype=np.int64)
        attrs = dict(h5_file["meta"].attrs)

    episode_starts = np.concatenate(([0], episode_ends[:-1]))
    episode_lengths = episode_ends - episode_starts

    print(f"Dataset: {args.dataset}")
    print(f"Task: {attrs.get('task', '<unknown>')}")
    print(f"Episodes: {len(episode_ends)}")
    print(f"Transitions: {obs.shape[0]}")
    print(f"Obs shape: {obs.shape}")
    print(f"Action shape: {actions.shape}")
    print(f"Episode length min/mean/max: {episode_lengths.min()} / {episode_lengths.mean():.1f} / {episode_lengths.max()}")
    print(f"Success: {success.sum()} / {len(success)}")
    print(f"Plans: {dict(Counter(plans))}")
    print(f"Attach count min/mean/max: {attach_count.min()} / {attach_count.mean():.1f} / {attach_count.max()}")
    print(f"Obs finite: {np.isfinite(obs).all()}")
    print(f"Action finite: {np.isfinite(actions).all()}")
    print(f"Obs mean preview: {_preview(obs.mean(axis=0))}")
    print(f"Obs std preview: {_preview(obs.std(axis=0))}")
    print(f"Action mean preview: {_preview(actions.mean(axis=0))}")
    print(f"Action std preview: {_preview(actions.std(axis=0))}")


if __name__ == "__main__":
    main()
