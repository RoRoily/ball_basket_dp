# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Summarize low-dimensional diffusion training and evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _float_or_none(value: str | int | float | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _load_training_metrics(metrics_csv: Path) -> dict:
    rows = []
    with metrics_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows.extend(reader)
    if not rows:
        return {}

    parsed_rows = []
    for row in rows:
        train_loss = _float_or_none(row.get("train_loss"))
        val_loss = _float_or_none(row.get("val_loss"))
        best_metric = _float_or_none(row.get("best_metric"))
        parsed_rows.append(
            {
                "epoch": int(row["epoch"]),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_metric": best_metric,
            }
        )

    last = parsed_rows[-1]
    best = min(parsed_rows, key=lambda row: row["best_metric"] if row["best_metric"] is not None else float("inf"))
    return {
        "epochs": last["epoch"],
        "last_train_loss": last["train_loss"],
        "last_val_loss": last["val_loss"],
        "best_epoch": best["epoch"],
        "best_metric": best["best_metric"],
    }


def _load_eval_metrics(run_dir: Path) -> dict:
    eval_files = sorted(run_dir.glob("*.json"))
    if not eval_files:
        return {}
    # Prefer explicit eval files, then fall back to the newest JSON.
    preferred = [path for path in eval_files if "eval" in path.name or "metric" in path.name]
    metrics_path = preferred[-1] if preferred else eval_files[-1]
    with metrics_path.open("r", encoding="utf-8") as json_file:
        metrics = json.load(json_file)
    return {
        "eval_file": metrics_path.name,
        "success_rate": metrics.get("success_rate"),
        "success_count": metrics.get("success_count"),
        "total_rollouts": metrics.get("total_rollouts"),
    }


def _parse_run_identity(run_dir: Path) -> dict:
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as json_file:
            config = json.load(json_file)
        return {"demo_count": config.get("demo_count"), "seed": config.get("seed")}

    text = str(run_dir)
    demo_match = re.search(r"demos_(\d+)", text)
    seed_match = re.search(r"seed_(\d+)", text)
    return {
        "demo_count": int(demo_match.group(1)) if demo_match else None,
        "seed": int(seed_match.group(1)) if seed_match else None,
    }


def _format_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.5f}"


def _format_success(success_count, total_rollouts, success_rate) -> str:
    if success_rate is None:
        return "-"
    return f"{success_rate:.3f} ({success_count}/{total_rollouts})"


def _find_run_dirs(paths: list[str]) -> list[Path]:
    run_dirs = []
    for path_text in paths:
        path = Path(path_text)
        if path.is_file() and path.name == "metrics.csv":
            run_dirs.append(path.parent)
        elif path.is_dir() and (path / "metrics.csv").exists():
            run_dirs.append(path)
        elif path.is_dir():
            run_dirs.extend(metrics_csv.parent for metrics_csv in path.rglob("metrics.csv"))
    return sorted(set(run_dirs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize low-dimensional diffusion training/eval runs.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["runs/lowdim_diffusion"],
        help="Run directories, metrics.csv files, or parent directories to scan.",
    )
    args = parser.parse_args()

    run_dirs = _find_run_dirs(args.paths)
    if not run_dirs:
        print("No runs found.")
        return

    rows = []
    for run_dir in run_dirs:
        identity = _parse_run_identity(run_dir)
        train_metrics = _load_training_metrics(run_dir / "metrics.csv")
        eval_metrics = _load_eval_metrics(run_dir)
        rows.append(
            {
                "run": str(run_dir),
                **identity,
                **train_metrics,
                **eval_metrics,
            }
        )

    header = ["run", "demos", "seed", "epochs", "last_train", "last_val", "best_epoch", "best_metric", "success"]
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))
    for row in rows:
        print(
            " | ".join(
                [
                    row["run"],
                    str(row.get("demo_count", "-")),
                    str(row.get("seed", "-")),
                    str(row.get("epochs", "-")),
                    _format_float(row.get("last_train_loss")),
                    _format_float(row.get("last_val_loss")),
                    str(row.get("best_epoch", "-")),
                    _format_float(row.get("best_metric")),
                    _format_success(row.get("success_count"), row.get("total_rollouts"), row.get("success_rate")),
                ]
            )
        )


if __name__ == "__main__":
    main()
