# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create CSV summaries and plots for visual diffusion experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def _float_or_none(value) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


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


def _parse_run_identity(run_dir: Path) -> tuple[int | None, int | None, int | None, str | None]:
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as json_file:
            config = json.load(json_file)
        return (
            config.get("demo_count"),
            config.get("image_size"),
            config.get("seed"),
            config.get("condition"),
        )

    text = str(run_dir)
    demo_match = re.search(r"demos_(\d+)", text)
    image_match = re.search(r"image_(\d+)", text)
    seed_match = re.search(r"seed_(\d+)", text)
    condition = None
    if "image_only" in run_dir.parts:
        condition = "image_only"
    elif "rgb_lowdim" in run_dir.parts:
        condition = "rgb_lowdim"
    demo_count = int(demo_match.group(1)) if demo_match else None
    image_size = int(image_match.group(1)) if image_match else None
    seed = int(seed_match.group(1)) if seed_match else None
    return demo_count, image_size, seed, condition


def _load_training_rows(metrics_csv: Path) -> list[dict]:
    rows = []
    with metrics_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": _float_or_none(row.get("train_loss")),
                    "val_loss": _float_or_none(row.get("val_loss")),
                    "best_metric": _float_or_none(row.get("best_metric")),
                }
            )
    return rows


def _load_eval_metrics(run_dir: Path) -> dict:
    eval_files = sorted(path for path in run_dir.glob("*.json") if path.name != "run_config.json")
    preferred = [path for path in eval_files if "eval" in path.name or "metric" in path.name]
    if not preferred:
        return {}
    with preferred[-1].open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def _summarize_run(run_dir: Path) -> tuple[dict, list[dict]]:
    training_rows = _load_training_rows(run_dir / "metrics.csv")
    if not training_rows:
        raise RuntimeError(f"No rows in {run_dir / 'metrics.csv'}")
    demo_count, image_size, seed, condition = _parse_run_identity(run_dir)
    last = training_rows[-1]
    best = min(
        training_rows,
        key=lambda row: row["best_metric"] if row["best_metric"] is not None else float("inf"),
    )
    eval_metrics = _load_eval_metrics(run_dir)
    summary = {
        "run_dir": str(run_dir),
        "condition": condition,
        "image_size": image_size,
        "demo_count": demo_count,
        "seed": seed,
        "epochs": last["epoch"],
        "last_train_loss": last["train_loss"],
        "last_val_loss": last["val_loss"],
        "best_epoch": best["epoch"],
        "best_metric": best["best_metric"],
        "success_rate": eval_metrics.get("success_rate"),
        "success_count": eval_metrics.get("success_count"),
        "total_rollouts": eval_metrics.get("total_rollouts"),
    }
    return summary, training_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, variance**0.5


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        if row["condition"] is None or row["image_size"] is None or row["demo_count"] is None:
            continue
        key = (str(row["condition"]), int(row["image_size"]), int(row["demo_count"]))
        grouped[key].append(row)

    aggregate_rows = []
    for condition, image_size, demo_count in sorted(grouped):
        group = grouped[(condition, image_size, demo_count)]
        success_values = [
            float(row["success_rate"]) for row in group if row.get("success_rate") is not None
        ]
        best_values = [float(row["best_metric"]) for row in group if row.get("best_metric") is not None]
        success_mean, success_std = _mean_std(success_values)
        best_mean, best_std = _mean_std(best_values)
        aggregate_rows.append(
            {
                "condition": condition,
                "image_size": image_size,
                "demo_count": demo_count,
                "num_seeds": len(group),
                "success_rate_mean": success_mean,
                "success_rate_std": success_std,
                "best_metric_mean": best_mean,
                "best_metric_std": best_std,
            }
        )
    return aggregate_rows


def _line_label(condition: str, image_size: int) -> str:
    return f"{condition}, {image_size}px"


def _plot(rows: list[dict], training_by_run: dict[str, list[dict]], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional environment package
        print(f"[WARN]: Could not import matplotlib, skipped PNG plots: {exc}")
        return

    aggregate_rows = _aggregate(rows)
    by_series: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in aggregate_rows:
        if row["success_rate_mean"] is None:
            continue
        by_series[(row["condition"], row["image_size"])].append(row)

    if by_series:
        plt.figure(figsize=(7, 4))
        for (condition, image_size), series_rows in sorted(by_series.items()):
            series_rows = sorted(series_rows, key=lambda item: item["demo_count"])
            xs = [row["demo_count"] for row in series_rows]
            means = [row["success_rate_mean"] for row in series_rows]
            stds = [row["success_rate_std"] for row in series_rows]
            plt.errorbar(xs, means, yerr=stds, marker="o", capsize=4, label=_line_label(condition, image_size))
        plt.xlabel("Number of demonstrations")
        plt.ylabel("Rollout success rate")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / "success_vs_demos.png", dpi=160)
        plt.close()

    plt.figure(figsize=(8, 5))
    plotted = False
    for row in rows:
        run_dir = row["run_dir"]
        training_rows = training_by_run[run_dir]
        val_values = [item["val_loss"] for item in training_rows]
        if all(value is None for value in val_values):
            values = [item["train_loss"] for item in training_rows]
            label_name = "train"
        else:
            values = val_values
            label_name = "val"
        epochs = [item["epoch"] for item in training_rows]
        label = (
            f"{row.get('condition')} {row.get('image_size')}px "
            f"demos={row.get('demo_count')} seed={row.get('seed')} {label_name}"
        )
        plt.plot(epochs, values, label=label, alpha=0.8)
        plotted = True
    if plotted:
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(output_dir / "loss_curves.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot visual diffusion experiment results.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["runs/visual_diffusion"],
        help="Run directories, metrics.csv files, or parent directories to scan.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for summary CSVs and PNG plots.")
    args = parser.parse_args()

    run_dirs = _find_run_dirs(args.paths)
    if not run_dirs:
        print("No visual runs found.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.paths[0]) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    training_by_run = {}
    for run_dir in run_dirs:
        summary, training_rows = _summarize_run(run_dir)
        rows.append(summary)
        training_by_run[str(run_dir)] = training_rows

    aggregate_rows = _aggregate(rows)
    _write_csv(output_dir / "summary.csv", rows)
    _write_csv(output_dir / "aggregate.csv", aggregate_rows)
    _plot(rows, training_by_run, output_dir)

    if (output_dir / "summary.csv").exists():
        print(f"[INFO]: wrote summary: {output_dir / 'summary.csv'}")
    if (output_dir / "aggregate.csv").exists():
        print(f"[INFO]: wrote aggregate: {output_dir / 'aggregate.csv'}")
    if (output_dir / "success_vs_demos.png").exists():
        print(f"[INFO]: wrote plot: {output_dir / 'success_vs_demos.png'}")
    if (output_dir / "loss_curves.png").exists():
        print(f"[INFO]: wrote plot: {output_dir / 'loss_curves.png'}")


if __name__ == "__main__":
    main()
