# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create CSV summaries and plots for low-dimensional diffusion experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


DIAGNOSTIC_METRICS = [
    "mean_initial_ball_to_basket_distance",
    "mean_final_ball_to_basket_distance",
    "mean_min_ball_to_basket_distance",
    "mean_ball_to_basket_improvement",
    "mean_initial_ee_to_ball_distance",
    "mean_final_ee_to_ball_distance",
    "mean_min_ee_to_ball_distance",
    "mean_max_ball_height",
    "mean_ball_height_gain",
    "ever_close_to_ball_rate",
    "ever_close_to_basket_rate",
    "ever_lifted_ball_rate",
]


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


def _parse_run_identity(run_dir: Path) -> tuple[int | None, int | None]:
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as json_file:
            config = json.load(json_file)
        return config.get("demo_count"), config.get("seed")

    text = str(run_dir)
    demo_match = re.search(r"demos_(\d+)", text)
    seed_match = re.search(r"seed_(\d+)", text)
    demo_count = int(demo_match.group(1)) if demo_match else None
    seed = int(seed_match.group(1)) if seed_match else None
    return demo_count, seed


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
    eval_files = sorted(run_dir.glob("*.json"))
    preferred = [path for path in eval_files if "eval" in path.name or "metric" in path.name]
    if not preferred:
        return {}
    with preferred[-1].open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def _summarize_run(run_dir: Path) -> tuple[dict, list[dict]]:
    training_rows = _load_training_rows(run_dir / "metrics.csv")
    if not training_rows:
        raise RuntimeError(f"No rows in {run_dir / 'metrics.csv'}")
    demo_count, seed = _parse_run_identity(run_dir)
    last = training_rows[-1]
    best = min(
        training_rows,
        key=lambda row: row["best_metric"] if row["best_metric"] is not None else float("inf"),
    )
    eval_metrics = _load_eval_metrics(run_dir)
    summary = {
        "run_dir": str(run_dir),
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
    for metric_name in DIAGNOSTIC_METRICS:
        summary[metric_name] = eval_metrics.get(metric_name)
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
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row["demo_count"] is not None:
            grouped[int(row["demo_count"])].append(row)

    aggregate_rows = []
    for demo_count in sorted(grouped):
        group = grouped[demo_count]
        success_values = [
            float(row["success_rate"]) for row in group if row.get("success_rate") is not None
        ]
        best_values = [float(row["best_metric"]) for row in group if row.get("best_metric") is not None]
        success_mean, success_std = _mean_std(success_values)
        best_mean, best_std = _mean_std(best_values)
        aggregate_row = {
            "demo_count": demo_count,
            "num_seeds": len(group),
            "success_rate_mean": success_mean,
            "success_rate_std": success_std,
            "best_metric_mean": best_mean,
            "best_metric_std": best_std,
        }
        for metric_name in DIAGNOSTIC_METRICS:
            values = [float(row[metric_name]) for row in group if row.get(metric_name) is not None]
            metric_mean, metric_std = _mean_std(values)
            aggregate_row[f"{metric_name}_mean"] = metric_mean
            aggregate_row[f"{metric_name}_std"] = metric_std
        aggregate_rows.append(aggregate_row)
    return aggregate_rows


def _plot(rows: list[dict], training_by_run: dict[str, list[dict]], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional environment package
        print(f"[WARN]: Could not import matplotlib, skipped PNG plots: {exc}")
        return

    aggregate_rows = _aggregate(rows)
    success_rows = [row for row in aggregate_rows if row["success_rate_mean"] is not None]
    if success_rows:
        xs = [row["demo_count"] for row in success_rows]
        means = [row["success_rate_mean"] for row in success_rows]
        stds = [row["success_rate_std"] for row in success_rows]
        plt.figure(figsize=(7, 4))
        plt.errorbar(xs, means, yerr=stds, marker="o", capsize=4)
        plt.xlabel("Number of demonstrations")
        plt.ylabel("Rollout success rate")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "success_vs_demos.png", dpi=160)
        plt.close()

    diagnostic_specs = [
        ("mean_min_ee_to_ball_distance", "Min hand-ball distance"),
        ("mean_final_ball_to_basket_distance", "Final ball-basket distance"),
        ("ever_close_to_ball_rate", "Ever close to ball rate"),
        ("ever_lifted_ball_rate", "Ever lifted ball rate"),
    ]
    if aggregate_rows:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        plotted_any = False
        for axis, (metric_name, title) in zip(axes.reshape(-1), diagnostic_specs):
            rows_with_metric = [
                row for row in aggregate_rows if row.get(f"{metric_name}_mean") is not None
            ]
            if not rows_with_metric:
                axis.set_visible(False)
                continue
            xs = [row["demo_count"] for row in rows_with_metric]
            means = [row[f"{metric_name}_mean"] for row in rows_with_metric]
            stds = [row[f"{metric_name}_std"] for row in rows_with_metric]
            axis.errorbar(xs, means, yerr=stds, marker="o", capsize=4)
            axis.set_title(title)
            axis.set_xlabel("Number of demonstrations")
            axis.grid(True, alpha=0.3)
            plotted_any = True
        if plotted_any:
            fig.tight_layout()
            fig.savefig(output_dir / "diagnostics_vs_demos.png", dpi=160)
        plt.close(fig)

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
        label = f"demos={row.get('demo_count')} seed={row.get('seed')} {label_name}"
        plt.plot(epochs, values, label=label, alpha=0.85)
        plotted = True
    if plotted:
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / "loss_curves.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot low-dimensional diffusion experiment results.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["runs/lowdim_diffusion"],
        help="Run directories, metrics.csv files, or parent directories to scan.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for summary CSVs and PNG plots.")
    args = parser.parse_args()

    run_dirs = _find_run_dirs(args.paths)
    if not run_dirs:
        print("No runs found.")
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
    if (output_dir / "diagnostics_vs_demos.png").exists():
        print(f"[INFO]: wrote plot: {output_dir / 'diagnostics_vs_demos.png'}")
    if (output_dir / "loss_curves.png").exists():
        print(f"[INFO]: wrote plot: {output_dir / 'loss_curves.png'}")


if __name__ == "__main__":
    main()
