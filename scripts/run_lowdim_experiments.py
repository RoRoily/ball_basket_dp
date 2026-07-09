# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run low-dimensional diffusion policy scale experiments.

This script orchestrates the existing pipeline:

1. collect raw scripted demonstrations;
2. filter demonstrations by quality;
3. inspect datasets;
4. train low-dimensional diffusion policies;
5. evaluate checkpoints with rollout metrics;
6. summarize runs.

It is meant to make small data-scale studies repeatable, for example comparing
5, 20, and 100 demonstrations.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _timestamp_name() -> str:
    return datetime.now().strftime("scale_%Y-%m-%d_%H-%M-%S")


def _bool_flag(enabled: bool, flag: str) -> list[str]:
    return [flag] if enabled else []


def _run(cmd: list[str], dry_run: bool) -> None:
    pretty = " ".join(shlex.quote(part) for part in cmd)
    print(f"\n[CMD] {pretty}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _write_json(path: Path, data: dict, dry_run: bool) -> None:
    print(f"\n[WRITE] {path}")
    if dry_run:
        print(json.dumps(data, indent=2, sort_keys=True))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2, sort_keys=True)


def _maybe_run(cmd: list[str], output_path: Path | None, skip_existing: bool, dry_run: bool) -> None:
    if output_path is not None and skip_existing and output_path.exists():
        print(f"\n[SKIP] {output_path} already exists.")
        return
    _run(cmd, dry_run)


def _python_script(script_name: str) -> list[str]:
    return [sys.executable, str(Path("scripts") / script_name)]


def _collection_attempts(num_demos: int, multiplier: int, explicit_attempts: int) -> int:
    if explicit_attempts > 0:
        return explicit_attempts
    return max(num_demos, num_demos * multiplier)


def _stages(requested: list[str]) -> set[str]:
    if "all" in requested:
        return {"collect", "filter", "inspect", "train", "eval", "summarize", "plot"}
    return set(requested)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BallBasket low-dimensional data-scale experiments.")
    parser.add_argument("--name", type=str, default=None, help="Experiment name. Defaults to a timestamp.")
    parser.add_argument("--demo_counts", type=int, nargs="+", default=[5, 20, 100], help="Demo counts to compare.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Training/evaluation seeds.")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=["all", "collect", "filter", "inspect", "train", "eval", "summarize", "plot"],
        help="Pipeline stages to run.",
    )
    parser.add_argument("--task", type=str, default="BallBasket-LowDim-v0", help="Isaac Lab task name.")
    parser.add_argument("--steps", type=int, default=430, help="Steps per demonstration/evaluation rollout.")
    parser.add_argument("--mode", choices=["auto", "drop", "throw"], default="auto", help="Scripted expert mode.")
    parser.add_argument("--virtual_grasp", action="store_true", default=True, help="Enable conditional virtual grasp.")
    parser.add_argument("--no_virtual_grasp", action="store_false", dest="virtual_grasp", help="Disable virtual grasp.")
    parser.add_argument("--keep_success_only", action="store_true", default=True, help="Filter for successful demos.")
    parser.add_argument(
        "--keep_all",
        action="store_false",
        dest="keep_success_only",
        help="Do not require success during offline filtering.",
    )
    parser.add_argument("--min_attach_count", type=int, default=1, help="Minimum attach count for filtered demos.")
    parser.add_argument(
        "--max_final_distance",
        type=float,
        default=-1.0,
        help="Maximum final ball-to-basket distance for filtered demos. Negative disables.",
    )
    parser.add_argument(
        "--max_demos_attempts_multiplier",
        type=int,
        default=10,
        help="Collection attempts are demo_count * this multiplier unless --max_demos_attempts is set.",
    )
    parser.add_argument("--max_demos_attempts", type=int, default=0, help="Explicit max attempts for each collection.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of episodes used for validation.")
    parser.add_argument("--ema_decay", type=float, default=0.995, help="Training EMA decay. 0 disables EMA.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch/Isaac device.")
    parser.add_argument("--num_envs_eval", type=int, default=8, help="Vectorized env count for evaluation.")
    parser.add_argument("--num_episodes_eval", type=int, default=5, help="Evaluation rollout batches.")
    parser.add_argument("--dataset_root", type=str, default="datasets/ball_basket_lowdim/experiments")
    parser.add_argument("--run_root", type=str, default="runs/lowdim_diffusion")
    parser.add_argument("--skip_existing", action="store_true", default=True, help="Skip outputs that already exist.")
    parser.add_argument("--overwrite", action="store_false", dest="skip_existing", help="Re-run existing outputs.")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Print commands without executing them.")
    args = parser.parse_args()

    if args.max_demos_attempts_multiplier < 1:
        raise ValueError("max_demos_attempts_multiplier must be >= 1.")

    experiment_name = args.name or _timestamp_name()
    dataset_dir = Path(args.dataset_root) / experiment_name
    run_root = Path(args.run_root) / experiment_name
    stages = _stages(args.stages)

    if not args.dry_run:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        run_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO]: experiment={experiment_name}")
    print(f"[INFO]: dataset_dir={dataset_dir}")
    print(f"[INFO]: run_root={run_root}")
    print(f"[INFO]: demo_counts={args.demo_counts}")
    print(f"[INFO]: seeds={args.seeds}")
    print(f"[INFO]: stages={sorted(stages)}")

    experiment_config = {
        "experiment": experiment_name,
        "demo_counts": args.demo_counts,
        "seeds": args.seeds,
        "task": args.task,
        "steps": args.steps,
        "mode": args.mode,
        "virtual_grasp": args.virtual_grasp,
        "keep_success_only": args.keep_success_only,
        "min_attach_count": args.min_attach_count,
        "max_final_distance": args.max_final_distance,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "val_ratio": args.val_ratio,
        "ema_decay": args.ema_decay,
        "device": args.device,
        "num_envs_eval": args.num_envs_eval,
        "num_episodes_eval": args.num_episodes_eval,
    }
    _write_json(run_root / "experiment_config.json", experiment_config, args.dry_run)

    for demo_count in args.demo_counts:
        raw_dataset = dataset_dir / f"raw_{demo_count}.hdf5"
        filtered_dataset = dataset_dir / f"filtered_{demo_count}.hdf5"
        eval_rollouts = args.num_envs_eval * args.num_episodes_eval
        attempts = _collection_attempts(
            demo_count, args.max_demos_attempts_multiplier, args.max_demos_attempts
        )

        print(f"\n[INFO]: === demo_count={demo_count} ===")

        if "collect" in stages:
            collect_cmd = (
                _python_script("collect_demos.py")
                + [
                    "--task",
                    args.task,
                    "--num_demos",
                    str(demo_count),
                    "--steps",
                    str(args.steps),
                    "--mode",
                    args.mode,
                    "--max_demos_attempts",
                    str(attempts),
                    "--output",
                    str(raw_dataset),
                    "--headless",
                    "--device",
                    args.device,
                ]
                + _bool_flag(args.virtual_grasp, "--virtual_grasp")
            )
            if args.keep_success_only:
                collect_cmd.append("--keep_success_only")
            if args.min_attach_count > 0:
                collect_cmd += ["--min_attach_count", str(args.min_attach_count)]
            if args.max_final_distance >= 0.0:
                collect_cmd += ["--max_final_distance", str(args.max_final_distance)]
            _maybe_run(collect_cmd, raw_dataset, args.skip_existing, args.dry_run)

        if "inspect" in stages:
            _maybe_run(_python_script("inspect_dataset.py") + [str(raw_dataset)], None, False, args.dry_run)

        if "filter" in stages:
            filter_cmd = [
                *_python_script("filter_dataset.py"),
                "--input",
                str(raw_dataset),
                "--output",
                str(filtered_dataset),
                "--min_attach_count",
                str(args.min_attach_count),
            ]
            if args.keep_success_only:
                filter_cmd.append("--keep_success_only")
            if args.max_final_distance >= 0.0:
                filter_cmd += ["--max_final_distance", str(args.max_final_distance)]
            _maybe_run(filter_cmd, filtered_dataset, args.skip_existing, args.dry_run)

        if "inspect" in stages:
            _maybe_run(_python_script("inspect_dataset.py") + [str(filtered_dataset)], None, False, args.dry_run)

        for seed in args.seeds:
            run_dir = run_root / f"demos_{demo_count}" / f"seed_{seed}"
            checkpoint = run_dir / "policy.pt"
            best_checkpoint = run_dir / "best.pt"
            eval_metrics = run_dir / f"eval_{eval_rollouts}_rollouts.json"
            run_config = {
                **experiment_config,
                "demo_count": demo_count,
                "seed": seed,
                "raw_dataset": str(raw_dataset),
                "filtered_dataset": str(filtered_dataset),
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "best_checkpoint": str(best_checkpoint),
                "eval_metrics": str(eval_metrics),
            }
            _write_json(run_dir / "run_config.json", run_config, args.dry_run)

            if "train" in stages:
                train_cmd = [
                    *_python_script("train_lowdim_diffusion.py"),
                    "--dataset",
                    str(filtered_dataset),
                    "--epochs",
                    str(args.epochs),
                    "--batch_size",
                    str(args.batch_size),
                    "--val_ratio",
                    str(args.val_ratio),
                    "--ema_decay",
                    str(args.ema_decay),
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                    "--output",
                    str(checkpoint),
                ]
                _maybe_run(train_cmd, checkpoint, args.skip_existing, args.dry_run)

            if "eval" in stages:
                eval_cmd = [
                    *_python_script("eval_lowdim_diffusion.py"),
                    "--task",
                    args.task,
                    "--checkpoint",
                    str(best_checkpoint),
                    "--num_envs",
                    str(args.num_envs_eval),
                    "--num_episodes",
                    str(args.num_episodes_eval),
                    "--steps",
                    str(args.steps),
                    "--metrics_path",
                    str(eval_metrics),
                    "--seed",
                    str(seed),
                    "--headless",
                    "--device",
                    args.device,
                ]
                _maybe_run(eval_cmd, eval_metrics, args.skip_existing, args.dry_run)

    if "summarize" in stages:
        _run(_python_script("summarize_lowdim_runs.py") + [str(run_root)], args.dry_run)
    if "plot" in stages:
        _run(
            _python_script("plot_lowdim_results.py")
            + [str(run_root), "--output_dir", str(run_root / "plots")],
            args.dry_run,
        )


if __name__ == "__main__":
    # Ensure child scripts resolve project-local imports when the command is run
    # from the repository root, which is the intended usage.
    os.chdir(Path(__file__).resolve().parents[1])
    main()
