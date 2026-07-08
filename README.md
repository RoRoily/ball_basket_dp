# Template for Isaac Lab Projects

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Ball Basket Low-Dimensional Pipeline

This project contains a first low-dimensional pipeline for `BallBasket-LowDim-v0`:

1. run the Isaac Lab environment;
2. collect scripted expert demonstrations into HDF5;
3. train a small DDPM-style low-dimensional diffusion policy;
4. deploy the checkpoint back into Isaac Lab and record a validation video.

On the server, start from the project root:

```bash
cd $HDD_BASE/workspace/ball_basket_dp
conda activate isaaclab
pip install -e source/ball_basket_dp
```

Collect a tiny debug dataset first:

```bash
python scripts/collect_demos.py \
  --task BallBasket-LowDim-v0 \
  --num_demos 5 \
  --steps 430 \
  --mode auto \
  --virtual_grasp \
  --output datasets/ball_basket_lowdim/debug_5.hdf5 \
  --headless
```

For larger runs, collect raw demonstrations first, then filter them:

```bash
python scripts/collect_demos.py \
  --task BallBasket-LowDim-v0 \
  --num_demos 100 \
  --steps 430 \
  --mode auto \
  --virtual_grasp \
  --output datasets/ball_basket_lowdim/raw_100.hdf5 \
  --headless

python scripts/filter_dataset.py \
  --input datasets/ball_basket_lowdim/raw_100.hdf5 \
  --output datasets/ball_basket_lowdim/train_success_100.hdf5 \
  --keep_success_only \
  --min_attach_count 1
```

You can also filter while collecting:

```bash
python scripts/collect_demos.py \
  --task BallBasket-LowDim-v0 \
  --num_demos 100 \
  --steps 430 \
  --mode auto \
  --virtual_grasp \
  --keep_success_only \
  --min_attach_count 1 \
  --max_demos_attempts 500 \
  --output datasets/ball_basket_lowdim/train_success_100.hdf5 \
  --headless
```

Inspect the dataset before training:

```bash
python scripts/inspect_dataset.py datasets/ball_basket_lowdim/debug_5.hdf5
```

Train the low-dimensional diffusion policy:

```bash
python scripts/train_lowdim_diffusion.py \
  --dataset datasets/ball_basket_lowdim/debug_5.hdf5 \
  --epochs 100 \
  --batch_size 256 \
  --val_ratio 0.1 \
  --device cuda \
  --output runs/lowdim_diffusion/debug_5/policy.pt
```

The training script writes checkpoints and a CSV loss log:

```bash
ls -lh runs/lowdim_diffusion/debug_5/
tail -n 5 runs/lowdim_diffusion/debug_5/metrics.csv
```

Validation is split by complete demonstration episodes, and normalization
statistics are estimated from the training episodes only.

Deploy the checkpoint and record a validation video plus rollout metrics:

```bash
python scripts/eval_lowdim_diffusion.py \
  --task BallBasket-LowDim-v0 \
  --checkpoint runs/lowdim_diffusion/debug_5/best.pt \
  --num_envs 1 \
  --num_episodes 1 \
  --steps 430 \
  --video \
  --video_dir videos/lowdim_diffusion \
  --metrics_path runs/lowdim_diffusion/debug_5/eval_metrics.json \
  --headless
```

For a quantitative check without recording video, evaluate more rollouts:

```bash
python scripts/eval_lowdim_diffusion.py \
  --task BallBasket-LowDim-v0 \
  --checkpoint runs/lowdim_diffusion/debug_5/best.pt \
  --num_envs 8 \
  --num_episodes 5 \
  --steps 430 \
  --metrics_path runs/lowdim_diffusion/debug_5/eval_40_rollouts.json \
  --headless
```

Summarize training and evaluation runs:

```bash
python scripts/summarize_lowdim_runs.py runs/lowdim_diffusion
```

Run a repeatable data-scale experiment:

```bash
python scripts/run_lowdim_experiments.py \
  --name scale_debug \
  --demo_counts 5 \
  --seeds 0 \
  --epochs 10 \
  --batch_size 128 \
  --ema_decay 0.995 \
  --num_envs_eval 4 \
  --num_episodes_eval 2 \
  --dry_run

python scripts/run_lowdim_experiments.py \
  --name scale_debug \
  --demo_counts 5 \
  --seeds 0 \
  --epochs 10 \
  --batch_size 128 \
  --ema_decay 0.995 \
  --num_envs_eval 4 \
  --num_episodes_eval 2
```

Once the debug experiment works, compare multiple dataset sizes:

```bash
python scripts/run_lowdim_experiments.py \
  --name scale_5_20_100 \
  --demo_counts 5 20 100 \
  --seeds 0 1 2 \
  --epochs 100 \
  --batch_size 256 \
  --ema_decay 0.995 \
  --num_envs_eval 8 \
  --num_episodes_eval 5
```

This writes per-run configs, training CSVs, rollout JSON metrics, and plots:

```bash
python scripts/summarize_lowdim_runs.py runs/lowdim_diffusion/scale_5_20_100
python scripts/plot_lowdim_results.py \
  runs/lowdim_diffusion/scale_5_20_100 \
  --output_dir runs/lowdim_diffusion/scale_5_20_100/plots
```

After the debug dataset works, collect more demonstrations by increasing
`--num_demos` to 100 or more and train a fresh checkpoint.

## Visual Diffusion Policy

The first visual policy scaffold uses rendered RGB frames plus the existing
low-dimensional proprio/context vector as diffusion conditions. This is not yet
a pure image-only policy; it is the practical next step for validating the
visual data, training, deployment, and video loop.

Collect a tiny visual debug dataset:

```bash
python scripts/collect_visual_demos.py \
  --task BallBasket-LowDim-v0 \
  --num_demos 5 \
  --steps 430 \
  --image_size 96 \
  --mode auto \
  --virtual_grasp \
  --output datasets/ball_basket_visual/debug_5.hdf5 \
  --headless
```

Inspect the visual dataset:

```bash
python scripts/inspect_dataset.py datasets/ball_basket_visual/debug_5.hdf5
```

Expected visual-specific output includes:

```text
Image shape: (N, 96, 96, 3)
Obs shape: (N, 43)
Action shape: (N, 4)
```

Train a small visual diffusion policy:

```bash
python scripts/train_visual_diffusion.py \
  --dataset datasets/ball_basket_visual/debug_5.hdf5 \
  --epochs 10 \
  --batch_size 32 \
  --val_ratio 0.2 \
  --device cuda \
  --output runs/visual_diffusion/debug_5/policy.pt
```

The script writes:

```text
runs/visual_diffusion/debug_5/policy.pt
runs/visual_diffusion/debug_5/best.pt
runs/visual_diffusion/debug_5/metrics.csv
```

Deploy the visual checkpoint and record a rollout video:

```bash
python scripts/eval_visual_diffusion.py \
  --task BallBasket-LowDim-v0 \
  --checkpoint runs/visual_diffusion/debug_5/best.pt \
  --num_envs 1 \
  --num_episodes 1 \
  --steps 430 \
  --video \
  --video_dir videos/visual_diffusion \
  --metrics_path runs/visual_diffusion/debug_5/eval_metrics.json \
  --headless
```

For a larger visual dataset, collect more successful demonstrations and train
again:

```bash
python scripts/collect_visual_demos.py \
  --task BallBasket-LowDim-v0 \
  --num_demos 100 \
  --steps 430 \
  --image_size 96 \
  --mode auto \
  --virtual_grasp \
  --keep_success_only \
  --min_attach_count 1 \
  --max_demos_attempts 500 \
  --output datasets/ball_basket_visual/train_success_100.hdf5 \
  --headless
```

Use `--image_only` with `train_visual_diffusion.py` only after the RGB +
low-dimensional version is stable.

## Physical Grasp Calibration

The virtual grasp pipeline is useful as a baseline, but physical grasping should
be calibrated separately with teleporting disabled. Sweep grasp heights and
offsets first:

```bash
python scripts/physical_grasp_eval.py \
  --task BallBasket-LowDim-v0 \
  --num_envs 8 \
  --trials 2 \
  --steps 260 \
  --descend_zs 0.06 0.075 0.09 \
  --close_zs 0.06 0.075 0.09 \
  --lift_zs 0.30 0.34 \
  --xy_offsets 0.0,0.0 0.02,0.0 -0.02,0.0 0.0,0.02 0.0,-0.02 \
  --output_dir runs/physical_grasp/sweep_debug \
  --headless
```

The script writes:

```text
runs/physical_grasp/sweep_debug/physical_grasp_metrics.csv
runs/physical_grasp/sweep_debug/physical_grasp_summary.json
```

Use the best row from the summary to record a no-virtual-grasp video:

```bash
python scripts/scripted_expert.py \
  --task BallBasket-LowDim-v0 \
  --num_envs 1 \
  --steps 260 \
  --mode drop \
  --grasp_offset_x 0.0 \
  --grasp_offset_y 0.0 \
  --descend_z 0.075 \
  --close_z 0.075 \
  --lift_z 0.34 \
  --video \
  --video_dir videos/physical_grasp \
  --headless
```

Only after lift/hold success is reasonable should you try collecting demos
without `--virtual_grasp`.

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/ball_basket_dp

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/ball_basket_dp/ball_basket_dp/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/ball_basket_dp"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
