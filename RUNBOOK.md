# Ball Basket Diffusion Policy 项目跑通指引

这份文档用于从零到一跑通当前 `ball_basket_dp` 项目。目标不是一次训练出很强的策略，而是把研究管线完整打通：

```text
使用仿真环境 -> 采集数据 -> 训练 -> 部署 -> 做验证 -> 做实验对比
```

当前项目已经包含三条主要路线：

1. `BallBasket-LowDim-v0` 低维状态任务
2. low-dimensional diffusion policy
3. visual diffusion policy

其中 visual policy 默认是 `RGB 图像 + low-dimensional proprio/context` 条件输入；`image_only` 是可选对照实验模式。

## 0. 项目与服务器约定

建议在服务器上保持下面的目录划分：

```text
SSD: conda、pip cache、tmp、Isaac cache
HDD: 项目代码、数据集、训练输出、视频、日志
```

本项目推荐路径：

```bash
export SSD_BASE=/mnt/dongxu-fs2/data-ssd/muhanzheng
export HDD_BASE=/mnt/dongxu-fs2/data-hdd/muhanzheng
export TMPDIR=$SSD_BASE/tmp
export PIP_CACHE_DIR=$SSD_BASE/pip-cache
```

项目代码位置：

```bash
cd $HDD_BASE/workspace/ball_basket_dp
```

如果 `echo $HDD_BASE` 为空，说明环境变量没有在当前 shell 生效。临时补上：

```bash
export SSD_BASE=/mnt/dongxu-fs2/data-ssd/muhanzheng
export HDD_BASE=/mnt/dongxu-fs2/data-hdd/muhanzheng
export TMPDIR=$SSD_BASE/tmp
export PIP_CACHE_DIR=$SSD_BASE/pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
```

如果希望每次登录自动生效，写入 `~/.bashrc`：

```bash
cat >> ~/.bashrc <<'EOF'

# Personal storage layout for Isaac Lab experiments
export SSD_BASE=/mnt/dongxu-fs2/data-ssd/muhanzheng
export HDD_BASE=/mnt/dongxu-fs2/data-hdd/muhanzheng
export TMPDIR=$SSD_BASE/tmp
export PIP_CACHE_DIR=$SSD_BASE/pip-cache
EOF
```

重新打开 shell，或执行：

```bash
source ~/.bashrc
```

这些变量写在你自己的 `~/.bashrc`，只影响你的用户，不影响服务器其他用户。

## 1. 每次开始实验前的标准准备

进入项目：

```bash
cd $HDD_BASE/workspace/ball_basket_dp
```

激活环境：

```bash
conda activate isaaclab
```

确认 Python 和 conda 指向正确：

```bash
which python
python --version
conda info --base
```

推荐看到类似：

```text
/mnt/dongxu-fs2/data-ssd/muhanzheng/conda-envs/isaaclab/bin/python
Python 3.11.x
/mnt/dongxu-fs2/data-ssd/muhanzheng/miniforge3
```

安装项目扩展：

```bash
pip install -e source/ball_basket_dp
```

如果你刚刚从 GitHub 拉了新代码，建议重新执行一次这个命令。

选择 GPU：

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=3
```

注意变量名是 `CUDA_VISIBLE_DEVICES`，不是 `CUDA_VISIBLE_DIVICES`。

## 2. Git 同步建议

在服务器上开始跑实验前：

```bash
git status
git pull --rebase
```

如果本地有新修改：

```bash
git status
git add <changed-files>
git commit -m "your commit message"
git pull --rebase
git push
```

如果看到：

```text
Your branch and 'origin/master' have diverged
```

先看两边分别有什么提交：

```bash
git log --oneline --left-right --graph master...origin/master
```

通常使用 rebase 合并远端新提交：

```bash
git pull --rebase origin master
```

有冲突时按文件解决冲突，然后：

```bash
git add <resolved-files>
git rebase --continue
git push
```

## 3. 确认任务注册成功

列出项目任务：

```bash
python scripts/list_envs.py
```

预期能看到：

```text
BallBasket-LowDim-v0
```

如果看不到，通常是以下原因：

1. 没有执行 `pip install -e source/ball_basket_dp`
2. 没有激活 `isaaclab` 环境
3. 当前目录不是项目根目录
4. Python 环境不是服务器上的 Isaac Lab 环境

## 4. 最小环境 smoke test

先跑一个短的 headless 环境测试：

```bash
python scripts/smoke_test_env.py \
  --task BallBasket-LowDim-v0 \
  --num_envs 16 \
  --steps 100 \
  --headless
```

预期看到类似：

```text
Gym observation space: Dict('policy': Box(..., (16, 43), float32))
Gym action space: Box(..., (16, 4), float32)
```

这个测试说明：

```text
Isaac Lab 能启动
任务注册正常
环境 reset/step 正常
obs/action 维度正常
```

## 5. 用 scripted expert 录一段验证视频

先用脚本专家生成一段可视化视频，确认场景和动作是否符合预期：

```bash
python scripts/scripted_expert.py \
  --task BallBasket-LowDim-v0 \
  --num_envs 1 \
  --steps 430 \
  --mode auto \
  --virtual_grasp \
  --video \
  --video_dir videos/scripted_expert \
  --headless
```

查找视频：

```bash
find videos/scripted_expert -type f | sort | tail
```

说明：

- `--virtual_grasp` 是调试用的虚拟抓取辅助。
- 它会在手靠近球后将球“绑定”到手上，因此视频里可能看到球被吸附/带走。
- 这不是完全真实物理抓取，但适合先跑通 diffusion policy 数据管线。

## 6. 低维 diffusion policy 跑通路径

低维管线使用状态向量，不使用 RGB 图像。它是最稳的第一条主线。

### 6.1 采集 5 条 debug demonstration

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

预期输出：

```text
[INFO]: saved demo ...
[INFO]: Wrote 5 demos to: ...
[INFO]: obs shape=(..., 43), action shape=(..., 4)
```

### 6.2 检查数据集

```bash
python scripts/inspect_dataset.py datasets/ball_basket_lowdim/debug_5.hdf5
```

重点看：

```text
Episodes: 5
Obs shape: (N, 43)
Action shape: (N, 4)
Obs finite: True
Action finite: True
```

如果 `Obs finite` 或 `Action finite` 是 `False`，不要训练，先排查采集脚本或仿真状态。

### 6.3 训练 low-dimensional diffusion

先用短训练验证：

```bash
python scripts/train_lowdim_diffusion.py \
  --dataset datasets/ball_basket_lowdim/debug_5.hdf5 \
  --epochs 10 \
  --batch_size 128 \
  --val_ratio 0.2 \
  --device cuda \
  --output runs/lowdim_diffusion/debug_5/policy.pt
```

预期生成：

```text
runs/lowdim_diffusion/debug_5/policy.pt
runs/lowdim_diffusion/debug_5/best.pt
runs/lowdim_diffusion/debug_5/metrics.csv
```

检查：

```bash
ls -lh runs/lowdim_diffusion/debug_5/
tail -n 5 runs/lowdim_diffusion/debug_5/metrics.csv
```

### 6.4 部署 low-dimensional checkpoint 并录视频

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

查找视频：

```bash
find videos/lowdim_diffusion -type f | sort | tail
```

查看指标：

```bash
cat runs/lowdim_diffusion/debug_5/eval_metrics.json
```

debug 数据只有 5 条，不要求视频动作很好。这个阶段的达标标准是：

```text
能生成 HDF5
能 inspect
能训练并保存 best.pt
能加载 best.pt 部署
能保存 eval metrics 和视频
```

## 7. 低维大一点的数据集与实验对比

采集更可靠的数据集：

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

训练：

```bash
python scripts/train_lowdim_diffusion.py \
  --dataset datasets/ball_basket_lowdim/train_success_100.hdf5 \
  --epochs 100 \
  --batch_size 256 \
  --val_ratio 0.1 \
  --ema_decay 0.995 \
  --device cuda \
  --output runs/lowdim_diffusion/train_success_100/policy.pt
```

多 rollout 定量验证：

```bash
python scripts/eval_lowdim_diffusion.py \
  --task BallBasket-LowDim-v0 \
  --checkpoint runs/lowdim_diffusion/train_success_100/best.pt \
  --num_envs 8 \
  --num_episodes 5 \
  --steps 430 \
  --metrics_path runs/lowdim_diffusion/train_success_100/eval_40_rollouts.json \
  --headless
```

一键跑低维规模实验：

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

画图：

```bash
python scripts/plot_lowdim_results.py \
  runs/lowdim_diffusion/scale_5_20_100 \
  --output_dir runs/lowdim_diffusion/scale_5_20_100/plots
```

结果文件：

```text
runs/lowdim_diffusion/scale_5_20_100/plots/summary.csv
runs/lowdim_diffusion/scale_5_20_100/plots/aggregate.csv
runs/lowdim_diffusion/scale_5_20_100/plots/success_vs_demos.png
runs/lowdim_diffusion/scale_5_20_100/plots/loss_curves.png
```

## 8. visual diffusion policy 跑通路径

视觉管线会保存 RGB 图像，并训练图像条件 diffusion policy。

默认模式：

```text
RGB 图像 + low-dimensional obs
```

可选对照：

```text
image_only
```

建议先把默认模式跑通，再尝试 `image_only`。

### 8.1 采集 5 条视觉 debug demonstration

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

预期：

```text
[INFO]: image shape=(N, 96, 96, 3), obs shape=(N, 43), action shape=(N, 4)
```

### 8.2 检查视觉数据集

```bash
python scripts/inspect_dataset.py datasets/ball_basket_visual/debug_5.hdf5
```

重点看：

```text
Image shape: (N, 96, 96, 3)
Obs shape: (N, 43)
Action shape: (N, 4)
Obs finite: True
Action finite: True
```

### 8.3 导出视觉预览

```bash
python scripts/preview_visual_dataset.py \
  datasets/ball_basket_visual/debug_5.hdf5 \
  --output_dir datasets/ball_basket_visual/debug_5_preview \
  --num_episodes 3 \
  --gif
```

查看输出：

```bash
find datasets/ball_basket_visual/debug_5_preview -maxdepth 1 -type f | sort
```

预期有：

```text
episode_0000_sheet.png
episode_0000.gif
preview_manifest.json
```

这一步很重要。训练视觉策略前，先确认画面里：

```text
机械臂可见
小球可见
篮子可见
目标区域没有被遮挡
画面不是全黑或过远
```

### 8.4 训练 visual diffusion

```bash
python scripts/train_visual_diffusion.py \
  --dataset datasets/ball_basket_visual/debug_5.hdf5 \
  --epochs 10 \
  --batch_size 32 \
  --val_ratio 0.2 \
  --device cuda \
  --output runs/visual_diffusion/debug_5/policy.pt
```

预期生成：

```text
runs/visual_diffusion/debug_5/policy.pt
runs/visual_diffusion/debug_5/best.pt
runs/visual_diffusion/debug_5/metrics.csv
```

### 8.5 部署 visual checkpoint 并录视频

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

注意：

```text
visual eval 当前只支持 --num_envs 1
```

原因是 `env.render()` 当前按一个主视角取图像。

## 9. visual 实验对比

先 dry-run 看命令是否符合预期：

```bash
python scripts/run_visual_experiments.py \
  --name visual_debug \
  --demo_counts 5 \
  --image_sizes 96 \
  --conditions rgb_lowdim \
  --seeds 0 \
  --epochs 10 \
  --batch_size 32 \
  --num_episodes_eval 2 \
  --dry_run
```

确认命令没问题后去掉 `--dry_run`：

```bash
python scripts/run_visual_experiments.py \
  --name visual_debug \
  --demo_counts 5 \
  --image_sizes 96 \
  --conditions rgb_lowdim \
  --seeds 0 \
  --epochs 10 \
  --batch_size 32 \
  --num_episodes_eval 2
```

对比 `rgb_lowdim` 和 `image_only`：

```bash
python scripts/run_visual_experiments.py \
  --name visual_ablation_5_20_100 \
  --demo_counts 5 20 100 \
  --image_sizes 96 \
  --conditions all \
  --seeds 0 1 2 \
  --epochs 100 \
  --batch_size 64 \
  --num_episodes_eval 5
```

重新画图：

```bash
python scripts/plot_visual_results.py \
  runs/visual_diffusion/visual_ablation_5_20_100 \
  --output_dir runs/visual_diffusion/visual_ablation_5_20_100/plots
```

重点看：

```text
success_vs_demos.png
loss_curves.png
summary.csv
aggregate.csv
```

一个合理的实验表述方式：

```text
在相同 demonstration 数量、相同训练 epoch、相同 seed 设置下，
比较 rgb_lowdim 和 image_only 的 rollout success_rate。
```

## 10. 真实物理抓取校准

虚拟抓取只是为了先跑通数据管线。真实物理抓取要单独校准。

先 sweep 抓取高度和 xy offset：

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

查看结果：

```bash
cat runs/physical_grasp/sweep_debug/physical_grasp_summary.json
column -s, -t < runs/physical_grasp/sweep_debug/physical_grasp_metrics.csv | head -30
```

用表现最好的参数录制真实抓取视频：

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

达标后，再尝试不使用 `--virtual_grasp` 采集数据。

## 11. 输出目录说明

常用输出：

```text
datasets/ball_basket_lowdim/
datasets/ball_basket_visual/
runs/lowdim_diffusion/
runs/visual_diffusion/
runs/physical_grasp/
videos/scripted_expert/
videos/lowdim_diffusion/
videos/visual_diffusion/
logs/
```

建议不要提交这些大文件到 Git。`.gitignore` 通常应该忽略：

```text
datasets/
runs/
videos/
logs/
*.hdf5
*.pt
```

## 12. 什么叫“项目跑通”

最低标准：

```text
1. python scripts/list_envs.py 能看到 BallBasket-LowDim-v0
2. smoke_test_env.py 能完成 step
3. scripted_expert.py 能录出视频
4. collect_demos.py 能生成 lowdim HDF5
5. train_lowdim_diffusion.py 能生成 policy.pt 和 best.pt
6. eval_lowdim_diffusion.py 能加载 checkpoint 并写 eval_metrics.json
7. collect_visual_demos.py 能生成带 images 的 HDF5
8. preview_visual_dataset.py 能导出 PNG/GIF
9. train_visual_diffusion.py 能生成 visual checkpoint
10. eval_visual_diffusion.py 能加载 visual checkpoint 并录视频
11. run_lowdim_experiments.py 或 run_visual_experiments.py 能 dry-run
12. plot_lowdim_results.py 或 plot_visual_results.py 能生成 summary/aggregate/plots
```

研究意义上的进一步标准：

```text
1. success_rate 随 demonstration 数量增加而提高
2. 多 seed 实验结果稳定
3. rgb_lowdim 和 image_only 有清晰对比
4. lowdim policy 和 visual policy 有定量表格
5. 真实物理抓取不依赖 virtual_grasp 后也有可观成功率
```

## 13. 常见问题

### 13.1 `which conda` 没输出，但 `conda info --base` 正常

这通常说明 `conda` 是 shell function，不一定是普通可执行文件。看：

```bash
type conda
conda info --base
```

只要 `conda activate isaaclab` 能用，`conda info --base` 指向 SSD 上的 miniforge，一般就没问题。

### 13.2 `isaacsim --no-window` Ctrl+C 没反应

Isaac Sim 有时退出慢。另开终端查进程：

```bash
ps -u $USER -o pid,ppid,stat,etime,cmd | grep -E "isaacsim|kit|omni" | grep -v grep
```

温和结束：

```bash
kill <PID>
```

仍然无响应时：

```bash
kill -9 <PID>
```

### 13.3 WebRTC 黑屏

你当前是通过跳板机访问服务器，WebRTC 常需要 TCP 和 UDP 都能连通。SSH 端口转发通常只能方便转 TCP，UDP 会比较麻烦。黑屏不影响 headless 训练、数据采集和视频录制。

建议继续使用：

```text
headless 运行 + RecordVideo 保存 mp4
```

### 13.4 训练视频里动作很差

如果只用了 5 条 demo，这是正常的。5 条 demo 的作用是 debug 管线，不是训练好策略。

提升方向：

```text
增加 demo 数量到 100/300/1000
过滤成功 demo
增加 epoch
做多 seed
看 metrics.csv 和 rollout success_rate
检查 scripted expert 数据质量
```

### 13.5 视觉策略学不好

先检查视觉数据：

```bash
python scripts/preview_visual_dataset.py <dataset.hdf5> --output_dir <preview-dir> --gif
```

常见原因：

```text
小球太小
篮子太小
相机视角不稳定
画面遮挡
demo 太少
image_only 难度太高
```

建议先使用默认 `rgb_lowdim`，确认成功后再比较 `image_only`。

### 13.6 `pip check` 有 Isaac Sim 依赖冲突

Isaac Sim/Isaac Lab 的 pip 包经常固定某些依赖版本。只要以下脚本能跑通，通常不用因为 `pip check` 里少量版本提示立刻重装环境：

```bash
python scripts/list_envs.py
python scripts/smoke_test_env.py --task BallBasket-LowDim-v0 --headless
```

如果出现 import error 或运行时崩溃，再针对具体包处理。

## 14. 推荐第一次完整跑通顺序

建议严格按下面顺序走：

```bash
# 1. 准备
cd $HDD_BASE/workspace/ball_basket_dp
conda activate isaaclab
pip install -e source/ball_basket_dp
export CUDA_VISIBLE_DEVICES=3

# 2. 环境
python scripts/list_envs.py
python scripts/smoke_test_env.py --task BallBasket-LowDim-v0 --num_envs 16 --steps 100 --headless

# 3. scripted expert 视频
python scripts/scripted_expert.py \
  --task BallBasket-LowDim-v0 \
  --num_envs 1 \
  --steps 430 \
  --mode auto \
  --virtual_grasp \
  --video \
  --video_dir videos/scripted_expert \
  --headless

# 4. lowdim debug 数据
python scripts/collect_demos.py \
  --task BallBasket-LowDim-v0 \
  --num_demos 5 \
  --steps 430 \
  --mode auto \
  --virtual_grasp \
  --output datasets/ball_basket_lowdim/debug_5.hdf5 \
  --headless

python scripts/inspect_dataset.py datasets/ball_basket_lowdim/debug_5.hdf5

# 5. lowdim 训练和部署
python scripts/train_lowdim_diffusion.py \
  --dataset datasets/ball_basket_lowdim/debug_5.hdf5 \
  --epochs 10 \
  --batch_size 128 \
  --val_ratio 0.2 \
  --device cuda \
  --output runs/lowdim_diffusion/debug_5/policy.pt

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

# 6. visual debug 数据
python scripts/collect_visual_demos.py \
  --task BallBasket-LowDim-v0 \
  --num_demos 5 \
  --steps 430 \
  --image_size 96 \
  --mode auto \
  --virtual_grasp \
  --output datasets/ball_basket_visual/debug_5.hdf5 \
  --headless

python scripts/inspect_dataset.py datasets/ball_basket_visual/debug_5.hdf5
python scripts/preview_visual_dataset.py \
  datasets/ball_basket_visual/debug_5.hdf5 \
  --output_dir datasets/ball_basket_visual/debug_5_preview \
  --num_episodes 3 \
  --gif

# 7. visual 训练和部署
python scripts/train_visual_diffusion.py \
  --dataset datasets/ball_basket_visual/debug_5.hdf5 \
  --epochs 10 \
  --batch_size 32 \
  --val_ratio 0.2 \
  --device cuda \
  --output runs/visual_diffusion/debug_5/policy.pt

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

走完这一串，就说明项目的核心研究管线已经跑通。

