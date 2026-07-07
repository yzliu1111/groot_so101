# Full fine-tune：SO101 合成数据准备与官方微调入口

读完这份文件，你应该能区分：

1. 原始 LeRobot v3 数据放在哪里。
2. GR00T 可读的 prepared v2.1 copy 放在哪里。
3. full fine-tune 为什么要分 LeRobot / GR00T 两个 Python 环境。
4. 本机 5060 Ti 上哪些路径和命令是固定答案。

如果是在一台新的 Ubuntu 机器上从零搭环境，先看：[ubuntu_env_setup/README_ZH.md](../ubuntu_env_setup/README_ZH.md)。那份文件是通用安装剧本；本文件只讲 full fine-tune 这一步本身。

## 机器视角

| 机器 | 推荐用途 | 不推荐做什么 |
|---|---|---|
| 本机 RTX 5060 Ti | 数据转换、prepared copy 生成、stats、loader smoke、小步数逻辑验证 | 不用它判断 2000-step full fine-tune 是否可行 |
| 远程 Linux GPU 训练机 | 复用 prepared 数据做训练 smoke 或正式训练；如果要重新转换数据，再补 LeRobot 环境 | 不要先从 `--global-batch-size 32` 盲跑 |

## 资产应该放在哪里

```text
本机：
/home/yzliu/smart_project/dataset/so101_lego_pick_0609_1722
/home/yzliu/smart_project/dataset/so101_lego_pick_0609_1722_mimic
/home/yzliu/smart_project/outputs/groot_so101_synthetic_datasets/
/home/yzliu/Isaac-GR00T-py312

远程 Linux GPU 训练机：
$SMART_PROJECT/outputs/groot_so101_synthetic_datasets/
$GROOT_ROOT
```

本机可以保留原始 `dataset/`；远程训练机第一轮只需要同步 prepared 数据，不必同步原始 v3 数据。

## SO101 合成数据微调路线

除了 zero-shot probe，本目录现在还增加了面向真实部署目标的 SO101 微调路线。这里的目标机械臂
明确是 SOARM101/SO101，不是 Franka。Franka 仍然只是为了分析 zero-shot embodiment mismatch
而加入的对照任务。

当前 `dataset/` 下的两份合成数据是 LeRobot v3 格式：

```text
dataset/so101_lego_pick_0609_1722
dataset/so101_lego_pick_0609_1722_mimic
```

它们的关键特征是：

- `robot_type` 是 `so101_follower`。
- `observation.state` 是 6D SO101 state。
- `action` 是 6D SO101 action。
- 前 5 维是 arm joints，第 6 维是 gripper。
- 图像有三路：`camera1/camera2/camera3`。

GR00T 训练侧当前需要的是 GR00T-flavored LeRobot v2.1 数据，并且需要额外的
`meta/modality.json`。所以新增的训练准备逻辑不是直接修改原始 `dataset/`，而是默认生成 prepared
副本：

```text
outputs/groot_so101_synthetic_datasets/
```

这些 prepared dataset 中会额外出现：

```text
meta/modality.json
meta/stats.json
meta/relative_stats.json
```

### 为什么是两阶段环境

本机环境是刻意隔离的；迁移到别的 Ubuntu 机器时也应该保持这个分层：

```text
LeRobot 数据/schema 转换 -> conda lerobot
GR00T stats / launch_finetune -> /home/yzliu/Isaac-GR00T-py312/.venv
Isaac closed-loop -> conda isaaclab
```

因此不要假设一个 Python 环境可以同时 import LeRobot、GR00T、IsaacLab。推荐流程是：

第一阶段：在 LeRobot 环境里准备数据。

```bash
cd /home/yzliu/smart_project
conda run -n lerobot python \
  experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
    --instruction "Pick up the red 2x4 lego brick." \
    --force-prepare \
    --skip-stats \
    --prepare-only
```

第二阶段：在 GR00T venv 里生成统计并启动 fine-tune 入口。

```bash
cd /home/yzliu/smart_project
/home/yzliu/Isaac-GR00T-py312/.venv/bin/python \
  experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
    --skip-prepare \
    --max-steps 2000 \
    --save-steps 500 \
    --global-batch-size 32
```

本机 RTX 5060 Ti 16GB 更适合做数据转换、loader smoke test、stats 生成和小步数逻辑验证；完整
GR00T 微调大概率仍然需要上云或使用 40GB+ 显存设备。

当前默认 GR00T checkout 是 `~/Isaac-GR00T-py312` / Python 3.12。旧
`~/Isaac-GR00T` / Python 3.10 只作为回滚和历史对照。

如果要迁移到另一台 Ubuntu 机器，直接看通用迁移指南；低显存策略和微调后部署看第三阶段文档：

```text
experiments/groot_n17_isaac_smart_task/ubuntu_env_setup/README_ZH.md
experiments/groot_n17_isaac_smart_task/lowmem_lora_freeze_so101/README_ZH.md
```

### v3 到 v2.1 转换

LeRobot v3 和 v2.1 最大区别是存储布局：

```text
v3:   data/chunk-000/file-000.parquet
      videos/observation.images.camera1/chunk-000/file-000.mp4
      meta/tasks.parquet
      meta/episodes/chunk-000/file-000.parquet

v2.1: data/chunk-000/episode_000000.parquet
      videos/chunk-000/observation.images.camera1/episode_000000.mp4
      meta/tasks.jsonl
      meta/episodes.jsonl
```

GR00T 官方仓库里有转换器：

```text
/home/yzliu/Isaac-GR00T-py312/scripts/lerobot_conversion/convert_v3_to_v2.py
```

这个转换器的 `convert_dataset()` 是原地转换：会把原始目录移动成 `_v3.0` 备份，再把 v2.1 写回原路径。
这不是删除数据，但会改变当前 `dataset/` 的目录形态。当前实验脚本默认使用 prepared copy，是为了减少误操作。

另外，GR00T 官方转换器依赖某个 LeRobot commit 的 API；本机 LeRobot 更新后可能出现
`load_info` 等 API 位置变化。这不是环境坏了，而是 Isaac/GR00T 和 LeRobot 更新节奏不同造成的正常
版本漂移。`train_so101_synthetic_groot.py` 会优先尝试官方转换器；如果依赖或 API 不兼容，会打印原因并
回退到本目录里的轻量非破坏性转换逻辑。

### GR00T modality config

新增文件：

```text
experiments/groot_n17_isaac_smart_task/full_finetune_so101/so101_synthetic_groot_config.py
```

这不是 LeRobot 官方格式，而是 GR00T 对自定义 embodiment 的训练配置。它注册：

```text
EmbodimentTag.NEW_EMBODIMENT
```

并声明：

```text
video:
  top
  wrist

state:
  single_arm  -> observation.state[0:5]
  gripper     -> observation.state[5:6]

action:
  single_arm  -> action[0:5], relative joint action
  gripper     -> action[5:6], absolute gripper target

language:
  annotation.human.task_description
```

训练相机映射是：

```text
observation.images.camera1 -> video.top
observation.images.camera3 -> video.wrist
observation.images.camera2 -> 不送入 GR00T 微调
```

这里把 `camera1` 命名为 `top`，是因为实际图像语义是 top/global 视角，而不是 front 视角。

如果只有 wrist 单相机数据，不要改双相机配置文件；改用独立的 wrist-only 配置：

```text
experiments/groot_n17_isaac_smart_task/full_finetune_so101/so101_synthetic_groot_wrist_only_config.py
```

准备数据时加：

```bash
--camera-layout wrist-only \
--wrist-camera-key observation.images.camera3
```

默认输出目录名会自动加 `_wrist_only` 后缀，例如：

```text
outputs/groot_so101_synthetic_datasets/so101_lego_pick_0609_1722_wrist_only
```

如果你的单相机数据里 wrist 图像不是 `observation.images.camera3`，只需要把
`--wrist-camera-key` 换成真实 feature key。训练和部署同一个 checkpoint 时必须加载同一份
wrist-only modality config。

## 本机 5060 Ti 固定 profile

本机路径固定如下。以后回到这台机器，先复制这一段：

```bash
export SMART_PROJECT=/home/yzliu/smart_project
export GROOT_ROOT=/home/yzliu/Isaac-GR00T-py312
export LEISAAC_ROOT=/home/yzliu/LeIsaac
export LEROBOT_ROOT="$SMART_PROJECT/lerobot"
```

固定资产：

```text
$SMART_PROJECT/dataset/so101_lego_pick_0609_1722
$SMART_PROJECT/dataset/so101_lego_pick_0609_1722_mimic
$SMART_PROJECT/outputs/groot_so101_synthetic_datasets
$SMART_PROJECT/outputs/groot_so101_synthetic_finetune
$GROOT_ROOT
$LEISAAC_ROOT
$LEROBOT_ROOT
```

本机环境自检：

```bash
conda run -n lerobot python -c "import pyarrow; print('lerobot env ok')"
"$GROOT_ROOT/.venv/bin/python" -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Isaac / LeIsaac 自检放在 zero-shot 文档里，因为它属于仿真部署链路，不属于 full fine-tune 数据准备。

## 本机 5060 Ti：推荐命令

先准备 prepared copy：

```bash
cd /home/yzliu/smart_project
conda run -n lerobot python \
  experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
    --instruction "Pick up the red 2x4 lego brick." \
    --force-prepare \
    --skip-stats \
    --prepare-only
```

只做 1 episode metadata/video smoke：

```bash
cd /home/yzliu/smart_project
conda run -n lerobot python \
  experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
    --max-episodes 1 \
    --force-prepare \
    --skip-stats \
    --prepare-only \
    --instruction "Pick up the red 2x4 lego brick."
```

在 GR00T venv 里跑 stats / 最小 fine-tune smoke：

```bash
cd /home/yzliu/smart_project
/home/yzliu/Isaac-GR00T-py312/.venv/bin/python \
  experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
    --skip-prepare \
    --max-steps 1 \
    --save-steps 1 \
    --global-batch-size 1 \
    --gradient-accumulation-steps 1
```
