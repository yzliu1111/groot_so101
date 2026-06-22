# Full fine-tune：SO101 合成数据准备与官方微调入口

读完这份文件，你应该能区分：

1. 原始 LeRobot v3 数据放在哪里。
2. GR00T 可读的 prepared v2.1 copy 放在哪里。
3. 本机 5060 Ti 和 remote 5090 在 full fine-tune 链路里各自承担什么。

## 机器视角

| 机器 | 推荐用途 | 不推荐做什么 |
|---|---|---|
| 本机 RTX 5060 Ti | 数据转换、prepared copy 生成、stats、loader smoke、小步数逻辑验证 | 不用它判断 2000-step full fine-tune 是否可行 |
| 公司 remote RTX 5090 | 复用 prepared 数据做训练 smoke；如果要重新转换数据，再补 LeRobot 环境 | 不要先从 `--global-batch-size 32` 盲跑 |

## 资产应该放在哪里

```text
本机：
/home/yzliu/smart_project/dataset/so101_lego_pick_0609_1722
/home/yzliu/smart_project/dataset/so101_lego_pick_0609_1722_mimic
/home/yzliu/smart_project/outputs/groot_so101_synthetic_datasets/
/home/yzliu/Isaac-GR00T

remote 5090：
/home/guest1/smart_project/outputs/groot_so101_synthetic_datasets/
/home/guest1/Isaac-GR00T
```

本机可以保留原始 `dataset/`；remote 第一轮只需要同步 prepared 数据，不必同步原始 v3 数据。

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

本机环境是刻意隔离的：

```text
LeRobot 数据/schema 转换 -> conda lerobot
GR00T stats / launch_finetune -> /home/yzliu/Isaac-GR00T/.venv
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
/home/yzliu/Isaac-GR00T/.venv/bin/python \
  experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
    --skip-prepare \
    --max-steps 2000 \
    --save-steps 500 \
    --global-batch-size 32
```

本机 RTX 5060 Ti 16GB 更适合做数据转换、loader smoke test、stats 生成和小步数逻辑验证；完整
GR00T 微调大概率仍然需要上云或使用 40GB+ 显存设备。

如果要迁移到另一台 Linux 训练机，直接看本文后面的 remote 章节；低显存策略看第三阶段文档：

```text
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
/home/yzliu/Isaac-GR00T/scripts/lerobot_conversion/convert_v3_to_v2.py
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
/home/yzliu/Isaac-GR00T/.venv/bin/python \
  experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
    --skip-prepare \
    --max-steps 1 \
    --save-steps 1 \
    --global-batch-size 1 \
    --gradient-accumulation-steps 1
```

## Remote 5090：基础迁移与 smoke

下面保留原 remote 手册里最有用的系统/用户层准备内容。低显存降显存策略不放在这里，统一看：

```text
experiments/groot_n17_isaac_smart_task/lowmem_lora_freeze_so101/README_ZH.md
```

### Remote 原手册摘录：另一台 Linux 训练机上的 GR00T SO101 微调准备

本文记录把当前 `experiments/groot_n17_isaac_smart_task` 微调链路迁移到另一台 Linux 设备时的步骤。范围先限定为：

- 安装和验证 Isaac-GR00T。
- 优先复用本机已经准备好的 SO101 prepared v2.1 数据集，先验证目标机能否跑 fine-tune。
- 必要时再准备 SO101 合成数据，把 LeRobot v3 转成 GR00T 可读的 v2.1 prepared copy。
- 生成 GR00T stats，并启动 fine-tune 入口做 smoke test 或正式训练。
- 做最小模型加载 / 推理 smoke test。

暂不覆盖真机部署、PolicyServer、IsaacLab 闭环控制和真实 SO101 上机。

### 1. 目标机状态记录

目标机信息后续可以边拿到边补：

| 项 | 当前记录 |
|---|---|
| 主机名 / 机器 | TODO |
| OS | TODO，例如 Ubuntu 22.04 |
| GPU / 显存 | TODO，例如 RTX 5090 32GB |
| `nvidia-smi` driver | TODO |
| `nvidia-smi` CUDA Version | TODO，注意这是驱动兼容上限，不等于 CUDA toolkit 已安装 |
| CUDA toolkit | TODO，例如已有 `/usr/local/cuda-13.0`，或后续安装 `/usr/local/cuda-12.8` |
| 日常登录账号 | `guest1`，当前已知没有 sudo 权限 |
| sudo 账号 | `smartscape`，用于系统层安装和修复 |
| 目标机当前已知环境 | 只有 Isaac Sim；没有 Isaac-GR00T / LeRobot / LeIsaac / IsaacLab |
| 项目目录 | 规划为 `/home/guest1/smart_project`，当前未必存在 |
| Isaac-GR00T 目录 | 规划为 `/home/guest1/Isaac-GR00T`，当前未必存在 |
| IsaacLab | 未安装；第一轮微调验证不需要 |
| LeRobot 环境 | 非第一优先级；当前先复用已经 prepared 的 v2.1 数据 |
| HuggingFace 权限 | 需要能访问 `nvidia/GR00T-N1.7-3B` 和 `nvidia/Cosmos-Reason2-2B` |

下面两个变量是“目标路径规划”，不是说目标机当前已经有这些目录。等第 6、7 节创建目录 / clone 仓库后，它们才会真实存在。

```bash
export SMART_PROJECT=/home/guest1/smart_project
export GROOT_ROOT=/home/guest1/Isaac-GR00T
```

如果目标机路径不同，只要同步替换这两个变量即可。

### 2. 当前优先策略

把目标机当成一张白纸处理。当前只知道它有 Isaac Sim；不要假设它已经有 IsaacLab、LeIsaac、LeRobot 或 Isaac-GR00T。

这次远程机的第一目标不是重新做数据转换，而是从零补齐最少训练环境，然后验证：

```text
已 prepared 的 SO101 v2.1 数据 + GR00T N1.7 fine-tune
在 5090 32GB 上能否跑过 smoke step，是否仍然 OOM。
```

因此第一轮只需要同步：

```text
$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/
$SMART_PROJECT/outputs/groot_so101_synthetic_datasets/
```

暂时不需要同步原始 `dataset/`，也不需要先安装完整 LeRobot。等确认训练机能进入训练 step 后，再决定是否在目标机补 v3 -> v2.1 转换能力。

Isaac Sim 暂时不参与这条链路。它说明目标机可能具备 NVIDIA 图形/仿真基础，但 fine-tune smoke test 不会启动 Isaac Sim。

账号分工：

| 账号 | sudo | 做什么 |
|---|---:|---|
| `smartscape` | 有 | 系统层：driver 检查、CUDA toolkit、ffmpeg、git-lfs、build-essential、libaio-dev |
| `guest1` | 无 | 用户层：clone GR00T、建 `.venv`、HF 登录、同步 prepared 数据、运行 stats/fine-tune |

不要把 GR00T 仓库和训练输出放在 `smartscape` 的 home 下再让 `guest1` 跑。除非专门配置共享组权限，否则容易出现 checkpoint、cache、HF token、uv cache 的权限问题。推荐 `guest1` 自己拥有：

```text
/home/guest1/Isaac-GR00T
/home/guest1/smart_project
```

### 3. 环境分层原则

不要把 IsaacLab、LeRobot、GR00T 全塞进同一个 Python 环境。

推荐分层：

```text
系统层       -> NVIDIA driver、CUDA toolkit、ffmpeg、git-lfs、build-essential
LeRobot/转换 -> 可选第二阶段，只有需要在目标机重新做 v3 -> v2.1 时才建
GR00T        -> Isaac-GR00T 自己的 .venv / uv 环境
IsaacLab     -> 当前目标机没有；只在后续仿真闭环时才考虑
Isaac Sim    -> 目标机已有；第一轮 fine-tune 不使用
```

当前第一轮 SO101 微调链路最关键的是：

```text
prepared v2.1 数据同步                      不依赖 LeRobot
GR00T stats / fine-tune                     在 guest1 的 Isaac-GR00T .venv 里跑
```

LeRobot v3 -> v2.1 转换是第二阶段能力，见后面的“可选：数据格式转换”。

### 4. CUDA / 系统层准备：先查，再决定是否 `smartscape`

目标机需要 NVIDIA driver 正常：

```bash
nvidia-smi
```

`nvidia-smi` 中的 `CUDA Version` 是驱动能力上限，不说明系统里已经有 `nvcc`。DeepSpeed 微调会找：

```text
$CUDA_HOME/bin/nvcc
```

当前项目主环境基本都是 CUDA 12 系，尤其 GR00T 是 PyTorch `cu128`，因此最稳妥的系统 toolkit 是 CUDA Toolkit 12.8。但如果目标机已经有 CUDA 13.0 toolkit，不是绝对不能跑，建议先用 `guest1` 做一次低成本 smoke test，再决定是否切到 `smartscape` 安装 12.8。

先用 `guest1` 检查目标机到底有没有 toolkit：

```bash
nvidia-smi
which nvcc || true
nvcc --version || true
ls -ld /usr/local/cuda /usr/local/cuda-* 2>/dev/null || true
readlink -f /usr/local/cuda || true
```

如果能看到类似 `/usr/local/cuda-13.0/bin/nvcc`，可以先不动系统安装，在第 5 节把 `CUDA_HOME` 指向这个 13.0 toolkit，然后直接跑第 10 节的 1-step smoke test。

这一步可能成功，也可能因为 PyTorch `cu128` / DeepSpeed CUDA op 编译检查报版本不匹配。只要失败信息指向 CUDA toolkit 版本、DeepSpeed 编译、Triton/CUDA extension build，再让 `smartscape` 安装 12.8 toolkit-only。不要因为 `nvidia-smi` 显示 CUDA 13.0 就去降级或更换 driver。

如果目标机没有可用 toolkit，或者 13.0 toolkit 的 smoke test 失败，使用 `smartscape` 账号安装 CUDA 12.8 toolkit-only 和系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  git git-lfs ffmpeg curl build-essential libaio-dev tmux htop

if ! apt-cache show cuda-toolkit-12-8 >/dev/null 2>&1; then
  UBUNTU_VERSION=$(. /etc/os-release && echo "${VERSION_ID//.}")
  curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/cuda-keyring_1.1-1_all.deb" -o /tmp/cuda-keyring.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
  rm /tmp/cuda-keyring.deb
  sudo apt-get update
fi

sudo apt-get install -y --no-install-recommends cuda-toolkit-12-8
```

验证：

```bash
/usr/local/cuda-12.8/bin/nvcc --version
```

如果 `/usr/local/cuda` 没有指向 12.8，不强行改系统 alternatives 也可以；`guest1` 训练时显式设置 `CUDA_HOME=/usr/local/cuda-12.8` 即可。

`smartscape` 做完系统层之后，切回 `guest1` 继续下面步骤。

### 5. `guest1` 执行线：用户层环境变量

`guest1` 每次开新 shell 先设置。`CUDA_HOME` 使用第 4 节确认可用的 toolkit 路径：如果先试目标机已有 CUDA 13.0，就指向 13.0；如果已经由 `smartscape` 安装了 12.8，或 13.0 smoke test 失败，就指向 12.8。

```bash
export SMART_PROJECT=/home/guest1/smart_project
export GROOT_ROOT=/home/guest1/Isaac-GR00T
export CUDA_HOME=/usr/local/cuda-13.0
# export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

验证：

```bash
whoami
which nvcc
nvcc --version
nvidia-smi
```

期望：

```text
whoami -> guest1
nvcc   -> $CUDA_HOME/bin/nvcc
```

如果 `guest1` 看不到 `nvcc`，但 `smartscape` 能看到，通常是环境变量没设；如果 `$CUDA_HOME/bin/nvcc` 本身不存在，需要回到第 4 节重新确认 toolkit 路径或让 `smartscape` 补系统层。

### 6. `guest1` 创建用户目录并安装 Isaac-GR00T

目标机如果还没有 GR00T：

```bash
export SMART_PROJECT=/home/guest1/smart_project
export GROOT_ROOT=/home/guest1/Isaac-GR00T

mkdir -p "$SMART_PROJECT"
cd /home/guest1
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd "$GROOT_ROOT"

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

export UV_HTTP_TIMEOUT=300
uv sync --python 3.10
uv pip install -e .
```

不建议 `guest1` 直接跑 GR00T 官方 dGPU 一键脚本，因为它包含 apt/sudo 系统安装部分：

```bash
cd "$GROOT_ROOT"
bash scripts/deployment/dgpu/install_deps.sh
```

如果确实要用它，应该让 `smartscape` 先完成系统层，`guest1` 再只执行 `uv sync` / `uv pip install -e .` 这类用户层命令。

HuggingFace 权限：

```bash
cd "$GROOT_ROOT"
uv run hf auth login
```

还需要在 HuggingFace 网页上同意访问：

```text
nvidia/GR00T-N1.7-3B
nvidia/Cosmos-Reason2-2B
```

GR00T 基础 import 验证：

```bash
cd "$GROOT_ROOT"
uv run python -c "import gr00t; print('GR00T import ok')"
```

GPU 真验证建议跑一个实际矩阵乘法，而不是只看 `torch.cuda.is_available()`：

```bash
cd "$GROOT_ROOT"
uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
x = torch.randn(4096, 4096, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("matmul ok:", float(y.sum()))
PY
```

再确认 PyTorch extension 系统看到的 `CUDA_HOME`。如果这里显示的是 13.0，就说明后续 DeepSpeed / CUDA op 编译会按 13.0 试跑；如果 1-step smoke test 因版本检查失败，再切到 12.8。

```bash
cd "$GROOT_ROOT"
uv run python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("extension CUDA_HOME:", CUDA_HOME)
PY
```

### 7. 同步项目代码和 prepared 数据

第一轮只需要目标机有：

```text
$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/
$SMART_PROJECT/outputs/groot_so101_synthetic_datasets/
```

在目标机 `guest1` 下先建项目子目录：

```bash
export SMART_PROJECT=/home/guest1/smart_project
mkdir -p "$SMART_PROJECT/experiments"
mkdir -p "$SMART_PROJECT/outputs"
```

从本机传输时，可以只同步实验目录和已经 prepared 的数据：

```bash
rsync -aP /home/yzliu/smart_project/experiments/groot_n17_isaac_smart_task/ \
  guest1@TARGET:/home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/

rsync -aP /home/yzliu/smart_project/outputs/groot_so101_synthetic_datasets/ \
  guest1@TARGET:/home/guest1/smart_project/outputs/groot_so101_synthetic_datasets/
```

如果目标机磁盘足够，也可以直接同步整个项目：

```bash
rsync -aP /home/yzliu/smart_project/ guest1@TARGET:/home/guest1/smart_project/
```

但不要为了第一轮微调验证同步原始 `dataset/`；已经 prepared 的 v2.1 数据足够启动训练。

验证 prepared 数据存在：

```bash
cd "$SMART_PROJECT"
find outputs/groot_so101_synthetic_datasets -maxdepth 4 -type f -name 'info.json' -print
find outputs/groot_so101_synthetic_datasets -type f -name '*.parquet' | wc -l
find outputs/groot_so101_synthetic_datasets -type f -name '*.mp4' | wc -l
```

当前本机 prepared 数据的参考数量是：

```text
parquet: 100
mp4:     200
```

### 8. 可选：准备 LeRobot / 转换环境

第一轮不需要这一节。只有后续决定在目标机重新从原始 LeRobot v3 数据生成 prepared v2.1 数据，才需要 LeRobot/转换环境。

目标机当前没有 LeRobot。到需要时，可以先建一个轻量转换环境：

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
python -m pip install pyarrow
```

系统层需要有 `ffmpeg`，因为 v3 -> v2.1 转换会切分 mp4：

```bash
sudo apt-get install -y ffmpeg
```

完整 LeRobot 不一定是转换的硬要求。当前 `train_so101_synthetic_groot.py` 会优先尝试 GR00T 官方 converter：

```text
$GROOT_ROOT/scripts/lerobot_conversion/convert_v3_to_v2.py
```

如果官方 converter 因 LeRobot API 漂移无法 import，会自动回退到本实验里的轻量非破坏性转换逻辑。

### 9. 可选：数据格式转换

第一轮不跑这一节；直接使用本机同步过去的 `outputs/groot_so101_synthetic_datasets/`。

默认源数据是：

```text
$SMART_PROJECT/dataset/so101_lego_pick_0609_1722
$SMART_PROJECT/dataset/so101_lego_pick_0609_1722_mimic
```

默认输出 prepared copy：

```text
$SMART_PROJECT/outputs/groot_so101_synthetic_datasets/
```

运行：

```bash
cd "$SMART_PROJECT"
conda run -n lerobot python \
  experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
    --groot-root "$GROOT_ROOT" \
    --instruction "Pick up the red 2x4 lego brick." \
    --force-prepare \
    --skip-stats \
    --prepare-only
```

如果目标机数据路径不同，用 `--source-dataset` 显式指定，可重复传多个：

```bash
cd "$SMART_PROJECT"
conda run -n lerobot python \
  experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
    --groot-root "$GROOT_ROOT" \
    --source-dataset /data/so101_lego_pick_0609_1722 \
    --source-dataset /data/so101_lego_pick_0609_1722_mimic \
    --prepared-root "$SMART_PROJECT/outputs/groot_so101_synthetic_datasets" \
    --instruction "Pick up the red 2x4 lego brick." \
    --force-prepare \
    --skip-stats \
    --prepare-only
```

相机映射默认是：

```text
observation.images.camera1 -> GR00T video.top
observation.images.camera3 -> GR00T video.wrist
observation.images.camera2 -> 当前不用于 GR00T 微调
```

如果目标数据的相机 key 不同，用：

```bash
--top-camera-key observation.images.camera1
--wrist-camera-key observation.images.camera3
```

验证 prepared 数据是否是 v2.1：

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("outputs/groot_so101_synthetic_datasets")
for p in sorted(root.iterdir()):
    if not (p / "meta" / "info.json").exists():
        continue
    info = json.load(open(p / "meta" / "info.json"))
    print(p)
    print("  codebase_version:", info.get("codebase_version"))
    print("  data_path:", info.get("data_path"))
    print("  video_path:", info.get("video_path"))
    print("  total_episodes:", info.get("total_episodes"))
    print("  features:", sorted(info.get("features", {}).keys()))
PY
```

期望看到：

```text
codebase_version: v2.1
data_path: data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
video_path: videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4
```

注意：v2.1 里图像通常是 mp4，不是一帧帧 png/jpg。数据目录应该类似：

```text
data/chunk-000/episode_000000.parquet
videos/chunk-000/observation.images.camera1/episode_000000.mp4
videos/chunk-000/observation.images.camera3/episode_000000.mp4
meta/info.json
meta/modality.json
meta/tasks.jsonl
meta/episodes.jsonl
```

### 10. `guest1` 跑 GR00T stats 和 fine-tune smoke test

进入目标机 GR00T 环境前，先设 CUDA toolkit，保持和第 5 节一致：

```bash
export SMART_PROJECT=/home/guest1/smart_project
export GROOT_ROOT=/home/guest1/Isaac-GR00T
export CUDA_HOME=/usr/local/cuda-13.0
# export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

先跑 1 step smoke test。这个命令的目标不是训练出模型，而是验证：

- prepared dataset 可被 GR00T loader 读取；
- `meta/modality.json` 可用；
- stats 可生成；
- model / processor / Trainer 能走到训练 step。

```bash
cd "$SMART_PROJECT"
"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
    --groot-root "$GROOT_ROOT" \
    --skip-prepare \
    --max-steps 1 \
    --save-steps 1 \
    --global-batch-size 1 \
    --dataloader-num-workers 0
```

如果已经在本机生成并同步了 `meta/stats.json` / `meta/relative_stats.json`，可以加 `--skip-stats` 节省时间；但第一轮目标机 smoke test 推荐不加，让目标机自己跑一次 stats，确认 GR00T 环境能完整读取数据。

如果看到类似下面的日志，说明已经进入训练入口：

```text
Creating custom train dataloader
0%|...| 0/1
Rank 0, Worker 0: Caching shard...
```

如果在此之后 OOM，通常说明训练策略太重，不是数据转换失败。
