# 另一台 Linux 训练机上的 GR00T SO101 微调准备手册

本文记录把当前 `experiments/groot_n17_isaac_smart_task` 微调链路迁移到另一台 Linux 设备时的步骤。范围先限定为：

- 安装和验证 Isaac-GR00T。
- 优先复用本机已经准备好的 SO101 prepared v2.1 数据集，先验证目标机能否跑 fine-tune。
- 必要时再准备 SO101 合成数据，把 LeRobot v3 转成 GR00T 可读的 v2.1 prepared copy。
- 生成 GR00T stats，并启动 fine-tune 入口做 smoke test 或正式训练。
- 做最小模型加载 / 推理 smoke test。

暂不覆盖真机部署、PolicyServer、IsaacLab 闭环控制和真实 SO101 上机。

## 1. 目标机状态记录

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

## 2. 当前优先策略

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

## 3. 环境分层原则

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

## 4. CUDA / 系统层准备：先查，再决定是否 `smartscape`

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

## 5. `guest1` 执行线：用户层环境变量

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

## 6. `guest1` 创建用户目录并安装 Isaac-GR00T

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

## 7. 同步项目代码和 prepared 数据

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

## 8. 可选：准备 LeRobot / 转换环境

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

## 9. 可选：数据格式转换

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
  experiments/groot_n17_isaac_smart_task/train_so101_synthetic_groot.py \
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
  experiments/groot_n17_isaac_smart_task/train_so101_synthetic_groot.py \
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

## 10. `guest1` 跑 GR00T stats 和 fine-tune smoke test

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
  experiments/groot_n17_isaac_smart_task/train_so101_synthetic_groot.py \
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

## 11. 5090 32GB 训练策略更新

当前官方默认 fine-tune 路线会训练大量 action-head / diffusion 相关参数。本机曾看到：

```text
Total parameters: 3,144,016,000
Trainable parameters: 1,620,515,968 (51.54%)
```

远程 5090 PC 已经验证过一次：CUDA 13.0 toolkit 至少能把训练跑到 OOM，没有出现明确 CUDA 版本相关报错；但默认 / 全量 action-head fine-tune 即使在 32GB 5090 上仍然 OOM。

因此当前不要继续在 `train_so101_synthetic_groot.py` 默认 fine-tune 路线上盲目调 batch size。下一步改用低显存入口：

```text
experiments/groot_n17_isaac_smart_task/train_so101_synthetic_groot_lowmem.py
```

详细说明见：

```text
experiments/groot_n17_isaac_smart_task/LOWMEM_LORA_FINETUNE_ZH.md
```

第一条推荐基线是 `projector-only`，它冻结 LLM、视觉骨干、diffusion Transformer 和 VLLN，只训练 action head 的 projector / encoder / decoder，保存出来仍是普通 GR00T checkpoint：

```bash
cd "$SMART_PROJECT"
"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/train_so101_synthetic_groot_lowmem.py \
    --strategy projector-only \
    --skip-stats \
    --max-steps 2000 \
    --save-steps 500 \
    --global-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --dataloader-num-workers 2 \
    --experiment-name so101_projector_only_bs1_acc16
```

如果还需要更少可训练参数，或 `projector-only` 表现不够，再试 `diffusion-lora` / `projector-plus-diffusion-lora`。LoRA 输出是 PEFT adapter，部署和推理加载要后续单独验证。

输出目录默认是：

```text
$SMART_PROJECT/outputs/groot_so101_synthetic_finetune/<experiment-name>/
```

如果要保留多组实验，显式改名字：

```bash
--experiment-name so101_synthetic_bs1_acc16_run01
```

## 12. 最小推理 / 模型加载 smoke test

这里的推理只用于验证 GR00T 安装和模型权重访问，不等同于 SO101 真机部署。

官方 demo smoke：

```bash
cd "$GROOT_ROOT"
uv run python scripts/deployment/standalone_inference_script.py \
  --model-path nvidia/GR00T-N1.7-3B \
  --dataset-path demo_data/droid_sample \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --traj-ids 1 2 \
  --inference-mode pytorch \
  --action-horizon 8
```

如果需要验证本项目 bridge 能加载 base model，而不做真实部署，可在目标机上启动：

```bash
cd "$SMART_PROJECT"
"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/groot_bridge_server.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --device cuda
```

这只说明 GR00T bridge 能加载模型并监听 socket。是否能在 IsaacLab/真机上闭环，是另一份部署文档要处理的问题。

微调 checkpoint 的推理验证流程等第一个有效 checkpoint 出来后再补。需要重点确认：

- checkpoint 目录里是否保存了 model 和 processor；
- `experiment_cfg/` 里的 modality config、statistics 是否随 checkpoint 一起可恢复；
- 自定义 `NEW_EMBODIMENT` 的 inference processor 是否需要额外传入同一个 `so101_synthetic_groot_config.py`。

## 13. 常见问题

### 13.1 `CUDA_HOME does not exist`

说明系统没有可供 DeepSpeed 探测或编译 CUDA op 的 toolkit。把 `CUDA_HOME` 指到第 4 节确认存在的 toolkit 路径，例如先试已有 13.0：

```bash
export CUDA_HOME=/usr/local/cuda-13.0
# export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

并确认：

```bash
$CUDA_HOME/bin/nvcc --version
```

### 13.2 `nvidia-smi` 显示 CUDA 13.0，能不能先试

可以先试，但要分清两件事：`nvidia-smi` 的 CUDA Version 是驱动兼容上限，不等于系统已经安装了 CUDA 13.0 toolkit；真正关键的是 `$CUDA_HOME/bin/nvcc` 是否存在。

如果目标机已有 `/usr/local/cuda-13.0/bin/nvcc`，先把 `CUDA_HOME` 指到 13.0 跑 1-step smoke test。能过就先继续，不急着装 12.8。若报 CUDA version mismatch、DeepSpeed CUDA op 编译失败，或 PyTorch extension build 明确不接受 13.0，再让 `smartscape` 安装 CUDA Toolkit 12.8。GR00T venv 的 PyTorch 是 `cu128`，所以 12.8 是更贴合当前训练栈的保底方案。

### 13.3 官方 converter import LeRobot 报 API 错

这是 LeRobot 和 GR00T 更新节奏不同造成的 API 漂移。当前脚本会自动回退到本地非破坏性 converter。只要最终 prepared dataset 是 v2.1，且有 `meta/modality.json`、mp4、parquet，就可以继续。

### 13.4 prepared 数据里为什么没有图片

LeRobot v2.1 这里的视频模态是 mp4：

```text
videos/chunk-000/<video_key>/episode_000000.mp4
```

不是一帧一张图片。训练时 loader 会按 parquet/timestamp 去视频里解码帧。

### 13.5 32GB 仍然 OOM

如果是默认 fine-tune 路线 OOM，这已经符合预期：默认训练策略太重。改用低显存文档里的三条路线：

- `projector-only`
- `diffusion-lora`
- `projector-plus-diffusion-lora`

如果这些路线的 `--global-batch-size 1` smoke test 仍然 OOM，记录峰值显存、策略名、trainable parameter 数量和完整报错，再决定是否向公司申请云端大显存资源。

### 13.6 `guest1` 没有 sudo 怎么办

正常。`guest1` 不应该负责系统安装。处理方式：

```text
smartscape -> 安装/修复系统层依赖
guest1     -> 运行 GR00T 用户层环境和训练
```

如果 `guest1` 遇到缺系统包的问题，例如 `ffmpeg`、`libaio-dev`、`nvcc` 缺失，把具体命令和报错交给 `smartscape` 执行第 4 节。

## 14. 迁移完成标准

目标机迁移到“可以开始认真微调”的最低标准：

- `nvidia-smi` 正常。
- `$CUDA_HOME/bin/nvcc --version` 正常。
- GR00T `.venv` import 正常。
- HF gated model 权限正常。
- prepared v2.1 数据已经同步到目标机。
- prepared copy 中有 parquet 和两路 mp4。
- GR00T stats 生成成功。
- `--max-steps 1 --global-batch-size 1` 能走进训练 step；若 OOM，需要记录峰值显存和报错位置，再决定冻结/LoRA 策略。
