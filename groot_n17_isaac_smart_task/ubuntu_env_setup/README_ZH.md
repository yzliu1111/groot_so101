# 通用 Ubuntu 环境搭建：GR00T / LeRobot / LeIsaac / IsaacLab

这份文件不是“某台目标机记录”。它是一份可以搬到任意 Ubuntu 22.04 终端执行的安装剧本。现在的原则是：先复刻本机 canonical 环境边界，再跑当前项目，不再在坏环境里反复猜包版本。

canonical 来源：

```text
/home/yzliu/Downloads/project_docs/CANONICAL_PROJECT_DOCS/ENV_ubuntu22_robot_learning_stack_v2_REQUIRED_SUMMARY.md
```

本文只抽取当前项目必须用到的部分。完整系统记录、驱动选择、CUDA 12.8 函数和 LeIsaac 安装实测，以上面那份 canonical 文档为准。

读完这份文件，你应该能做到：

1. 在一台新的 Ubuntu GPU 机器上补齐当前项目需要的 Python 环境。
2. 分清哪些步骤需要 sudo，哪些步骤普通用户就能做。
3. 分清 train-only、LeIsaac 仿真推理、SO101 微调后推理分别需要哪些资产。
4. 避免把本机路径或某台历史目标机的路径硬编码进迁移命令。

## 0. 先填这些坑

所有后续命令都依赖这组变量。先在目标机 shell 里填好：

```bash
# 当前项目目录。里面应该有 experiments/、outputs/ 等。
export SMART_PROJECT="/home/guest1/smart_project"

# Isaac-GR00T 仓库目录。当前默认使用 Python 3.12 checkout。
export GROOT_ROOT="/home/guest1/Isaac-GR00T-py312"

# LeIsaac / LeRobot 源码目录。
# 本机 canonical:     export LEISAAC_ROOT="$HOME/LeIsaac"
# 5090 目标机当前:   export LEISAAC_ROOT="$SMART_PROJECT/leisaac"
export LEISAAC_ROOT="$SMART_PROJECT/leisaac"
export LEISAAC_ASSETS_ROOT="$LEISAAC_ROOT/assets"
export LEROBOT_ROOT=""

# Conda 环境名。这里指“运行 Isaac/LeIsaac runner 的 env”，不要求名字必须叫 leisaac。
# 本机 canonical:     export LEISAAC_ENV="leisaac"
# 目标机若还没单独建 leisaac env：填当前实际使用的 Isaac env，例如 isaaclab；
# 如果旧 env 已经被 pip 搅乱，则后面第 7 节建议新建 leisaac_clean。
export LEISAAC_ENV="leisaac"
export LEROBOT_ENV="lerobot"

# Isaac Sim / Kit EULA。后面所有 Isaac / LeIsaac 命令都默认继承这个值。
export OMNI_KIT_ACCEPT_EULA=YES

# 旧 Isaac Sim binary 根目录。canonical 主线不需要；只做旧 binary 诊断时再填。
export ISAACSIM_ROOT=""

# GR00T 训练 / DeepSpeed 需要真实 CUDA toolkit，也就是 $CUDA_HOME/bin/nvcc。
# 推理和 Isaac runner 可以先没有 nvcc；训练前必须补齐。
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
if [ ! -x "$CUDA_HOME/bin/nvcc" ] && [ -x /usr/local/cuda/bin/nvcc ]; then
  export CUDA_HOME=/usr/local/cuda
fi

if [ -n "${CUDA_HOME:-}" ] && [ -d "$CUDA_HOME" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
```

自检：

```bash
whoami
pwd
echo "$SMART_PROJECT"
echo "$GROOT_ROOT"
echo "$LEISAAC_ROOT"
echo "$LEISAAC_ASSETS_ROOT"
echo "$LEISAAC_ENV"
echo "$LEROBOT_ROOT"
echo "$ISAACSIM_ROOT"
echo "$OMNI_KIT_ACCEPT_EULA"
echo "$CUDA_HOME"
which nvcc || true
nvcc --version || true
```

如果 `SMART_PROJECT`、`GROOT_ROOT` 或 `LEISAAC_ENV` 为空，或者路径不存在，不要继续。`LEROBOT_ROOT`、`ISAACSIM_ROOT` 可以按任务需要留空。
如果要跑 GR00T 训练，`CUDA_HOME/bin/nvcc` 必须存在；如果只跑 Isaac runner / GR00T bridge 推理，可以先没有 nvcc。

当前已知机器 profile：

```bash
# 本机 5060 Ti canonical profile
export SMART_PROJECT=/home/yzliu/smart_project
export GROOT_ROOT=/home/yzliu/Isaac-GR00T-py312
export LEISAAC_ROOT=/home/yzliu/LeIsaac
export LEISAAC_ASSETS_ROOT="$LEISAAC_ROOT/assets"
export LEISAAC_ENV=leisaac
export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_HOME=/usr/local/cuda-12.8
```

```bash
# 5090 目标机当前 profile：LeIsaac repo 在 smart_project 里面
export SMART_PROJECT=/home/guest1/smart_project
export GROOT_ROOT=/home/guest1/Isaac-GR00T-py312
export LEISAAC_ROOT="$SMART_PROJECT/leisaac"
export LEISAAC_ASSETS_ROOT="$LEISAAC_ROOT/assets"
export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_HOME=/usr/local/cuda-13.0

# 如果目标机还没有单独 leisaac env，就填当前实际的 Isaac runtime env。
# 如果当前 env 已经坏了，不要继续修，后面第 7 节改用 leisaac_clean。
export LEISAAC_ENV="leisaac"
```

公司 AWS / 训练机如果系统 CUDA toolkit 是 13.2，也不要直接套 Thor / Spark 的
deployment 脚本，除非那台机器真的是对应的 aarch64 平台。官方 GR00T N1.7 当前区分是：

```text
x86_64 dGPU，例如 H100/L40/4090/5090/AWS GPU -> Python 3.12 + root pyproject + PyTorch cu128 wheels
Jetson Thor / DGX Spark aarch64                    -> Python 3.12 + scripts/deployment/thor 或 spark 的 cu13 栈
```

所以 AWS x86 GPU 上仍先走本文件第 5 节的 `uv sync --python 3.12`。如果只是推理或
普通训练入口，看到 `torch.version.cuda` 是 `12.8` 是正常的，它表示 PyTorch wheel
的用户态 CUDA 运行时，不等于机器不能有 `/usr/local/cuda-13.2`。只有当 DeepSpeed /
CUDA extension 明确需要 `nvcc` 时，才把 `CUDA_HOME` 指向那台机器真实存在的 toolkit，
例如：

```bash
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

如果编译类错误显示 CUDA 13.2 与 PyTorch cu128 wheel 不兼容，优先记录错误并改用官方
dGPU/容器路径或 CUDA 12.8 toolkit，不要把 Thor / Spark 的 aarch64 cu13 依赖栈强行装到
x86 AWS 主机。

不要把下面三类东西混成一个 Python 环境：

```text
GR00T venv：训练 / bridge server / 模型推理
Isaac runtime conda：Isaac Sim + IsaacLab + LeIsaac SO101 runtime，也就是 $LEISAAC_ENV
LeRobot conda：LeRobot dataset / 数据转换 / 可视化，也就是 $LEROBOT_ENV
```

canonical 本机已经验证过：Isaac runtime env 里 **不安装 LeRobot** 是正确状态。`lerobot: NOT FOUND` 在 `$LEISAAC_ENV` 中不是错误。

当前 GR00T 默认是 `~/Isaac-GR00T-py312` / Python 3.12。旧的
`~/Isaac-GR00T` / Python 3.10 只保留作回滚点，不再作为新实验的默认 `GROOT_ROOT`。

## 1. 先决定这台机器要承担什么

不同任务需要的环境不同：

| 目标 | 必须有 | 可以暂时没有 |
|---|---|---|
| 只训练 / lowmem 微调 | `SMART_PROJECT`、`GROOT_ROOT`、prepared 数据、GR00T venv | Isaac Sim、IsaacLab、LeIsaac |
| 重新做 v3 -> v2.1 数据转换 | 上面这些 + `conda activate "$LEROBOT_ENV"` | Isaac Sim、IsaacLab |
| 在 LeIsaac 场景里部署推理 | GR00T venv + `conda activate leisaac` + Isaac Sim + IsaacLab + LeIsaac | LeRobot |
| 真实 SO101 推理 | GR00T venv + 真实机器人/相机 runner | IsaacLab 可选，当前仓库还没有完整真机 runner |

这也是为什么不要把所有东西塞进一个 Python 环境。后面所有命令按这三类终端分工：

| 终端 | Python 环境 | 用来跑什么 | 不要拿来跑什么 |
|---|---|---|---|
| GR00T 终端 | `$GROOT_ROOT/.venv/bin/python` | stats、训练、bridge server | Isaac runner |
| LeRobot 终端 | `conda activate "$LEROBOT_ENV"` | v3 -> v2.1 数据转换 | GR00T 训练、Isaac runner |
| LeIsaac 终端 | `conda activate leisaac` | LeIsaac / IsaacLab runner | GR00T 训练、LeRobot 数据转换 |

后面涉及 Python 环境的命令块都会写清楚它属于哪个终端。只要你切换任务，就先回到对应终端模板，不要把上一个环境的 `PYTHONPATH`、`PYTHONNOUSERSITE` 或 Isaac Sim hook 顺手带过去。

## 2. 系统层依赖

如果你有 sudo：

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  git git-lfs curl rsync unzip ffmpeg build-essential cmake pkg-config tmux htop
git lfs install
```

如果你没有 sudo，把下面这行发给管理员：

```text
Please install: git git-lfs curl rsync unzip ffmpeg build-essential cmake pkg-config tmux htop
```

GPU / CUDA 先只检查，不要一上来重装：

```bash
nvidia-smi
which nvcc || true
nvcc --version || true
ls -ld /usr/local/cuda /usr/local/cuda-* 2>/dev/null || true
```

`nvidia-smi` 里的 `CUDA Version` 是 driver 支持上限，不等于系统已有 CUDA toolkit。只有训练时明确需要编译 CUDA extension，再要求 `$CUDA_HOME/bin/nvcc`。

如果这台机器要跑 GR00T fine-tune / DeepSpeed 训练，继续设置并验证 CUDA toolkit：

```bash
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
if [ ! -x "$CUDA_HOME/bin/nvcc" ] && [ -x /usr/local/cuda/bin/nvcc ]; then
  export CUDA_HOME=/usr/local/cuda
fi

if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "ERROR: GR00T training needs CUDA_HOME/bin/nvcc; current CUDA_HOME=$CUDA_HOME" >&2
  echo "Install CUDA Toolkit or set CUDA_HOME to the real toolkit path." >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
"$CUDA_HOME/bin/nvcc" --version
```

如果只跑 GR00T bridge 推理或 Isaac / LeIsaac runner，这一步可以暂时跳过；不要为了推理把 driver 或 toolkit 重装一遍。

## 3. 同步项目资产

从源机器同步到目标机时，也用变量挖坑：

```bash
export SOURCE_PROJECT="__FILL_SOURCE_SMART_PROJECT__"
export TARGET_HOST="__FILL_USER_AT_HOST__"
export TARGET_PROJECT="__FILL_TARGET_SMART_PROJECT__"

ssh "$TARGET_HOST" "mkdir -p '$TARGET_PROJECT/experiments' '$TARGET_PROJECT/outputs'"

rsync -aP --exclude='__pycache__' --exclude='.pytest_cache' \
  "$SOURCE_PROJECT/experiments/groot_n17_isaac_smart_task/" \
  "$TARGET_HOST:$TARGET_PROJECT/experiments/groot_n17_isaac_smart_task/"

rsync -aP --exclude='__pycache__' --exclude='.pytest_cache' \
  "$SOURCE_PROJECT/outputs/groot_so101_synthetic_datasets/" \
  "$TARGET_HOST:$TARGET_PROJECT/outputs/groot_so101_synthetic_datasets/"
```

如果这台机器要跑 LeIsaac 仿真，再同步：

```bash
export SOURCE_LEISAAC_ROOT="__FILL_SOURCE_LEISAAC_ROOT__"
export TARGET_LEISAAC_ROOT="__FILL_TARGET_LEISAAC_ROOT__"

ssh "$TARGET_HOST" "mkdir -p '$TARGET_LEISAAC_ROOT'"

rsync -aP --exclude='__pycache__' --exclude='.pytest_cache' \
  "$SOURCE_LEISAAC_ROOT/" \
  "$TARGET_HOST:$TARGET_LEISAAC_ROOT/"
```

如果这台机器要重新做数据转换，再同步：

```bash
export SOURCE_LEROBOT_ROOT="__FILL_SOURCE_LEROBOT_ROOT__"
export TARGET_LEROBOT_ROOT="__FILL_TARGET_LEROBOT_ROOT_OR_EMPTY__"

ssh "$TARGET_HOST" "mkdir -p '$TARGET_LEROBOT_ROOT'"

rsync -aP --exclude='__pycache__' --exclude='.pytest_cache' \
  "$SOURCE_LEROBOT_ROOT/" \
  "$TARGET_HOST:$TARGET_LEROBOT_ROOT/"

rsync -aP \
  "$SOURCE_PROJECT/dataset/" \
  "$TARGET_HOST:$TARGET_PROJECT/dataset/"
```

目标机上检查：

```bash
find "$SMART_PROJECT/experiments/groot_n17_isaac_smart_task" -maxdepth 2 -type f | head
find "$SMART_PROJECT/outputs/groot_so101_synthetic_datasets" -type f -name 'info.json' -print
```

## 4. Conda / Miniforge

如果目标机已有当前用户可写的 conda：

```bash
conda --version
conda info --base
```

如果没有，装到用户目录：

```bash
curl -L -o /tmp/Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

bash /tmp/Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda init bash
```

新开 shell 后：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
```

## 5. GR00T 环境

如果目标机还没有 Isaac-GR00T：

```bash
cd "$(dirname "$GROOT_ROOT")"
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T "$(basename "$GROOT_ROOT")"
cd "$GROOT_ROOT"

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

export UV_HTTP_TIMEOUT=300
uv sync --python 3.12
```

如果已经有 `$GROOT_ROOT/.venv`，只做验证：

```bash
"$GROOT_ROOT/.venv/bin/python" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

HuggingFace 权限：

```bash
"$GROOT_ROOT/.venv/bin/hf" auth login
```

需要能访问：

```text
nvidia/GR00T-N1.7-3B
nvidia/Cosmos-Reason2-2B
```

GR00T import 验证：

```bash
"$GROOT_ROOT/.venv/bin/python" -c "import gr00t; print('GR00T import ok')"
```

官方 open-loop smoke：

```bash
cd "$GROOT_ROOT"
uv run python scripts/deployment/standalone_inference_script.py \
  --model-path nvidia/GR00T-N1.7-3B \
  --dataset-path demo_data/droid_sample \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --traj-ids 1 2 \
  --inference-mode pytorch \
  --execution-horizon 8
```

## 6. LeRobot 数据转换环境

只有需要从原始 LeRobot v3 数据重新生成 prepared v2.1 时才需要这一节。
如果不做数据转换，`LEROBOT_ROOT` 可以一直留空。要做这一节时先填：

```bash
export LEROBOT_ROOT="__FILL_TARGET_LEROBOT_REPO__"
export LEROBOT_ENV="${LEROBOT_ENV:-lerobot}"
```

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "$LEROBOT_ENV" python=3.12 -y
conda activate "$LEROBOT_ENV"

python -m pip install "pip<26" "setuptools<81" "wheel<0.47"
cd "$LEROBOT_ROOT"
python -m pip install -e ".[dataset]"
python -m pip install pyarrow av opencv-python-headless imageio tqdm
```

验证：

```bash
python -c "import lerobot, pyarrow, av; print('lerobot ok')"
```

准备 SO101 prepared 数据：

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

## 7. Isaac Sim / IsaacLab / LeIsaac 环境

只有要跑 LeIsaac 仿真推理时才需要这一节。

当前 Week 2 的实际卡点在这一节：projector-only 微调权重加载并通过 API URL / bridge 暴露这一层已经跑通，但 LeIsaac / Isaac runner 和 viewport 没有起来。因此排查顺序要先回到本节的 runtime / SimulationApp / task smoke / viewport，而不是先怀疑 checkpoint 是否有效。

这节现在按本机 canonical v2 环境复刻，不再以“修坏的 `isaaclab` env”为主线。目标状态不是强制目录名或 env 名完全一样，而是版本边界和职责边界一致：

```text
conda env        = $LEISAAC_ENV；本机 canonical 是 leisaac，目标机可用现有 Isaac env 或新建 leisaac_clean
Python           = 3.11
repo             = $LEISAAC_ROOT；本机 canonical 是 ~/LeIsaac，5090 目标机当前是 $SMART_PROJECT/leisaac
LeIsaac          = 0.4.0
IsaacSim         = 5.1.0.0
IsaacLab         = 2.3.0
Torch            = 2.7.0+cu128
TorchVision      = 0.22.0+cu128
NumPy            = 1.26.0
Packaging        = 23.0
Wheel            = 0.41.3
Setuptools       = 80.9.0
LeRobot in env   = 不安装；NOT FOUND 是正确状态
```

关键原则：

```text
不要在 Isaac runtime env，也就是 `$LEISAAC_ENV` 里安装 LeRobot。
不要使用无约束的 pip install -U pip setuptools wheel。
不要把已有的坏 isaaclab 环境当成修复对象。
不要把 2025 年旧 ~/isaacsim binary 默认为可用 IsaacSim 5.1。
```

### 7.1 先填路径

本机 canonical：

```bash
export SMART_PROJECT=/home/yzliu/smart_project
export LEISAAC_ROOT="$HOME/LeIsaac"
export LEISAAC_ASSETS_ROOT="$LEISAAC_ROOT/assets"
export LEISAAC_ENV="leisaac"
```

5090 目标机当前事实：

```bash
export SMART_PROJECT=/home/guest1/smart_project
export LEISAAC_ROOT="$SMART_PROJECT/leisaac"
export LEISAAC_ASSETS_ROOT="$LEISAAC_ROOT/assets"
export LEISAAC_ENV="leisaac"
```

检查：

```bash
echo "$SMART_PROJECT"
echo "$LEISAAC_ROOT"
echo "$LEISAAC_ENV"
```

如果你想把 LeIsaac 放到别处，就只改 `LEISAAC_ROOT`，不要改项目代码。`LEISAAC_ENV` 也只是 env 名，不要求必须叫 `leisaac`。

### 7.2 准备 LeIsaac repo

如果目标机已经有 LeIsaac repo，不管它在 `~/LeIsaac` 还是 `$SMART_PROJECT/leisaac`，只看 `$LEISAAC_ROOT`：

```bash
cd "$LEISAAC_ROOT"
git status --short || true
test -f source/leisaac/pyproject.toml
test -d dependencies/IsaacLab/source/isaaclab
```

如果目标机还没有 LeIsaac：

```bash
cd "$HOME"
git clone --recurse-submodules https://github.com/LightwheelAI/leisaac.git "$LEISAAC_ROOT"

cd "$LEISAAC_ROOT"
git submodule update --init --recursive
```

如果网络不能直接 clone，就从本机同步 LeIsaac repo 到目标机的 `$LEISAAC_ROOT`。

### 7.3 选择或创建 Isaac runtime conda 环境

如果目标机还没有单独 `leisaac` env，但已经有一个你正在用的 Isaac runtime env，先把它的名字填给 `LEISAAC_ENV`：

```bash
conda env list
export LEISAAC_ENV="leisaac"
```

然后可以直接跳到 7.5 做只读验证。不要先往这个 env 里乱装包。

如果这个已有 env 已经被反复 pip 修过，或者 `pip check` / import 已经明显冲突，不要继续修它。换新名字创建干净环境，例如：

```bash
export LEISAAC_ENV="leisaac_clean"
```

创建干净环境：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME

conda create -n "$LEISAAC_ENV" python=3.11 -y
conda activate leisaac
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

如果目标机有 CUDA 13.0 toolkit，可以启用；没有也先不要因为推理/runner 强行装 toolkit：

```bash
if [ -d /usr/local/cuda-13.0 ]; then
  export CUDA_HOME=/usr/local/cuda-13.0
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
```

### 7.4 按 canonical 版本边界安装

这一节只用于新建的干净 `$LEISAAC_ENV`。如果你是在目标机现有 Isaac env 里排查，先跑 7.5；只有确认要重建环境时再回到这里。

先写 constraints。这一步的目的不是“锁死一切”，而是防止 resolver 把 `packaging` / `wheel` / `setuptools` 拉到已经验证会冲突的组合：

```bash
cat > /tmp/leisaac_canonical_constraints.txt <<'EOF'
packaging==23.0
wheel==0.41.3
setuptools==80.9.0
numpy==1.26.0
protobuf==7.35.1
flatdict==4.0.1
gymnasium==1.2.0
prettytable==3.3.0
hidapi==0.14.0.post2
EOF

export PIP_CONSTRAINT=/tmp/leisaac_canonical_constraints.txt
export PIP_NO_BUILD_ISOLATION=1
```

安装基础边界：

```bash
python -m pip install "setuptools==80.9.0" "wheel==0.41.3" "packaging==23.0" toml
```

安装 PyTorch cu128。5090 / 5060 Ti 都是 Blackwell 路线，优先用 cu128 wheel：

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

安装 IsaacLab + pip IsaacSim 5.1：

```bash
python -m pip install "isaaclab[isaacsim,all]==2.3.0" \
  --extra-index-url https://pypi.nvidia.com
```

安装 LeIsaac 自身。这里不要安装 `[lerobot]` extra：

```bash
python -m pip install --no-build-isolation "flatdict==4.0.1"
python -m pip install --no-build-isolation -e "$LEISAAC_ROOT/source/leisaac"
```

如果这里出现大规模 resolver 冲突，停止。不要继续 `pip install -U`，不要把 LeRobot 装进来。最常见原因是 env 不是空的，或者 `PIP_CONSTRAINT` 没生效。

### 7.5 基础验证

先看 pip 依赖状态：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate leisaac
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -m pip check
```

期望：

```text
No broken requirements found.
```

再检查核心 import。最后的 `lerobot: NOT FOUND` 是正确结果：

```bash
python - <<'PY'
import importlib

for name in (
    "numpy",
    "torch",
    "isaacsim",
    "isaaclab",
    "isaaclab_tasks",
    "isaaclab_assets",
    "isaaclab_rl",
    "isaaclab_mimic",
    "leisaac",
):
    module = importlib.import_module(name)
    print(name, "FOUND", getattr(module, "__file__", "<namespace>"))

try:
    importlib.import_module("lerobot")
except ModuleNotFoundError:
    print("lerobot NOT FOUND (expected in leisaac env)")
else:
    raise SystemExit("lerobot should not be installed in leisaac env")
PY
```

再用项目 probe 启动 Isaac app：

```bash
export SMART_PROJECT="/home/guest1/smart_project"
export LEISAAC_ROOT="$SMART_PROJECT/leisaac"
export LEISAAC_ASSETS_ROOT="$LEISAAC_ROOT/assets"
export OMNI_KIT_ACCEPT_EULA=YES
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python "$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/ubuntu_env_setup/isaac_stack_probe.py" --launch-app
```

如果这里报 `omni.physx` / `omni.physics`，不要 pip install 这些名字。它们是 Isaac / Kit extension，不是普通 Python 包。优先确认 IsaacSim 是 `5.1.0.0`，并确认 `python -m pip check` 没有冲突。

### 7.6 补 LeIsaac 资产

LeIsaac repo 本体不包含所有 USD 资产。最小 `LiftCube-Direct` smoke 需要这两个：

```bash
export SMART_PROJECT="/home/guest1/smart_project"
export LEISAAC_ROOT="$SMART_PROJECT/leisaac"
export LEISAAC_ASSETS_ROOT="$LEISAAC_ROOT/assets"

cd "$LEISAAC_ROOT"

curl -L -o /tmp/table_with_cube_asset.zip \
  "https://github.com/LightwheelAI/leisaac/releases/download/v0.1.2/table_with_cube.zip"
unzip -o /tmp/table_with_cube_asset.zip -d assets

mkdir -p assets/scenes
if [ -d assets/table_with_cube ] && [ ! -e assets/scenes/table_with_cube ]; then
  mv assets/table_with_cube assets/scenes/table_with_cube
fi

mkdir -p assets/robots
curl -L -o assets/robots/so101_follower.usd \
  "https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd"
```

检查：

```bash
test -f "$LEISAAC_ROOT/assets/scenes/table_with_cube/scene.usd"
test -f "$LEISAAC_ROOT/assets/robots/so101_follower.usd"
```

`test -f ...` 成功时不会输出任何内容；没有输出通常就是通过。想看到明确结果可以这样跑：

```bash
test -f "$LEISAAC_ROOT/assets/scenes/table_with_cube/scene.usd" && echo "table_with_cube scene OK"
test -f "$LEISAAC_ROOT/assets/robots/so101_follower.usd" && echo "so101_follower USD OK"
```

### 7.7 生成 SmartTask portable scene

如果要跑 `LeIsaac-SO101-SmartTask-v0`，先把历史机器里的绝对 USD 引用转成 repo-local 引用。这个步骤不覆盖原始 `scene.usd`，只生成：

```text
$LEISAAC_ROOT/assets/scenes/smart_scene/scene_portable.usda
```

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate leisaac
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export SMART_PROJECT="/home/guest1/smart_project"
export LEISAAC_ROOT="$SMART_PROJECT/leisaac"
export LEISAAC_ASSETS_ROOT="$LEISAAC_ROOT/assets"
export OMNI_KIT_ACCEPT_EULA=YES

python "$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/ubuntu_env_setup/make_smart_scene_portable.py" --write --check
test -f "$LEISAAC_ROOT/assets/scenes/smart_scene/scene_portable.usda" && echo "SmartTask portable scene OK"
```

如果脚本打印 `pxr is not importable before Kit startup; launching headless Isaac app once.`，这是正常 fallback：pip IsaacSim 环境里 USD/pxr 绑定常常要在 Kit 初始化后才进入 Python 路径。

成功时至少应看到这两行：

```text
[ OK ] No absolute /home or /Users USD paths remain after rewrite.
[ OK ] Wrote portable scene: /home/guest1/smart_project/leisaac/assets/scenes/smart_scene/scene_portable.usda
```

如果没有看到 `Wrote portable scene`，不要继续跑 7.8/推理，先定位转换脚本为什么没有写出 `scene_portable.usda`。

如果检查输出里仍然有 `/home/ubuntu/...`、`/Users/...` 这类绝对路径，先停下来确认缺的是哪个资产。不要用创建旧路径 symlink 的方式长期绕过。

当前随仓库带的 `assets/red_2x4_lego_brick.usd` 还会引用缺失的内部 layer `red_2x4_lego_brick_03.usd`。转换脚本会从 portable scene 里移除这条坏 payload；实验 runner 默认用 `--smart-target-asset cuboid` 在当前进程里显式生成一个红色 2x4 尺寸 cuboid 作为目标物体，不修改 LeIsaac 源码，先保证闭环链路能跑通。

如果之后拿到了完整可 compose 的红色 LEGO USD，不需要改代码，在 runner 命令后追加：

```bash
--smart-target-asset /path/to/complete_red_2x4_lego_brick.usd
```

如果要完全依赖 LeIsaac scene parser，不加 fallback，可以显式传 `--smart-target-asset scene`。

runner 会在日志里打印：

```text
[runner] SmartTask target cfg ...
[runner] target object state prim_path=... root_pos_w=...
```

如果可视化里看不到替代块，先用放大尺寸和显式位置确认：

```bash
--smart-target-pos 0.0 0.25 0.07 \
--smart-target-cuboid-size 0.06 0.03 0.02
```

确认位置正确后，再把尺寸改回接近真实 2x4 LEGO：

```bash
--smart-target-cuboid-size 0.0318 0.0158 0.0096
```

### 7.8 LeIsaac task smoke

task registry 必须在 `AppLauncher` 启动后 import：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate leisaac
export SMART_PROJECT="/home/guest1/smart_project"
export LEISAAC_ROOT="$SMART_PROJECT/leisaac"
export LEISAAC_ASSETS_ROOT="$LEISAAC_ROOT/assets"
export OMNI_KIT_ACCEPT_EULA=YES
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python - <<'PY'
from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "enable_cameras": True})
simulation_app = app_launcher.app

import gymnasium as gym
import leisaac.tasks  # noqa: F401

env_ids = sorted(k for k in gym.registry.keys() if k.startswith("LeIsaac-"))
print("LeIsaac env count:", len(env_ids))
for env_id in env_ids:
    print(env_id)

simulation_app.close()
PY
```

期望能看到 15 个左右的 `LeIsaac-...` task。然后跑最小 make/reset/step：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate leisaac
export SMART_PROJECT="/home/guest1/smart_project"
export LEISAAC_ROOT="$SMART_PROJECT/leisaac"
export LEISAAC_ASSETS_ROOT="$LEISAAC_ROOT/assets"
export OMNI_KIT_ACCEPT_EULA=YES
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python - <<'PY'
import numpy as np
from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "enable_cameras": True})
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import leisaac.tasks  # noqa: F401

task = "LeIsaac-SO101-LiftCube-Direct-v0"
env_cfg = parse_env_cfg(task, device="cuda", num_envs=1)
env = gym.make(task, cfg=env_cfg)
print("env make OK")
obs, info = env.reset()
print("env reset OK")
print("action_space:", env.action_space)

action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
for step in range(5):
    obs, reward, terminated, truncated, info = env.step(action)
    print("step", step, "OK")

env.close()
simulation_app.close()
PY
```

只有这个通过，才说明 LeIsaac + IsaacSim + IsaacLab + SO101 assets 的 runtime 闭环真的通了。

### 7.9 Isaac runner 终端模板

这个终端只用于 LeIsaac / Isaac runner，不跑 GR00T 训练：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME
conda activate leisaac
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export SMART_PROJECT="/home/guest1/smart_project"
export LEISAAC_ROOT="$SMART_PROJECT/leisaac"
export LEISAAC_ASSETS_ROOT="$LEISAAC_ROOT/assets"
export OMNI_KIT_ACCEPT_EULA=YES

cd "$SMART_PROJECT"
export PYTHONPATH="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task:$LEISAAC_ROOT/source/leisaac:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

python experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/run_smart_task_closed_loop.py \
  --robot so101 \
  --control-mode joint \
  --smart-target-asset cuboid \
  --debug-cameras-only \
  --headless
```

不要用：

```bash
conda run -n "$LEISAAC_ENV" env PYTHONPATH="..." python ...
```

这个写法容易绕开或覆盖 Isaac / Kit 启动需要的环境状态。

### 7.10 旧 `~/isaacsim` binary 只做诊断

如果目标机上已经有 `~/isaacsim`，先确认版本：

```bash
export ISAACSIM_ROOT="$HOME/isaacsim"
test -f "$ISAACSIM_ROOT/VERSION" && head -n 1 "$ISAACSIM_ROOT/VERSION"
test -f "$ISAACSIM_ROOT/python.sh" && echo "python.sh ok"
```

如果它是 Isaac Sim 4.5.x，不要拿它配 LeIsaac 0.4.0 / IsaacLab 2.3.0。当前 canonical 路线使用 pip IsaacSim 5.1.0.0。

只读诊断可以这样跑：

```bash
"$ISAACSIM_ROOT/python.sh" - <<'PY'
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})
print("Isaac Sim binary SimulationApp ok")
simulation_app.close()
PY
```

这个通过只能说明旧 binary 自己能启动，不代表当前 LeIsaac runtime 应该绑定它。

## 8. Smoke tests

### 8.1 GR00T 训练 smoke

这个命令属于 **GR00T 训练终端**，不是 Isaac runner 终端。

最稳妥做法是开一个新终端，只设置 `SMART_PROJECT` 和 `GROOT_ROOT`，然后跑 `$GROOT_ROOT/.venv/bin/python`。不要先 `conda activate leisaac`，不要带 Isaac Sim 的 `PYTHONPATH`。

```bash
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH
export SMART_PROJECT="/home/guest1/smart_project"
export GROOT_ROOT="/home/guest1/Isaac-GR00T-py312"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
if [ ! -x "$CUDA_HOME/bin/nvcc" ] && [ -x /usr/local/cuda/bin/nvcc ]; then
  export CUDA_HOME=/usr/local/cuda
fi
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "ERROR: GR00T training needs CUDA_HOME/bin/nvcc; current CUDA_HOME=$CUDA_HOME" >&2
  return 1 2>/dev/null || exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
"$CUDA_HOME/bin/nvcc" --version

cd "$SMART_PROJECT"
"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/lowmem_lora_freeze_so101/train_so101_synthetic_groot_lowmem.py \
    --strategy projector-only \
    --skip-stats \
    --max-steps 1 \
    --save-steps 1 \
    --global-batch-size 1 \
    --gradient-accumulation-steps 1 \
    --dataloader-num-workers 0 \
    --experiment-name migration_projector_only_smoke
```

如果这个命令报 `importing the numpy C-extensions failed`，第一判断不是重装 numpy，而是你把 IsaacLab/Isaac Sim 的环境变量带进了 GR00T venv。开新终端重跑上面的命令，或至少确认：

```bash
echo "PYTHONPATH=${PYTHONPATH:-}"
echo "PYTHONHOME=${PYTHONHOME:-}"
"$GROOT_ROOT/.venv/bin/python" - <<'PY'
import sys
import numpy
print(sys.executable)
print(numpy.__file__)
PY
```

### 8.2 Zero-shot LeIsaac 仿真推理

终端 1：GR00T bridge。它必须使用 GR00T venv。

```bash
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH
export SMART_PROJECT="/home/guest1/smart_project"
export GROOT_ROOT="/home/guest1/Isaac-GR00T-py312"

cd "$SMART_PROJECT"
"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/groot_bridge_server.py \
    --deployment-mode zero-shot-oxe \
    --model-path nvidia/GR00T-N1.7-3B \
    --device cuda
```

终端 2：Isaac runner。只有这个终端进入 `$LEISAAC_ENV`。

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME
export SMART_PROJECT="/home/guest1/smart_project"
export LEISAAC_ENV="leisaac"
conda activate leisaac
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export LEISAAC_ROOT="$SMART_PROJECT/leisaac"
export LEISAAC_ASSETS_ROOT="$LEISAAC_ROOT/assets"
export OMNI_KIT_ACCEPT_EULA=YES
cd "$SMART_PROJECT"
export PYTHONPATH="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task:$LEISAAC_ROOT/source/leisaac:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

python experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/run_smart_task_closed_loop.py \
  --deployment-mode zero-shot-oxe \
  --robot so101 \
  --control-mode joint \
  --smart-target-asset cuboid \
  --dry-run \
  --max-policy-calls 1 \
  --instruction "Pick up the red 2x4 lego brick."
```

### 8.3 SO101 微调 checkpoint 的 LeIsaac 仿真推理

终端 1：GR00T bridge，使用 GR00T venv。`--deployment-mode so101-finetuned` 会自动使用
`NEW_EMBODIMENT`；`--camera-layout` 决定加载 wrist-only、dual 或 triple modality config。

权重文件可以只留在目标 Ubuntu 机器上，不需要同步回本机。下面的 bridge 暴露的是
host/port 形式的本地 TCP API，默认 `127.0.0.1:5577`；如果你已有 API 地址，就把 Isaac runner
里的 `BRIDGE_HOST/BRIDGE_PORT` 改成对应值。不要把这个 pickle socket 直接开放给不可信网络；
跨机器优先用 SSH tunnel。

如果已有记录是 `http://host:port/...` 形式的 URL，当前 runner 不能直接填完整 URL；只取
`host` 和 `port`。只有在额外实现 HTTP adapter 后，才使用 HTTP URL。

```bash
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH
export SMART_PROJECT="/home/guest1/smart_project"
export GROOT_ROOT="/home/guest1/Isaac-GR00T-py312"
export CHECKPOINT="__FILL_FINETUNED_CHECKPOINT_DIR__"

cd "$SMART_PROJECT"
"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/groot_bridge_server.py \
    --deployment-mode so101-finetuned \
    --camera-layout dual \
    --model-path "$CHECKPOINT" \
    --host 127.0.0.1 \
    --port 5577 \
    --device cuda
```

终端 2：Isaac runner，使用 `$LEISAAC_ENV`。

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME
export SMART_PROJECT="/home/guest1/smart_project"
export LEISAAC_ENV="leisaac"
conda activate leisaac
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export LEISAAC_ROOT="$SMART_PROJECT/leisaac"
export LEISAAC_ASSETS_ROOT="$LEISAAC_ROOT/assets"
export OMNI_KIT_ACCEPT_EULA=YES
cd "$SMART_PROJECT"
export PYTHONPATH="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task:$LEISAAC_ROOT/source/leisaac:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export BRIDGE_HOST="${BRIDGE_HOST:-127.0.0.1}"
export BRIDGE_PORT="${BRIDGE_PORT:-5577}"

python experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/run_smart_task_closed_loop.py \
  --deployment-mode so101-finetuned \
  --camera-layout dual \
  --front-observation-key camera3 \
  --wrist-observation-key camera2 \
  --robot so101 \
  --control-mode joint \
  --smart-target-asset cuboid \
  --bridge-host "$BRIDGE_HOST" \
  --bridge-port "$BRIDGE_PORT" \
  --dry-run \
  --max-policy-calls 1 \
  --instruction "Pick up the red 2x4 lego brick."
```

上面是 dual 示例。wrist-only 只给 runner 传 `--wrist-observation-key`；triple 还要加
`--left-observation-key camera1`。bridge 与 runner 的 `--camera-layout` 必须和训练 checkpoint 一致。

## 9. 坏环境处理

如果已经按旧命令把 `isaaclab` 或 `leisaac` env 搅乱，不要继续在这个 env 里修。保留它作为事故现场，换新名字走第 7 节：

```bash
export LEISAAC_ENV=leisaac_clean
```

旧 env 可以先不删。只有确认新 env 能跑后，再考虑清理：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null || true
conda env list
```

需要清理时再明确删除旧环境：

```bash
conda env remove -n isaaclab -y
```

不要把“删旧 env”当作修复动作本身；真正的判断顺序仍然是：

```text
clean leisaac env + canonical pins
-> python -m pip check
-> isaac_stack_probe.py
-> isaac_stack_probe.py --launch-app
-> LeIsaac task registry / LiftCube make-reset-step
```

## 10. 常见错误

`Permission denied` 跑 `./isaaclab.sh`：

```text
当前主线不再要求运行 ./isaaclab.sh。
如果你手工运行它，Permission denied 通常只是脚本没有执行位：
```

```bash
chmod u+x isaaclab.sh
```

`No module named isaacsim`：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME
conda activate leisaac
python -c "import isaacsim; print(getattr(isaacsim, '__file__', '<namespace>'))"
python -m pip show isaacsim isaacsim-core isaaclab
```

`No module named isaaclab`：

```text
先不要跑 bash ./isaaclab.sh -i none。
按第 7.4 节确认已经装过 isaaclab[isaacsim,all]==2.3.0。
如果 pip show isaaclab 也没有，说明你当前不是 `$LEISAAC_ENV`，或者安装步骤没有完成。
```

`No module named gymnasium`：

```text
这只是普通 Python 运行依赖缺失。
在当前 canonical 路线里，`isaaclab[isaacsim,all]==2.3.0` 应该会带上 gymnasium。
如果缺失，先看 `python -m pip check`，不要直接无约束升级。
```

`flatdict` / `pkg_resources` 报错：

```text
不要无约束升级 setuptools。
当前 canonical 边界是 setuptools==80.9.0，并在安装 flatdict / LeIsaac 时使用 --no-build-isolation。
```

`packaging 26.2` 和 `isaaclab-rl requires packaging<24` 冲突：

```text
这说明 env 已经进入 pip resolver 冲突状态。
不要继续 pip install -U，也不要把 LeRobot 装进 leisaac。
换新 `$LEISAAC_ENV` 跑第 7 节。
```

`No module named numpy`，但 `pip install numpy` 又说已经存在：

```text
如果错误发生在 Isaac runner，通常是你用了 conda run -n "$LEISAAC_ENV" env PYTHONPATH=... 覆盖了当前 shell 状态，或者当前终端不是 `$LEISAAC_ENV`。
按第 7.9 节的 Isaac runner 终端模板重跑。
```

`importing the numpy C-extensions failed`：

```text
如果错误发生在 GR00T 训练或 GR00T bridge，通常是你把 Isaac runner 终端里的 PYTHONPATH 带进了 GR00T venv。
开新终端，按第 8.1 或第 8.2 的终端 1 模板重跑。
```

`CXXABI_1.3.15 not found` / `omni.kit.test` 刷很多无关错误：

```text
优先按第 7 节命令模板重跑，确认是在启动 python 之前设置：
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

这通常是系统 /lib 里的 libstdc++.so.6 抢在 conda env 前面被加载，不是先补 Python 包。
```

`omni.physx` / `omni.physics` 相关错误：

```text
不要 pip install omni.physx 或 omni.physics。
先确认 `python -m pip show isaacsim isaacsim-core` 是 5.1.0.0，并跑第 7.5 的 `isaac_stack_probe.py --launch-app`。
如果旧 `~/isaacsim` 是 4.5.x，不要把它混进当前 LeIsaac 0.4.0 / IsaacLab 2.3.0 runtime。
```

真实 SO101 推理：

```text
当前仓库已经有 GR00T bridge 和 LeIsaac 仿真 runner。
当前仓库还没有完整真实 SO101 hardware runner。
真实 SO101 部署需要新增一个进程：读真实 top/wrist 图像 + SO101 state -> 按 NEW_EMBODIMENT schema 请求 bridge -> 把 action.single_arm/action.gripper 安全限幅后发给真实机器人。
```
