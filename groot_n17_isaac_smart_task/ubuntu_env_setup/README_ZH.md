# Ubuntu 目标机搭建与 smoke test

这份指南面向 Ubuntu 22.04 x86_64 GPU 机器。目标不是把所有框架塞进一个环境，而是先决定
机器承担什么任务，再只安装对应环境。

## 1. 先决定机器用途

| 用途 | 必须有 | 不需要 |
|---|---|---|
| GR00T 训练 | project、prepared 数据、GR00T venv、训练时可用的 CUDA toolkit | Isaac/LeIsaac、LeRobot |
| 重新转换数据 | 上面项目文件 + LeRobot conda env + 原始 v3 数据 | Isaac/LeIsaac |
| LeIsaac 仿真部署 | GR00T venv + Isaac/LeIsaac conda env + assets | LeRobot |

三个终端必须分开：

```text
GR00T：  $GROOT_ROOT/.venv/bin/python
LeRobot：conda activate $LEROBOT_ENV
LeIsaac：conda activate $LEISAAC_ENV
```

## 2. 项目同步后：每个目标机新终端先执行

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
```

这条命令会打印当前 project、GR00T 和 conda env 名称，但不会激活任何 conda 环境。
项目尚未同步时先执行第 4 节；同步完成后，后面每个可独立开始的阶段都会就近重复这条命令。

## 3. 系统检查

有 sudo 时：

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  git git-lfs curl rsync unzip ffmpeg build-essential cmake pkg-config tmux htop
git lfs install
```

`guest1` 没有 sudo 时，把上面的包列表交给 `smartscape` 安装。

先检查 GPU，不要先重装 driver/CUDA：

```bash
nvidia-smi
which nvcc || true
nvcc --version || true
ls -ld /usr/local/cuda /usr/local/cuda-* 2>/dev/null || true
```

`nvidia-smi` 的 CUDA Version 是 driver 支持上限，不代表 `$CUDA_HOME/bin/nvcc` 存在。
bridge 推理和 Isaac runner 可以没有 toolkit；DeepSpeed/CUDA extension 训练才需要 nvcc。

## 4. 同步项目

从源机器执行：

```bash
export SOURCE_PROJECT="__FILL_SOURCE_SMART_PROJECT__"
export TARGET_HOST="guest1@__FILL_TARGET_HOST__"
export TARGET_PROJECT=/home/guest1/smart_project

ssh "$TARGET_HOST" "mkdir -p '$TARGET_PROJECT/experiments' '$TARGET_PROJECT/outputs'"

rsync -aP --exclude='__pycache__' --exclude='.pytest_cache' \
  "$SOURCE_PROJECT/experiments/groot_n17_isaac_smart_task/" \
  "$TARGET_HOST:$TARGET_PROJECT/experiments/groot_n17_isaac_smart_task/"

rsync -aP \
  "$SOURCE_PROJECT/outputs/groot_so101_synthetic_datasets/" \
  "$TARGET_HOST:$TARGET_PROJECT/outputs/groot_so101_synthetic_datasets/"
```

要跑 LeIsaac 时还要同步或 clone `leisaac/`；重新转换数据时才同步原始 `dataset/` 和
LeRobot repo。

同步完成后，登录目标机检查。即使这是一个全新的终端，也可以直接从这里开始：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
test -d "$SMART_PROJECT/experiments/groot_n17_isaac_smart_task"
find "$SMART_PROJECT/outputs/groot_so101_synthetic_datasets" \
  -path '*/meta/info.json' -print
```

## 5. Miniforge

已有 conda 就跳过安装：

```bash
conda --version
conda info --base
```

没有时安装到用户目录：

```bash
curl -L -o /tmp/Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda init bash
```

## 6. GR00T 环境

本节可以从新目标机终端独立开始：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
```

没有 repo 时：

```bash
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T "$GROOT_ROOT"
cd "$GROOT_ROOT"
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
export UV_HTTP_TIMEOUT=300
uv sync --python 3.12
```

已有 repo/venv 时不要重复安装，直接验证：

```bash
"$GROOT_ROOT/.venv/bin/python" - <<'PY'
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
assert sys.version_info[:2] == (3, 12)
PY

"$GROOT_ROOT/.venv/bin/python" -c "import gr00t; print('GR00T import OK')"
```

需要 gated model 时登录：

```bash
"$GROOT_ROOT/.venv/bin/hf" auth login
```

训练需要 nvcc 时，把 `CUDA_HOME` 指到真实 toolkit，并尽量匹配 `torch.version.cuda`：

```bash
export CUDA_HOME="__FILL_REAL_CUDA_TOOLKIT_PATH__"
test -x "$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
"$CUDA_HOME/bin/nvcc" --version
```

训练入口的 dry-run 和 smoke 见
[full_finetune_so101/README_ZH.md](../full_finetune_so101/README_ZH.md) 与
[lowmem_lora_freeze_so101/README_ZH.md](../lowmem_lora_freeze_so101/README_ZH.md)。

## 7. LeRobot 环境（只有重新转换数据才需要）

新目标机终端先恢复路径，再创建或激活 LeRobot env：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
export LEROBOT_ROOT="__FILL_LEROBOT_REPO__"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "$LEROBOT_ENV" python=3.12 -y
conda activate "$LEROBOT_ENV"

python -m pip install "pip<26" "setuptools<81" "wheel<0.47"
cd "$LEROBOT_ROOT"
python -m pip install -e ".[dataset]"
python -m pip install pyarrow av opencv-python-headless imageio tqdm
python -c "import lerobot, pyarrow, av; print('LeRobot import OK')"
```

转换命令见 full-finetune README。不要把 LeRobot 安装进 `$LEISAAC_ENV`。

## 8. LeIsaac / Isaac runtime

本章可以从新目标机终端独立开始：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
```

### 8.1 准备 repo

已有 `$LEISAAC_ROOT` 时：

```bash
test -f "$LEISAAC_ROOT/source/leisaac/pyproject.toml"
test -d "$LEISAAC_ROOT/dependencies/IsaacLab/source/isaaclab"
```

没有时：

```bash
git clone --recurse-submodules https://github.com/LightwheelAI/leisaac.git "$LEISAAC_ROOT"
cd "$LEISAAC_ROOT"
git submodule update --init --recursive
```

### 8.2 先检查现有 env

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LEISAAC_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -m pip check
```

如果现有 env 已被反复 pip 修改且 `pip check`/import 冲突，不要继续修；改用新名字：

```bash
export LEISAAC_ENV=leisaac_clean
conda create -n "$LEISAAC_ENV" python=3.11 -y
conda activate "$LEISAAC_ENV"
```

如果最后采用了 `leisaac_clean`，以后新终端在 `source terminal_env.sh` 前先执行
`export LEISAAC_ENV=leisaac_clean`；下面的 probe、assets 和 smoke 命令都会读取这个变量。

### 8.3 只在新 env 中安装 canonical 边界

```bash
python -m pip install \
  "setuptools==80.9.0" "wheel==0.41.3" "packaging==23.0" \
  "numpy==1.26.0" toml

python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install "isaaclab[isaacsim,all]==2.3.0" \
  --extra-index-url https://pypi.nvidia.com

python -m pip install --no-build-isolation "flatdict==4.0.1"
python -m pip install --no-build-isolation -e "$LEISAAC_ROOT/source/leisaac"
```

不要安装 LeRobot，不要执行无约束的 `pip install -U pip setuptools wheel`。

### 8.4 基础 probe

```bash
export LEISAAC_ENV=leisaac  # 若第 8.2 节创建了 leisaac_clean，这里改成 leisaac_clean
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LEISAAC_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$SMART_PROJECT"

python experiments/groot_n17_isaac_smart_task/ubuntu_env_setup/isaac_stack_probe.py
python experiments/groot_n17_isaac_smart_task/ubuntu_env_setup/isaac_stack_probe.py --launch-app
```

第二条必须看到：

```text
[ OK ] AppLauncher started SimulationApp
[ OK ] import leisaac
[ OK ] SimulationApp closed
```

`omni.physx`/`omni.physics` 是 Kit extension；失败时不要把它们当普通 pip 包安装。

## 9. Assets 和 portable SmartTask scene

本节可能在另一个终端执行，因此先完整恢复 Isaac 终端状态：

```bash
export LEISAAC_ENV=leisaac  # 若第 8.2 节创建了 leisaac_clean，这里改成 leisaac_clean
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LEISAAC_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$SMART_PROJECT"
```

LeIsaac repo 不一定包含完整 assets。至少检查：

```bash
test -f "$LEISAAC_ROOT/assets/robots/so101_follower.usd"
test -f "$LEISAAC_ROOT/assets/scenes/smart_scene/scene.usd"
```

缺少 SO101 USD 时，从 LeIsaac release 获取：

```bash
mkdir -p "$LEISAAC_ROOT/assets/robots"
curl -L -o "$LEISAAC_ROOT/assets/robots/so101_follower.usd" \
  "https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd"
```

把 SmartTask 中的历史绝对路径转换成 repo-local USDA；原始 `scene.usd` 不会被覆盖：

```bash
python "$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/ubuntu_env_setup/make_smart_scene_portable.py" \
  --write --check

test -f "$LEISAAC_ROOT/assets/scenes/smart_scene/scene_portable.usda"
```

成功标志：

```text
[ OK ] No absolute /home or /Users USD paths remain after rewrite.
[ OK ] Wrote portable scene: .../scene_portable.usda
```

LEGO USD 缺内部 layer 的问题由 experiments runner 默认 `--smart-target-asset cuboid` 处理，
不修改 `leisaac/`。

## 10. SmartTask smoke

这里经常会另开终端。下面这段不依赖第 2 节残留的 shell 状态，可以直接复制：

```bash
export LEISAAC_ENV=leisaac  # 若第 8.2 节创建了 leisaac_clean，这里改成 leisaac_clean
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LEISAAC_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task:$LEISAAC_ROOT/source/leisaac:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
cd "$SMART_PROJECT"
```

创建 SmartTask、reset 相机并检查目标物，不连接 GR00T：

```bash
python experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/run_smart_task_closed_loop.py \
  --deployment-mode so101-finetuned \
  --so101-checkpoint-joint-units lerobot_motor_units \
  --camera-layout dual \
  --front-observation-key camera3 \
  --wrist-observation-key camera2 \
  --robot so101 \
  --control-mode joint \
  --smart-target-asset cuboid \
  --debug-cameras-only \
  --headless
```

这里显式保留 sim 的默认单位值，但 `--debug-cameras-only` 会在动作单位转换前退出，所以
这一步只验证环境、场景和相机，不验证 motor/degree 转换链路。真机数据 checkpoint 的后续
部署命令应把单位参数改成 `degrees`，并确保 bridge 与 runner 一致；真正的单位合约验证从
bridge dry-run 握手和 one-step 开始。

只有这一步通过，才进入 bridge dry-run 和 one-step。后续命令见
[zero_shot_isaac_smart_task/README_ZH.md](../zero_shot_isaac_smart_task/README_ZH.md)。

## 11. 常见错误

| 现象 | 停在哪一层 |
|---|---|
| GR00T 中 numpy/import 异常 | 新开终端，清掉 Isaac `PYTHONPATH/PYTHONHOME` |
| `CUDA_HOME does not exist` | 只在训练终端设置真实 toolkit 路径 |
| Isaac `CXXABI` 错误 | 确保 `$CONDA_PREFIX/lib` 位于 `LD_LIBRARY_PATH` 最前 |
| `lerobot` 出现在 Isaac env | 环境混装；重建干净 env，不继续补包 |
| `omni.physx` import 失败 | 先跑 `isaac_stack_probe --launch-app`，不要 pip install omni 包 |
| portable scene 仍有绝对路径 | 停止，先补缺失 asset/修 scene 引用 |
| SmartTask camera/asset 失败 | 不进入模型调试，先修 runtime/scene |
| checkpoint 能加载但动作不对 | 回到 camera dry-run、单位契约和 one-step |

## 12. 最小验收顺序

```text
GR00T import/GPU
-> Isaac import
-> SimulationApp
-> portable scene
-> SmartTask camera smoke
-> bridge dry-run
-> one policy action
```

任何一步失败都停在该层，不要跳到后面猜模型效果。

需要理解代码中的进程、相机、action 和时间约定时，再看
[TECHNICAL_CONTRACTS_ZH.md](../TECHNICAL_CONTRACTS_ZH.md)；跑本手册不要求先读技术手册。
