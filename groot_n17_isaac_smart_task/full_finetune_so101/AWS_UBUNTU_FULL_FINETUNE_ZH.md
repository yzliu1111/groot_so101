# AWS Ubuntu：SO101 full fine-tune 迁移说明

这份说明只覆盖 AWS Ubuntu 上的 LeRobot v3 数据准备、GR00T stats 和
`full_finetune_so101` 训练。不安装 Isaac Sim / LeIsaac，不在 AWS 部署 checkpoint，
也不进入 low-memory / LoRA 路线。

实际目标机 profile：Ubuntu 24.04 DLAMI、单卡 H100、NVIDIA 驱动/toolkit CUDA 13.2。
CUDA 13.2 不需要先卸载或降级；下面会保留系统 toolkit，并单独验证 GR00T venv 中 PyTorch、
Triton 与 FFmpeg 的运行链路。

当前 NVIDIA GR00T `main` 的 dGPU 路线要求 Python 3.12。这里不强制固定历史 commit，
但每次训练都要记录实际 commit，避免将来无法解释同名参数或 loss 的变化。

官方入口：

- [Isaac-GR00T README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/README.md)
- [Python 版本与依赖声明](https://github.com/NVIDIA/Isaac-GR00T/blob/main/pyproject.toml)
- [官方 dGPU 安装脚本](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/dgpu/install_deps.sh)
- [官方 CUDA 13.x Triton 补丁](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/patch_triton_cuda13.sh)
- [fine-tune 参数定义](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/configs/finetune_config.py)
- [stats.py](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/data/stats.py)

## 1. 推荐迁移边界

推荐优先级：

1. 最省事：在本地把 v3 转成 prepared v2.1，再只上传 prepared 数据和实验代码。
2. 如果只上传 v3 到 AWS：在 AWS 使用独立 LeRobot 环境完成阶段 A，再使用 GR00T
   Python 3.12 venv 完成阶段 B。
3. 不建议把“转换 + stats + 长训练”第一次就塞进一个命令；任何中间失败都会让恢复点
   和所用环境不够清楚。

项目脚本的控制流确实是：

```text
发现 v3 数据
  -> 未指定 --skip-prepare 时转换为 prepared v2.1
  -> 校验 6D schema、关节顺序、modality、episode parquet/video 完整性
  -> 未指定 --skip-stats 时生成 stats
  -> 未指定 --prepare-only 时启动训练
```

所以，从代码顺序看，上传 v3 且不加 `--skip-prepare`，会先转换，再生成 stats，最后训练。
但同一个 Python 解释器还必须同时满足转换器和 GR00T 的依赖；正式 AWS 运行仍建议拆成下面
两个阶段。

wrapper 会阻断错误维数/关节顺序、残缺 prepared copy、相机 layout 不匹配、parquet 行数不符、
缺 episode video、危险的 prepared 删除路径以及无效训练参数。选择多个数据集时还必须显式传
`--allow-multiple-datasets`，用于确认它们属于同一 action/state 坐标系。单位本身不写在
LeRobot metadata 中，代码无法自动证明 degree 与 motor unit 一致，所以 real/sim 的目录隔离
仍是必须遵守的数据契约。

## 2. AWS 目录约定

下面假定 AWS 登录用户为 `ubuntu`：

```text
/home/ubuntu/smart_project
/home/ubuntu/Isaac-GR00T
```

项目和数据至少要形成下面的目录：

```text
~/smart_project/
├── experiments/groot_n17_isaac_smart_task/
├── dataset/aws_v3/
│   ├── real/                               # 只放 degree 坐标的真机 v3 数据
│   └── sim/                                # 只放 LeRobot motor unit 的 sim v3 数据
└── outputs/
```

原始 v3 数据保留在 `dataset/`，prepared copy 和 checkpoint 都写到 `outputs/`。
real 与 sim 必须分开转换、分开 stats、分开训练。**不要**把 `SOURCE_ROOT` 设成同时包含两者的
`dataset/aws_v3`；`modality.json` 不记录或转换单位，loss 也不能发现这种混合错误。

下面示例先跑 real。跑 sim 时开启新终端，仅把 `DATA_DOMAIN` 改为 `sim`，并使用新的输出目录：

```bash
source "$HOME/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh" target
export DATA_DOMAIN=real
export SOURCE_ROOT="$SMART_PROJECT/dataset/aws_v3/$DATA_DOMAIN"
export PREPARED_ROOT="$SMART_PROJECT/outputs/groot_so101_synthetic_datasets/aws_v3/$DATA_DOMAIN"
test "$SMART_PROJECT" = "$HOME/smart_project"
test -d "$SOURCE_ROOT"
```

## 3. Ubuntu 24.04、FFmpeg 与 CUDA 13.2

这些是必装项，不要等训练报错后才补 FFmpeg：

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  curl git git-lfs ffmpeg libaio-dev
git lfs install

grep '^PRETTY_NAME=' /etc/os-release
git --version
ffmpeg -version | head -n 1
ffmpeg -hide_banner -decoders 2>/dev/null | grep -i av1
```

当前 GR00T 的 `torchcodec` 要求 FFmpeg 4--7。Ubuntu 24.04 apt 默认的 FFmpeg 6.x 在范围内；
如果这里实际显示 8.x，先换回 FFmpeg 7 或更低，不要开始训练。

确认 DLAMI 的 CUDA 13.2 toolkit。`nvidia-smi` 中的 CUDA 版本只是驱动兼容上限，`nvcc`
才表明本机 toolkit：

```bash
nvidia-smi

if test -x /usr/local/cuda-13.2/bin/nvcc; then
  export CUDA_HOME=/usr/local/cuda-13.2
elif test -x /usr/local/cuda/bin/nvcc; then
  export CUDA_HOME=/usr/local/cuda
else
  echo "ERROR: CUDA toolkit nvcc not found" >&2
  exit 1
fi

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$CUDA_HOME/bin/nvcc" --version
```

如果 `/usr/local/cuda-13.2` 存在而通用链接 `/usr/local/cuda` 不存在，先补上链接。官方 dGPU
安装脚本用这个通用路径判断是否已有 toolkit；补链接可以避免它误装一份 CUDA 12.8：

```bash
if test -d /usr/local/cuda-13.2 && ! test -e /usr/local/cuda && ! test -L /usr/local/cuda; then
  sudo ln -s /usr/local/cuda-13.2 /usr/local/cuda
fi
test -x /usr/local/cuda/bin/nvcc
```

最后检查磁盘和内存：

```bash
df -h "$HOME"
free -h
command -v conda || true
```

## 4. 安装当前 GR00T Python 3.12 环境

如果目录不存在：

```bash
git clone --recurse-submodules \
  https://github.com/NVIDIA/Isaac-GR00T.git "$HOME/Isaac-GR00T"
```

如果目录已经存在：

```bash
git -C "$HOME/Isaac-GR00T" pull --ff-only
git -C "$HOME/Isaac-GR00T" submodule update --init --recursive
```

恢复 CUDA 变量，然后运行仓库当前自带的官方 dGPU 安装脚本。它会再次确认 `ffmpeg`、
`libaio-dev`，安装 `uv`，并用 Python 3.12 创建 `.venv`；重复 apt 安装是无害的：

```bash
cd "$HOME/Isaac-GR00T"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash scripts/deployment/dgpu/install_deps.sh

export PATH="$HOME/.local/bin:$PATH"
uv --version
uv run python -c "import gr00t; print('GR00T installed successfully')"
```

当前 `main` 在 x86 dGPU 上安装的是 Python 3.12 环境和 PyTorch CUDA 12.8 wheel。这里的
`cu128` 是 venv 的 PyTorch runtime，不要求卸掉 DLAMI 的 13.2 驱动/toolkit；新驱动可以运行
较旧的 CUDA runtime。先打印实际安装结果，不要只凭 README 或 AMI 名称猜：

```bash
cd "$HOME/Isaac-GR00T"
uv run python -c \
  "import sys, torch, triton; print(sys.version); print('torch=', torch.__version__, 'torch_cuda=', torch.version.cuda); print('triton=', triton.__version__); print('gpu=', torch.cuda.get_device_name(0)); assert sys.version_info[:2] == (3, 12); assert torch.cuda.is_available()"
uv run python -c "import torchcodec; print('torchcodec import OK')"
```

### CUDA 13.2：固定执行仓库补丁

这台 DLAMI 按已经实跑成功的路线处理：安装完成后固定执行
`scripts/patch_triton_cuda13.sh`，不把它降级为“出错才尝试”的可选步骤。补丁前先打印一次
PTX 映射，只用于记录安装状态；即使当前 Triton 已经输出 `ptx= 92`，也继续执行补丁：

```bash
cd "$HOME/Isaac-GR00T"
uv run python -c \
  "from triton.backends.nvidia.compiler import ptx_get_version; print('ptx=', ptx_get_version('13.2'))"
```

执行补丁并保存日志。当前脚本是幂等的：如果 `compiler.py` 已原生包含 CUDA 13 分支，会打印
`already patched`，同时仍安装可跨 `uv run` 使用的 `.pth` 启动 hook：

```bash
cd "$HOME/Isaac-GR00T"
set -o pipefail
uv run bash scripts/patch_triton_cuda13.sh 2>&1 | tee "$HOME/groot_cuda13_patch.log"

uv run python -c \
  "from triton.backends.nvidia.compiler import ptx_get_version; print('ptx=', ptx_get_version('13.2'))"
```

补丁后的探针必须输出 `ptx= 92`；若脚本失败或仍不是 92，就停止，不要开始 stats / 训练。

最后实际触发一次 GPU 上的 Triton 编译，而不只是 import：

```bash
cd "$HOME/Isaac-GR00T"
uv run python -c \
  "import torch; f=torch.compile(lambda x: torch.sin(x)+1); x=torch.randn(1024, device='cuda'); y=f(x); torch.cuda.synchronize(); print('torch.compile OK', float(y.mean()))"
```

只有补丁、补丁后 PTX 探针和 `torch.compile` 三项都通过，才进入数据 stats / 1-step。新建或
彻底重建 `.venv` 后重新执行本节补丁；若补丁后仍失败，保留完整报错、版本输出和
`~/groot_cuda13_patch.log`，不要继续长训练。

记录而不固定当前版本：

```bash
git -C "$HOME/Isaac-GR00T" rev-parse HEAD | tee "$HOME/groot_commit_used.txt"
"$HOME/Isaac-GR00T/.venv/bin/python" -c \
  "import sys, torch, gr00t; print(sys.version); print(torch.__version__); print(torch.cuda.is_available()); assert sys.version_info[:2] == (3, 12); assert torch.cuda.is_available()"
```

首次下载模型前要先获得 Hugging Face gated model 权限，并登录：

```bash
cd "$HOME/Isaac-GR00T"
uv run hf auth login
```

不要把 `HF_TOKEN` 写进 README、shell history 或训练日志。

## 5. 上传后检查 v3 数据

```bash
find "$SOURCE_ROOT" -path '*/meta/info.json' -print | sort
du -sh "$SOURCE_ROOT"
```

每个 v3 叶子数据集都应包含 `meta/info.json`，其中
`codebase_version` 为 `v3.0`。同一次递归转换的数据集必须使用一致的相机 feature key。

当前已核对的真机与 sim wrist-only 数据都只包含
`observation.images.camera1`。所以下面的完整流程显式使用
`--camera-layout wrist-only --dataset-wrist-camera-key observation.images.camera1`；不要省略这两个
参数，因为 wrapper 的代码默认值是 `dual`，并把 wrist 默认映射到
`observation.images.camera3`。如果换成确实同时包含 top/front 与 wrist 的双相机数据，再把各阶段
统一改成 `--camera-layout dual`，并显式传入实际的两个 camera feature key。

这两类数据的结构相同，但前五维的原始数值坐标不同：真机数据是 degree；sim 数据是
LeIsaac 的 LeRobot motor unit，前五维目标范围为 `[-100, 100]`。两边 gripper 都使用
`[0, 100]` 范围。`meta/modality.json` 只声明维度切片和相机映射，不声明、也不会转换物理
单位。因此 real checkpoint 回到原真机、sim checkpoint 回到原 sim 没有这一层的跨域问题；
不要因为两者都叫 `single_arm` 就直接认定可无条件混训或互换部署。

两份已上传 meta 的任务文本都是 `pick up block`，只是列名不同：真机使用 `task`，sim 使用
`__index_level_0__`。wrapper 已兼容这两种 v3 写法，下面不传 `--instruction`，从而保留各自源
任务文本。只有确认源任务文本错误时才用 `--instruction` 覆盖；不要仅为了命令看起来完整就
改写语言条件。

如果数据目录很大，上传完成后至少对 `meta/` 文件和数据总大小做一次两端核对；不要在 AWS
上用移动或原地覆盖的方式转换唯一一份数据。

## 6. 阶段 A：在独立 LeRobot 环境转换 v3

如果已经上传 prepared v2.1，跳到第 7 节。

AWS 上没有 `lerobot` 环境时，可以创建一个只用于本项目非破坏性转换的独立环境。当前
wrapper 自带轻量 fallback converter；它只需要 Python 3.12、`pyarrow` 和系统 `ffmpeg`，
不需要把 LeRobot 安装进 GR00T venv：

```bash
conda create -n lerobot -c conda-forge -y python=3.12 pyarrow
conda run -n lerobot python -c \
  "import sys, pyarrow; print(sys.version); print(pyarrow.__version__); assert sys.version_info[:2] == (3, 12)"
```

DLAMI 通常已有 conda；如果 `command -v conda` 没有输出，不必额外安装整套 Miniconda，也可以
复用已经安装的 `uv` 建立同等隔离的轻量转换环境：

```bash
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 "$HOME/.venvs/lerobot-convert"
uv pip install --python "$HOME/.venvs/lerobot-convert/bin/python" pyarrow
"$HOME/.venvs/lerobot-convert/bin/python" -c \
  "import sys, pyarrow; print(sys.version); print(pyarrow.__version__); assert sys.version_info[:2] == (3, 12)"
```

使用这个 fallback 时，把下方转换命令开头的 `conda run -n lerobot python` 换成
`"$HOME/.venvs/lerobot-convert/bin/python"`，其他参数不变。

如果已有经过验证的 LeRobot conversion env，直接复用，不要重装。转换环境不能与 GR00T
venv 或 Isaac 环境混用。

转换命令：

```bash
source "$HOME/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh" target
export DATA_DOMAIN=real
export SOURCE_ROOT="$SMART_PROJECT/dataset/aws_v3/$DATA_DOMAIN"
export PREPARED_ROOT="$SMART_PROJECT/outputs/groot_so101_synthetic_datasets/aws_v3/$DATA_DOMAIN"
cd "$SMART_PROJECT"

conda run -n lerobot python \
  experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
    --source-root "$SOURCE_ROOT" \
    --allow-multiple-datasets \
    --prepared-root "$PREPARED_ROOT" \
    --camera-layout wrist-only \
    --dataset-wrist-camera-key observation.images.camera1 \
    --skip-stats \
    --prepare-only
```

第一次建议先对一份数据加 `--max-episodes 1` 做转换 smoke。确认后换一个干净的
`--prepared-root` 跑完整转换。只有明确要重建 prepared copy 时才加 `--force-prepare`；
它不会删除源 v3，但会覆盖对应 prepared 输出。

转换完成后检查：

```bash
find "$PREPARED_ROOT" -path '*/meta/info.json' -print | sort
find "$PREPARED_ROOT" -path '*/meta/modality.json' -print | sort
find "$PREPARED_ROOT" -path '*/data/chunk-*/episode_*.parquet' -print -quit
find "$PREPARED_ROOT" -path '*/videos/chunk-*/*.mp4' -print -quit
```

新转换会为每个 prepared 数据集写入 `meta/modality.json`。如果文件缺失，不要开始 stats
或训练。

## 7. 阶段 B1：GR00T dry-run

即使加了 `--skip-prepare`，当前 wrapper 仍会解析并验证源 v3 数据，因此下面保留同一个
`--source-root`。它不会重新转换源数据。

```bash
source "$HOME/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh" target
export DATA_DOMAIN=real
export SOURCE_ROOT="$SMART_PROJECT/dataset/aws_v3/$DATA_DOMAIN"
export PREPARED_ROOT="$SMART_PROJECT/outputs/groot_so101_synthetic_datasets/aws_v3/$DATA_DOMAIN"
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$SMART_PROJECT"

"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
    --source-root "$SOURCE_ROOT" \
    --allow-multiple-datasets \
    --prepared-root "$PREPARED_ROOT" \
    --camera-layout wrist-only \
    --dataset-wrist-camera-key observation.images.camera1 \
    --skip-prepare \
    --skip-stats \
    --dry-run \
    --max-steps 1 \
    --save-steps 1 \
    --global-batch-size 1 \
    --gradient-accumulation-steps 1
```

检查打印出的 dataset path、modality config、camera layout、base model 和 output directory。

## 8. 阶段 B2：stats + 1-step smoke

第一次运行时不要加 `--skip-stats`：

上传的两类源 v3 `meta/stats.json` 来自不同统计实现：真机 image count 按像素累计，sim 的
image count 是抽样数量。这不表示图像损坏，也不是要把两者的 image mean/std 强行算成一样。
当前 GR00T `stats.py` 会根据 prepared `meta/info.json` 重算 action、state 等低维 float 特征的
统计；由于 wrist-only modality 把前五维配置为 relative joint action，它还会生成
`meta/relative_stats.json`。这两个文件必须与本次 GR00T 代码、prepared 数据和 modality
配置配套，所以第一次 stats 不能跳过。

如果从新终端开始本节，先完整恢复路径、数据域和 CUDA 13.2：

```bash
source "$HOME/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh" target
export DATA_DOMAIN=real
export SOURCE_ROOT="$SMART_PROJECT/dataset/aws_v3/$DATA_DOMAIN"
export PREPARED_ROOT="$SMART_PROJECT/outputs/groot_so101_synthetic_datasets/aws_v3/$DATA_DOMAIN"
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$SMART_PROJECT"
```

```bash
"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
    --source-root "$SOURCE_ROOT" \
    --allow-multiple-datasets \
    --prepared-root "$PREPARED_ROOT" \
    --camera-layout wrist-only \
    --dataset-wrist-camera-key observation.images.camera1 \
    --skip-prepare \
    --output-dir "$SMART_PROJECT/outputs/groot_so101_synthetic_finetune/aws_${DATA_DOMAIN}_smoke" \
    --max-steps 1 \
    --save-steps 1 \
    --global-batch-size 1 \
    --gradient-accumulation-steps 1 \
    --dataloader-num-workers 0 \
    --experiment-name "aws_${DATA_DOMAIN}_so101_smoke"
```

通过标准：

- stats 对每个 prepared 数据集成功完成；
- 模型和 optimizer 成功初始化；
- 完成 1 个 optimizer step；
- loss 为有限值，没有 OOM、NaN、CUDA extension 或数据 key 错误；
- dataloader 能实际解码一批视频，没有 `torchcodec` / FFmpeg / AV1 错误；
- checkpoint / trainer state 按预期写入 smoke 输出目录。

stats 完成后确认每个 prepared 数据集都有普通低维 stats 和 relative-action stats：

```bash
find "$PREPARED_ROOT" -path '*/meta/stats.json' -print | sort
find "$PREPARED_ROOT" -path '*/meta/relative_stats.json' -print | sort
```

只有本次 prepared 数据和 modality 没有变化、并且上面的 stats 已成功生成时，后续 pilot 与
正式训练才使用 `--skip-stats`。

当前已核对的两份 v3 数据都使用 AV1。旧训练日志证明当时的环境能读它，但 NVIDIA 当前 main
只把 H.264 列为保证支持的编码，AV1 能否工作取决于 AWS 上的 `torchcodec` 与 FFmpeg build。
如果 1-step 已经成功，就不要为了格式统一额外转码；如果明确报 AV1 解码错误，只在
`$PREPARED_ROOT` 副本上使用 NVIDIA 当前仓库提供的转换脚本，绝对不要对唯一的源 v3 原地运行：

```bash
cd "$GROOT_ROOT"
uv run python examples/SimplerEnv/convert_av1_to_h264.py "$PREPARED_ROOT" --jobs 8
```

转码后换一个新的 smoke `--output-dir`，重新执行本节 1-step；通过后再进入 H100 pilot。

## 9. 阶段 B3：单卡 H100 20-step 吞吐 pilot

NVIDIA 当前
[硬件建议](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md)
给 1x H100 的 quick-start `global_batch_size=32`。1-step 通过后，先用
20 steps 确认峰值显存、吞吐和 loss 都稳定：

```bash
source "$HOME/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh" target
export DATA_DOMAIN=real
export SOURCE_ROOT="$SMART_PROJECT/dataset/aws_v3/$DATA_DOMAIN"
export PREPARED_ROOT="$SMART_PROJECT/outputs/groot_so101_synthetic_datasets/aws_v3/$DATA_DOMAIN"
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$SMART_PROJECT"
```

```bash
"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
    --source-root "$SOURCE_ROOT" \
    --allow-multiple-datasets \
    --prepared-root "$PREPARED_ROOT" \
    --camera-layout wrist-only \
    --dataset-wrist-camera-key observation.images.camera1 \
    --skip-prepare \
    --skip-stats \
    --output-dir "$SMART_PROJECT/outputs/groot_so101_synthetic_finetune/aws_${DATA_DOMAIN}_h100_pilot" \
    --global-batch-size 32 \
    --gradient-accumulation-steps 1 \
    --max-steps 20 \
    --save-steps 20 \
    --save-total-limit 1 \
    --dataloader-num-workers 4 \
    --learning-rate 1e-4 \
    --experiment-name "aws_${DATA_DOMAIN}_so101_h100_pilot"
```

如果 batch 32 OOM，先退到 `--global-batch-size 16`，不要同时增加 accumulation。batch 32
稳定后，没有必要为了“更激进”盲目追到 64；先让正式 baseline 与 NVIDIA 单卡 H100 配置
保持一致。

## 10. 阶段 B4：正式训练

参数选择见 [TRAINING_PARAMETER_REFERENCE_ZH.md](TRAINING_PARAMETER_REFERENCE_ZH.md)。
以下使用单卡 H100、有效 batch 32；`MAX_STEPS` 要按所有 participating dataset 的总有效
帧数调整。约 27 万有效帧可先用 10000，约 7000 有效帧应改成 1000：

```bash
source "$HOME/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh" target
export DATA_DOMAIN=real
export SOURCE_ROOT="$SMART_PROJECT/dataset/aws_v3/$DATA_DOMAIN"
export PREPARED_ROOT="$SMART_PROJECT/outputs/groot_so101_synthetic_datasets/aws_v3/$DATA_DOMAIN"
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$SMART_PROJECT"

export MAX_STEPS=10000

"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
    --source-root "$SOURCE_ROOT" \
    --allow-multiple-datasets \
    --prepared-root "$PREPARED_ROOT" \
    --camera-layout wrist-only \
    --dataset-wrist-camera-key observation.images.camera1 \
    --skip-prepare \
    --skip-stats \
    --output-dir "$SMART_PROJECT/outputs/groot_so101_synthetic_finetune/aws_${DATA_DOMAIN}_run_001" \
    --global-batch-size 32 \
    --gradient-accumulation-steps 1 \
    --max-steps "$MAX_STEPS" \
    --save-steps 500 \
    --save-total-limit 5 \
    --dataloader-num-workers 4 \
    --learning-rate 1e-4 \
    --experiment-name "aws_${DATA_DOMAIN}_so101_run_001"
```

已经完成 W&B 登录并希望远程监控时，再显式加 `--use-wandb`。

不要复用旧 run 的 `--output-dir`。增加 batch、改变 accumulation 或改变数据组合时，使用新
目录和新实验名，并记录：

```text
项目 experiments commit
Isaac-GR00T commit
源数据目录及帧数
prepared 目录
camera layout
global batch / accumulation / max steps / learning rate
GPU 型号与数量
```

## 11. 训练中与训练后检查

loss 下降只能证明当前数据能够驱动当前 objective 的优化，不能单独证明相机、动作单位、
语言标注或部署语义正确。正式训练至少保留以下证据：

1. 随机抽查 prepared 图像、语言和 6D action，确认没有错位或常量列。
2. 保存 loss、learning rate、grad norm、step time 和 GPU memory 日志。
3. 比较至少三个相邻 checkpoint，而不是只拿最低 training loss 的最后一个。
4. 将 checkpoint 同步回本地后做 held-out / open-loop 检查；AWS 本身不需要启动部署。

`loss ≈ 0.03` 的解释和停止条件见参数参考文档。它可以是“值得开始评估 checkpoint”的信号，
但不是训练管道或策略成功的保证书。
