# SO101 数据准备与 full fine-tune

这份 README 用于把 LeRobot v3 数据非破坏性地准备成 GR00T v2.1 数据，然后调用官方
`launch_finetune.py`。本机适合数据准备和 smoke；正式训练优先放到大显存机器。

## 1. 两个环境，不要混用

```text
LeRobot v3 -> prepared v2.1：conda lerobot
GR00T stats / fine-tune：     $GROOT_ROOT/.venv (Python 3.12)
```

原始 `dataset/` 不会被覆盖；prepared copy 写到：

```text
$SMART_PROJECT/outputs/groot_so101_synthetic_datasets/
```

训练输出写到：

```text
$SMART_PROJECT/outputs/groot_so101_synthetic_finetune/
```

## 2. 新终端恢复路径

目标机：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
```

本机：

```bash
source /home/yzliu/physical_ai/company_project/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh local
```

上面二选一。首次使用或换机器后验证：

```bash
test -d "$SMART_PROJECT"
test -x "$GROOT_ROOT/.venv/bin/python"
"$GROOT_ROOT/.venv/bin/python" -c \
  "import sys, torch; print(sys.version); print(torch.__version__); assert sys.version_info[:2] == (3, 12)"
```

## 3. 选择相机布局

| layout | prepared 输入 | config |
|---|---|---|
| `dual`（默认） | `camera1 -> video.top`，`camera3 -> video.wrist` | `so101_synthetic_groot_config.py` |
| `wrist-only` | 指定一条 wrist camera | `so101_synthetic_groot_wrist_only_config.py` |
| `triple` | top + left + wrist | `so101_synthetic_groot_triple_config.py` |

训练、bridge 和 runner 必须使用同一个 layout。

## 4. 阶段 A：在 LeRobot 环境准备数据

如果从这里新开终端，先恢复路径。目标机直接执行；本机使用注释中的替代行：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
# source /home/yzliu/physical_ai/company_project/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh local
```

默认数据源：

```text
$SMART_PROJECT/dataset/so101_lego_pick_0609_1722
$SMART_PROJECT/dataset/so101_lego_pick_0609_1722_mimic
```

默认 dual 准备命令：

```bash
cd "$SMART_PROJECT"
conda run -n lerobot python \
  experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
    --camera-layout dual \
    --instruction "Pick up the red 2x4 lego brick." \
    --force-prepare \
    --skip-stats \
    --prepare-only
```

先做 1 episode smoke 时加：

```bash
--max-episodes 1
```

wrist-only：

```bash
conda run -n lerobot python \
  experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
    --camera-layout wrist-only \
    --dataset-wrist-camera-key observation.images.camera3 \
    --force-prepare \
    --skip-stats \
    --prepare-only
```

triple：

```bash
conda run -n lerobot python \
  experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
    --camera-layout triple \
    --dataset-front-camera-key observation.images.camera1 \
    --dataset-left-camera-key observation.images.camera2 \
    --dataset-wrist-camera-key observation.images.camera3 \
    --force-prepare \
    --skip-stats \
    --prepare-only
```

自定义任务树使用 `--source-root`；脚本会递归发现含 `meta/info.json` 且
`codebase_version == v3.0` 的叶子目录：

```bash
--source-root "$SMART_PROJECT/dataset/custom" \
--prepared-root "$SMART_PROJECT/outputs/groot_so101_synthetic_datasets/custom"
```

同一次递归转换中的数据集必须使用一致的 camera feature key。

### 阶段 A 成功标志

每个 prepared 数据集至少应包含：

```text
meta/info.json
meta/modality.json
meta/stats.json 或后续可生成 stats 的完整数据
data/chunk-*/episode_*.parquet
videos/chunk-*/...
```

不要直接用官方原地转换器处理唯一一份源数据；它会移动原目录并写回 v2.1。本脚本默认使用
prepared copy，就是为了避免这类误操作。

## 5. 阶段 B：在 GR00T venv 检查训练命令

这个阶段可以在新的训练终端独立开始。先恢复路径并清掉 Isaac 环境变量：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
# source /home/yzliu/physical_ai/company_project/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh local
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH
cd "$SMART_PROJECT"
```

只构造命令，不跑 stats/训练：

```bash
"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
    --camera-layout dual \
    --skip-prepare \
    --skip-stats \
    --dry-run \
    --max-steps 1 \
    --save-steps 1 \
    --global-batch-size 1 \
    --gradient-accumulation-steps 1
```

检查输出里的 dataset path、modality config、camera layout 和 output directory。

## 6. 1-step full fine-tune smoke

目标机没有 stats 时不要加 `--skip-stats`；已有 stats 时可加它。

从新终端直接执行本节时，先恢复训练终端状态：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
# source /home/yzliu/physical_ai/company_project/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh local
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH
cd "$SMART_PROJECT"
```

```bash
"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
    --camera-layout dual \
    --skip-prepare \
    --max-steps 1 \
    --save-steps 1 \
    --global-batch-size 1 \
    --gradient-accumulation-steps 1 \
    --dataloader-num-workers 0 \
    --experiment-name so101_full_finetune_smoke
```

如果 1 step OOM，停止 full fine-tune，转到
[lowmem_lora_freeze_so101/README_ZH.md](../lowmem_lora_freeze_so101/README_ZH.md)，不要先增加 batch size。

## 7. 较长训练起点

只在 1-step 通过后执行：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
# source /home/yzliu/physical_ai/company_project/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh local
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH
cd "$SMART_PROJECT"
```

```bash
"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
    --camera-layout dual \
    --skip-prepare \
    --max-steps 2000 \
    --save-steps 500 \
    --global-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --dataloader-num-workers 2 \
    --experiment-name so101_full_finetune_bs1_acc16
```

## 8. 动作语义：部署前必须知道

prepared 数据的 `action` 是绝对 LeRobot motor target。训练 config 对 arm 使用 relative
representation，只是 processor 内部表示；`Gr00tPolicy.get_action()` 会在返回前
`decode_action()`，所以部署侧收到的仍是绝对数据集空间目标。

部署命令和单位换算见
[zero_shot_isaac_smart_task/README_ZH.md](../zero_shot_isaac_smart_task/README_ZH.md)。
跨文件的数据、相机和 action 约定见
[TECHNICAL_CONTRACTS_ZH.md](../TECHNICAL_CONTRACTS_ZH.md)。

## 9. 常见失败

| 现象 | 先检查 |
|---|---|
| LeRobot 转换 import 失败 | 是否在 conda `lerobot`，不是 GR00T/Isaac env |
| 找不到 v3 数据 | `meta/info.json` 和 `codebase_version` |
| stats 找不到 modality | layout 与 config 是否一致 |
| GR00T import 失败 | 是否用 `$GROOT_ROOT/.venv/bin/python` |
| 1-step OOM | 转 low-memory，不要盲目加 batch size |
| 部署相机不匹配 | bridge/runner/layout 是否与 checkpoint 一致 |
