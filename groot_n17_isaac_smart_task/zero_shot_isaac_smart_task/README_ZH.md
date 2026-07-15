# GR00T bridge + LeIsaac SmartTask 部署指南

这份 README 只回答一件事：如何把 GR00T action 安全地送进 LeIsaac SmartTask。

主线是 SO101 微调 checkpoint；`zero-shot-oxe` 和 Franka 只用于对照。真实 SO101
硬件部署不在这里，当前 runner 驱动的是 Isaac 中的机器人。

## 1. 先选路线

| 目的 | bridge 参数 | runner 参数 |
|---|---|---|
| 部署 SO101 微调权重 | `--deployment-mode so101-finetuned` | 同左，`--robot so101 --control-mode joint` |
| base model 对照 | `--deployment-mode zero-shot-oxe` | 同左，SO101 或 Franka |

第一次运行固定按下面的顺序：

```text
相机检查 -> bridge dry-run -> 一个 policy action -> 再增加 horizon/calls
```

## 2. 每个新终端先恢复路径

目标机：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
```

本机：

```bash
source /home/yzliu/physical_ai/company_project/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh local
```

上面二选一。脚本只恢复路径变量，不激活 conda。首次使用或换机器后确认目录和 Python：

```bash
test -d "$SMART_PROJECT"
test -x "$GROOT_ROOT/.venv/bin/python"
"$GROOT_ROOT/.venv/bin/python" -c \
  "import sys; print(sys.version); assert sys.version_info[:2] == (3, 12)"
```

## 3. SO101 微调 checkpoint：终端 1 启动 bridge

新 bridge 终端先恢复路径。目标机直接执行；本机使用下面注释中的替代行：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
# source /home/yzliu/physical_ai/company_project/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh local

export CHECKPOINT="__FILL_FINETUNED_CHECKPOINT_DIR__"

conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH

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

看到 modality、camera layout 和 `action_decoding` 后再开终端 2。bridge 使用本地
pickle socket，默认只监听 `127.0.0.1`；不要直接暴露到不可信网络。

## 4. 终端 2 准备 Isaac runner

runner 终端是独立的新终端，也要先恢复路径。目标机直接执行；本机使用注释中的替代行：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
# source /home/yzliu/physical_ai/company_project/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh local

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LEISAAC_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$SMART_PROJECT"
export PYTHONPATH="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task:$LEISAAC_ROOT/source/leisaac:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export BRIDGE_HOST="${BRIDGE_HOST:-127.0.0.1}"
export BRIDGE_PORT="${BRIDGE_PORT:-5577}"
```

不要在这个 conda env 里安装 LeRobot，也不要在这个终端运行 GR00T 训练。

### 4.1 只检查相机和目标物

这一步不连接 bridge：

```bash
python experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/run_smart_task_closed_loop.py \
  --deployment-mode so101-finetuned \
  --camera-layout dual \
  --front-observation-key camera3 \
  --wrist-observation-key camera2 \
  --robot so101 \
  --control-mode joint \
  --smart-target-asset cuboid \
  --debug-cameras-only \
  --headless
```

确认日志包含：

```text
camera3 -> GR00T video.top
camera2 -> GR00T video.wrist
target object state prim_path=... root_pos_w=...
```

### 4.2 请求一次 action，但不执行

```bash
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
  --max-policy-calls 1
```

必须看到 `action.single_arm`、`action.gripper` 和解码后的绝对 LeRobot motor-unit
契约；契约不一致时 runner 会拒绝继续。

### 4.3 第一次只执行一个 policy action

```bash
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
  --no-headless \
  --max-policy-calls 1 \
  --action-horizon 1 \
  --policy-action-hz 30 \
  --so101-arm-target-scale 0.3 \
  --so101-max-arm-step-rad 0.05
```

先看运动方向和日志里的单位转换。方向正确后，保持安全参数不变，依次增加：

```text
action-horizon:   1 -> 2 -> 4 -> 完整 chunk
max-policy-calls: 1 -> 2 -> 4
```

## 5. 三种相机布局

bridge 和 runner 的 `--camera-layout` 必须与训练 checkpoint 一致。

| layout | bridge | runner live mapping |
|---|---|---|
| `wrist-only` | `--camera-layout wrist-only` | `--wrist-observation-key camera2` |
| `dual` | `--camera-layout dual` | `camera3 -> top`，`camera2 -> wrist` |
| `triple` | `--camera-layout triple` | `camera3 -> top`，`camera1 -> left`，`camera2 -> wrist` |

prepared 数据里的历史 key 与 live key 不是同一概念：

```text
训练语义：video.top / video.left / video.wrist
当前 live：camera3 / camera1 / camera2
```

## 6. 动作单位和四个安全参数

SO101 微调部署链路是：

```text
Isaac radians
-> LeRobot motor units 作为 GR00T state
-> GR00T + decode_action
-> decoded absolute motor target
-> motor limits
-> radians
-> 每个 policy action 的 radian 限幅
-> Isaac runtime joint limits
-> env.step()
```

不要把 `single_arm` 当 radian，也不要执行
`current_radians + returned_action`。processor 内部可以使用 relative 表示，但
`Gr00tPolicy.get_action()` 对外返回的是解码后的数据集空间绝对目标。

| 参数 | 含义 |
|---|---|
| `--max-policy-calls` | 最多请求多少个 action chunk |
| `--action-horizon` | 每个 chunk 执行多少个 policy action；`0` 表示完整 chunk |
| `--so101-arm-target-scale` | 当前姿态朝绝对目标移动的比例，范围 `[0,1]` |
| `--so101-max-arm-step-rad` | 每个 policy action 的最大关节变化，单位 radian |

旧参数 `--so101-arm-delta-scale` 和 `--so101-max-arm-delta` 只保留兼容，不要再写进新命令。

当前 SO101 数据是 30 Hz，SmartTask env 是 60 Hz，所以 runner 会把每个 policy target
保持两个 `env.step()`。`0.05 rad` 是第一次仿真方向检查值，不是硬件速度规格；正式阈值应由
已验证的安全速度乘以 `1 / policy_action_hz` 推导。

准备读或修改这条链路时，看
[SmartTask 代码阅读约定](../TECHNICAL_CONTRACTS_ZH.md)；实际运行不要求先读完技术手册。

## 7. LEGO USD 补丁

同事的 LEGO USD 缺内部 layer。runner 默认使用：

```bash
--smart-target-asset cuboid
```

它只在当前 runner 进程中创建红色 2x4 尺寸 cuboid，不修改 `leisaac/`。拿到完整 USD 后改为：

```bash
--smart-target-asset /absolute/path/to/complete_lego.usd
```

如果要完全依赖 scene parser，显式使用 `--smart-target-asset scene`。

## 8. Zero-shot 对照

新 bridge 终端先恢复路径，然后改为 zero-shot。目标机直接执行；本机使用注释中的替代行：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
# source /home/yzliu/physical_ai/company_project/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh local

"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/groot_bridge_server.py \
    --deployment-mode zero-shot-oxe \
    --model-path nvidia/GR00T-N1.7-3B \
    --device cuda
```

终端 2 使用：

```text
SO101 joint:  --deployment-mode zero-shot-oxe --robot so101 --control-mode joint
SO101 EEF:    --deployment-mode zero-shot-oxe --robot so101 --control-mode eef
Franka joint: --deployment-mode zero-shot-oxe --robot franka --control-mode joint
Franka EEF:   --deployment-mode zero-shot-oxe --robot franka --control-mode eef
```

这些是 embodiment 对照，不是 SO101 微调部署主线。

## 9. 常见失败：停在哪一层

| 现象 | 先检查 |
|---|---|
| bridge import/checkpoint 失败 | GR00T Python 3.12 venv、checkpoint 路径 |
| camera key 不存在 | `--debug-cameras-only` 和 layout/live mapping |
| 看不到 LEGO | `--smart-target-asset cuboid`、日志里的 `root_pos_w` |
| action contract mismatch | bridge 与 runner 是否来自同一版代码 |
| joint limit mismatch | 是否加载了另一套 SO101 USD；不要继续执行 |
| 动作方向错误 | 立即停在 one-step，不要增加 horizon |
| 30/60 Hz 不整除 | 修正 `--policy-action-hz`，不要绕过报错 |

## 10. 代码边界与测试

```text
groot_bridge_server.py       GR00T 进程
run_smart_task_closed_loop.py Isaac/LeIsaac 主流程
wire.py                      本地 socket 协议
action_chunk.py              action shape/时间维校验与切片
action_timing.py             policy/env 频率对齐
pose_math.py                 quaternion/rot6d/EEF 数学
so101_joint_units.py         motor units/radians/limits
tests/                       不启动 SimulationApp 的回归测试
```

运行测试：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
# source /home/yzliu/physical_ai/company_project/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh local
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LEISAAC_ENV"
cd "$SMART_PROJECT"
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/tests -v
```

环境搭建看 [ubuntu_env_setup/README_ZH.md](../ubuntu_env_setup/README_ZH.md)，数据准备看
[full_finetune_so101/README_ZH.md](../full_finetune_so101/README_ZH.md)，低显存训练看
[lowmem_lora_freeze_so101/README_ZH.md](../lowmem_lora_freeze_so101/README_ZH.md)。
