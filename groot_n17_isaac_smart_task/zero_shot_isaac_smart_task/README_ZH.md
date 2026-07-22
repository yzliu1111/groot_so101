# GR00T bridge + LeIsaac SmartTask 部署指南

这份 README 只回答一件事：如何把 GR00T action 安全地送进 LeIsaac SmartTask。

主线是 SO101 微调 checkpoint；`zero-shot-oxe` 和 Franka 只用于对照。真实 SO101
硬件部署不在这里，当前 runner 驱动的是 Isaac 中的机器人。

## 1. 先选路线

| 目的 | bridge 参数 | runner 参数 |
|---|---|---|
| sim 数据微调权重 | `--deployment-mode so101-finetuned` | 同左，`--so101-checkpoint-joint-units lerobot_motor_units --robot so101 --control-mode joint` |
| 真机数据微调权重 | `--deployment-mode so101-finetuned` | 同左，`--so101-checkpoint-joint-units degrees --robot so101 --control-mode joint` |
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

bridge 只负责 decode 回 checkpoint 数据集坐标，不做单位换算。看到 modality、camera
layout 和 `action_decoding` 后再开终端 2。bridge 使用本地 pickle socket，默认只监听
`127.0.0.1`；不要直接暴露到不可信网络。

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
# sim checkpoint 用 lerobot_motor_units；真机 checkpoint 用 degrees。
export SO101_CHECKPOINT_JOINT_UNITS="lerobot_motor_units"
# Gym task 决定 LeIsaac env config / scene；instruction 要与 checkpoint 的训练文本一致。
export ISAAC_TASK="LeIsaac-SO101-SmartTask-v0"
export TASK_INSTRUCTION="Pick up the red 2x4 lego brick."
```

不要在这个 conda env 里安装 LeRobot，也不要在这个终端运行 GR00T 训练。

### 4.1 选择 task / scene

runner 只保留一个选择入口：`--task` 接受 Gym task ID，所选 env config 负责 scene、相机、
目标物和 termination。bridge 不再接收 task。`--instruction` 显式值优先；当前基础
SO101/Franka task 省略时保留旧默认训练文本，其他 task 省略时读取所选 env config 的
`task_description`。checkpoint metadata 不保存 Gym task ID，所以不能从权重自动推断场景。

当前 checkout 的真实状态：

| 数据 / 场景 | 应选 task | 当前可用性 |
|---|---|---|
| 基础红色 2x4 抓取/抬起 | `LeIsaac-SO101-SmartTask-v0` | env cfg 已实现 |
| 多物体中选红色 2x4 并放入木盘 | `LeIsaac-SO101-SmartTask-Red-v0` | 只有 Gym 注册，`SmartTaskRedEnvCfg` 缺失 |
| 多物体中选蓝色 2x4 并放入木盘 | `LeIsaac-SO101-SmartTask-Blue-v0` | 只有 Gym 注册，`SmartTaskBlueEnvCfg` 缺失 |
| 多物体中选小红色 2x2 并放入木盘 | `LeIsaac-SO101-SmartTask-SmallRed-v0` | 只有 Gym 注册，`SmartTaskSmallRedEnvCfg` 缺失 |
| 无木盘、桌面单块抓取 | 当前没有对应 task ID | 尚不能选择 |

因此这次 deploy 接口已经能随 `--task` 切换与当前 observation/action/target 契约兼容、且可
实例化的 SmartTask 场景；但上表其余场景
不能仅靠 runner 补全。拿到包含对应 env cfg 的 LeIsaac 版本后，只需修改
`ISAAC_TASK` / `TASK_INSTRUCTION`，无需再改 bridge 或新增一套 scene 参数。部署时使用普通 task，
不要使用为数据生成准备的 `*-Mimic-v0`。

### 4.2 只检查相机和目标物

这一步不连接 bridge：

```bash
python experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/run_smart_task_closed_loop.py \
  --deployment-mode so101-finetuned \
  --so101-checkpoint-joint-units "$SO101_CHECKPOINT_JOINT_UNITS" \
  --task "$ISAAC_TASK" \
  --instruction "$TASK_INSTRUCTION" \
  --camera-layout dual \
  --front-observation-key camera3 \
  --wrist-observation-key camera2 \
  --robot so101 \
  --control-mode joint \
  --smart-target-asset auto \
  --debug-cameras-only \
  --headless
```

确认日志包含：

```text
camera3 -> GR00T video.top
camera2 -> GR00T video.wrist
target object state prim_path=... root_pos_w=...
```

### 4.3 请求一次 action，但不执行

```bash
python experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/run_smart_task_closed_loop.py \
  --deployment-mode so101-finetuned \
  --so101-checkpoint-joint-units "$SO101_CHECKPOINT_JOINT_UNITS" \
  --task "$ISAAC_TASK" \
  --instruction "$TASK_INSTRUCTION" \
  --camera-layout dual \
  --front-observation-key camera3 \
  --wrist-observation-key camera2 \
  --robot so101 \
  --control-mode joint \
  --smart-target-asset auto \
  --bridge-host "$BRIDGE_HOST" \
  --bridge-port "$BRIDGE_PORT" \
  --dry-run \
  --max-policy-calls 1
```

必须看到 `action.single_arm`、`action.gripper`，以及 bridge 报告
`dataset_action_units=checkpoint_dataset_coordinates`；runner 日志里的 `arm_units` 必须与
训练数据来源一致。

### 4.4 第一次只执行一个 policy action

```bash
python experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/run_smart_task_closed_loop.py \
  --deployment-mode so101-finetuned \
  --so101-checkpoint-joint-units "$SO101_CHECKPOINT_JOINT_UNITS" \
  --task "$ISAAC_TASK" \
  --instruction "$TASK_INSTRUCTION" \
  --camera-layout dual \
  --front-observation-key camera3 \
  --wrist-observation-key camera2 \
  --robot so101 \
  --control-mode joint \
  --smart-target-asset auto \
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

## 6. checkpoint 单位和四个安全参数

| checkpoint 来源 | 前五个 arm state/action | gripper |
|---|---|---|
| LeIsaac sim | LeRobot motor units | LeRobot `[0,100]` |
| 真机 `use_degrees=True` | degrees | LeRobot `[0,100]` |

`--so101-checkpoint-joint-units` 只在 runner 中选择前五维转换。runner 不会从 checkpoint
自动推断；当前 LeRobot meta 也没有 unit 字段。`get_action()` 返回 decode 后的数据集空间
绝对目标，所以不要把 `single_arm` 当 radians，也不要执行
`current_radians + returned_action`。统一安全链路是：数据集限位 → 转 radians → 单步限幅 →
Isaac runtime limits → `env.step()`。更完整的代码契约见
[TECHNICAL_CONTRACTS_ZH.md](../TECHNICAL_CONTRACTS_ZH.md)。

| 参数 | 含义 |
|---|---|
| `--so101-checkpoint-joint-units` | checkpoint 前五个 arm state/action 的单位；sim 用 `lerobot_motor_units`，真机用 `degrees` |
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

当前基础 task 的 LEGO USD 缺内部 layer。runner 默认使用：

```bash
--smart-target-asset auto
```

对 `LeIsaac-SO101-SmartTask-v0`，`auto` 只在当前 runner 进程中创建红色 2x4
cuboid，不修改 `leisaac/`。对其他 task，`auto` 等于 `scene`：完全使用该 task 自己的 scene，
不会注入红色目标。这样切到 Blue / SmallRed / 多物体场景时不会静默选错积木。

基础 task 拿到完整 USD 后可改为：

```bash
--smart-target-asset /absolute/path/to/complete_lego.usd
```

如果基础 task 要完全依赖 scene parser，显式使用 `--smart-target-asset scene`。非基础 task
不接受旧红色 cuboid 补丁；其资产应在对应 env config 中声明。

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
| 看不到 LEGO | 基础 task 检查 `--smart-target-asset auto`；其他 task 检查其 env cfg/scene；再看日志里的 `root_pos_w` |
| action contract mismatch | bridge 与 runner 是否来自同一版代码，bridge 是否报告 decoded dataset action |
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
so101_joint_units.py         motor units/degrees/radians/limits
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
