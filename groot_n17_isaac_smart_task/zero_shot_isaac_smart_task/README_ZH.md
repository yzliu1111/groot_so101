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
# task 只选基础 env；scene profile 单独选择部署布局。
export ISAAC_TASK="LeIsaac-SO101-SmartTask-v0"
export ISAAC_SCENE_PROFILE="table-red24"
export TARGET_OBJECT_KEY="auto"
# instruction 必须与 checkpoint 的训练文本一致。
export TASK_INSTRUCTION="Pick up the red 2x4 lego brick."
```

不要在这个 conda env 里安装 LeRobot，也不要在这个终端运行 GR00T 训练。

### 4.1 选择基础 task 与 scene profile

两个入口的职责固定分开：

- `--task` 只选择机器人、action、observation、camera 等基础 Gym env；SO101 三种部署场景都使用
  `LeIsaac-SO101-SmartTask-v0`。
- `--scene-profile` 只选择部署前的木盘和 LEGO 布局，不依赖额外 Gym task ID。
- bridge 不接收 task 或 scene profile；`--instruction` 仍必须与 checkpoint 的训练文本一致。

| `--scene-profile` | 部署场景 | target 规则 |
|---|---|---|
| `tray-red24` | 桌上有木盘，单个红色 2x4 位于木盘内 | `auto` 自动选择红色 2x4 |
| `table-red24` | 桌上没有木盘，单个红色 2x4 位于桌面 | `auto` 自动选择红色 2x4 |
| `multi-lego-tray` | 桌上有木盘，红 2x4、红 2x2、蓝 2x4 位于木盘一侧 | 必须显式传 `--target-object-key` |
| `task-default` | 保留所选 task 原生 scene 和旧资产兼容路径 | 从 env cfg 解析 |

因此，无木盘的桌面单块场景已经可直接选择：

```text
--task LeIsaac-SO101-SmartTask-v0 --scene-profile table-red24
```

`multi-lego-tray` 的 target 可选
`red_2x4_lego_brick`、`red_2x2_lego_brick` 或 `blue_2x4_lego_brick`，并且必须显式传
与 checkpoint 训练文本一致的 `--instruction`。多块位置直接沿用当前 LeIsaac scene 的 authored
布局，不再增加 x/y 方向参数。

例如选择蓝色 2x4：

```bash
export ISAAC_SCENE_PROFILE="multi-lego-tray"
export TARGET_OBJECT_KEY="blue_2x4_lego_brick"
export TASK_INSTRUCTION="Pick up the blue 2x4 lego brick and place it in the tray."
```

三个显式 profile 当前只与基础 SO101 task 组合；不要使用数据生成用的 `*-Mimic-v0`。
checkpoint metadata 不保存 Gym task ID 或 scene profile，所以 runner 不会从权重猜场景。

### 4.2 恢复 LeIsaac SmartTask 相机

当前 `leisaac/main` 只保留单相机配置，仅暴露一个腕部 `camera1`。要使用
`dual` / `triple` checkpoint，需要手工恢复 LeIsaac 已有的三路相机定义；runner 的 key 参数
只能映射已经存在的 observation，不能创建缺失的 sensor。

修改下面这个文件：

```text
leisaac/source/leisaac/leisaac/tasks/smart_task/smart_task_env_cfg.py
```

不要只取消注释，因为当前 wrist 和被注释的 left 块都叫 `camera1`。真正的约束只有两个：

1. `SmartTaskSceneCfg` 中启用所需的 wrist、left、front/top sensor，并给它们互不重复的字段名。
2. `SmartTaskObservationsCfg.PolicyCfg` 为这些 sensor 暴露同名 observation term；每个
   `SceneEntityCfg(...)` 必须与 scene sensor 名称一致。

`camera1/2/3` 本身没有固定语义。改动最少的一种做法是保留现有 wrist=`camera1`，将启用的
left 块命名为 `camera2`，front/top 继续使用 `camera3`：

```text
camera1 = wrist
camera2 = left
camera3 = front/top
```

这只是 LeIsaac 文件的一个最小改动示例，不是 runner 的硬编码约定。也可以使用其他名称或
排列，只要各 key 唯一，并在启动 runner 时用 `--isaac-front/left/wrist-camera-key` 明确映射。
同时保留两个 `__post_init__()` 中删除父类旧 `front` / `wrist` / `left` 属性的逻辑。

`wrist-only` 只需暴露一条腕部 observation；`dual` 需要两条不同的 front/top 与 wrist；
`triple` 需要三条不同的 front/top、left、wrist。现有场景 USD 已包含 left/front 的 xform，
腕部 camera prim 由 `TiledCameraCfg` 创建，不需要另外修改 USD 路径。

### 4.3 只检查相机和目标物

这一步不连接 bridge：

```bash
python experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/run_smart_task_closed_loop.py \
  --deployment-mode so101-finetuned \
  --so101-checkpoint-joint-units "$SO101_CHECKPOINT_JOINT_UNITS" \
  --task "$ISAAC_TASK" \
  --scene-profile "$ISAAC_SCENE_PROFILE" \
  --target-object-key "$TARGET_OBJECT_KEY" \
  --instruction "$TASK_INSTRUCTION" \
  --camera-layout dual \
  --isaac-front-camera-key camera3 \
  --isaac-wrist-camera-key camera1 \
  --robot so101 \
  --control-mode joint \
  --debug-cameras-only \
  --headless
```

确认日志包含：

```text
camera3 -> GR00T video.top
camera1 -> GR00T video.wrist
target object state prim_path=... root_pos_w=...
```

### 4.4 请求一次 action，但不执行

```bash
python experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/run_smart_task_closed_loop.py \
  --deployment-mode so101-finetuned \
  --so101-checkpoint-joint-units "$SO101_CHECKPOINT_JOINT_UNITS" \
  --task "$ISAAC_TASK" \
  --scene-profile "$ISAAC_SCENE_PROFILE" \
  --target-object-key "$TARGET_OBJECT_KEY" \
  --instruction "$TASK_INSTRUCTION" \
  --camera-layout dual \
  --isaac-front-camera-key camera3 \
  --isaac-wrist-camera-key camera1 \
  --robot so101 \
  --control-mode joint \
  --bridge-host "$BRIDGE_HOST" \
  --bridge-port "$BRIDGE_PORT" \
  --dry-run \
  --max-policy-calls 1
```

必须看到 `action.single_arm`、`action.gripper`，以及 bridge 报告
`dataset_action_units=checkpoint_dataset_coordinates`；runner 日志里的 `arm_units` 必须与
训练数据来源一致。

### 4.5 第一次只执行一个 policy action

```bash
python experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/run_smart_task_closed_loop.py \
  --deployment-mode so101-finetuned \
  --so101-checkpoint-joint-units "$SO101_CHECKPOINT_JOINT_UNITS" \
  --task "$ISAAC_TASK" \
  --scene-profile "$ISAAC_SCENE_PROFILE" \
  --target-object-key "$TARGET_OBJECT_KEY" \
  --instruction "$TASK_INSTRUCTION" \
  --camera-layout dual \
  --isaac-front-camera-key camera3 \
  --isaac-wrist-camera-key camera1 \
  --robot so101 \
  --control-mode joint \
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
| `wrist-only` | `--camera-layout wrist-only` | 例：`--isaac-wrist-camera-key camera1` |
| `dual` | `--camera-layout dual` | 例：`--isaac-front-camera-key camera3 --isaac-wrist-camera-key camera1` |
| `triple` | `--camera-layout triple` | 在 dual 示例上增加 `--isaac-left-camera-key camera2` |

prepared 数据里的历史 key 与 live key 不是同一概念：

```text
训练语义：video.top / video.left / video.wrist
live key：由当前 LeIsaac env cfg 决定，通过 --isaac-*-camera-key 显式映射
```

这里的 `dataset-*` 和 `isaac-*` 参数故意使用不同前缀：训练侧参数指定 LeRobot dataset
feature key，部署侧参数指定 `obs["policy"]` 中的 live key。runner 会同时检查 key 是否存在，
以及所选角色是否映射到不同的 live camera；例如 triple 中把三路都指定成 `camera1` 会在请求
模型前直接报错。

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

## 7. Scene profile 的 USD 组合

三个显式 scene profile 都由 `experiments/` 在 runner 进程内组合，不修改 `leisaac/`：

- 生成 file-backed USDA wrapper，让 LeIsaac parser 与 live scene 读取同一份组合结果。
- wrapper 停用原 scene 里的 legacy LEGO，并清空其中写死到同事机器的 payload；两个仅用于
  viewport 显示的 camera preview mesh 也会清空绝对 reference，真实 Camera prim 不受影响。
- 红 2x4 和红 2x2 使用 repo 内完整 USD；蓝 2x4 复用完整红 2x4 几何，再以
  `PreviewSurfaceCfg` 覆盖为 LeIsaac authored 的纯蓝材质。
- 木盘继续使用 scene 中完整的 `plate.usd`；`table-red24` 会真正 deactivate 木盘 prim，
  不是只隐藏画面。

wrapper 写入 `zero_shot_isaac_smart_task/runs/scene_profiles/`，该目录已被 git ignore。
显式 profile 自己拥有资产和 pose，因此会拒绝 `--smart-target-*` 以及自定义
`--smart-scene-usd`，避免两套配置叠加。

`task-default` 只为旧命令兼容，仍沿用原 task scene / cuboid fallback；新三种部署场景不要
再传 legacy 资产参数。

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
| 场景物体不对 | 检查 `scene profile=...`、wrapper、objects/tray 日志、`target object=...` 和 `root_pos_w` |
| action contract mismatch | bridge 与 runner 是否来自同一版代码，bridge 是否报告 decoded dataset action |
| joint limit mismatch | 是否加载了另一套 SO101 USD；不要继续执行 |
| 动作方向错误 | 立即停在 one-step，不要增加 horizon |
| 30/60 Hz 不整除 | 修正 `--policy-action-hz`，不要绕过报错 |

## 10. 代码边界与测试

```text
groot_bridge_server.py       GR00T 进程
run_smart_task_closed_loop.py Isaac/LeIsaac 主流程
scene_profiles.py             部署布局、USD wrapper、真实 LEGO 资产与 target 重绑
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
