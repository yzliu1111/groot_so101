# GR00T bridge + LeIsaac SmartTask 部署指南

这份 README 只回答一件事：如何把 GR00T action 安全地送进 LeIsaac SmartTask。

主线是 SO101 微调 checkpoint；`zero-shot-oxe` 和 Franka 只用于对照。真实 SO101
硬件部署不在这里，当前 runner 驱动的是 Isaac 中的机器人。

慢速远程连接可直接跳转：
[选场景](#41-选择基础-task-与-scene-profile) ·
[查相机/物体](#43-只检查相机和目标物) ·
[dry-run](#44-请求一次-action但不执行) ·
[one-step](#45-定义一次执行函数第一次只跑一个-policy-action) ·
[逐档增加](#46-按-24完整-chunk更多-calls-逐档增加)

## 1. 快速决策（先看这里）

| 目的 | bridge 参数 | runner 参数 |
|---|---|---|
| sim 数据微调权重 | `--deployment-mode so101-finetuned` | 同左，`--so101-checkpoint-joint-units lerobot_motor_units --robot so101 --control-mode joint` |
| 真机数据微调权重 | `--deployment-mode so101-finetuned` | 同左，`--so101-checkpoint-joint-units degrees --robot so101 --control-mode joint` |
| base model 对照 | `--deployment-mode zero-shot-oxe --camera-layout dual` | 同左；SO101 或 Franka；不要传 `degrees` |

SO101 微调部署的三种显式场景都固定使用：

```text
--task LeIsaac-SO101-SmartTask-v0
```

不要把 `--task` 和 `--scene-profile` 当成两个可以任意组合的菜单：

| 你要验证什么 | `--scene-profile` | `--target-object-key` | profile 配置的 success |
|---|---|---|---|
| 有木盘，单个红色 2x4，抓取/抬起 | `tray-red24` | `auto` | `object_lifted` |
| 无木盘，单个红色 2x4，抓取/抬起 | `table-red24` | `auto` | `object_lifted` |
| 有木盘，多块 LEGO，选一块放入木盘 | `multi-lego-tray` | 必须显式选择 | `object_placed_on_tray` |

最后一列是 profile 写入 env cfg 的判据，不是默认运行的自动停止条件。SmartTask 原生判据目前
过松，runner 默认将 success termination 关闭；本手册的正常命令也不加
`--use-env-success-termination`。先用画面、目标物状态和 metrics 验证，不要看到表格就假定
episode 会在成功时自动停止。

第一次运行固定按下面的顺序，不要跳步：

```text
相机和目标物检查
-> runner dry-run（会向 bridge 请求 action，但不执行）
-> calls=1, horizon=1（只执行 1 个 policy action）
-> calls=1, horizon=2
-> calls=1, horizon=4
-> calls=1, horizon=0（完整执行 1 个返回 chunk）
-> 再增加 max-policy-calls
```

最容易误解的参数在这里先说清楚：

```text
--action-horizon 0      = 完整执行 bridge 实际返回的 chunk，不是“执行 0 步”
--max-policy-calls 1    = 最多请求 1 个 chunk，不是“执行 1 步”
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
# 必须与 checkpoint 的训练 camera layout 一致：wrist-only / dual / triple。
export CAMERA_LAYOUT="dual"

if [[ -d "$CHECKPOINT" && "$CHECKPOINT" != *"__FILL_"* ]]; then
  printf '[OK] checkpoint=%s\n' "$CHECKPOINT"
else
  printf '[FAIL] fill CHECKPOINT with an existing finetuned checkpoint directory\n' >&2
  false
fi

case "$CAMERA_LAYOUT" in
  wrist-only|dual|triple) printf '[OK] bridge camera-layout=%s\n' "$CAMERA_LAYOUT" ;;
  *) printf '[FAIL] invalid CAMERA_LAYOUT=%s\n' "$CAMERA_LAYOUT" >&2; false ;;
esac
```

必须看到两行 `[OK]`，再复制启动块：

```bash
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH

cd "$SMART_PROJECT"
"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/groot_bridge_server.py \
    --deployment-mode so101-finetuned \
    --camera-layout "$CAMERA_LAYOUT" \
    --model-path "$CHECKPOINT" \
    --host 127.0.0.1 \
    --port 5577 \
    --device cuda
```

bridge 只负责 decode 回 checkpoint 数据集坐标，不做单位换算。看到以下日志后再开终端 2：

```text
[bridge] policy loaded
[bridge] camera layout: ...
[bridge] modality: ...
[bridge] action decoding: ...
[bridge] listening on 127.0.0.1:5577
```

没有出现 `listening`，或者打印的 camera layout / modality 与 checkpoint 不一致时，停在
bridge 排障，不要继续启动 runner。bridge 使用本地 pickle socket，默认只监听
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
# 必须与终端 1 的 bridge 和 checkpoint 一致。
export CAMERA_LAYOUT="dual"
# 下面是恢复三路相机后的示例 live key；所选 layout 不使用的 key 会被 runner 忽略。
export ISAAC_FRONT_CAMERA_KEY="camera3"
export ISAAC_LEFT_CAMERA_KEY="camera2"
export ISAAC_WRIST_CAMERA_KEY="camera1"

if [[ "${CONDA_DEFAULT_ENV:-}" == "$LEISAAC_ENV" &&
      -n "${CONDA_PREFIX:-}" &&
      -x "$CONDA_PREFIX/bin/python" &&
      -d "$SMART_PROJECT" &&
      -d "$LEISAAC_ROOT" ]]; then
  printf '[OK] runner env: conda=%s python=%s\n' \
    "$CONDA_DEFAULT_ENV" "$CONDA_PREFIX/bin/python"
  printf '[OK] runner env: task-root=%s\n' "$SMART_PROJECT"
  printf '[OK] runner env: leisaac-root=%s\n' "$LEISAAC_ROOT"
  printf '[OK] runner env: camera-layout=%s\n' "$CAMERA_LAYOUT"
else
  printf '[FAIL] conda activation or runner paths are invalid\n' >&2
  printf '[FAIL] expected_env=%s actual_env=%s prefix=%s\n' \
    "$LEISAAC_ENV" "${CONDA_DEFAULT_ENV:-<unset>}" "${CONDA_PREFIX:-<unset>}" >&2
  false
fi
```

必须看到四行 `[OK]`。出现 `[FAIL]`，或 conda 激活失败时，都停在这里。不要在这个
conda env 里安装 LeRobot，也不要在这个终端运行 GR00T 训练。

### 4.1 选择基础 task 与 scene profile

五个概念不要混在一起：

```text
--task               选择基础 Gym env
--scene-profile      选择木盘和 LEGO 布局；multi 还会重绑 env cfg 的 success 判据
--target-object-key  指定 runner 追踪和重绑的目标 LEGO
--instruction        checkpoint 的语言条件，必须与训练文本一致
checkpoint           决定模型实际学会了哪项任务
```

当前部署组合如下：

| `--task` | `--scene-profile` | target | instruction | 当前结果 |
|---|---|---|---|---|
| `LeIsaac-SO101-SmartTask-v0` | `tray-red24` | `auto` | 显式写 checkpoint 原文 | 支持：有盘单红 pick/lift |
| 同上 | `table-red24` | `auto` | 显式写 checkpoint 原文 | 支持：无盘单红 pick/lift |
| 同上 | `multi-lego-tray` | 三选一，不能是 `auto` | 必须显式写 checkpoint 原文 | 支持：选定目标；profile 配置 place-on-tray 判据 |
| 同上 | `task-default` | 从旧 env cfg 解析 | 旧默认或显式值 | 仅用于旧 scene / cuboid fallback |
| Blue / SmallRed task ID | `task-default` | 当前 LeIsaac cfg 尚未完整重绑 | 不要依赖自动值 | 暂不列为已验证部署路线 |
| 任意非基础 task | 三种显式 profile | — | — | runner 会拒绝 |
| 任意 `*-Mimic-v0` | 任意 profile | — | — | runner 会拒绝 |

bridge CLI 不接收 task、scene profile 或 target；runner 会把 instruction 放进 observation
发给 bridge。以下场景预设中只复制本次要跑的一组。

上表的 success 判据默认不参与 episode 自动终止；如需专门评估 env termination，先确认
判据本身不再过松，再显式增加 `--use-env-success-termination`。普通部署验证不要加。

#### A. 有木盘，单个红色 2x4，抓取/抬起

```bash
export ISAAC_TASK="LeIsaac-SO101-SmartTask-v0"
export ISAAC_SCENE_PROFILE="tray-red24"
export TARGET_OBJECT_KEY="auto"
export TASK_INSTRUCTION="Pick up the red 2x4 lego brick."
```

#### B. 无木盘，单个红色 2x4，抓取/抬起

```bash
export ISAAC_TASK="LeIsaac-SO101-SmartTask-v0"
export ISAAC_SCENE_PROFILE="table-red24"
export TARGET_OBJECT_KEY="auto"
export TASK_INSTRUCTION="Pick up the red 2x4 lego brick."
```

有盘和无盘单块场景都只有红色 2x4，并使用相同完整位姿；无盘场景只关闭木盘。它们不复用
多块 pick-and-place 场景的红色 2x4 坐标。

#### C. 有木盘，多块 LEGO，选择蓝色 2x4 放入木盘

```bash
export ISAAC_TASK="LeIsaac-SO101-SmartTask-v0"
export ISAAC_SCENE_PROFILE="multi-lego-tray"
export TARGET_OBJECT_KEY="blue_2x4_lego_brick"
export TASK_INSTRUCTION="Pick up the blue 2x4 lego brick and place it in the tray."
```

上面的英文只有在它确实是 checkpoint 训练原文时才能直接使用。要改选红色目标时，不要只
替换一个残留变量；从下面再完整复制一组。

```bash
export ISAAC_TASK="LeIsaac-SO101-SmartTask-v0"
export ISAAC_SCENE_PROFILE="multi-lego-tray"
export TARGET_OBJECT_KEY="red_2x4_lego_brick"
export TASK_INSTRUCTION="__FILL_EXACT_CHECKPOINT_INSTRUCTION__"
```

```bash
export ISAAC_TASK="LeIsaac-SO101-SmartTask-v0"
export ISAAC_SCENE_PROFILE="multi-lego-tray"
export TARGET_OBJECT_KEY="red_2x2_lego_brick"
export TASK_INSTRUCTION="__FILL_EXACT_CHECKPOINT_INSTRUCTION__"
```

多块位置继续沿用 LeIsaac authored multi-LEGO 布局。`multi-lego-tray` 不允许
`TARGET_OBJECT_KEY=auto`，也不允许省略 `TASK_INSTRUCTION`。

选完后立即检查，确认没有残留上一轮终端变量或未填写占位符：

```bash
if [[ -n "${ISAAC_TASK:-}" &&
      -n "${ISAAC_SCENE_PROFILE:-}" &&
      -n "${TARGET_OBJECT_KEY:-}" &&
      -n "${TASK_INSTRUCTION:-}" &&
      "$TASK_INSTRUCTION" != *"__FILL_"* ]]; then
  printf '[OK] deploy choice: task=%s profile=%s target=%s\n' \
    "$ISAAC_TASK" "$ISAAC_SCENE_PROFILE" "$TARGET_OBJECT_KEY"
  printf '[OK] deploy choice: instruction=%s\n' "$TASK_INSTRUCTION"
else
  printf '[FAIL] scene preset is incomplete; copy one complete block and fill its instruction\n' >&2
  false
fi
```

以下组合不要运行：

```text
显式 scene profile + Blue / SmallRed / Franka task
显式 scene profile + 任意 *-Mimic-v0 task
显式 scene profile + --smart-scene-usd 自定义路径
显式 scene profile + 任意 --smart-target-* legacy override
multi-lego-tray + --target-object-key auto
multi-lego-tray + 省略 --instruction
so101-finetuned + --robot franka
so101-finetuned + --control-mode eef
zero-shot-oxe + --camera-layout wrist-only/triple
zero-shot-oxe + --so101-checkpoint-joint-units degrees
bridge / runner / checkpoint 使用不同 camera layout
```

checkpoint metadata 不保存 Gym task ID 或 scene profile，所以 runner 不会从权重猜场景。

### 4.2 首次使用或 LeIsaac 更新后：确认 SmartTask 相机

如果目标机已经暴露了当前 layout 所需的不同 camera key，直接
[跳到 4.3 做只读相机检查](#43-只检查相机和目标物)。只有 camera key 缺失或
`leisaac/main` 更新覆盖了本地相机定义时，才做本节的一次性源码恢复。

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
排列，只要各 key 唯一，并在启动 runner 时用 `--isaac-front-camera-key`、
`--isaac-left-camera-key`、`--isaac-wrist-camera-key` 明确映射。同时保留两个
`__post_init__()` 中删除父类旧 `front` / `wrist` / `left` 属性的逻辑。

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
  --camera-layout "$CAMERA_LAYOUT" \
  --isaac-front-camera-key "$ISAAC_FRONT_CAMERA_KEY" \
  --isaac-left-camera-key "$ISAAC_LEFT_CAMERA_KEY" \
  --isaac-wrist-camera-key "$ISAAC_WRIST_CAMERA_KEY" \
  --robot so101 \
  --control-mode joint \
  --debug-cameras-only \
  --headless
```

三个显式 scene profile 会打印前两行 profile 日志；`task-default` 不会。只有当前路线应有的
日志以及后续各项都正确才继续：

```text
[runner] scene profile=... source_scene=... wrapper=...
[runner] scene profile objects=... tray_active=...
[runner] Isaac env created successfully
[runner] Isaac obs['policy'][...] -> GR00T video....
[runner] target object=...
[runner] target object state prim_path=... root_pos_w=...
[runner] camera-debug end
```

本手册默认的 dual 映射还应逐字包含：

```text
[runner] Isaac obs['policy']['camera3'] -> GR00T video.top
[runner] Isaac obs['policy']['camera1'] -> GR00T video.wrist
```

如果使用 wrist-only 或 triple，以第 5 节表格中该 layout 实际启用的角色为准。

场景日志应与所选预设一致：

| profile | objects | `tray_active` | red 2x4 初始位置 |
|---|---|---:|---|
| `tray-red24` | 只有 `red_2x4_lego_brick` | `True` | 约 `(0.0, 0.25, 0.025)` |
| `table-red24` | 只有 `red_2x4_lego_brick` | `False` | 与 `tray-red24` 相同 |
| `multi-lego-tray` | red 2x4、red 2x2、blue 2x4 | `True` | 保留 multi authored 坐标 |

出现 `task setup failed` / `gym.make failed`、camera key 不存在、两种相机角色复用同一个 key，
或者 objects / `tray_active` / target / `root_pos_w` 不符合预设时，停在这里。该步骤不连接
bridge，任一项不对都不要进入 dry-run。

### 4.4 请求一次 action，但不执行

```bash
python experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/run_smart_task_closed_loop.py \
  --deployment-mode so101-finetuned \
  --so101-checkpoint-joint-units "$SO101_CHECKPOINT_JOINT_UNITS" \
  --task "$ISAAC_TASK" \
  --scene-profile "$ISAAC_SCENE_PROFILE" \
  --target-object-key "$TARGET_OBJECT_KEY" \
  --instruction "$TASK_INSTRUCTION" \
  --camera-layout "$CAMERA_LAYOUT" \
  --isaac-front-camera-key "$ISAAC_FRONT_CAMERA_KEY" \
  --isaac-left-camera-key "$ISAAC_LEFT_CAMERA_KEY" \
  --isaac-wrist-camera-key "$ISAAC_WRIST_CAMERA_KEY" \
  --robot so101 \
  --control-mode joint \
  --bridge-host "$BRIDGE_HOST" \
  --bridge-port "$BRIDGE_PORT" \
  --dry-run \
  --max-policy-calls 1
```

成功标记：

```text
[runner] bridge modality: ...
[runner] bridge action decoding: ...
[runner] action keys: ...
[runner] action summary: ...
[runner] dry-run metrics ...
```

`action keys` 必须包含裸 key `single_arm` 和 `gripper`，例如：

```text
[runner] action keys: ['gripper', 'single_arm']
```

bridge 必须报告
`policy_api_output=decoded_dataset_action` 和
`dataset_action_units=checkpoint_dataset_coordinates`；runner 日志里的 `arm_units` 必须与
训练数据来源一致。出现 action contract mismatch、camera layout mismatch、NaN / inf 或错误
单位时，停在 dry-run。

`--dry-run` 仍会按 `--max-policy-calls` 请求 chunk，但不会执行 action；因此这里显式限制为
`--max-policy-calls 1`。`--action-horizon` 在 dry-run 中不参与执行。

### 4.5 定义一次执行函数，第一次只跑一个 policy action

下面的函数只需要在当前 runner 终端定义一次。后面递增 horizon / calls 时只改两行数字，
不再复制整条长命令；如果新开了终端，就从第 4 节重新恢复变量并重新定义。

```bash
run_selected_scene_chunk() {
  : "${MAX_POLICY_CALLS:?set MAX_POLICY_CALLS before running}"
  : "${ACTION_HORIZON:?set ACTION_HORIZON before running}"
  : "${SO101_CHECKPOINT_JOINT_UNITS:?redo the terminal 2 setup}"
  : "${ISAAC_TASK:?copy one complete scene preset}"
  : "${ISAAC_SCENE_PROFILE:?copy one complete scene preset}"
  : "${TARGET_OBJECT_KEY:?copy one complete scene preset}"
  : "${TASK_INSTRUCTION:?copy one complete scene preset}"
  : "${CAMERA_LAYOUT:?redo the terminal 2 setup}"
  : "${ISAAC_FRONT_CAMERA_KEY:?redo the terminal 2 setup}"
  : "${ISAAC_LEFT_CAMERA_KEY:?redo the terminal 2 setup}"
  : "${ISAAC_WRIST_CAMERA_KEY:?redo the terminal 2 setup}"
  : "${BRIDGE_HOST:?redo the terminal 2 setup}"
  : "${BRIDGE_PORT:?redo the terminal 2 setup}"
  python experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/run_smart_task_closed_loop.py \
    --deployment-mode so101-finetuned \
    --so101-checkpoint-joint-units "$SO101_CHECKPOINT_JOINT_UNITS" \
    --task "$ISAAC_TASK" \
    --scene-profile "$ISAAC_SCENE_PROFILE" \
    --target-object-key "$TARGET_OBJECT_KEY" \
    --instruction "$TASK_INSTRUCTION" \
    --camera-layout "$CAMERA_LAYOUT" \
    --isaac-front-camera-key "$ISAAC_FRONT_CAMERA_KEY" \
    --isaac-left-camera-key "$ISAAC_LEFT_CAMERA_KEY" \
    --isaac-wrist-camera-key "$ISAAC_WRIST_CAMERA_KEY" \
    --robot so101 \
    --control-mode joint \
    --bridge-host "$BRIDGE_HOST" \
    --bridge-port "$BRIDGE_PORT" \
    --no-headless \
    --max-policy-calls "$MAX_POLICY_CALLS" \
    --action-horizon "$ACTION_HORIZON" \
    --policy-action-hz 30 \
    --so101-arm-target-scale 0.3 \
    --so101-max-arm-step-rad 0.05
}
```

现在只执行一档：

```bash
export MAX_POLICY_CALLS=1
export ACTION_HORIZON=1
run_selected_scene_chunk
```

这条命令最多执行：

```text
1 个 chunk 请求 × 每个 chunk 1 个 policy action = 1 个 policy action
```

成功标记：

```text
[runner] request action 1/1
[runner] executing 1 action steps from this chunk
[runner] first LeIsaac command (...)
[runner] after policy call 1 metrics ...
```

先看运动方向和日志里的单位转换。方向错误、joint limit mismatch、单位错误、动作突然跳变，
或者目标物 / camera / instruction 与 checkpoint 不一致时，立即停止，不要增加 horizon。

### 4.6 按 2、4、完整 chunk、更多 calls 逐档增加

下面每个代码块单独执行；不要一次性全贴。每一档结束后先看画面和日志，确认正确才执行下一块。

先执行一个 chunk 的前 2 步：

```bash
export MAX_POLICY_CALLS=1
export ACTION_HORIZON=2
run_selected_scene_chunk
```

确认 2 步正确后，执行前 4 步：

```bash
export MAX_POLICY_CALLS=1
export ACTION_HORIZON=4
run_selected_scene_chunk
```

确认 4 步正确后，完整执行一个返回 chunk：

```bash
export MAX_POLICY_CALLS=1
export ACTION_HORIZON=0
run_selected_scene_chunk
```

这里 `ACTION_HORIZON=0` 表示执行 bridge 实际返回的全部 action，不是执行 0 步。runner 会打印
真实长度，例如：

```text
[runner] executing 16 action steps from this chunk
```

当前 SO101 modality 通常返回 16-step chunk，但以日志里的实际数字为准。

完整单个 chunk 正确后，才增加请求次数：

```bash
export MAX_POLICY_CALLS=2
export ACTION_HORIZON=0
run_selected_scene_chunk
```

```bash
export MAX_POLICY_CALLS=4
export ACTION_HORIZON=0
run_selected_scene_chunk
```

每一档应看到与变量一致的 `request action X/Y`、实际
`executing N action steps from this chunk` 和 `after policy call X metrics`。方向、单位、
目标、幅度或 camera/instruction 任一项不对，立即停止，不要执行下一档。

组合关系如下：

| 参数 | 最多执行什么 |
|---|---|
| `--max-policy-calls 1 --action-horizon 1` | 1 个 policy action |
| `--max-policy-calls 1 --action-horizon 4` | 一个 chunk 的前 4 步 |
| `--max-policy-calls 1 --action-horizon 0` | 完整执行 1 个返回 chunk |
| `--max-policy-calls 4 --action-horizon 1` | 最多 4 个 policy action，每个来自一次新请求 |
| `--max-policy-calls 4 --action-horizon 0` | 最多完整执行 4 个 chunk，不是只执行 4 步 |

安全递增顺序：

```text
calls=1,horizon=1
-> calls=1,horizon=2
-> calls=1,horizon=4
-> calls=1,horizon=0
-> calls=2,horizon=0
-> calls=4,horizon=0
```

一次只扩大一个维度；只有上一档运动方向、目标、单位和幅度都正确，才进入下一档。

## 5. SO101 微调的三种相机布局

bridge 和 runner 的 `--camera-layout` 必须与训练 checkpoint 一致。终端 1 和终端 2 是两个
独立 shell，所以两个终端都要设置相同的 `CAMERA_LAYOUT`。

下表的 wrist-only / dual / triple 只适用于 `so101-finetuned`。`zero-shot-oxe` 固定使用
`dual`，传 wrist-only 或 triple 会在启动参数检查阶段直接报错。

| layout | bridge | runner live mapping |
|---|---|---|
| `wrist-only` | `--camera-layout wrist-only` | 例：`--isaac-wrist-camera-key camera1` |
| `dual` | `--camera-layout dual` | 例：`--isaac-front-camera-key camera3 --isaac-wrist-camera-key camera1` |
| `triple` | `--camera-layout triple` | 在 dual 示例上增加 `--isaac-left-camera-key camera2` |

prepared 数据里的历史 key 与 live key 不是同一概念：

```text
训练语义：video.top / video.left / video.wrist
live key：由当前 LeIsaac env cfg 决定，通过三个完整的 --isaac-...-camera-key 参数显式映射
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
| `--max-policy-calls` | 整个 run 最多请求多少个 action chunk；不是总 action 步数 |
| `--action-horizon` | 每个 chunk 最多执行多少个 policy action；`0` 表示完整返回 chunk，不是 0 步 |
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
- `tray-red24` 与 `table-red24` 都只有一个红色 2x4，并共享 single-pick 的完整位姿；两者
  只在木盘是否启用上不同。
- `multi-lego-tray` 保留独立的三块 LEGO authored 坐标，并把 env cfg 中的 success 判据
  重绑为 `object_placed_on_tray`；所以它不只是一个纯视觉 scene 开关。runner 默认仍禁用
  success termination，除非显式传 `--use-env-success-termination`。

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
    --camera-layout dual \
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

这些路线都必须使用 `--camera-layout dual`，也不能使用
`--so101-checkpoint-joint-units degrees`。它们是 embodiment 对照，不是 SO101 微调部署主线。

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
