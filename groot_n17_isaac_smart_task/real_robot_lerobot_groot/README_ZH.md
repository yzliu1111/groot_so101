# 使用 LeRobot 在 SO101 真机部署 GR00T N1.7

本目录直接使用 LeRobot 的：

- `SO101Follower` 读取校准后的 6D 电机位置和腕部相机；
- GR00T N1.7 policy / processor 加载原始 NVIDIA checkpoint；
- `SO101Follower.send_action()` 下发绝对电机位置目标。

当前入口只支持 wrist-only checkpoint。默认是 **dry-run**：会连接真机、读取相机和
电机、执行模型推理，但不会发送电机目标。

## 最快验证卡

目标不是第一轮就做完整评测，而是用最短路径回答两个问题：

1. LeRobot 能否在目标机读取 SO101 + wrist camera，并让 checkpoint 返回合法 action？
2. 这个 action 能否在真机上按正确方向执行，并完成一次已训练任务？

### 终端 0：一次性变量

下面只需要替换 `SMART_PROJECT`、`ROBOT_PORT`、`ROBOT_ID` 和 `WRIST_CAMERA`：

```bash
export SMART_PROJECT=/home/guest1/smart_project
export PYTHON=/home/guest1/miniforge3/envs/lerobot/bin/python
export LEROBOT_BIN=/home/guest1/miniforge3/envs/lerobot/bin

export CHECKPOINT="$SMART_PROJECT/outputs/aws-0715/groot_so101_synthetic_finetune/aws_real_run_001_wrist_pick/aws_real_so101_run_001_wrist_pick/checkpoint-10000"
export DEPLOY="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/real_robot_lerobot_groot/run_so101_groot_real.py"
export RUN_DIR="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/real_robot_lerobot_groot/runs/$(date +%Y%m%d_%H%M%S)"

export ROBOT_PORT=/dev/ttyACM0
export ROBOT_ID=so101_follower_arm
export WRIST_CAMERA=/dev/v4l/by-id/REPLACE_WITH_REAL_CAMERA

mkdir -p "$RUN_DIR"
test -d "$CHECKPOINT" && test -f "$DEPLOY" && test -x "$PYTHON"
```

最后一条命令退出码为 `0` 才继续。

### 终端 1：60 秒内完成设备确认

```bash
"$LEROBOT_BIN/lerobot-find-port"
"$LEROBOT_BIN/lerobot-find-cameras" opencv
```

把结果填回终端 0 的 `ROBOT_PORT` 和 `WRIST_CAMERA`。优先使用稳定的
`/dev/v4l/by-id/...`，不要依赖重启后可能变化的 `/dev/video0` 编号。

### 第一次：dry-run，不动机器人

```bash
"$PYTHON" "$DEPLOY" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-arm-units degrees \
  --robot-port "$ROBOT_PORT" \
  --robot-id "$ROBOT_ID" \
  --wrist-camera "$WRIST_CAMERA" \
  --instruction "pick up block" \
  --parameter-dtype fp32 \
  --max-policy-calls 1 \
  --trace-jsonl "$RUN_DIR/dry_run.jsonl"
```

PASS 标记：

```text
[real] policy loaded; hardware has not been connected yet
[real] mode=DRY-RUN
[real] policy call 1/1 chunk=(16, 6)
[real] dry-run first target=...
[real] robot disconnected
```

并检查：

```bash
wc -l "$RUN_DIR/dry_run.jsonl"
```

应为 `2` 行：一行 `run_config`，一行 `policy_call`。机器人不应运动。

### 第二次：单 chunk 真机执行

将机器人放回训练数据的起始姿态，清空机械臂工作范围，手放在急停位置：

```bash
"$PYTHON" "$DEPLOY" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-arm-units degrees \
  --robot-port "$ROBOT_PORT" \
  --robot-id "$ROBOT_ID" \
  --wrist-camera "$WRIST_CAMERA" \
  --instruction "pick up block" \
  --parameter-dtype fp32 \
  --max-policy-calls 1 \
  --action-horizon 16 \
  --max-arm-delta 5 \
  --max-gripper-delta 10 \
  --execute \
  --trace-jsonl "$RUN_DIR/execute_1chunk.jsonl"
```

输入 `EXECUTE` 后观察一个 chunk。关节方向、腕部画面语义或夹爪方向任一不对，立即
`Ctrl+C`，不要进入连续闭环。

### 第三次：直接得到完整验证结果

单 chunk 方向正确后，把同一条命令中的：

```text
--max-policy-calls 1
```

改为：

```text
--max-policy-calls 20
```

同时把 trace 改为：

```text
--trace-jsonl "$RUN_DIR/execute_20calls.jsonl"
```

这就是当前最快且仍保留必要安全门的正式验证路径：

```text
设备确认 -> 1-call dry-run -> 1 chunk execute -> 20 calls execute
```

## 详细说明

### 1. 真机预检

在目标机的 LeRobot 环境中确认设备：

```bash
lerobot-find-port
lerobot-find-cameras opencv
```

必须使用与采集数据时相同的 follower calibration ID。没有 calibration 文件时先运行：

```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=so101_follower_arm
```

### 2. Dry-run

real001 checkpoint 的前五个 arm 维度是 degrees，gripper 是 0–100：

```bash
CHECKPOINT=/path/to/checkpoint-10000

python real_robot_lerobot_groot/run_so101_groot_real.py \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-arm-units degrees \
  --robot-port /dev/ttyACM0 \
  --robot-id so101_follower_arm \
  --wrist-camera /dev/v4l/by-id/REPLACE_WITH_REAL_CAMERA \
  --instruction "pick up block" \
  --parameter-dtype fp32 \
  --max-policy-calls 1 \
  --trace-jsonl runs/real_so101_dry_run_001.jsonl
```

通过条件：

- checkpoint、机器人和 wrist camera 均成功连接；
- 输出 `(1, 16, 6)` 等价的 decoded action chunk；
- state 和 action 全部 finite；
- trace 中 `execute=false`；
- 机器人没有动作。

### 3. 第一次受控执行

先将机器人放在与训练 episode 起点接近的姿态，清空工作区，准备物理急停。第一次只执行
一个 chunk，并保留默认逐 tick 限幅：

```bash
python real_robot_lerobot_groot/run_so101_groot_real.py \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-arm-units degrees \
  --robot-port /dev/ttyACM0 \
  --robot-id so101_follower_arm \
  --wrist-camera /dev/v4l/by-id/REPLACE_WITH_REAL_CAMERA \
  --instruction "pick up block" \
  --parameter-dtype fp32 \
  --max-policy-calls 1 \
  --action-horizon 16 \
  --max-arm-delta 5 \
  --max-gripper-delta 10 \
  --execute \
  --trace-jsonl runs/real_so101_execute_001.jsonl
```

程序会要求手工输入 `EXECUTE`。`Ctrl+C` 会停止 rollout 并 disconnect；默认 disconnect
时关闭 follower torque。只有在已经验证外部急停和保持风险后，才考虑
`--keep-torque-on-disconnect`。

### 4. 扩展到连续闭环

单 chunk 的方向、相机语义、关节顺序和夹爪开合全部通过后，再逐步增加：

```text
--max-policy-calls 2
--max-policy-calls 5
--max-policy-calls 20
```

不要在第一次真机运行时直接跳到 20 calls，也不要为了追求轨迹速度先关闭逐步限幅。
