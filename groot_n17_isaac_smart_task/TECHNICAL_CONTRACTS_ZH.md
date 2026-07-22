# SmartTask 代码阅读约定

这是一份代码地图，不是操作手册。运行命令看各阶段 README；准备修改 bridge、runner、
相机布局或 action 语义时，先确认下面这些跨文件约定。

## 1. 推荐阅读顺序

1. [训练 modality config](full_finetune_so101/so101_synthetic_groot_config.py)
2. [GR00T bridge](zero_shot_isaac_smart_task/groot_bridge_server.py)
3. [Isaac runner](zero_shot_isaac_smart_task/run_smart_task_closed_loop.py) 的 `main()`
4. [action chunk](zero_shot_isaac_smart_task/action_chunk.py)、
   [时间对齐](zero_shot_isaac_smart_task/action_timing.py)、
   [SO101 单位](zero_shot_isaac_smart_task/so101_joint_units.py) 和
   [位姿数学](zero_shot_isaac_smart_task/pose_math.py)
5. [纯 Python 回归测试](zero_shot_isaac_smart_task/tests/)

先看 config，确定 checkpoint 的输入输出 schema；再沿 bridge 到 runner 看运行时数据流。
四个小模块是从 runner 拆出的可独立测试边界，不需要启动 SimulationApp。

## 2. 进程和环境边界

| 工作 | 进程 / 环境 | 不应导入 |
|---|---|---|
| v3 数据准备 | conda `lerobot` | Isaac / GR00T |
| stats、训练、bridge | `$GROOT_ROOT/.venv`，Python 3.12 | Isaac / LeIsaac |
| Isaac runner | conda `leisaac` 或已验证的 IsaacLab env | GR00T / LeRobot |

bridge 与 runner 通过 `wire.py` 的本地 TCP pickle 协议通信。默认只监听
`127.0.0.1:5577`；它不是面向不可信网络的服务协议。

## 3. 相机约定：角色稳定，live key 可以变化

checkpoint schema 使用语义角色：

```text
video.top / video.left / video.wrist
```

当前 SmartTask live observation 默认映射：

```text
camera3 -> top
camera1 -> left
camera2 -> wrist
```

prepared dataset 的 `camera1/camera2/camera3` 与 live key 不能按名字直接等同。新增 layout
时必须同时更新训练 config、bridge modality 校验、runner live mapping 和测试。bridge 与
runner 在执行动作前会比较 layout 和 modality，不一致就停止。

## 4. SO101 action 约定

| checkpoint 来源 | 前五维 arm state/action | 第六维 gripper |
|---|---|---|
| LeIsaac sim | LeRobot motor units | LeRobot `[0,100]` |
| 真机 `use_degrees=True` | degrees | LeRobot `[0,100]` |

bridge 只保证 `get_action()` 已 decode 回 checkpoint 数据集的绝对坐标；单位选择只存在于
runner 的 `--so101-checkpoint-joint-units {lerobot_motor_units,degrees}`。默认 motor 兼容旧
sim 命令；真机 checkpoint 必须显式传 `degrees`。runner 不会从 checkpoint/meta 自动推断。

关键点：processor 的 `use_relative_action` 可以表示内部训练变换，但不能据此把
`get_action()` 的返回值重新解释成 delta。部署侧禁止执行：

```python
current_radians + returned_action
```

`so101_joint_units.py` 的实际保护顺序是：

```text
拒绝错误 shape / NaN / infinity
-> 按所选 checkpoint 坐标 clip：arm motor limits 或 arm degree/USD limits；gripper始终 [0,100]
-> 绝对 dataset target 转 radians
-> arm_target_scale 从当前姿态向目标插值（只作用于前 5 个 arm joints）
-> max_arm_step_rad 限制一次 policy action 的 arm radian 变化
-> clip 到 Isaac runtime soft joint limits
```

runner 还会核对当前 USD 的 6 个 joint 名称、顺序和范围。更换 SO101 USD、关节顺序、
符号或范围时，必须同步修改 converter 和测试，不能绕过校验。

## 5. action chunk 和时间约定

GR00T action 数组统一为 `(B, T, D)`。`action_chunk.py` 负责：

- 所有已知 action key 的 `T` 必须一致；
- 时间维不能为空；
- 数值必须有限；
- runner 每次只切出一个 `(B, 1, D)` policy action。

两个容易混淆的参数：

```text
action_horizon   一个返回 chunk 最多执行多少个 policy action
max_policy_calls 整个 run 最多向 bridge 请求多少个 chunk
```

policy 时间基准与 Isaac step 分开。当前数据 30 Hz、env 60 Hz，所以同一个 policy target
保持两个 `env.step()`。`action_timing.py` 只接受整数倍；非整数比例会报错，而不是静默取整。

`max_arm_step_rad` 是每个 policy action 的位移保护，不是机器人硬件速度规格。正式阈值应由
验证过的安全速度和 `policy_action_hz` 推导。

## 6. 路线边界

- `so101-finetuned + joint + lerobot_motor_units`：sim 数据 checkpoint，执行绝对 motor target。
- `so101-finetuned + joint + degrees`：真机数据 checkpoint，执行绝对 arm degree target；gripper仍为 `[0,100]`。
- `zero-shot-oxe`：embodiment 对照，不代表存在通用高维 action 到 SO101 的 adapter。
- `eef`：把相对 EEF-frame `xyz + rot6d` 与当前末端位姿组合，再交给 IK action cfg。
- Franka：用于更接近 OXE/DROID embodiment 的对照，不等于 SO101 checkpoint 部署。

关于 LeIsaac 为什么没有通用 embodiment adapter 的检索证据，见
[LEISAAC_ACTION_ADAPTER_AUDIT_ZH.md](zero_shot_isaac_smart_task/LEISAAC_ACTION_ADAPTER_AUDIT_ZH.md)；
它是审计记录，不是必读流程。

## 7. SmartTask 场景契约

部署时五个概念必须分开：

```text
--task               选择机器人/action/observation/camera 等基础 Gym env
--scene-profile      选择部署前的木盘和 LEGO 布局
scene wrapper / USD  把基础 scene 与完整 repo-local 资产组合成 live scene
--instruction        checkpoint 的语言条件，必须与训练文本一致
--target-object-key  multi 场景选择哪个刚体；同时用于 metric/debug/termination
```

`tray-red24`、`table-red24` 和 `multi-lego-tray` 都在 `experiments/` 内实现，不依赖额外
task ID，也不修改 `leisaac/`。显式 profile 会停用旧 LEGO composition arc，加载完整红 2x4 /
红 2x2 USD；蓝 2x4 复用红 2x4 几何并覆盖为蓝色材质。`multi-lego-tray` 必须显式指定 target
和 instruction，物体 pose 沿用 LeIsaac authored layout。

`task-default` 才保留旧 scene / cuboid fallback 兼容语义；不要把 `--smart-target-*`
与显式 scene profile 混用。

## 8. 修改完成的最低检查

```text
训练 config 与 camera layout 对齐
bridge ping 报告 modality + action_decoding
runner task / scene profile / target / instruction 与 checkpoint 对齐
runner units 与 checkpoint 的训练数据来源一致
runner camera-only 与 scene rigid-object 列表通过
纯 Python tests 通过
dry-run 通过
最后才执行 one policy action
```

新增 action schema、时间重采样或机器人资产时，应先把规则写进独立小模块和测试，再接回
runner；不要把新分支直接堆进 `main()`。
