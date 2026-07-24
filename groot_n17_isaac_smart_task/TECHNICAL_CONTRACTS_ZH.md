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

当前 `leisaac/main` 实际只暴露 wrist=`camera1`。部署 README 采用“保留这条现有 wrist，
再恢复其余相机”的最小改动示例：

```text
camera3 -> top
camera2 -> left
camera1 -> wrist
```

这不是相机编号的固定语义。runner 的历史 `leisaac-current` preset 使用过
camera3=top、camera1=left、camera2=wrist；部署命令通过三个 `--isaac-...-camera-key`
显式参数覆盖 preset，因此应以 env cfg 实际暴露的 key 和 runner 启动日志为准。

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

### 4.1 SO101 初始姿态与首帧时序

SO101 微调部署的首个 proprioception 和腕部图像必须来自同一个、与 checkpoint 数据来源
匹配的 reset pose。runner 的 `--so101-initial-pose auto` 只在
`so101-finetuned + so101 + joint` 路线解析为 `dataset-start`：

```text
lerobot_motor_units -> sim 数据 50 集 frame_index=5 的逐维中位数
degrees              -> real 数据 50 集 frame_index=0 的逐维中位数
其他路线             -> LeIsaac authored USD default
```

`dataset-first-frame` 仅保留为旧命令的兼容别名，也会解析成 `dataset-start`。sim 的 frame 0
是复位瞬态；当前 frame 5 参考比它晚约 `0.17 s`，腕部更朝向桌面但仍早于约 frame 10 的
稳定姿态。选择其他帧时必须使用 `--so101-initial-dataset-state` 显式记录六维值。

实现时序固定为：

```text
解析 checkpoint units 和 initial-pose
-> dataset state 转 Isaac radians，并拒绝越过 USD limits
-> 将 arm/gripper JointPositionActionCfg.use_default_offset 显式设为 False
-> 只修改 parse_env_cfg() 返回的当前 env_cfg.scene.robot.init_state.joint_pos
-> 设置 rerender_on_reset=True
-> gym.make()
-> 确认两个实例化 action term 的 runtime offset 都是 0
-> env.reset()
-> 校验实际 joint_pos 与 preset 的最大误差不超过 0.0001 rad
-> 读取同一姿态下的新 camera observation
-> 填充 FrameHistory
-> bridge inference
-> policy action loop
```

IsaacLab 的 `JointPositionActionCfg.use_default_offset` 默认为 `True`；它会把
`asset.data.default_joint_pos` 加到 raw action。由于本 runner 已将模型输出转换成绝对
radians，非零 reset pose 不能同时充当 action offset，否则每一步会得到
`absolute_target + reset_pose`。arm 和 gripper 两项都必须关闭该默认行为，并在 env 创建后
fail closed 验证。

不能在 `env.reset()` 后只调用 `set_joint_position_target()`：它只设置 action/actuator target，
不会立即把 articulation state、FK 和腕部 camera 一起重建。也不能用全零 action 做
“settle”，因为当前 `JointPositionAction` 把 action 当绝对 radians，会将 preset 拉回零姿态。
初始姿态/reset/rerender 不计入 `action_horizon`、`max_policy_calls` 或 viewport capture。

内置值是带明确 frame provenance 的 dataset-start 中位数，不是全数据中位数。若换用另一批
real/sim 数据，必须通过
`--so101-initial-dataset-state` 显式覆盖或更新有 provenance 的 preset；不能假定所有
`degrees` checkpoint 共用一个 home pose。gripper 在任何模式下仍按 LeRobot `[0,100]`
解释。

## 5. action chunk 和时间约定

GR00T action 数组统一为 `(B, T, D)`。`action_chunk.py` 负责：

- 所有已知 action key 的 `T` 必须一致；
- 时间维不能为空；
- 数值必须有限；
- runner 每次只切出一个 `(B, 1, D)` policy action。

两个容易混淆的参数：

```text
action_horizon   一个返回 chunk 最多执行多少个 policy action；0 = 完整返回 chunk
max_policy_calls 整个 run 最多向 bridge 请求多少个 chunk；不是总 action 步数
```

例如 `max_policy_calls=4, action_horizon=0` 表示最多完整执行 4 个 chunk，不是执行 4 步。
dry-run 仍按 `max_policy_calls` 请求 chunk，但不执行 action，因此 action horizon 不参与执行。

policy 时间基准与 Isaac step 分开。当前数据 30 Hz、env 60 Hz，所以同一个 policy target
保持两个 `env.step()`。`action_timing.py` 只接受整数倍；非整数比例会报错，而不是静默取整。

`max_arm_step_rad` 是每个 policy action 的位移保护，不是机器人硬件速度规格。正式阈值应由
验证过的安全速度和 `policy_action_hz` 推导。

## 6. 路线边界

- `so101-finetuned + joint + lerobot_motor_units`：sim 数据 checkpoint，执行绝对 motor target。
- `so101-finetuned + joint + degrees`：真机数据 checkpoint，执行绝对 arm degree target；gripper仍为 `[0,100]`。
- `zero-shot-oxe`：embodiment 对照，只允许 dual camera layout 和 motor-unit 选项；不代表存在
  通用高维 action 到 SO101 的 adapter。
- `eef`：把相对 EEF-frame `xyz + rot6d` 与当前末端位姿组合，再交给 IK action cfg。
- Franka：用于更接近 OXE/DROID embodiment 的对照，不等于 SO101 checkpoint 部署。

关于 LeIsaac 为什么没有通用 embodiment adapter 的检索证据，见
[LEISAAC_ACTION_ADAPTER_AUDIT_ZH.md](zero_shot_isaac_smart_task/LEISAAC_ACTION_ADAPTER_AUDIT_ZH.md)；
它是审计记录，不是必读流程。

## 7. SmartTask 场景契约

部署时五个概念必须分开：

```text
--task               选择机器人/action/observation/camera 等基础 Gym env
--scene-profile      选择木盘和 LEGO 布局；multi 还会重绑 env cfg 的 success 判据
scene wrapper / USD  把基础 scene 与完整 repo-local 资产组合成 live scene
--instruction        checkpoint 的语言条件，必须与训练文本一致
--target-object-key  multi 场景选择哪个刚体；同时用于 metric/debug/termination
```

`tray-red24`、`table-red24` 和 `multi-lego-tray` 都在 `experiments/` 内实现，不依赖额外
task ID，也不修改 `leisaac/`。显式 profile 会停用旧 LEGO composition arc，加载完整红 2x4 /
红 2x2 USD；蓝 2x4 复用红 2x4 几何并覆盖为蓝色材质。`multi-lego-tray` 必须显式指定 target
和 instruction，物体 pose 沿用 LeIsaac authored layout。

`tray-red24` 和 `table-red24` 都只有红色 2x4，并共享 single-pick 的完整 pose；前者启用木盘，
后者停用木盘。`multi-lego-tray` 使用另一套三块 LEGO pose，并将 env cfg 的 success 判据
改为 `object_placed_on_tray`。runner 默认禁用原生 success termination，因为当前判据过松；
只有显式传 `--use-env-success-termination` 才会让该判据参与 episode 自动终止。

`task-default` 才保留旧 scene / cuboid fallback 兼容语义；不要把 `--smart-target-*`
与显式 scene profile 混用。

## 8. 修改完成的最低检查

```text
训练 config 与 camera layout 对齐
bridge ping 报告 modality + action_decoding
runner task / scene profile / target / instruction 与 checkpoint 对齐
runner units 与 checkpoint 的训练数据来源一致
runner arm/gripper use_default_offset=False，runtime offsets 全零
runner initial-pose source / units 正确，reset verification 不超过 0.0001 rad
runner camera-only 与 scene rigid-object 列表通过
纯 Python tests 通过
dry-run 通过
最后才执行 one policy action
```

新增 action schema、时间重采样或机器人资产时，应先把规则写进独立小模块和测试，再接回
runner；不要把新分支直接堆进 `main()`。
