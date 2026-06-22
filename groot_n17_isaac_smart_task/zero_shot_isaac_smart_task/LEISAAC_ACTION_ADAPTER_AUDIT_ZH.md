# LeIsaac Action Adapter 审计结论

本文档记录对 `leisaac/` 目录的代码检索结论，问题是：

> LeIsaac 是否已经提供了把其他模型的更高维 action 输出，自动降维/投影/适配到 SO101/soarm 6D action 的 API？

结论是：没有发现这样的通用 API。

更准确地说，LeIsaac 当前已有的是：

1. SO101/LeRobot action 与 IsaacLab SO101 joint action 之间的数值换算。
2. 不同 teleop device 对应的 action manager 配置。
3. policy inference 脚本中对 GR00T N1.5/N1.6、LeRobot、OpenPI 的服务端调用封装。

但这些机制都默认模型输出已经符合 SO101/LeRobot 的 action schema。它们不是把 Franka/DROID/GR00T 高维输出自动投影到 SO101 结构的 embodiment adapter。

## 1. Policy client 默认吃的是 SO101/LeRobot 6D action

文件：

```text
leisaac/source/leisaac/leisaac/policy/service_policy_clients.py
```

GR00T N1.5 client 的核心逻辑是：

```python
concat_action = np.concatenate(
    [action_chunk["action.single_arm"], action_chunk["action.gripper"]],
    axis=1,
)
concat_action = convert_lerobot_action_to_leisaac(concat_action)
return torch.from_numpy(concat_action[:, None, :])
```

这里的语义是：

```text
action.single_arm: 5D
action.gripper:    1D
concat_action:     6D
```

也就是说，policy server 返回的 action 本身就应该是 SO101/LeRobot 风格的 `5 arm joints + 1 gripper`。

GR00T N1.6 client 也是同样思路：

```python
concat_action = np.concatenate(
    [action_chunk["single_arm"], action_chunk["gripper"]],
    axis=-1,
)
concat_action = concat_action.squeeze(0)
concat_action = convert_lerobot_action_to_leisaac(concat_action)
```

LeRobot async client 也是：

```python
action_list = [action.get_action()[None, :] for action in action_chunk]
concat_action = torch.cat(action_list, dim=0)
concat_action = convert_lerobot_action_to_leisaac(concat_action)
```

OpenPI client 也是：

```python
action_chunk = self.infer(obs_dict)["actions"]
processed_action = convert_lerobot_action_to_leisaac(action_chunk)
```

这些 client 都没有做 “7D/8D/9D 高维模型输出 -> SO101 6D” 的降维或逆运动学映射。

## 2. `convert_lerobot_action_to_leisaac()` 只是数值范围换算

文件：

```text
leisaac/source/leisaac/leisaac/utils/robot_utils.py
```

相关函数：

```python
def convert_lerobot_action_to_leisaac(action: torch.Tensor | np.ndarray) -> np.ndarray:
    ...
    for idx, joint_name in enumerate(joint_limits):
        motor_limit_range = motor_limits[joint_name]
        joint_limit_range = joint_limits[joint_name]
        motor_range = motor_limit_range[1] - motor_limit_range[0]
        joint_range = joint_limit_range[1] - joint_limit_range[0]
        motor_degree = action[:, idx] - motor_limit_range[0]
        processed_degree = motor_degree / motor_range * joint_range + joint_limit_range[0]
        processed_radius = processed_degree / 180.0 * torch.pi
        processed_action[:, idx] = processed_radius
```

这个函数做的是：

```text
LeRobot/SO101 motor range
-> SO101 USD joint limit range
-> IsaacLab 使用的 radians
```

它不是运动学 adapter，也不是 embodiment adapter。

它不会理解：

```text
Franka 7 joints
DROID relative EEF action
EEF pose + gripper
GR00T N1.7 OXE/DROID action
```

更不会把这些高维 action 自动变成：

```text
SO101 shoulder_pan
SO101 shoulder_lift
SO101 elbow_flex
SO101 wrist_flex
SO101 wrist_roll
SO101 gripper
```

## 3. SO101 的 action 配置是 teleop/action manager，不是 policy 降维 adapter

文件：

```text
leisaac/source/leisaac/leisaac/devices/action_process.py
```

这里根据 `teleop_device` 初始化不同 action manager：

```python
if device in ["so101leader", "lekiwi-leader"]:
    action_cfg.arm_action = mdp.JointPositionActionCfg(...)
    action_cfg.gripper_action = mdp.JointPositionActionCfg(...)
```

这一路是 SO101 leader 的 joint-position 控制，总维度是：

```text
5 arm joints + 1 gripper = 6D
```

文件里也有 keyboard/gamepad/state_machine 的 IK 配置：

```python
elif device in ["keyboard", "gamepad", "lekiwi-keyboard", "lekiwi-gamepad"]:
    action_cfg.arm_action = mdp.DifferentialInverseKinematicsActionCfg(...)
    action_cfg.gripper_action = mdp.RelativeJointPositionActionCfg(...)
```

以及：

```python
elif device in ["so101_state_machine"]:
    action_cfg.arm_action = mdp.DifferentialInverseKinematicsActionCfg(...)
    action_cfg.gripper_action = mdp.BinaryJointPositionActionCfg(...)
```

但这些是给 teleop 或 state machine 使用的 action manager 选择。它们没有被 policy client 用作 “模型高维输出到 SO101 的自动降维接口”。

## 4. `policy_inference.py` 不做 action schema 适配

文件：

```text
leisaac/scripts/evaluation/policy_inference.py
```

核心执行逻辑是：

```python
actions = policy.get_action(obs_dict).to(env.device)
for i in range(min(args_cli.policy_action_horizon, actions.shape[0])):
    action = actions[i, :, :]
    obs_dict, _, reset_terminated, reset_time_outs, _ = env.step(action)
```

这里 `policy.get_action()` 返回什么 action tensor，`env.step(action)` 就执行什么 action tensor。

脚本只按 action horizon 逐步执行 chunk，不会把 action 维度重新解释为另一个 embodiment。

也就是说，如果 policy client 返回的是 6D SO101 action，它就执行 6D SO101 action；如果要执行 Franka/DROID/EEF action，需要另写对应的 action cfg 和桥接逻辑。

## 5. 离线 dataset 转换不是运行时 adapter

文件：

```text
leisaac/scripts/convert/lerobot2isaaclab.py
```

这里有：

```python
def denormalize_lerobot_to_isaaclab_radians(joint_values_lerobot: np.ndarray) -> np.ndarray:
    ...
    dimension = joint_values_lerobot.shape[1]
    if dimension < 6:
        raise ValueError(f"Expected D>=6, got D={dimension}")
    ...
    if dimension == 12:
        # bimanual
        ...
    else:
        # single arm
        for joint_index in range(6):
            ...
```

这段用于把 LeRobot dataset 里的 action 列转换成 IsaacLab 可 replay 的 HDF5 数据。

它最多处理：

```text
6D single-arm SO101
12D bi-arm SO101
```

并且主要做数值范围和单位转换。它不是 policy inference 运行时调用的高维 action adapter。

## 6. 对当前 GR00T N1.7 实验的含义

这说明，LeIsaac 原生 policy deployment 路线更接近：

```text
模型已经针对 SO101/LeRobot schema 微调
-> 输出 SO101 6D action
-> LeIsaac 做数值换算
-> env.step()
```

而不是：

```text
模型输出 Franka/DROID/OXE 风格 action
-> LeIsaac 自动降维/投影/重定向到 SO101
-> env.step()
```

因此，如果直接把 GR00T N1.7 的 OXE/DROID-style 输出塞进 SO101 任务，中间并没有 LeIsaac 官方提供的成熟 embodiment adapter 来保证语义对齐。

这也是为什么我们当前做了两条路线：

1. SO101 路线：可以验证 LeIsaac/LeRobot/SO101 生态里的 6D action 链路，但对 GR00T N1.7 原生能力有较大 embodiment mismatch。
2. Franka 路线：绕过 SO101 形态限制，用更接近 GR00T OXE/DROID 预训练 embodiment 的 Panda/EEF/IK 路线测试 zero-shot 能力。

## 7. 简短结论

LeIsaac 目前没有发现“高维模型 action 输出自动降维适配 SO101”的通用 API。

已有的转换函数主要是：

```text
SO101 LeRobot 数值范围 <-> SO101 IsaacLab joint radians
```

而不是：

```text
Franka/DROID/GR00T high-dimensional action -> SO101 embodiment action
```

所以，如果要让 GR00T N1.7 的原生 OXE/DROID 输出合理控制 SO101，需要自己设计 adapter；但这个 adapter 会成为实验中的额外变量，影响 zero-shot 失败时的解释清晰度。
