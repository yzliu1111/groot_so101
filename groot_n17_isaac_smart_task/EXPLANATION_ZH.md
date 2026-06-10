# GR00T N1.7 + LeIsaac SmartTask 实验说明

这份文档解释 `experiments/groot_n17_isaac_smart_task` 这个实验目录做了什么、为什么这么做、如何复用 LeIsaac 里已有的 Isaac 场景，以及 Isaac 侧和 GR00T 侧是如何桥接起来的。

## 1. 实验目标

你的同事项目里已经有一套 LeIsaac SmartTask：

- 场景资产在 `leisaac/assets/scenes/smart_scene`。
- SO101 follower 机器人 USD 在 `leisaac/assets/robots`。
- 任务注册名是 `LeIsaac-SO101-SmartTask-v0`。
- 数据采集链路曾经使用真实 SOARM/SO101 demo arm 联动 Isaac 中的 SO101 follower，再把图像和机器人状态整理成 HDF5 / LeRobot 数据。

但这次实验的目标不是继续验证 LeRobot / SmolVLA 路线，而是验证：

> 在尽量绕开 LeRobot policy stack 的情况下，GR00T N1.7 base model 能不能在这个 Isaac 场景里做 zero-shot pick。

换句话说，我们想回答的问题是：

- GR00T N1.7 在没有针对 SO101 和这个乐高任务微调的情况下，会不会输出有意义动作？
- 如果动作失败，失败是模型 zero-shot 能力边界，还是 LeRobot/LeIsaac 中间层造成的摩擦？
- 如果绕开 LeRobot policy stack，直接把 Isaac observation 给 GR00T，再把 GR00T action 落到 Isaac 机器人上，结果会不会更真实地反映 N1.7 的潜力？

## 2. 为什么没有直接改 LeIsaac policy client

LeIsaac 的历史定位是把 Isaac 和 LeRobot 桥接起来：

1. 用 USD 资产在 Isaac 中加载 SO101 机器人和 SmartTask 场景。
2. 连接真实 SOARM/SO101 demo arm，让真实机械臂动作联动仿真中的 follower。
3. 通过 teleop 演示完成 pick。
4. 采集 Isaac 中的相机图像和机器人 state。
5. 转换成 LeRobot 数据集格式，用于 LeRobot 生态里的模型训练/部署。

这条链路对采集数据和使用 LeRobot 模型非常有价值，但对 GR00T N1.7 zero-shot 验证有两个问题：

- 当前 LeRobot / LeIsaac 代码里没有原生、成熟的 N1.7 部署路径。
- N1.7 的官方模型更接近 DROID/Franka/通用 EEF action schema，而不是 SO101 原生 6D joint schema。

如果强行把 N1.7 塞进 LeRobot policy client，失败时很难判断：

- 是 N1.7 zero-shot 本身不行？
- 是 SO101 embodiment 不匹配？
- 是 LeRobot/LeIsaac adapter 映射错了？
- 是 action schema 被错误压缩了？

所以这个实验目录采用了更干净的结构：

- 复用 LeIsaac 已经搭好的 Isaac scene、robot、task。
- 不改 LeIsaac 主工程的 policy client。
- 不把 GR00T 依赖装进 IsaacLab 环境。
- 不把 IsaacLab 依赖装进 GR00T venv。
- 用一个很小的本地 socket bridge 连接两个隔离环境。

## 3. 目录结构

```text
experiments/groot_n17_isaac_smart_task/
├── README.md
├── EXPLANATION_ZH.md
├── wire.py
├── groot_bridge_server.py
└── run_smart_task_closed_loop.py
```

核心文件含义：

- `wire.py`：Isaac 进程和 GR00T 进程之间的通信协议。负责 socket、msgpack、numpy array 序列化。
- `groot_bridge_server.py`：运行在 GR00T venv 中，加载 `nvidia/GR00T-N1.7-3B`，对 Isaac 侧提供 `ping/get_action/reset/shutdown` 接口。
- `run_smart_task_closed_loop.py`：运行在 `conda isaaclab` 中，启动 LeIsaac SmartTask，构造 GR00T observation，接收 action，并驱动 Isaac 中的 SO101。
- `README.md`：实际运行命令。
- `EXPLANATION_ZH.md`：当前这份中文说明材料。

## 4. 两个 Python 环境如何隔离

本实验明确分成两个进程：

### 4.1 GR00T 进程

运行命令：

```bash
cd /home/yzliu/smart_project
/home/yzliu/Isaac-GR00T/.venv/bin/python \
  experiments/groot_n17_isaac_smart_task/groot_bridge_server.py \
  --model-path nvidia/GR00T-N1.7-3B \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --device cuda
```

这个进程只负责：

- import GR00T。
- 加载 GR00T N1.7 模型和 processor。
- 接收 observation。
- 调用 `policy.get_action()`。
- 返回 action dict。

它不 import IsaacLab，也不创建 Isaac scene。

### 4.2 Isaac 进程

运行命令示例：

```bash
cd /home/yzliu/smart_project
conda run -n isaaclab env \
  PYTHONPATH=/home/yzliu/smart_project/leisaac/source/leisaac \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  python experiments/groot_n17_isaac_smart_task/run_smart_task_closed_loop.py \
    --control-mode eef \
    --no-headless \
    --render-sleep-s 0.08 \
    --keep-open-s 60 \
    --max-policy-calls 1 \
    --instruction "Pick up the red 2x4 lego brick."
```

这个进程只负责：

- import IsaacLab / LeIsaac。
- 创建 `LeIsaac-SO101-SmartTask-v0`。
- 从 Isaac observation 中读取图像、关节状态、末端位姿。
- 通过 socket 向 GR00T bridge 请求 action。
- 把 action 转成 LeIsaac 可执行命令。
- `env.step(command)` 驱动仿真机器人。

它不 import GR00T。

## 5. 如何复用 LeIsaac 已有场景

Isaac 侧代码没有手动打开 USD 文件，而是复用 LeIsaac 已经注册好的 task。

关键代码在 `run_smart_task_closed_loop.py`：

```python
import leisaac
env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env = gym.make(args.task, cfg=env_cfg).unwrapped
```

这里发生了几件事：

1. `import leisaac` 会触发 LeIsaac package 的任务注册副作用。
2. `args.task` 默认是 `LeIsaac-SO101-SmartTask-v0`。
3. `parse_env_cfg()` 读取这个 task 对应的 IsaacLab env config。
4. `gym.make()` 根据这个 config 创建 IsaacLab env。

因此这个实验复用了 LeIsaac 中已有的：

- SmartTask scene config。
- SO101 follower robot config。
- camera observation config。
- red 2x4 lego brick object。
- end-effector frame transformer。
- action manager 配置机制。

这也是为什么实验目录没有复制 USD 文件。USD 资产仍然来自 LeIsaac 主工程：

- `leisaac/assets/scenes/smart_scene/scene.usd`
- `leisaac/assets/robots/so101_follower.usd`

## 6. Observation 如何从 LeIsaac 转成 GR00T

GR00T N1.7 base model 使用的是 OXE/DROID embodiment：

```text
OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
```

bridge 启动后会打印 modality，大致是：

```text
video:
  exterior_image_1_left
  wrist_image_left

state:
  eef_9d
  gripper_position
  joint_position

action:
  eef_9d
  gripper_position
  joint_position

language:
  annotation.language.language_instruction
```

Isaac 侧的 `build_oxe_observation()` 把 LeIsaac observation 映射成这个 schema。

### 6.1 图像映射

当前使用两路图像：

```text
LeIsaac camera1 -> GR00T video.exterior_image_1_left
LeIsaac camera3 -> GR00T video.wrist_image_left
```

GR00T OXE/DROID config 的 video delta indices 是 `[-15, 0]`，意思是模型希望看到一个历史帧和一个当前帧。实验里用 `FrameHistory` 保存相机历史，然后取最旧帧和最新帧，组成：

```text
shape = (1, 2, H, W, 3)
```

含义：

- `1`：batch size。
- `2`：两帧图像。
- `H, W, 3`：RGB 图像。

### 6.2 EEF 状态映射

LeIsaac 给的末端状态是：

```text
ee_frame_state = [x, y, z, qw, qx, qy, qz]
```

GR00T 需要的是：

```text
eef_9d = [x, y, z, rot6d...]
```

所以代码会把四元数 `quat(wxyz)` 转成旋转矩阵前两行 flatten 后的 `rot6d`。

### 6.3 Joint 状态映射

SO101 的 joint state 是 6 维：

```text
[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
```

OXE/DROID 的 `joint_position` 是 7 维。实验里做了一个非常粗的 padding：

```text
SO101 6D -> DROID 7D
[j0, j1, j2, j3, j4, j5, 0]
```

这只是为了让 schema 对上，不代表语义完全正确。

### 6.4 Gripper 状态映射

SO101 第 6 个 joint 是 gripper，所以：

```text
state.gripper_position = joint_pos[5]
```

shape 写成 `(1, 1, 1)`，符合 GR00T 的 batch/time/dim 约定。

### 6.5 Language 映射

语言指令传成：

```python
{
  "annotation.language.language_instruction": [[instruction]]
}
```

这里的双层 list 表示：

- batch size = 1。
- language timestep = 1。

## 7. GR00T bridge 如何工作

`groot_bridge_server.py` 中的核心是：

```python
policy = Gr00tPolicy(
    embodiment_tag=embodiment_tag,
    model_path=args.model_path,
    device=args.device,
    strict=not args.no_strict,
)
```

这个 policy 会：

1. 加载 GR00T 模型权重。
2. 加载 processor。
3. 校验输入 observation 是否符合 modality schema。
4. 把 observation 编码成模型输入。
5. 调用模型生成 action。
6. decode action 到物理数值空间。

Isaac 侧每次请求：

```python
reply = request(
    args.bridge_host,
    args.bridge_port,
    {"endpoint": "get_action", "observation": groot_obs},
    args.timeout_s,
)
```

bridge 侧收到后执行：

```python
action, info = policy.get_action(request["observation"], request.get("options"))
```

然后把 action 返回 Isaac 侧。

## 8. 通信协议为什么用 socket + msgpack

实验没有用 ROS、ZMQ、HTTP 或 gRPC，而是用了最小的 TCP socket + msgpack。

原因是：

- 不给任何环境安装新依赖。
- 不引入额外服务框架。
- 只需要本机进程间通信。
- observation/action 主要是 numpy array，msgpack 加一点自定义 ndarray 编码就够用。

`wire.py` 中的协议是：

```text
8 字节 header：payload 长度
N 字节 payload：msgpack 编码后的 dict
```

其中 numpy array 会被编码成：

```python
{
  "__ndarray__": True,
  "dtype": "...",
  "shape": (...),
  "data": b"..."
}
```

接收端再用 dtype、shape、data 还原成 numpy array。

## 9. Action 如何从 GR00T 转回 LeIsaac

这是这个实验最关键的部分。目前支持两条路线。

### 9.1 Joint 路线

运行参数：

```bash
--control-mode joint
```

LeIsaac action mode：

```text
so101leader
```

GR00T 输出：

```text
action.joint_position: shape = (1, 40, 7)
```

实验处理：

```text
取前 6 维 -> LeIsaac SO101 6D joint command
```

问题：

- GR00T 的 joint_position 是 DROID/Franka 类机器人语义。
- SO101 是完全不同的 5DoF arm + gripper。
- 直接取前 6 维会非常粗糙，容易动作小、不朝任务目标去。

实际日志也显示，joint 路线里模型并非完全没 action，但执行到 SO101 上后动作幅度和语义都不理想。

### 9.2 EEF 路线

运行参数：

```bash
--control-mode eef
```

LeIsaac action mode：

```text
mimic_so101leader
```

这个模式在 LeIsaac 里使用 Differential IK，action 格式是：

```text
[x, y, z, qw, qx, qy, qz, gripper]
```

GR00T 输出：

```text
action.eef_9d: shape = (1, 40, 9)
action.gripper_position: shape = (1, 40, 1)
```

实验处理：

1. 读取当前末端位姿：

```text
current_eef = [x, y, z, qw, qx, qy, qz]
```

2. 把 GR00T 的 `eef_9d` 当作相对 EEF delta：

```text
delta = [dx, dy, dz, rot6d...]
```

3. 组合当前 pose 和 delta：

```text
target_pose = current_pose @ delta_pose
```

4. 拼上 gripper：

```text
command = [target_x, target_y, target_z, target_qw, target_qx, target_qy, target_qz, target_gripper]
```

5. 调用：

```python
env.step(command)
```

这条路线更接近跨机器人迁移，因为末端位姿比原始关节更接近机器人无关的动作表达。

实际观察中，EEF 路线明显比 joint 路线动作更大，说明 GR00T 输出的 EEF action 确实更有信号。

## 10. 当前实验观察和结论

目前观察到：

- joint 路线动作很小或语义不明显。
- EEF 路线动作明显更大。
- 但在当前 SO101 + SmartTask + OXE/DROID zero-shot setup 下，没有可靠抓起红色 2x4 lego。

这说明：

1. GR00T N1.7 并不是完全没有输出动作。
2. 之前“看起来没动作”的主要原因，是 joint action schema 落到 SO101 上非常别扭。
3. EEF/IK 路线更能释放 GR00T 的动作意图。
4. 但 zero-shot 到 SO101/lego pick 仍然没有成功。

更准确的结论应该是：

> N1.7 base model 在这个 LeIsaac SmartTask 场景中有非空、较强的末端动作意图；EEF 控制路线比 joint 硬映射合理得多。但在没有针对 SO101 embodiment 和该任务微调的情况下，当前 setup 没有实现可靠 zero-shot pick。

## 11. 为什么这不是“模型完全不行”的证明

当前失败仍然混合了多个因素：

- GR00T base model 的预训练 embodiment 更偏 DROID/Franka/通用 EEF，不是 SO101。
- SO101 自由度少，IK 能力受限。
- SO101 gripper 的尺度和 DROID/Franka gripper 语义未必一致。
- 这个 SmartTask 场景和红色 lego pick 的视觉/几何分布未必在 base model 舒适区。
- 当前 EEF relative/absolute 解读仍然是实验假设，需要更多日志验证。

所以当前结果更适合被理解为：

> 原生 N1.7 路线已经跑通，动作通路也有信号；但要得到真正 pick 成功，可能需要 embodiment 对齐、少量微调、或者换更接近官方支持的 Franka/DROID 类机器人场景。

## 12. 下一步建议

建议后续按优先级做：

1. 记录一轮 EEF 路线日志，观察 `jaw_to_lego` 是否下降、gripper 是否闭合。
2. 如果 jaw 没朝物体去，说明 policy/视觉语义不够。
3. 如果 jaw 接近但没夹住，说明 gripper/IK/接触参数是主要问题。
4. 尝试更接近 GR00T 官方 embodiment 的 Franka/Panda 场景，减少 SO101 morphology mismatch。
5. 如果公司后续有真实 SO101/SOARM 数据，可以考虑对 N1.7 做小规模 embodiment finetune。

## 13. 复现命令

终端 1：启动 GR00T bridge。

```bash
cd /home/yzliu/smart_project
/home/yzliu/Isaac-GR00T/.venv/bin/python \
  experiments/groot_n17_isaac_smart_task/groot_bridge_server.py \
  --model-path nvidia/GR00T-N1.7-3B \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --device cuda
```

终端 2：启动 Isaac viewport，并走 EEF 控制路线。

```bash
cd /home/yzliu/smart_project
conda run -n isaaclab env \
  PYTHONPATH=/home/yzliu/smart_project/leisaac/source/leisaac \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  python experiments/groot_n17_isaac_smart_task/run_smart_task_closed_loop.py \
    --control-mode eef \
    --no-headless \
    --render-sleep-s 0.08 \
    --keep-open-s 60 \
    --max-policy-calls 1 \
    --instruction "Pick up the red 2x4 lego brick."
```
