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
├── franka_smart_task/
│   ├── __init__.py
│   └── franka_smart_task_env_cfg.py
├── wire.py
├── groot_bridge_server.py
├── run_smart_task_closed_loop.py
├── so101_synthetic_groot_config.py
└── train_so101_synthetic_groot.py
```

核心文件含义：

- `wire.py`：Isaac 进程和 GR00T 进程之间的通信协议。负责 socket、msgpack、numpy array 序列化。
- `groot_bridge_server.py`：运行在 GR00T venv 中，加载 `nvidia/GR00T-N1.7-3B`，对 Isaac 侧提供 `ping/get_action/reset/shutdown` 接口。
- `run_smart_task_closed_loop.py`：运行在 `conda isaaclab` 中，启动 LeIsaac SmartTask 或本目录注册的 Franka SmartTask，构造 GR00T observation，接收 action，并驱动 Isaac 中的机器人。
- `so101_synthetic_groot_config.py`：GR00T `NEW_EMBODIMENT` 的 SO101 modality config，用于合成数据微调。
- `train_so101_synthetic_groot.py`：SO101 合成数据准备和 GR00T fine-tune 启动脚本。
- `franka_smart_task/__init__.py`：注册新的 gymnasium task id：`Groot-Franka-SmartTask-v0`。
- `franka_smart_task/franka_smart_task_env_cfg.py`：定义 Franka 版本 SmartTask 的 IsaacLab env config。
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
    --max-policy-calls 8 \
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

### 4.3 为什么 runner 会主动插入 copied IsaacLab 依赖路径

这台机器上既有你复制过来的公司项目：

```text
/home/yzliu/smart_project/leisaac/dependencies/IsaacLab/source
```

也可能有机器上原本安装/checkout 的 IsaacLab。为了不污染本机环境，也为了让实验尽量使用公司项目复制进来的版本，`run_smart_task_closed_loop.py` 在 import IsaacLab 之前会把这些路径插到 `sys.path` 最前面：

```text
leisaac/dependencies/IsaacLab/source/isaaclab
leisaac/dependencies/IsaacLab/source/isaaclab_assets
leisaac/dependencies/IsaacLab/source/isaaclab_tasks
leisaac/dependencies/IsaacLab/source/isaaclab_mimic
```

这一步的含义是：

- 不需要 pip install。
- 不改 conda env。
- 不改 shell 全局配置。
- 当前 runner 进程里优先 import workspace 内复制的 IsaacLab / IsaacLab Assets / IsaacLab Tasks。

也就是说，隔离边界仍然是两个进程和两个虚拟环境；只是 Isaac runner 进程内部把公司项目自带源码放在更高优先级。

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

## 5.1 Franka 版本 task 是怎么隔离出来的

为了验证“GR00T N1.7 是否具备 zero-shot 能力”，SO101 版本实验有一个很大的干扰项：

> GR00T N1.7 的 OXE/DROID action schema 更接近 7DoF Franka/DROID 类机器人，而 SO101 是更小、更弱、关节语义完全不同的演示/教育机械臂。

所以新加的 Franka 路线不是为了替代 SO101 路线，而是为了做一个更公平的对照实验：

- 如果 SO101 抓不到，但 Franka 能明显更接近目标，说明 embodiment mismatch 很可能是主要问题。
- 如果 Franka 也完全没有朝物体去，说明问题更可能在 zero-shot 视觉/语言/任务理解本身。
- 如果 Franka 能抓起来，那就能更强地证明 N1.7 base model 确实有比较强的 zero-shot pick 能力。

### 5.1.1 task 注册方式

新增文件：

```text
experiments/groot_n17_isaac_smart_task/franka_smart_task/__init__.py
```

里面调用：

```python
gym.register(
    id="Groot-Franka-SmartTask-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "franka_smart_task.franka_smart_task_env_cfg:FrankaSmartTaskEnvCfg",
    },
)
```

这里没有修改 LeIsaac 主工程的注册表，而是利用 Python import 的副作用，在 experiments 目录里额外注册一个 task id。runner 里执行：

```python
import franka_smart_task
```

之后，`gym.make("Groot-Franka-SmartTask-v0", cfg=env_cfg)` 就可以创建 Franka 版本环境。

### 5.1.2 复用哪些东西

Franka task 仍然复用 LeIsaac 的 `SmartTaskSceneCfg` / observation schema，但
`__post_init__()` 不再直接走 `SmartTaskEnvCfg.__post_init__()`。原因是
`SmartTaskEnvCfg.__post_init__()` 里包含 SO101 的 robot init 假设；Franka 版现在
显式调用更底层的 `SingleArmTaskEnvCfg.__post_init__()`，然后只复刻 SmartTask 需要的
场景资产解析和 lego 随机化。

所以 Franka 路线继续复用：

- SmartScene USD。
- 红色 2x4 lego brick。
- 外部相机 observation key。
- wrist camera observation key。
- `camera1/camera3/joint_pos/ee_frame_state` 这套 policy observation 结构。
- reward / termination / scene entity 命名的大部分约定。
- IsaacLab ManagerBasedRLEnv 创建方式。

但机器人相关的 init pose、wrist camera、EEF frame、gripper action 和 action manager
都按 Franka embodiment 重新定义。它不是另起炉灶写一个新仿真项目，而是：

```text
SmartScene / lego / observation key：复用 LeIsaac
SO101 外部相机物理机位：复用 LeIsaac
Franka 外部主视角物理机位：覆盖为 Franka overview camera
robot / EEF / wrist / gripper / action：按 Franka 语义定义
```

### 5.1.3 替换哪些东西

Franka task 主要替换四类配置：

1. 机器人资产：

```python
robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

这里的 `FRANKA_PANDA_HIGH_PD_CFG` 来自你复制进来的 IsaacLab Assets：

```text
leisaac/dependencies/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/franka.py
```

`prim_path="{ENV_REGEX_NS}/Robot"` 保持了 LeIsaac 里 `SceneEntityCfg("robot")` 的命名习惯。也就是说，下游 observation/action/termination 仍然可以通过 scene entity 名字 `"robot"` 找到机器人。

Franka 的 root 初始朝向被设置为绕世界 Z 轴逆时针 90 度：

```python
self.scene.robot.init_state.rot = (0.70710678, 0.0, 0.0, 0.70710678)
```

这个四元数的含义是 `yaw=+90deg`。这样做的目的不是改变 Franka 官方 ready pose
的关节形状，而是把整台 Panda 在 SmartScene 中转向，使 ready pose 下的夹爪朝向
更自然地对准 lego 所在的任务方向。

2. 末端坐标系：

Franka 没有 SO101 的 follower link 命名，所以 `ee_frame` 改成跟踪 `panda_hand`，并额外保留两个 target frame：

- `end_effector`：用于 `ee_frame_state`，给 GR00T 构造 `state.eef_9d`。
- `grasp_center`：用于诊断指标和 LeIsaac 原始 `object_grasped` 里对 `target_pos_w[:, 1, :]` 的访问约定。

3. wrist 相机和 top/global 外部相机：

当前 GR00T OXE/DROID probe 真正送给模型的是两路图像：

```text
camera1 -> video.exterior_image_1_left
camera3 -> video.wrist_image_left
```

其中：

- `camera1` 是送给 GR00T 的 top/global 外部视角 key。虽然历史 USD/变量命名里可能出现
  `front`，但从实际图像语义看，它更应该被理解为俯视/全局观察相机，而不是机器人正前方相机。
  SO101 路线中它仍然来自 SmartScene USD 里的原始固定外部相机。Franka 路线中不能直接照搬
  这个物理机位，因为它是按 SO101/soarm 的体型和高度设计的；换成更高大的 Panda 后，这个
  相机不适合作为 Franka 全局观察。因此 Franka 路线把同一个 `camera1` observation key 覆盖为新的
  `Scene/franka_overview_camera`，它是一个更高、更远的斜上方 overview camera，用来同时
  看到 Franka、桌面盘面和 lego 目标区域。
- `camera3` 是 Franka wrist camera：
  `Robot/panda_hand/wrist_camera`。它保留腕部/夹爪视角语义，跟着 `panda_hand` 运动。
- `camera2` 是 SmartScene USD 里的左侧固定相机，仍保留在 policy observation 中，但当前
  OXE/DROID GR00T probe 没有把它送给模型。

这里有一个重要结论：如果 Franka ready pose 的整体朝向和 lego 所在方向不一致，应该优先旋转
Franka root，而不是把 wrist 相机单独旋成“看目标”的相机；后者会破坏真实 wrist camera 的语义。
当前版本已经把 Franka root 逆时针旋转 90 度。wrist camera 仍然挂在 `panda_hand` 下，所以它会
随着机器人末端自然旋转，不需要额外修改 hand-relative offset。初始目标定位仍主要交给主外部相机
`camera1`，wrist 相机主要服务于靠近目标和闭合夹爪阶段。

Franka 版还显式去掉了 SO101 模板里遗留的 robot-mounted `front` sensor，避免 Isaac stage 中
多出一个与当前 GR00T 输入无关的机器人前置相机。

4. action 配置：

Franka 版本支持两种 action mode：

```text
franka_ik    = 7D EEF pose target + 1D binary gripper
franka_joint = 7D arm joint target + 1D binary gripper
```

runner 根据参数自动选择：

```text
--robot franka --control-mode eef   -> franka_ik
--robot franka --control-mode joint -> franka_joint
```

这让 Franka 路线和 SO101 路线共享同一个 bridge / observation 构造 / action chunk 执行逻辑，但 action manager 是各自独立的。

### 5.1.4 Franka 为什么更适合这个 zero-shot 对照

GR00T bridge 使用的 embodiment tag 是：

```text
OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
```

它的 action/state 里有：

```text
state.joint_position: 7D
action.joint_position: 7D
action.eef_9d: 9D
action.gripper_position: 1D
```

SO101 的 arm + gripper 只有 6 维，而且第 6 维是 gripper，不是标准 7DoF arm joint。把 7D DROID/Franka action 硬压到 SO101 上，语义一定会损失。

Franka Panda 则天然是：

```text
7D arm joints + parallel gripper
```

所以它可以更直接地承接 GR00T 的 `joint_position` 和 `eef_9d + gripper_position` 输出。这样如果失败，至少可以少一个“机器人形态差太多”的解释。

### 5.1.5 Franka cfg 当前采用的原则

Franka route 当前的原则是：

- 不再把 SO101 的 robot init、历史 camera/sensor 配置、wrist offset 直接套到 Franka 上。
- 使用 copied IsaacLab 里的 `FRANKA_PANDA_HIGH_PD_CFG`。
- 使用 copied IsaacLab Franka stack task 的 ready joint pose 作为桌面任务起点。
- Franka root 绕世界 Z 轴逆时针旋转 90 度，让 ready pose 下的夹爪方向更贴近 SmartScene
  中 lego 的任务方向。
- wrist camera 使用 Franka wrist/gripper 视角语义，不强行初始看 lego。
- `camera1` 这个 observation key 承担 top/global 目标观测；SO101 使用原 SmartScene
  固定相机，Franka 使用专属的 `franka_overview_camera`。
- reset 后启用 `rerender_on_reset=True`，避免相机 observation 仍是旧缓存帧。

这也意味着 Franka 路线验证的是“更接近 GR00T OXE/DROID embodiment 的机器人，在同一个
SmartScene 任务里是否更自然”，而不是“把 SO101 的所有 cfg 名字换成 Franka”。

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

这里的 `camera1` 在本 SmartTask 数据里应理解为 top/global 视角。映射到
`video.exterior_image_1_left` 只是因为 OXE/DROID embodiment 的外部相机 key 叫这个名字，
不表示它是 front camera。

`camera2` 仍然存在于 Isaac policy observation 中，但当前没有送入 GR00T。这是因为
`OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` embodiment 的 video modality 只声明了
`exterior_image_1_left` 和 `wrist_image_left` 两路。

调试相机时可以运行：

```bash
cd /home/yzliu/smart_project
conda run -n isaaclab env \
  PYTHONPATH=/home/yzliu/smart_project/experiments/groot_n17_isaac_smart_task:/home/yzliu/smart_project/leisaac/source/leisaac \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  python experiments/groot_n17_isaac_smart_task/run_smart_task_closed_loop.py \
    --robot franka \
    --control-mode eef \
    --headless \
    --debug-cameras-only
```

这个模式不会连接 GR00T bridge，只会 reset Isaac env，一次性打印：

- policy 中 `camera1/camera2/camera3` 的 shape、像素范围、均值和方差。
- InteractiveScene 中实际注册了哪些 sensors。
- USD stage 中实际有哪些 camera prim。
- wrist camera 和 `panda_hand` 的世界坐标关系。
- 当前三路 policy 图像 PNG，默认保存在 `runs/camera_debug/`。

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

Franka 的 arm 正好是 7DoF，所以：

```text
Franka panda_joint1..7 -> DROID 7D joint_position
```

这比 SO101 padding 合理得多，但仍然要注意：GR00T 输出的是 DROID/OXE 训练语义下的
relative joint action，不等于“任何 7DoF 机器人都能直接完美执行”。

### 6.4 Gripper 状态映射

SO101 第 6 个 joint 是 gripper，所以：

```text
state.gripper_position = joint_pos[5]
```

shape 写成 `(1, 1, 1)`，符合 GR00T 的 batch/time/dim 约定。

Franka 则有两个 finger joint，当前实验取两个 finger joint 的平均开口作为：

```text
state.gripper_position = mean(panda_finger_joint1, panda_finger_joint2)
```

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

Isaac action mode：

```text
SO101  -> so101leader
Franka -> franka_joint
```

GR00T 输出：

```text
action.joint_position: shape = (1, 40, 7)
```

实验处理：

```text
SO101:  取前 6 维 -> LeIsaac SO101 6D joint command
Franka: 7D relative joint action + current joint state -> Franka 7D joint target
```

问题：

- GR00T 的 joint_position 是 DROID/Franka 类机器人语义。
- SO101 是完全不同的 5DoF arm + gripper。
- SO101 直接取前 6 维会非常粗糙，容易动作小、不朝任务目标去。
- Franka joint route 形态更匹配，但仍受 GR00T joint action 语义和当前场景分布影响。

实际日志也显示，joint 路线里模型并非完全没 action，但执行到 SO101 上后动作幅度和语义都不理想。

### 9.2 EEF 路线

运行参数：

```bash
--control-mode eef
```

Isaac action mode：

```text
SO101  -> mimic_so101leader
Franka -> franka_ik
```

这两个模式都使用 Differential IK，action 格式是：

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

对 Franka 来说，最后一维 gripper 不是 SO101 的连续 gripper joint target，而是
`BinaryJointPositionActionCfg` 的开/合命令：正数表示 open，负数表示 close。

这条路线更接近跨机器人迁移，因为末端位姿比原始关节更接近机器人无关的动作表达。

实际观察中，EEF 路线明显比 joint 路线动作更大，说明 GR00T 输出的 EEF action 确实更有信号。

## 10. 当前实验观察和结论

目前观察到：

- joint 路线动作很小或语义不明显。
- EEF 路线动作明显更大。
- 但在当前 SO101 + SmartTask + OXE/DROID zero-shot setup 下，没有可靠抓起红色 2x4 lego。
- Franka 路线已经隔离出来，并且不再照搬 SO101 的 robot/camera/wrist cfg。
- Franka 的 top/global 外部相机 `camera1` 已改成
  Franka 专属的 `franka_overview_camera`；它能看到 Panda 全局姿态、桌面和目标物。
- wrist 相机保持末端/夹爪视角语义，不再强行旋向 lego。

这说明：

1. GR00T N1.7 并不是完全没有输出动作。
2. 之前“看起来没动作”的主要原因，是 joint action schema 落到 SO101 上非常别扭。
3. EEF/IK 路线更能释放 GR00T 的动作意图。
4. Franka 路线更适合判断 N1.7 在 OXE/DROID 风格 embodiment 下的真实表现。
5. 但 zero-shot 到这个 SmartScene/lego pick 任务是否能成功，仍需要继续用 Franka EEF/IK 长一点的闭环视频来观察。

更准确的结论应该是：

> N1.7 base model 在这个 LeIsaac SmartTask 场景中有非空、较强的末端动作意图；EEF 控制路线比 joint 硬映射合理得多。但在没有针对 SO101 embodiment 和该任务微调的情况下，当前 setup 没有实现可靠 zero-shot pick。

对 Franka 路线更准确的说法是：

> Franka task 已经变成更干净的 embodiment 对照：场景仍是同一个 SmartScene，但 robot/EEF/wrist/gripper/action 都按 Franka 语义处理。接下来 Franka 的失败或成功，更能说明 GR00T N1.7 在相对接近 OXE/DROID embodiment 的设置下是否具备 zero-shot pick 能力。

## 11. 为什么这不是“模型完全不行”的证明

当前失败仍然混合了多个因素：

- GR00T base model 的预训练 embodiment 更偏 DROID/Franka/通用 EEF，不是 SO101。
- SO101 自由度少，IK 能力受限。
- SO101 gripper 的尺度和 DROID/Franka gripper 语义未必一致。
- 这个 SmartTask 场景和红色 lego pick 的视觉/几何分布未必在 base model 舒适区。
- 当前 EEF relative/absolute 解读仍然是实验假设，需要更多日志验证。
- Franka 版本虽然减少了 SO101 morphology mismatch，并把主外部相机改成了适合 Panda
  的 overview view，但 SmartScene 的物体布局和夹爪初始朝向仍然不是 GR00T 官方任务配置。

所以当前结果更适合被理解为：

> 原生 N1.7 路线已经跑通，动作通路也有信号；但要得到真正 pick 成功，可能需要 embodiment 对齐、少量微调、或者换更接近官方支持的 Franka/DROID 类机器人场景。

## 12. 下一步建议

建议后续按优先级做：

1. 优先跑 Franka + EEF/IK 的 8-call 视频，观察末端是否朝盘面/lego 区域运动。
2. 同时保留 SO101 + EEF/IK 作为对照，比较“同一模型输出落在不同 embodiment 上”的差异。
3. 用 `--debug-cameras-only` 保存每次运行前的 `camera1/camera3`，确认模型看到的是正确 top/global 视角和 wrist 视角。
4. 如果 Franka 末端没有朝目标区移动，优先怀疑视觉/语言/zero-shot 能力边界或 observation schema。
5. 如果 Franka 末端能接近但夹不住，再检查 gripper 阈值、接触参数、EEF delta 缩放和 IK 跟踪。
6. 对当前已有 SO101 合成数据，先跑通 GR00T `NEW_EMBODIMENT` 微调链路；真机部署前再补真实
   SO101/SOARM 数据或做 HG-DAgger 类修正数据采集。

## 13. SO101 合成数据微调路线

除了 zero-shot probe，本目录现在还增加了面向真实部署目标的 SO101 微调路线。这里的目标机械臂
明确是 SOARM101/SO101，不是 Franka。Franka 仍然只是为了分析 zero-shot embodiment mismatch
而加入的对照任务。

当前 `dataset/` 下的两份合成数据是 LeRobot v3 格式：

```text
dataset/so101_lego_pick_0609_1722
dataset/so101_lego_pick_0609_1722_mimic
```

它们的关键特征是：

- `robot_type` 是 `so101_follower`。
- `observation.state` 是 6D SO101 state。
- `action` 是 6D SO101 action。
- 前 5 维是 arm joints，第 6 维是 gripper。
- 图像有三路：`camera1/camera2/camera3`。

GR00T 训练侧当前需要的是 GR00T-flavored LeRobot v2.1 数据，并且需要额外的
`meta/modality.json`。所以新增的训练准备逻辑不是直接修改原始 `dataset/`，而是默认生成 prepared
副本：

```text
outputs/groot_so101_synthetic_datasets/
```

这些 prepared dataset 中会额外出现：

```text
meta/modality.json
meta/stats.json
meta/relative_stats.json
```

### 13.1 为什么是两阶段环境

本机环境是刻意隔离的：

```text
LeRobot 数据/schema 转换 -> conda lerobot
GR00T stats / launch_finetune -> /home/yzliu/Isaac-GR00T/.venv
Isaac closed-loop -> conda isaaclab
```

因此不要假设一个 Python 环境可以同时 import LeRobot、GR00T、IsaacLab。推荐流程是：

第一阶段：在 LeRobot 环境里准备数据。

```bash
cd /home/yzliu/smart_project
conda run -n lerobot python \
  experiments/groot_n17_isaac_smart_task/train_so101_synthetic_groot.py \
    --instruction "Pick up the red 2x4 lego brick." \
    --force-prepare \
    --skip-stats \
    --prepare-only
```

第二阶段：在 GR00T venv 里生成统计并启动 fine-tune 入口。

```bash
cd /home/yzliu/smart_project
/home/yzliu/Isaac-GR00T/.venv/bin/python \
  experiments/groot_n17_isaac_smart_task/train_so101_synthetic_groot.py \
    --skip-prepare \
    --max-steps 2000 \
    --save-steps 500 \
    --global-batch-size 32
```

本机 RTX 5060 Ti 16GB 更适合做数据转换、loader smoke test、stats 生成和小步数逻辑验证；完整
GR00T 微调大概率仍然需要上云或使用 40GB+ 显存设备。

### 13.2 v3 到 v2.1 转换

LeRobot v3 和 v2.1 最大区别是存储布局：

```text
v3:   data/chunk-000/file-000.parquet
      videos/observation.images.camera1/chunk-000/file-000.mp4
      meta/tasks.parquet
      meta/episodes/chunk-000/file-000.parquet

v2.1: data/chunk-000/episode_000000.parquet
      videos/chunk-000/observation.images.camera1/episode_000000.mp4
      meta/tasks.jsonl
      meta/episodes.jsonl
```

GR00T 官方仓库里有转换器：

```text
/home/yzliu/Isaac-GR00T/scripts/lerobot_conversion/convert_v3_to_v2.py
```

这个转换器的 `convert_dataset()` 是原地转换：会把原始目录移动成 `_v3.0` 备份，再把 v2.1 写回原路径。
这不是删除数据，但会改变当前 `dataset/` 的目录形态。当前实验脚本默认使用 prepared copy，是为了减少误操作。

另外，GR00T 官方转换器依赖某个 LeRobot commit 的 API；本机 LeRobot 更新后可能出现
`load_info` 等 API 位置变化。这不是环境坏了，而是 Isaac/GR00T 和 LeRobot 更新节奏不同造成的正常
版本漂移。`train_so101_synthetic_groot.py` 会优先尝试官方转换器；如果依赖或 API 不兼容，会打印原因并
回退到本目录里的轻量非破坏性转换逻辑。

### 13.3 GR00T modality config

新增文件：

```text
experiments/groot_n17_isaac_smart_task/so101_synthetic_groot_config.py
```

这不是 LeRobot 官方格式，而是 GR00T 对自定义 embodiment 的训练配置。它注册：

```text
EmbodimentTag.NEW_EMBODIMENT
```

并声明：

```text
video:
  top
  wrist

state:
  single_arm  -> observation.state[0:5]
  gripper     -> observation.state[5:6]

action:
  single_arm  -> action[0:5], relative joint action
  gripper     -> action[5:6], absolute gripper target

language:
  annotation.human.task_description
```

训练相机映射是：

```text
observation.images.camera1 -> video.top
observation.images.camera3 -> video.wrist
observation.images.camera2 -> 不送入 GR00T 微调
```

这里把 `camera1` 命名为 `top`，是因为实际图像语义是 top/global 视角，而不是 front 视角。

## 14. 复现命令

终端 1：启动 GR00T bridge。

```bash
cd /home/yzliu/smart_project
/home/yzliu/Isaac-GR00T/.venv/bin/python \
  experiments/groot_n17_isaac_smart_task/groot_bridge_server.py \
  --model-path nvidia/GR00T-N1.7-3B \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --device cuda
```

终端 2A：启动原始 SO101 SmartTask viewport，并走 joint-space baseline。

```bash
cd /home/yzliu/smart_project
conda run -n isaaclab env \
  PYTHONPATH=/home/yzliu/smart_project/experiments/groot_n17_isaac_smart_task:/home/yzliu/smart_project/leisaac/source/leisaac \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  python experiments/groot_n17_isaac_smart_task/run_smart_task_closed_loop.py \
    --robot so101 \
    --control-mode joint \
    --no-headless \
    --render-sleep-s 0.08 \
    --keep-open-s 60 \
    --max-policy-calls 8 \
    --capture-video \
    --capture-name so101_joint_probe \
    --instruction "Pick up the red 2x4 lego brick."
```

终端 2B：启动原始 SO101 SmartTask viewport，并走 EEF/IK 控制路线。

```bash
cd /home/yzliu/smart_project
conda run -n isaaclab env \
  PYTHONPATH=/home/yzliu/smart_project/experiments/groot_n17_isaac_smart_task:/home/yzliu/smart_project/leisaac/source/leisaac \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  python experiments/groot_n17_isaac_smart_task/run_smart_task_closed_loop.py \
    --robot so101 \
    --control-mode eef \
    --no-headless \
    --render-sleep-s 0.08 \
    --keep-open-s 60 \
    --max-policy-calls 8 \
    --capture-video \
    --capture-name so101_eef_probe \
    --instruction "Pick up the red 2x4 lego brick."
```

终端 2C：启动 Franka SmartTask viewport，并走 joint-space baseline。

```bash
cd /home/yzliu/smart_project
conda run -n isaaclab env \
  PYTHONPATH=/home/yzliu/smart_project/experiments/groot_n17_isaac_smart_task:/home/yzliu/smart_project/leisaac/source/leisaac \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  python experiments/groot_n17_isaac_smart_task/run_smart_task_closed_loop.py \
    --robot franka \
    --control-mode joint \
    --no-headless \
    --render-sleep-s 0.08 \
    --keep-open-s 60 \
    --max-policy-calls 8 \
    --capture-video \
    --capture-name franka_joint_probe \
    --instruction "Pick up the red 2x4 lego brick."
```

终端 2D：启动 Franka SmartTask viewport，并走 EEF/IK 控制路线。

```bash
cd /home/yzliu/smart_project
conda run -n isaaclab env \
  PYTHONPATH=/home/yzliu/smart_project/experiments/groot_n17_isaac_smart_task:/home/yzliu/smart_project/leisaac/source/leisaac \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  python experiments/groot_n17_isaac_smart_task/run_smart_task_closed_loop.py \
    --robot franka \
    --control-mode eef \
    --no-headless \
    --render-sleep-s 0.08 \
    --keep-open-s 60 \
    --max-policy-calls 8 \
    --capture-video \
    --capture-name franka_eef_probe \
    --instruction "Pick up the red 2x4 lego brick."
```

这四条命令会得到一个完整的 2x2 对照：

```text
SO101  + joint
SO101  + EEF/IK
Franka + joint
Franka + EEF/IK
```

录制实现方式：

- runner 调用 Isaac Sim 自带的 `omni.kit.capture.viewport`。
- mp4 编码依赖同环境已有的 `omni.videoencoding`。
- 默认只录“主动控制阶段”，不录最后 `--keep-open-s` 的静止观察阶段。
- 默认输出目录是 `experiments/groot_n17_isaac_smart_task/runs/captures/`。
- 默认视频参数是 `960x540 / 15fps / 2Mbps`，目的是让文件不要太大。

如果文件仍然太大，可以调低：

```bash
--capture-bitrate-mbps 1.0
--capture-width 768
--capture-height 432
--capture-every-nth-frames 2
```

这四条 Isaac 命令共用同一个 GR00T bridge。差别只在：

- SO101 joint：验证“DROID/Franka 风格 7D joint action 硬落到 SO101 joint space”的 baseline。
- SO101 EEF：验证“原 LeIsaac 机器人 + EEF/IK action 落地”。
- Franka joint：验证“7D joint action 在 Franka 上是否比 SO101 更自然”。
- Franka EEF：验证“更接近 GR00T 官方 embodiment 的机器人 + EEF/IK action 落地”。
