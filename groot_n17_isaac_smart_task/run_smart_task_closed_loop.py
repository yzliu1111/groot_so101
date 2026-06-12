"""在 LeIsaac SmartTask 中调用 GR00T N1.7 bridge，并闭环驱动 Isaac 机器人。

运行环境：
    conda run -n isaaclab ...

这个文件是整个实验的 Isaac 侧主程序。它做四件事：

1. 启动 Isaac Sim / IsaacLab，并加载 LeIsaac 已经注册好的 SmartTask。
2. 从 SmartTask observation 中取出图像、关节状态、末端位姿。
3. 把这些 observation 改写成 GR00T N1.7 OXE/DROID embodiment 需要的 schema。
4. 调用 GR00T bridge 拿 action，再把 action 转成 LeIsaac 可以执行的命令。

注意：
    这里不是把 SO101 伪装成“真正的 DROID/Franka”。它只是一个 zero-shot
    探针：尽量复用 LeIsaac 的场景和 USD 资产，绕开 LeRobot policy stack，
    看 GR00T N1.7 base model 在这个场景里会给出什么样的动作。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch


# 当前文件所在目录：
# /home/yzliu/smart_project/experiments/groot_n17_isaac_smart_task
SCRIPT_DIR = Path(__file__).resolve().parent

# 项目根目录：
# /home/yzliu/smart_project
REPO_ROOT = SCRIPT_DIR.parents[1]

# LeIsaac 的 Python package 源码目录。运行脚本时虽然 README 里已经设置
# PYTHONPATH，但这里再插一次，方便直接运行或调试。
LEISAAC_SRC = REPO_ROOT / "leisaac" / "source" / "leisaac"

# LeIsaac 复制项目中自带的 IsaacLab 依赖目录。用户这份 workspace 是把公司
# 项目的 leisaac / lerobot 整体复制过来的，所以这里优先把 copied IsaacLab
# source 放进 sys.path，避免无意中依赖机器上另一份 IsaacLab 源码。
LEISAAC_ISAACLAB_SRC = REPO_ROOT / "leisaac" / "dependencies" / "IsaacLab" / "source"
LEISAAC_ISAACLAB_PACKAGES = (
    LEISAAC_ISAACLAB_SRC / "isaaclab",
    LEISAAC_ISAACLAB_SRC / "isaaclab_assets",
    LEISAAC_ISAACLAB_SRC / "isaaclab_tasks",
    LEISAAC_ISAACLAB_SRC / "isaaclab_mimic",
)

# 让 Python 可以 import 同目录的 wire.py。
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# 让 Python 可以 import leisaac package。
if str(LEISAAC_SRC) not in sys.path:
    sys.path.insert(0, str(LEISAAC_SRC))

# 优先使用 LeIsaac 复制进来的 IsaacLab package。不存在的目录跳过，避免硬依赖。
for package_path in reversed(LEISAAC_ISAACLAB_PACKAGES):
    if package_path.exists() and str(package_path) not in sys.path:
        sys.path.insert(0, str(package_path))

# `request()` 是本实验自定义的 socket 请求函数：
# Isaac runner 通过它向 GR00T bridge 发送 observation，并等待 action。
from wire import request  # noqa: E402

# AppLauncher 要在 sys.path 调整后再 import，这样 isaaclab/isaaclab_assets 会优先
# 来自 copied LeIsaac dependencies，而不是机器上其它源码副本。
from isaaclab.app import AppLauncher  # noqa: E402


def quat_wxyz_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """把 Isaac/LeIsaac 的 wxyz 四元数转换成 GR00T 使用的 rot6d。

    参数：
    - `quat`：shape `(4,)`，顺序是 `(w, x, y, z)`。

    返回：
    - shape `(6,)`，表示旋转矩阵前两行 flatten 后的 6D rotation。

    背景：
    - LeIsaac observation 里的 `ee_frame_state` 是 `xyz + quat(wxyz)`。
    - GR00T OXE/DROID schema 的 `eef_9d` 是 `xyz + rot6d`。
    - 所以构造 GR00T state 时必须做这个转换。
    """

    w, x, y, z = quat

    # 四元数需要先归一化，否则转出的旋转矩阵可能不是正交矩阵。
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-8:
        # 异常情况下用单位旋转兜底：矩阵前两行是 [1,0,0] 和 [0,1,0]。
        return np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)

    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # 标准 wxyz 四元数到 3x3 旋转矩阵的公式。
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )

    # GR00T repo 的 pose.py 里使用的是“旋转矩阵前两行 flatten”的 rot6d。
    return rot[:2, :].reshape(-1)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """把 wxyz 四元数转换为 3x3 旋转矩阵。

    这个函数用于 EEF 控制路线：GR00T 返回的 eef_9d 被当作“当前末端坐标系下
    的相对位姿增量”，需要先把当前末端四元数转成矩阵，然后做矩阵乘法组合。
    """

    w, x, y, z = quat.astype(np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)

    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """把 3x3 旋转矩阵转换回 Isaac/LeIsaac 需要的 wxyz 四元数。

    LeIsaac 的 `mimic_so101leader` IK action 格式是：
        `[x, y, z, qw, qx, qy, qz, gripper]`

    因此当我们把 GR00T 的 rot6d 组合成目标旋转矩阵后，还需要转成四元数。
    """

    trace = float(np.trace(rot))

    # 下面是常见的 matrix -> quaternion 数值稳定写法：
    # 根据矩阵 trace 和对角线最大项选择不同分支，避免接近 180 度旋转时除数太小。
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(max(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2], 1e-8)) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(max(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2], 1e-8)) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1], 1e-8)) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s

    quat = np.array([w, x, y, z], dtype=np.float32)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """把 GR00T 的 rot6d 转回 3x3 旋转矩阵。

    参数：
    - `rot6d`：shape `(6,)`，可看作两个 3D 行向量。

    返回：
    - shape `(3, 3)` 的旋转矩阵。

    做法：
    - 第一行归一化。
    - 第二行减去在第一行方向上的投影，再归一化。
    - 第三行用叉乘补出来。

    这和 GR00T repo 里 pose.py 的思路一致。
    """

    rows = rot6d.astype(np.float64).reshape(2, 3)

    row1 = rows[0]
    row1_norm = np.linalg.norm(row1)
    if row1_norm < 1e-8:
        row1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        row1 = row1 / row1_norm

    row2 = rows[1] - np.dot(rows[1], row1) * row1
    row2_norm = np.linalg.norm(row2)
    if row2_norm < 1e-8:
        row2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        row2 = row2 / row2_norm

    row3 = np.cross(row1, row2)
    return np.vstack([row1, row2, row3]).astype(np.float32)


def compose_pose_delta(base_pos: np.ndarray, base_quat: np.ndarray, delta_9d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把 GR00T 的 EEF 相对动作组合到当前末端位姿上。

    参数：
    - `base_pos`：当前末端位置，shape `(3,)`，在 robot frame 下。
    - `base_quat`：当前末端旋转，shape `(4,)`，wxyz。
    - `delta_9d`：GR00T 输出的一步 `eef_9d` action，shape `(9,)`。

    返回：
    - `target_pos`：目标末端位置，shape `(3,)`。
    - `target_quat`：目标末端旋转，shape `(4,)`，wxyz。

    重要说明：
    - OXE/DROID config 里 `eef_9d` 被标成 relative EEF action。
    - 这里采用的实验假设是：`delta_9d` 是相对于当前 EEF frame 的变换。
    - 如果后续确认 GR00T processor 已经把它 decode 成 absolute action，
      这一层组合就需要改掉；目前日志显示这条路线比 joint 路线更能释放动作。
    """

    base_rot = quat_wxyz_to_matrix(base_quat)
    delta_pos = delta_9d[:3]
    delta_rot = rot6d_to_matrix(delta_9d[3:9])

    # 平移增量先从 EEF frame 转到 robot frame，再加到当前 robot-frame 位置上。
    target_pos = base_pos + base_rot @ delta_pos

    # 旋转增量通过矩阵乘法组合：T_target = T_base @ T_delta。
    target_rot = base_rot @ delta_rot

    return target_pos.astype(np.float32), matrix_to_quat_wxyz(target_rot)


def eef_state_to_9d(ee_frame_state: torch.Tensor) -> np.ndarray:
    """把 LeIsaac 的 `ee_frame_state` observation 转成 GR00T 的 `eef_9d` state。

    LeIsaac：
        `ee_frame_state` shape 通常是 `(num_envs, 7)`，内容是
        `[x, y, z, qw, qx, qy, qz]`。

    GR00T：
        `state.eef_9d` 需要 `[x, y, z, rot6d...]`，总共 9 维。
    """

    state = ee_frame_state.detach().cpu().numpy()[0].astype(np.float32)
    pos = state[:3]
    quat = state[3:7]
    return np.concatenate([pos, quat_wxyz_to_rot6d(quat)]).astype(np.float32)


def image_to_numpy(image: torch.Tensor) -> np.ndarray:
    """把 Isaac observation 里的相机 tensor 转成 GR00T bridge 可传输的 numpy 图像。

    输入：
    - Isaac/LeIsaac 给出的 camera tensor，形状通常是 `(1, H, W, 3)`。

    输出：
    - 去掉 batch 维之后的 `np.uint8` 图像，形状 `(H, W, 3)`。
    """

    return image.detach().cpu().numpy()[0].astype(np.uint8)


class FrameHistory:
    """保存相机历史帧，用来构造 GR00T 需要的两帧视频输入。

    OXE/DROID 的 video modality 使用 `delta_indices=[-15, 0]`，也就是希望看到
    一个历史帧和一个当前帧。这里没有真实 15-step 历史采样，而是用一个简单的
    deque 保存 warmup/运行过程中的帧，取最旧帧和最新帧组成 `(old, cur)`。
    """

    def __init__(self, horizon: int):
        # 每个 camera key 对应一个固定长度队列。
        self._frames: dict[str, deque[np.ndarray]] = {}

        # `horizon` 控制最多保留多少帧；README 默认 warmup_frames=16。
        self.horizon = horizon

    def push(self, key: str, frame: np.ndarray) -> None:
        """向某一路相机历史中追加一帧。"""

        if key not in self._frames:
            self._frames[key] = deque(maxlen=self.horizon)
        self._frames[key].append(frame)

    def pair(self, key: str) -> np.ndarray:
        """返回 GR00T video 输入需要的两帧，并加上 batch 维。

        返回 shape：
            `(1, 2, H, W, 3)`

        含义：
            batch=1，time=2，后面是 RGB 图像。
        """

        frames = self._frames[key]
        old = frames[0]
        cur = frames[-1]
        return np.stack([old, cur], axis=0)[None, ...].astype(np.uint8)


def build_oxe_observation(
    policy_obs: dict[str, torch.Tensor],
    history: FrameHistory,
    instruction: str,
    robot: str,
) -> dict[str, Any]:
    """把 LeIsaac SmartTask observation 映射到 GR00T N1.7 OXE/DROID schema。

    输入：
    - `policy_obs`：LeIsaac env 返回的 `obs["policy"]`。
    - `history`：相机历史缓存。
    - `instruction`：语言指令，例如 "Pick up the red 2x4 lego brick."

    输出：
    - GR00T bridge 可直接传给 `policy.get_action()` 的 nested observation dict。

    SO101 mapping：
    - `camera1` -> `video.exterior_image_1_left`
    - `camera3` -> `video.wrist_image_left`
    - `ee_frame_state` -> `state.eef_9d`
    - SO101 6D joint state pad 到 7D -> `state.joint_position`
    - SO101 gripper joint -> `state.gripper_position`

    Franka mapping：
    - `camera1` / `camera3` 同上。
    - Franka 前 7 个 panda_joint -> `state.joint_position`。
    - 两个 panda_finger joint 的平均值 -> `state.gripper_position`。

    注意：
    - 这只是 probe，不表示 SO101 就是 DROID robot。
    - joint_position 的 7D 是为了符合 OXE/DROID schema；SO101 实际只有 6 个
      joint，其中第 6 个是 gripper。
    """

    exterior = image_to_numpy(policy_obs["camera1"])
    wrist = image_to_numpy(policy_obs["camera3"])

    # GR00T OXE/DROID schema 只使用两路 video key：外部左视角和腕部左视角。
    history.push("exterior_image_1_left", exterior)
    history.push("wrist_image_left", wrist)

    joint_all = policy_obs["joint_pos"].detach().cpu().numpy()[0].astype(np.float32)

    if robot == "franka":
        # Franka Panda arm 正好是 7DoF，直接对齐 GR00T OXE/DROID 的 7D joint_position。
        joint_7_values = joint_all[:7]
        # Panda hand 有两个 finger joint，取平均开口作为单标量 gripper state。
        gripper_value = float(joint_all[7:].mean()) if joint_all.shape[0] > 7 else 0.04
    else:
        # LeIsaac SO101 joint_pos shape 是 `(1, 6)`：
        # [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        # GR00T OXE/DROID 的 joint_position 是 7D。这里把 SO101 的 6D 放前 6 维，
        # 第 7 维补 0。这是一个很粗的兼容层，也是 joint 控制路线效果差的主要风险。
        joint_7_values = np.zeros(7, dtype=np.float32)
        joint_7_values[: min(6, joint_all.shape[0])] = joint_all[:6]
        gripper_value = float(joint_all[5]) if joint_all.shape[0] > 5 else 0.0

    joint_7 = joint_7_values.reshape(1, 1, 7).astype(np.float32)

    # gripper_position 单独作为 state key，shape 必须是 `(B, T, D)`。
    gripper = np.array([[[gripper_value]]], dtype=np.float32)

    # 当前末端状态转成 eef_9d，shape 从 `(9,)` 扩成 `(1,1,9)`。
    eef_9d = eef_state_to_9d(policy_obs["ee_frame_state"])[None, None, :]

    return {
        "video": {
            "exterior_image_1_left": history.pair("exterior_image_1_left"),
            "wrist_image_left": history.pair("wrist_image_left"),
        },
        "state": {
            "eef_9d": eef_9d,
            "gripper_position": gripper,
            "joint_position": joint_7,
        },
        "language": {
            # GR00T policy 对 language 的 batch/time 期望是 list[list[str]]。
            "annotation.language.language_instruction": [[instruction]],
        },
    }


def joint_action_to_leisaac_tensor(
    action: dict[str, np.ndarray],
    env_device: str,
    fallback_joint: torch.Tensor,
    robot: str,
) -> torch.Tensor:
    """把 GR00T 的 `action.joint_position` 转成 LeIsaac so101leader 6D action。

    用于默认 `--control-mode joint`。

    输入：
    - `action["joint_position"]`：GR00T 返回的 `(B,T,7)` action chunk。
    - `env_device`：Isaac env 的 torch device，例如 `cuda:0`。
    - `fallback_joint`：如果 action 不含 joint_position，就保持当前关节状态。

    输出：
    - SO101: shape `(1,6)`，直接作为 6D joint target。
    - Franka: shape `(1,8)`，7D arm joint target + 1D binary gripper sign。

    局限：
    - 这条路线把 DROID 7D joint action 的前 6 维硬映射到 SO101。
    - 它不理解不同机器人关节语义差异，所以只作为 baseline probe。
    """

    if "joint_position" not in action:
        print("[runner] warning: GR00T action lacks joint_position; holding current pose", flush=True)
        if robot == "franka":
            hold = np.zeros((1, 8), dtype=np.float32)
            current = fallback_joint.detach().cpu().numpy()[0].astype(np.float32)
            hold[0, :7] = current[:7]
            hold[0, 7] = 1.0
            return torch.from_numpy(hold).to(env_device)
        return fallback_joint.clone()

    joint = np.asarray(action["joint_position"], dtype=np.float32)
    if joint.ndim != 3:
        raise ValueError(f"expected joint_position shape (B,T,D), got {joint.shape}")

    if robot == "franka":
        command = np.zeros((1, 8), dtype=np.float32)
        current = fallback_joint.detach().cpu().numpy()[0].astype(np.float32)
        # OXE/DROID config 标记 joint_position 是 relative action；Franka 7DoF
        # 和这个分支的 7D 更匹配，所以这里按 current + delta 解释。
        command[0, :7] = current[:7] + joint[0, 0, :7]
        command[0, 7] = 1.0
        if "gripper_position" in action:
            gripper_value = float(np.asarray(action["gripper_position"], dtype=np.float32)[0, 0, 0])
            command[0, 7] = 1.0 if gripper_value > 0.5 else -1.0
    else:
        command = np.zeros((1, 6), dtype=np.float32)
        usable = min(6, joint.shape[-1])
        command[0, :usable] = joint[0, 0, :usable]
    return torch.from_numpy(command).to(env_device)


def eef_action_to_leisaac_tensor(
    action: dict[str, np.ndarray],
    env_device: str,
    policy_obs: dict[str, torch.Tensor],
    robot: str,
) -> torch.Tensor:
    """把 GR00T 的 `eef_9d + gripper_position` 转成 LeIsaac IK action。

    用于 `--control-mode eef`。

    LeIsaac 的 `mimic_so101leader` action 格式：
        shape `(1, 8)`：
        `[x, y, z, qw, qx, qy, qz, gripper]`

    当前做法：
    - 读取当前 `ee_frame_state` 作为 base pose。
    - 把 GR00T 一步 `eef_9d` 当作相对 EEF delta。
    - 组合成目标 EEF pose。
    - 把 `gripper_position` 作为 gripper 目标值拼到最后一维。
    """

    if "eef_9d" not in action:
        print("[runner] warning: GR00T action lacks eef_9d; holding current EEF pose", flush=True)
        ee_state = policy_obs["ee_frame_state"].detach().cpu().numpy()[0].astype(np.float32)
        joint = policy_obs["joint_pos"].detach().cpu().numpy()[0].astype(np.float32)
        if robot == "franka":
            gripper = np.array([1.0], dtype=np.float32)
        else:
            gripper = joint[5:6].astype(np.float32)
        command = np.concatenate([ee_state[:7], gripper])[None, :]
        return torch.from_numpy(command.astype(np.float32)).to(env_device)

    eef = np.asarray(action["eef_9d"], dtype=np.float32)
    if eef.ndim != 3:
        raise ValueError(f"expected eef_9d shape (B,T,9), got {eef.shape}")

    # 当前末端位姿，LeIsaac 给的是 robot frame 下的 xyz + quat(wxyz)。
    ee_state = policy_obs["ee_frame_state"].detach().cpu().numpy()[0].astype(np.float32)

    # 组合当前末端位姿和 GR00T 返回的一步相对 EEF action。
    target_pos, target_quat = compose_pose_delta(ee_state[:3], ee_state[3:7], eef[0, 0, :9])

    if "gripper_position" in action:
        gripper = float(np.asarray(action["gripper_position"], dtype=np.float32)[0, 0, 0])
    else:
        joint = policy_obs["joint_pos"].detach().cpu().numpy()[0].astype(np.float32)
        gripper = float(joint[7:].mean()) if robot == "franka" and joint.shape[0] > 7 else float(joint[5])

    if robot == "franka":
        # Franka task 的 gripper action 是 BinaryJointPositionAction：
        # 正数表示 open，负数表示 close。GR00T gripper_position 通常在 [0, 1]
        # 一带，这里用 0.5 做最简单的开合阈值。
        gripper = 1.0 if gripper > 0.5 else -1.0

    command = np.concatenate([target_pos, target_quat, np.array([gripper], dtype=np.float32)])[None, :]
    return torch.from_numpy(command.astype(np.float32)).to(env_device)


def tensor_values(tensor: torch.Tensor) -> list[float]:
    """把 torch tensor 的第一个 env 数值转成便于日志打印的 Python list。"""

    return tensor.detach().cpu().numpy()[0].round(4).tolist()


def smart_task_metrics(env: Any, robot_kind: str) -> dict[str, float]:
    """从 Isaac scene 中读取几个用于判断 pick 进展的诊断指标。

    指标：
    - `lego_z_minus_base`：乐高块高度相对于机器人 base 的差值。
    - `jaw_to_lego`：gripper jaw frame 到乐高块 root 的距离。
    - `gripper`：SO101 gripper joint 当前值，或 Franka finger 平均开口。

    这些不是严格成功判据，只是帮助我们观察动作有没有朝正确方向发展。
    """

    lego = env.scene["red_2x4_lego_brick"]
    robot = env.scene["robot"]
    ee_frame = env.scene["ee_frame"]

    lego_pos = lego.data.root_pos_w[0]
    jaw_pos = ee_frame.data.target_pos_w[0, 1]
    robot_base_name = "panda_link0" if robot_kind == "franka" else "base"
    base_idx = robot.data.body_names.index(robot_base_name)
    base_z = robot.data.body_pos_w[0, base_idx, 2]
    if robot_kind == "franka" and robot.data.joint_pos.shape[1] > 7:
        gripper = robot.data.joint_pos[0, 7:].mean()
    else:
        gripper = robot.data.joint_pos[0, -1]

    return {
        "lego_z_minus_base": float((lego_pos[2] - base_z).detach().cpu()),
        "jaw_to_lego": float(torch.linalg.vector_norm(lego_pos - jaw_pos).detach().cpu()),
        "gripper": float(gripper.detach().cpu()),
    }


def print_metrics(env: Any, prefix: str, robot_kind: str) -> None:
    """用统一格式打印 SmartTask 诊断指标。"""

    metrics = smart_task_metrics(env, robot_kind)
    print(
        "[runner] "
        f"{prefix} metrics "
        f"lego_z_minus_base={metrics['lego_z_minus_base']:.4f} "
        f"jaw_to_lego={metrics['jaw_to_lego']:.4f} "
        f"gripper={metrics['gripper']:.4f}",
        flush=True,
    )


def summarize_action(action: dict[str, np.ndarray]) -> str:
    """压缩打印 GR00T action dict 的 key、shape、min、max。

    这个函数的目的不是完整 dump action，而是快速判断：
    - GR00T 是否真的返回了动作。
    - 哪些 action 分支幅度明显。
    - action chunk 的时间长度是多少。
    """

    parts = []
    for key in sorted(action):
        value = np.asarray(action[key])
        if value.size:
            parts.append(
                f"{key}: shape={tuple(value.shape)} min={float(value.min()):.4f} max={float(value.max()):.4f}"
            )
        else:
            parts.append(f"{key}: shape={tuple(value.shape)} empty")
    return "; ".join(parts)


def parse_args() -> argparse.Namespace:
    """解析 Isaac runner 的命令行参数。"""

    parser = argparse.ArgumentParser()

    # 选择实验机器人。so101 是原 LeIsaac SmartTask；franka 是本 experiments
    # 目录新增注册的 Groot-Franka-SmartTask-v0。
    parser.add_argument("--robot", choices=("so101", "franka"), default="so101")

    # IsaacLab task id。None 表示按 --robot 自动选择默认 task。
    parser.add_argument("--task", default=None)

    # GR00T bridge 的地址和端口。默认对应 README 中 bridge 的启动参数。
    parser.add_argument("--bridge-host", default="127.0.0.1")
    parser.add_argument("--bridge-port", type=int, default=5577)

    # 传给 GR00T 的自然语言任务指令。
    parser.add_argument("--instruction", default="Pick up the red 2x4 lego brick.")

    # IsaacLab env 的设备。一般用 cuda。
    parser.add_argument("--device", default="cuda")

    # 两种 action 落地方式：
    # - joint：把 GR00T joint_position 前 6 维当 SO101 关节命令。
    # - eef：把 GR00T eef_9d + gripper_position 转成 LeIsaac IK 命令。
    parser.add_argument(
        "--control-mode",
        choices=("joint", "eef"),
        default="joint",
        help="joint uses GR00T joint_position as SO101 joints; eef uses GR00T eef_9d + gripper_position through LeIsaac IK.",
    )

    # 请求 GR00T 的次数。每次请求会返回一个 action chunk。
    parser.add_argument("--max-policy-calls", type=int, default=1)

    # 每个 action chunk 执行多少步。0 表示执行完整 chunk。
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=0,
        help="Number of action steps to execute from each GR00T chunk. 0 means execute the full returned chunk.",
    )

    # 在第一次推理前先积累多少帧相机历史。
    parser.add_argument("--warmup-frames", type=int, default=16)

    # bridge socket 请求超时时间。
    parser.add_argument("--timeout-s", type=float, default=120.0)

    # IsaacLab env seed；默认 None 表示使用环境默认随机性。
    parser.add_argument("--seed", type=int, default=None)

    # 如果 env 报 terminated/timed_out，是否仍继续执行动作。
    parser.add_argument(
        "--ignore-terminations",
        action="store_true",
        help="Keep stepping even if the current SmartTask success termination fires.",
    )

    # SmartTask 原生 success termination 目前过松，所以默认禁用。
    parser.add_argument(
        "--use-env-success-termination",
        action="store_true",
        help="Use SmartTask's built-in success termination. Disabled by default because it is currently too loose.",
    )

    # viewport 模式下每个 step 后 sleep 一下，让动作肉眼可见。
    parser.add_argument(
        "--render-sleep-s",
        type=float,
        default=0.01,
        help="Sleep between simulated action steps so motion is visible in viewport mode.",
    )

    # Isaac Sim 是否 headless。`BooleanOptionalAction` 会自动支持
    # `--headless` 和 `--no-headless` 两个相反开关。
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use --no-headless to open the Isaac Sim viewport.",
    )

    # 主动控制结束后继续保持窗口多久，方便观察最终状态。
    parser.add_argument(
        "--keep-open-s",
        type=float,
        default=0.0,
        help="Keep Isaac Sim open for this many seconds after the run, useful with --no-headless.",
    )

    # 只构造 observation 并请求 GR00T，不执行动作。
    parser.add_argument("--dry-run", action="store_true", help="Build and send one obs, but do not step actions.")

    args = parser.parse_args()
    if args.task is None:
        args.task = "Groot-Franka-SmartTask-v0" if args.robot == "franka" else "LeIsaac-SO101-SmartTask-v0"
    return args


def keep_open(simulation_app: Any, env: Any, seconds: float) -> None:
    """主动控制结束后继续渲染 Isaac Sim viewport。"""

    if seconds <= 0:
        return

    end_time = time.time() + seconds
    print(f"[runner] keeping Isaac Sim open for {seconds:.1f}s", flush=True)

    while simulation_app.is_running() and time.time() < end_time:
        # `env.sim.render()` 手动推进渲染，让 viewport 保持刷新。
        env.sim.render()
        time.sleep(1.0 / 30.0)


def action_chunk_len(action: dict[str, np.ndarray]) -> int:
    """从 GR00T action dict 中推断 action chunk 的时间长度 T。"""

    for key in ("joint_position", "eef_9d", "gripper_position"):
        if key in action:
            arr = np.asarray(action[key])
            if arr.ndim >= 2:
                return int(arr.shape[1])
    raise ValueError(f"cannot infer action chunk length from keys: {sorted(action.keys())}")


def main() -> None:
    """Isaac runner 主流程。"""

    args = parse_args()

    # AppLauncher 必须尽早创建；它会启动 Isaac Sim app。
    # `headless` 控制是否打开 viewport。
    # `enable_cameras=True` 是必须的，否则 camera observation 不会渲染。
    app_launcher = AppLauncher({"headless": args.headless, "enable_cameras": True})
    simulation_app = app_launcher.app

    # 这些 import 依赖 Isaac Sim / IsaacLab 初始化，所以放在 AppLauncher 之后。
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim

    # import leisaac 的副作用是注册 LeIsaac 的 gymnasium task id。
    import leisaac  # noqa: F401
    # import franka_smart_task 的副作用是注册 experiments 里的 Franka task id。
    import franka_smart_task  # noqa: F401

    # 读取 task 默认配置，并指定 device / num_envs。
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)

    # control-mode 决定 LeIsaac action manager 使用哪套 action cfg：
    # - so101leader：JointPositionAction，action 维度 6。
    # - mimic_so101leader：Differential IK pose + gripper，action 维度 8。
    if args.robot == "franka":
        teleop_device = "franka_ik" if args.control_mode == "eef" else "franka_joint"
    else:
        teleop_device = "mimic_so101leader" if args.control_mode == "eef" else "so101leader"
    env_cfg.use_teleop_device(teleop_device)

    env_cfg.seed = args.seed

    # 关闭 recorder，避免这个实验无意中写 LeIsaac 数据集。
    env_cfg.recorders = None

    # 关闭 time_out termination，方便我们自己控制运行长度。
    env_cfg.terminations.time_out = None

    # SmartTask 当前 success 判断太松，会出现几乎没动就成功的问题。
    # 默认禁用它；用户显式加 --use-env-success-termination 时才恢复。
    if not args.use_env_success_termination and hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None

    # 创建 gymnasium env，并拿 unwrapped 环境方便访问 scene、sim、cfg 等属性。
    env = gym.make(args.task, cfg=env_cfg).unwrapped

    # 相机历史缓存，用于构造 GR00T 的两帧 video 输入。
    history = FrameHistory(horizon=max(args.warmup_frames, 2))

    try:
        # 先 ping bridge，确认 GR00T 进程已经启动，并打印 modality schema。
        print("[runner] ping bridge", flush=True)
        ping = request(args.bridge_host, args.bridge_port, {"endpoint": "ping"}, args.timeout_s)
        if not ping.get("ok"):
            raise RuntimeError(ping)

        print(f"[runner] bridge modality: {ping.get('modality')}", flush=True)
        print(
            f"[runner] robot={args.robot} task={args.task} control mode={args.control_mode} "
            f"teleop_device={teleop_device}",
            flush=True,
        )

        # reset Isaac env，拿到第一帧 observation。
        obs, _ = env.reset()
        policy_obs = obs["policy"]

        print(f"[runner] initial joint_pos values={tensor_values(policy_obs['joint_pos'])}", flush=True)
        print_metrics(env, "initial", args.robot)

        # warmup：重复把当前 observation 放入 history，确保 history 至少有两帧。
        for _ in range(args.warmup_frames):
            build_oxe_observation(policy_obs, history, args.instruction, args.robot)

        stop_run = False

        # 外层循环：每次向 GR00T 请求一个 action chunk。
        for call_idx in range(args.max_policy_calls):
            groot_obs = build_oxe_observation(
                policy_obs,
                history,
                args.instruction,
                args.robot,
            )

            print(f"[runner] request action {call_idx + 1}/{args.max_policy_calls}", flush=True)

            # 向 bridge 发起 get_action 请求。这里的 observation 会被 msgpack + numpy
            # 编码成 bytes，经本地 TCP 发给 GR00T venv 里的 bridge。
            reply = request(
                args.bridge_host,
                args.bridge_port,
                {"endpoint": "get_action", "observation": groot_obs},
                args.timeout_s,
            )

            if not reply.get("ok"):
                raise RuntimeError(reply.get("traceback") or reply.get("error"))

            action = reply["action"]
            print(f"[runner] action keys: {sorted(action.keys())}", flush=True)
            print(f"[runner] action summary: {summarize_action(action)}", flush=True)

            if args.dry_run:
                print_metrics(env, "dry-run", args.robot)
                continue

            # 决定这个 chunk 执行多少步。
            # 默认 action_horizon=0，表示完整执行 GR00T 返回的 40 步。
            num_action_steps = action_chunk_len(action) if args.action_horizon <= 0 else args.action_horizon
            print(f"[runner] executing {num_action_steps} action steps from this chunk", flush=True)

            # 内层循环：逐步执行 action chunk。
            for step_idx in range(num_action_steps):
                if args.control_mode == "eef":
                    # EEF 路线：取当前时间步的 eef_9d 和 gripper_position。
                    step_action = {}
                    if "eef_9d" in action and action["eef_9d"].shape[1] > step_idx:
                        step_action["eef_9d"] = action["eef_9d"][:, step_idx : step_idx + 1, :]
                    if "gripper_position" in action and action["gripper_position"].shape[1] > step_idx:
                        step_action["gripper_position"] = action["gripper_position"][:, step_idx : step_idx + 1, :]
                    command = eef_action_to_leisaac_tensor(step_action, env.device, policy_obs, args.robot)
                else:
                    # Joint 路线：取当前时间步的 joint_position。
                    step_action = {}
                    if "joint_position" in action and action["joint_position"].shape[1] > step_idx:
                        step_action["joint_position"] = action["joint_position"][:, step_idx : step_idx + 1, :]
                    if "gripper_position" in action and action["gripper_position"].shape[1] > step_idx:
                        step_action["gripper_position"] = action["gripper_position"][:, step_idx : step_idx + 1, :]
                    if not step_action:
                        step_action = action
                    command = joint_action_to_leisaac_tensor(step_action, env.device, policy_obs["joint_pos"], args.robot)

                # 只在每个 chunk 的第一步打印 command，避免日志过大。
                if step_idx == 0:
                    cmd_np = command.detach().cpu().numpy()
                    if args.control_mode == "eef":
                        # EEF mode 下，当前值是 `[xyz, quat, gripper]`。
                        joint_np = policy_obs["joint_pos"].detach().cpu().numpy()
                        if args.robot == "franka":
                            gripper_np = np.ones((joint_np.shape[0], 1), dtype=np.float32)
                        else:
                            gripper_np = joint_np[:, 5:6]
                        cur_np = np.concatenate(
                            [
                                policy_obs["ee_frame_state"].detach().cpu().numpy()[:, :7],
                                gripper_np,
                            ],
                            axis=1,
                        )
                    elif args.robot == "franka":
                        joint_np = policy_obs["joint_pos"].detach().cpu().numpy()
                        cur_np = np.concatenate(
                            [joint_np[:, :7], np.ones((joint_np.shape[0], 1), dtype=np.float32)],
                            axis=1,
                        )
                    else:
                        # Joint mode 下，当前值就是 6D joint_pos。
                        cur_np = policy_obs["joint_pos"].detach().cpu().numpy()

                    delta_np = cmd_np - cur_np
                    print(
                        "[runner] first LeIsaac command "
                        f"shape={tuple(cmd_np.shape)} min={float(cmd_np.min()):.4f} max={float(cmd_np.max()):.4f} "
                        f"values={cmd_np[0].round(4).tolist()} "
                        f"delta={delta_np[0].round(4).tolist()}",
                        flush=True,
                    )

                # LeIsaac 原本会根据 gripper 距离物体动态调整 effort limit。
                # 因为我们绕开了 LeIsaac 的 teleop loop，所以这里手动调用一次。
                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, teleop_device)

                # 真正把 action 送进 IsaacLab env。
                # 返回值：obs, reward, terminated, timed_out, info。
                obs, _, terminated, timed_out, _ = env.step(command)
                policy_obs = obs["policy"]

                # 用新 observation 更新相机历史。
                build_oxe_observation(policy_obs, history, args.instruction, args.robot)

                if terminated[0] or timed_out[0]:
                    print(f"[runner] episode ended: terminated={terminated[0]} timed_out={timed_out[0]}", flush=True)
                    if not args.ignore_terminations:
                        stop_run = True
                        break

                if not simulation_app.is_running():
                    stop_run = True
                    break

                if args.render_sleep_s > 0:
                    time.sleep(args.render_sleep_s)

            print(f"[runner] latest joint_pos values={tensor_values(policy_obs['joint_pos'])}", flush=True)
            print_metrics(env, f"after policy call {call_idx + 1}", args.robot)

            if stop_run:
                break

        keep_open(simulation_app, env, args.keep_open_s)

    finally:
        # 无论正常结束还是异常退出，都关闭 env 和 Isaac app。
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
