"""在 LeIsaac SmartTask 中调用 GR00T N1.7 bridge，并闭环驱动 Isaac 机器人。

运行环境：
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate leisaac
    export LEISAAC_ROOT="$HOME/LeIsaac"
    python run_smart_task_closed_loop.py ...

这个文件是整个实验的 Isaac 侧主程序。它做四件事：

1. 启动 Isaac Sim / IsaacLab，并加载 LeIsaac 已经注册好的 SmartTask。
2. 从 SmartTask observation 中取出图像、关节状态、末端位姿。
3. 按 deployment mode 把这些 observation 改写成 GR00T 需要的 schema：
   zero-shot 使用 OXE/DROID；SO101 微调 checkpoint 使用 NEW_EMBODIMENT。
4. 调用 GR00T bridge 拿 action，再把 action 转成 LeIsaac 可以执行的命令。

注意：
    这里不是把 SO101 伪装成“真正的 DROID/Franka”。它只是一个 zero-shot
    探针：尽量复用 LeIsaac 的场景和 USD 资产，绕开 LeRobot policy stack，
    看 GR00T N1.7 base model 在这个场景里会给出什么样的动作。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


# 当前文件所在目录：
# $SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task
SCRIPT_DIR = Path(__file__).resolve().parent

# 当前实验目录：
# $SMART_PROJECT/experiments/groot_n17_isaac_smart_task
EXPERIMENT_ROOT = SCRIPT_DIR.parent

# 项目根目录：
# $SMART_PROJECT
REPO_ROOT = EXPERIMENT_ROOT.parents[1]

# LeIsaac 的 repo 根目录。本实验默认优先使用当前项目里的同事 LeIsaac copy；
# 如果确实要对比外部 checkout，再显式设置 LEISAAC_ROOT。
LOCAL_LEISAAC_ROOT = REPO_ROOT / "leisaac"
DEFAULT_LEISAAC_ROOT = LOCAL_LEISAAC_ROOT if LOCAL_LEISAAC_ROOT.exists() else Path.home() / "LeIsaac"
LEISAAC_ROOT = Path(os.environ.get("LEISAAC_ROOT", DEFAULT_LEISAAC_ROOT)).expanduser().resolve()

# LeIsaac 内部的 ASSETS_ROOT 会尝试从 git root 推断；当前 workspace 根 .git
# 可能不是有效仓库，所以这里显式固定到所选 LeIsaac repo 的 assets。
os.environ.setdefault("LEISAAC_ASSETS_ROOT", str((LEISAAC_ROOT / "assets").resolve()))

# LeIsaac 的 Python package 源码目录。运行脚本时 README 里也会设置 PYTHONPATH，
# 这里再插一次，方便直接运行或调试。
LEISAAC_SRC = LEISAAC_ROOT / "source" / "leisaac"

# LeIsaac 仓库自带的 IsaacLab 依赖目录。优先把这份 source 放进 sys.path，
# 避免无意中依赖机器上另一份 IsaacLab 源码。
LEISAAC_ISAACLAB_SRC = LEISAAC_ROOT / "dependencies" / "IsaacLab" / "source"
LEISAAC_ISAACLAB_PACKAGES = (
    LEISAAC_ISAACLAB_SRC / "isaaclab",
    LEISAAC_ISAACLAB_SRC / "isaaclab_assets",
    LEISAAC_ISAACLAB_SRC / "isaaclab_tasks",
    LEISAAC_ISAACLAB_SRC / "isaaclab_mimic",
)

# Current colleague LeIsaac and the older leisaac2 copy expose the same policy
# observation keys, but their camera semantics differ:
# - leisaac-current: camera1=left, camera2=wrist, camera3=front/top.
# - leisaac2-legacy: camera1=front/top, camera2=left, camera3=template wrist.
# The runner prints the resolved mapping so the log shows what GR00T saw.
CAMERA_PROFILE_DEFAULTS = {
    "leisaac-current": {
        "exterior": "camera3",
        "top": "camera3",
        "wrist": "camera2",
    },
    "leisaac2-legacy": {
        "exterior": "camera1",
        "top": "camera1",
        "wrist": "camera3",
    },
    "franka-current": {
        "exterior": "camera1",
        "top": "camera1",
        "wrist": "camera2",
    },
}

TARGET_OBJECT_CANDIDATES = (
    "red_2x4_lego_brick_pick",
    "red_2x4_lego_brick",
)
SMART_TARGET_OBJECT_KEY = "red_2x4_lego_brick_pick"
SMART_TARGET_CUBOID_SIZE = (0.0318, 0.0158, 0.0096)
SMART_TARGET_CUBOID_MASS = 0.02
SMART_TARGET_MANAGED_PRIM_PATH = f"{{ENV_REGEX_NS}}/Scene/{SMART_TARGET_OBJECT_KEY}_managed"

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

    def latest(self, key: str) -> np.ndarray:
        """Return the latest frame as `(B=1, T=1, H, W, C)`."""

        frame = self._frames[key][-1]
        return frame[None, None, ...].astype(np.uint8)


def get_policy_camera(policy_obs: dict[str, torch.Tensor], key: str, role: str) -> torch.Tensor:
    """Return a selected policy camera tensor with a useful error message."""

    if key not in policy_obs:
        camera_keys = sorted(name for name in policy_obs if name.startswith("camera"))
        raise KeyError(
            f"{role} camera key {key!r} is not in policy obs; "
            f"available camera keys are {camera_keys}"
        )
    return policy_obs[key]


def _sensor_cfg_prim_path(sensor: Any) -> str:
    cfg = getattr(sensor, "cfg", None)
    return str(getattr(cfg, "prim_path", "") or "")


def detect_camera_profile(env: Any) -> str:
    """Infer whether the active LeIsaac task uses current or legacy camera names."""

    scene = getattr(env, "scene", None)
    scene_sensors = getattr(scene, "sensors", {}) if scene is not None else {}

    # The legacy SmartTask exposes the template wrist sensor as scene sensor
    # "wrist" and maps policy camera3 to it.
    if "wrist" in scene_sensors:
        return "leisaac2-legacy"

    # Current colleague LeIsaac renamed the robot-mounted wrist sensor to camera2.
    camera2_prim = _sensor_cfg_prim_path(scene_sensors.get("camera2"))
    if "wrist" in camera2_prim or "/Robot/" in camera2_prim:
        return "leisaac-current"

    # Prefer the current layout when auto-detection is inconclusive because the
    # project-level leisaac directory is expected to track the latest colleague code.
    return "leisaac-current"


def resolve_camera_mapping(args: argparse.Namespace, env: Any) -> tuple[str, dict[str, str]]:
    """Resolve camera profile defaults plus explicit per-role overrides."""

    profile = args.camera_profile
    if profile == "auto":
        profile = "franka-current" if args.robot == "franka" else detect_camera_profile(env)

    defaults = CAMERA_PROFILE_DEFAULTS[profile]
    mapping = {
        "exterior": args.exterior_camera_key or defaults["exterior"],
        "top": args.top_camera_key or defaults["top"],
        "wrist": args.wrist_camera_key or defaults["wrist"],
    }
    return profile, mapping


def rigid_object_names(env: Any) -> list[str]:
    """Return rigid object names from the active IsaacLab scene."""

    scene = getattr(env, "scene", None)
    if scene is None:
        return []

    rigid_objects = getattr(scene, "rigid_objects", None)
    if isinstance(rigid_objects, dict):
        return sorted(rigid_objects.keys())

    private_rigid_objects = getattr(scene, "_rigid_objects", None)
    if isinstance(private_rigid_objects, dict):
        return sorted(private_rigid_objects.keys())

    return []


def resolve_target_object_name(env: Any, requested_name: str) -> str:
    """Resolve the SmartTask target object across current and legacy scenes."""

    names = rigid_object_names(env)
    if requested_name != "auto":
        if requested_name in names:
            return requested_name
        raise KeyError(
            f"target object {requested_name!r} is not in scene rigid objects; "
            f"available rigid objects are {names}"
        )

    for candidate in TARGET_OBJECT_CANDIDATES:
        if candidate in names:
            return candidate

    fuzzy_matches = [
        name for name in names
        if "red" in name.lower() and "lego" in name.lower() and "brick" in name.lower()
    ]
    if fuzzy_matches:
        return sorted(fuzzy_matches)[0]

    raise KeyError(
        "could not auto-detect SmartTask target object; "
        f"tried {list(TARGET_OBJECT_CANDIDATES)}, available rigid objects are {names}"
    )


def is_leisaac_smart_task(task: str) -> bool:
    """Return whether this runner task is a LeIsaac SO101 SmartTask variant."""

    return task.startswith("LeIsaac-SO101-SmartTask")


def resolve_smart_scene_usd(value: str) -> Path:
    """Resolve the scene USD used by LeIsaac SmartTask in this runner process."""

    scene_dir = LEISAAC_ROOT / "assets" / "scenes" / "smart_scene"
    if value == "auto":
        portable_scene = scene_dir / "scene_portable.usda"
        source_scene = scene_dir / "scene.usd"
        resolved = portable_scene if portable_scene.exists() else source_scene
    else:
        resolved = Path(value).expanduser()
        if not resolved.is_absolute():
            resolved = (REPO_ROOT / resolved).resolve()
        else:
            resolved = resolved.resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"SmartTask scene USD does not exist: {resolved}")
    return resolved


def smart_target_pose_from_scene(scene_usd_path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Read the target pose from the SmartTask scene, falling back to a table-top pose."""

    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(scene_usd_path))
        if stage is not None:
            for prim_path in (
                f"/world/{SMART_TARGET_OBJECT_KEY}",
                f"/World/{SMART_TARGET_OBJECT_KEY}",
                f"/{SMART_TARGET_OBJECT_KEY}",
            ):
                prim = stage.GetPrimAtPath(prim_path)
                if not prim or not prim.IsValid():
                    continue
                matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                if matrix.Orthonormalize(issueWarning=True):
                    rot = matrix.ExtractRotationQuat()
                    quat = (rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2])
                else:
                    quat = (1.0, 0.0, 0.0, 0.0)
                pos = matrix.ExtractTranslation()
                return (pos[0], pos[1], pos[2]), quat
    except Exception as exc:  # noqa: BLE001 - asset fallback should still be usable.
        print(f"[runner] warning: failed reading SmartTask target pose from {scene_usd_path}: {exc}", flush=True)

    return (0.0, 0.0, 0.02), (1.0, 0.0, 0.0, 0.0)


def add_smart_target_cfg(env_cfg: Any, target_asset: str, target_prim_path: str, scene_usd_path: Path) -> None:
    """Register SmartTask's target object without modifying the LeIsaac source tree."""

    if target_asset == "scene":
        return

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    if target_asset == "cuboid":
        spawn_cfg = sim_utils.CuboidCfg(
            size=SMART_TARGET_CUBOID_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=SMART_TARGET_CUBOID_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.7),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.78, 0.05, 0.02),
                roughness=0.55,
            ),
        )
    else:
        target_usd_path = Path(target_asset).expanduser()
        if not target_usd_path.is_absolute():
            target_usd_path = (REPO_ROOT / target_usd_path).resolve()
        else:
            target_usd_path = target_usd_path.resolve()
        if not target_usd_path.exists():
            raise FileNotFoundError(f"--smart-target-asset USD path does not exist: {target_usd_path}")
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=str(target_usd_path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=SMART_TARGET_CUBOID_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )

    pos, rot = smart_target_pose_from_scene(scene_usd_path)
    setattr(
        env_cfg.scene,
        SMART_TARGET_OBJECT_KEY,
        RigidObjectCfg(
            prim_path=target_prim_path,
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
        ),
    )


def install_smart_task_asset_patch(args: argparse.Namespace) -> Path | None:
    """Patch LeIsaac SmartTask config in-process for portable scene and target fallback."""

    if not is_leisaac_smart_task(args.task):
        return None

    scene_usd_path = resolve_smart_scene_usd(args.smart_scene_usd)

    import leisaac.assets.scenes.smart_scene as smart_scene
    import leisaac.tasks.smart_task.smart_task_env_cfg as smart_task_cfg

    original_parse = smart_task_cfg.parse_usd_and_create_subassets

    def parse_usd_and_create_subassets_with_target(usd_path, env_cfg, *parse_args, **parse_kwargs):
        result = original_parse(str(scene_usd_path), env_cfg, *parse_args, **parse_kwargs)
        add_smart_target_cfg(env_cfg, args.smart_target_asset, args.smart_target_prim_path, scene_usd_path)
        return result

    # Keep the imported module constants consistent for __post_init__ and debug logs.
    smart_scene.SMART_SCENE_USD_PATH = str(scene_usd_path)
    smart_scene.SMART_SCENE_CFG.spawn.usd_path = str(scene_usd_path)
    smart_task_cfg.SMART_SCENE_USD_PATH = str(scene_usd_path)
    smart_task_cfg.SMART_SCENE_CFG.spawn.usd_path = str(scene_usd_path)
    smart_task_cfg.parse_usd_and_create_subassets = parse_usd_and_create_subassets_with_target

    print(
        "[runner] SmartTask asset patch "
        f"scene_usd={scene_usd_path} target_asset={args.smart_target_asset} "
        f"target_prim_path={args.smart_target_prim_path}",
        flush=True,
    )
    return scene_usd_path


def build_oxe_observation(
    policy_obs: dict[str, torch.Tensor],
    history: FrameHistory,
    instruction: str,
    robot: str,
    camera_mapping: dict[str, str],
) -> dict[str, Any]:
    """把 LeIsaac SmartTask observation 映射到 GR00T N1.7 OXE/DROID schema。

    输入：
    - `policy_obs`：LeIsaac env 返回的 `obs["policy"]`。
    - `history`：相机历史缓存。
    - `instruction`：语言指令，例如 "Pick up the red 2x4 lego brick."

    输出：
    - GR00T bridge 可直接传给 `policy.get_action()` 的 nested observation dict。

    SO101 mapping depends on `--camera-profile`:
    - `camera_mapping["exterior"]` -> `video.exterior_image_1_left`
    - `camera_mapping["wrist"]` -> `video.wrist_image_left`
    - `ee_frame_state` -> `state.eef_9d`
    - SO101 6D joint state pad 到 7D -> `state.joint_position`
    - SO101 gripper joint -> `state.gripper_position`

    Franka mapping uses the same selected exterior/wrist camera keys, then:
    - Franka 前 7 个 panda_joint -> `state.joint_position`。
    - 两个 panda_finger joint 的平均值 -> `state.gripper_position`。

    注意：
    - 这只是 probe，不表示 SO101 就是 DROID robot。
    - joint_position 的 7D 是为了符合 OXE/DROID schema；SO101 实际只有 6 个
      joint，其中第 6 个是 gripper。
    """

    exterior = image_to_numpy(get_policy_camera(policy_obs, camera_mapping["exterior"], "exterior/top"))
    wrist = image_to_numpy(get_policy_camera(policy_obs, camera_mapping["wrist"], "wrist"))

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


def build_so101_new_embodiment_observation(
    policy_obs: dict[str, torch.Tensor],
    history: FrameHistory,
    instruction: str,
    robot: str,
    camera_mapping: dict[str, str],
) -> dict[str, Any]:
    """Map LeIsaac SO101 observations to the trained NEW_EMBODIMENT schema."""

    if robot != "so101":
        raise ValueError("SO101 NEW_EMBODIMENT schema only supports --robot so101")

    top = image_to_numpy(get_policy_camera(policy_obs, camera_mapping["top"], "top"))
    wrist = image_to_numpy(get_policy_camera(policy_obs, camera_mapping["wrist"], "wrist"))
    history.push("top", top)
    history.push("wrist", wrist)

    joint_all = policy_obs["joint_pos"].detach().cpu().numpy()[0].astype(np.float32)
    single_arm = np.zeros((1, 1, 5), dtype=np.float32)
    single_arm[0, 0, : min(5, joint_all.shape[0])] = joint_all[:5]
    gripper_value = float(joint_all[5]) if joint_all.shape[0] > 5 else 0.0
    gripper = np.array([[[gripper_value]]], dtype=np.float32)

    return {
        "video": {
            "top": history.latest("top"),
            "wrist": history.latest("wrist"),
        },
        "state": {
            "single_arm": single_arm,
            "gripper": gripper,
        },
        "language": {
            "annotation.human.task_description": [[instruction]],
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


def so101_new_embodiment_action_to_leisaac_tensor(
    action: dict[str, np.ndarray],
    env_device: str,
    fallback_joint: torch.Tensor,
    arm_delta_scale: float = 1.0,
    max_arm_delta: float | None = None,
    gripper_min: float | None = None,
    gripper_max: float | None = None,
) -> torch.Tensor:
    """Convert trained SO101 NEW_EMBODIMENT actions to LeIsaac 6D joint targets."""

    current = fallback_joint.detach().cpu().numpy()[0].astype(np.float32)
    command = current.copy()

    if "single_arm" in action:
        arm = np.asarray(action["single_arm"], dtype=np.float32)
        if arm.ndim != 3:
            raise ValueError(f"expected single_arm shape (B,T,D), got {arm.shape}")
        usable = min(5, arm.shape[-1])
        # The SO101 config marks single_arm as relative joint action, so convert
        # it into the absolute joint target expected by LeIsaac JointPositionAction.
        arm_delta = arm[0, 0, :usable] * float(arm_delta_scale)
        if max_arm_delta is not None:
            bound = abs(float(max_arm_delta))
            arm_delta = np.clip(arm_delta, -bound, bound)
        command[:usable] = current[:usable] + arm_delta
    else:
        print("[runner] warning: GR00T action lacks single_arm; holding arm joints", flush=True)

    if "gripper" in action:
        gripper = np.asarray(action["gripper"], dtype=np.float32)
        if gripper.ndim != 3:
            raise ValueError(f"expected gripper shape (B,T,D), got {gripper.shape}")
        gripper_value = float(gripper[0, 0, 0])
        if gripper_min is not None or gripper_max is not None:
            lower = -np.inf if gripper_min is None else float(gripper_min)
            upper = np.inf if gripper_max is None else float(gripper_max)
            gripper_value = float(np.clip(gripper_value, lower, upper))
        command[5] = gripper_value
    else:
        print("[runner] warning: GR00T action lacks gripper; holding gripper", flush=True)

    return torch.from_numpy(command[None, :]).to(env_device)


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


def tensor_first_row(tensor: torch.Tensor) -> np.ndarray:
    """把 torch tensor 的第一个 env 拿到 CPU numpy，供诊断日志使用。"""

    return tensor.detach().cpu().numpy()[0]


def format_vec(values: np.ndarray, precision: int = 4) -> str:
    """把短向量格式化成日志友好的 `[x, y, z]` 字符串。"""

    rounded = np.asarray(values, dtype=np.float64).round(precision).tolist()
    return str(rounded)


def get_usd_stage(env: Any) -> Any | None:
    """取得当前 Isaac Sim 的 USD stage。

    优先从 `env.sim.stage` 读取；如果 IsaacLab 版本没有暴露这个属性，再退回
    Omniverse 全局 USD context。这个函数只用于 debug，不影响正常控制。
    """

    stage = getattr(getattr(env, "sim", None), "stage", None)
    if stage is not None:
        return stage

    try:
        import omni.usd

        return omni.usd.get_context().get_stage()
    except Exception:  # noqa: BLE001 - debug helper should be best-effort.
        return None


def usd_world_pose(stage: Any, prim_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """读取某个 USD prim 的世界位姿。

    返回：
    - `pos_w`：世界坐标位置，shape `(3,)`。
    - `quat_wxyz`：世界坐标旋转四元数，shape `(4,)`，顺序 `(w, x, y, z)`。

    这里直接查 USD Xform，而不是依赖 camera sensor 的 `data.pos_w`，因为有些
    IsaacLab 版本默认 `update_latest_camera_pose=False`，sensor data 不一定实时更新。
    """

    if stage is None or not prim_path:
        return None

    try:
        from pxr import UsdGeom

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None

        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()
        rotation = matrix.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        pos_w = np.array([translation[0], translation[1], translation[2]], dtype=np.float64)
        quat_wxyz = np.array(
            [rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2]],
            dtype=np.float64,
        )
        return pos_w, quat_wxyz
    except Exception as exc:  # noqa: BLE001 - keep debug non-fatal.
        print(f"[runner] camera-debug warning: failed reading USD pose for {prim_path}: {exc}", flush=True)
        return None


def usd_camera_prim_paths(stage: Any) -> list[str]:
    """列出当前 USD stage 中所有 UsdGeom.Camera prim 的路径。"""

    if stage is None:
        return []

    try:
        from pxr import UsdGeom

        return [prim.GetPath().pathString for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)]
    except Exception as exc:  # noqa: BLE001 - keep debug non-fatal.
        print(f"[runner] camera-debug warning: failed traversing USD cameras: {exc}", flush=True)
        return []


def sensor_prim_paths(sensor: Any) -> list[str]:
    """从 IsaacLab sensor 对象中尽量取出实例化后的 prim path。"""

    view = getattr(sensor, "_view", None)
    prim_paths = getattr(view, "prim_paths", None)
    if prim_paths is None:
        return []
    return [str(path) for path in prim_paths]


def save_policy_camera_frames(policy_obs: dict[str, torch.Tensor], output_dir: str | Path) -> None:
    """把当前 policy observation 里的 camera 图像保存成 png。

    这一步用于确认“GR00T 实际拿到的图像内容”。它保存的是 observation tensor，
    不是 viewport 截屏，所以最适合排查相机是否被挡住、方向是否反了、腕部相机
    是否跟着手爪移动。
    """

    output_path = Path(output_dir).expanduser()
    if not output_path.is_absolute():
        output_path = SCRIPT_DIR / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001 - PIL may be absent in minimal envs.
        print(f"[runner] camera-debug warning: cannot save png frames because PIL import failed: {exc}", flush=True)
        return

    saved = []
    for key in sorted(policy_obs):
        if not key.startswith("camera"):
            continue
        frame = image_to_numpy(policy_obs[key])
        file_path = output_path / f"{key}.png"
        Image.fromarray(frame).save(file_path)
        saved.append(str(file_path))

    print(f"[runner] camera-debug saved policy camera frames: {saved}", flush=True)


def print_camera_debug(
    env: Any,
    policy_obs: dict[str, torch.Tensor],
    robot_kind: str,
    frame_dir: str | None,
    camera_mapping: dict[str, str] | None = None,
    target_object_name: str | None = None,
) -> None:
    """打印 SmartTask/Franka 相机诊断信息。

    诊断目标：
    - 确认 policy observation 中实际有哪些 camera key。
    - 确认 InteractiveScene 中实例化了哪些 camera sensor。
    - 确认 USD stage 中为什么会看到多个 camera prim。
    - 对 Franka 尤其确认 `wrist` camera 和 `panda_hand` 的世界位置关系。
    """

    print("[runner] camera-debug begin", flush=True)
    if camera_mapping is not None:
        print(f"[runner] camera-debug selected mapping: {camera_mapping}", flush=True)

    for key in sorted(policy_obs):
        if not key.startswith("camera"):
            continue
        tensor = policy_obs[key]
        frame = image_to_numpy(tensor)
        print(
            "[runner] camera-debug policy "
            f"{key}: tensor_shape={tuple(tensor.shape)} frame_shape={tuple(frame.shape)} "
            f"dtype={frame.dtype} min={int(frame.min())} max={int(frame.max())} "
            f"mean={float(frame.mean()):.2f} std={float(frame.std()):.2f}",
            flush=True,
        )

    scene_sensors = getattr(env.scene, "sensors", {})
    print(f"[runner] camera-debug scene sensor names: {sorted(scene_sensors.keys())}", flush=True)
    for name, sensor in sorted(scene_sensors.items()):
        cfg = getattr(sensor, "cfg", None)
        cfg_prim = getattr(cfg, "prim_path", None)
        paths = sensor_prim_paths(sensor)
        print(
            "[runner] camera-debug sensor "
            f"{name}: class={sensor.__class__.__name__} cfg_prim_path={cfg_prim} instantiated_paths={paths}",
            flush=True,
        )

    stage = get_usd_stage(env)
    camera_paths = usd_camera_prim_paths(stage)
    print(f"[runner] camera-debug USD camera count={len(camera_paths)}", flush=True)
    for path in camera_paths:
        pose = usd_world_pose(stage, path)
        if pose is None:
            print(f"[runner] camera-debug USD camera {path}: pose=<unavailable>", flush=True)
            continue
        pos_w, quat_wxyz = pose
        print(
            "[runner] camera-debug USD camera "
            f"{path}: pos_w={format_vec(pos_w)} quat_wxyz={format_vec(quat_wxyz)}",
            flush=True,
        )

    robot = env.scene["robot"]
    print(f"[runner] camera-debug robot body names: {robot.data.body_names}", flush=True)

    body_names_to_report = ["panda_link0", "panda_hand", "panda_leftfinger", "panda_rightfinger"]
    if robot_kind != "franka":
        body_names_to_report = ["base", "gripper", "jaw"]

    body_positions: dict[str, np.ndarray] = {}
    for body_name in body_names_to_report:
        if body_name not in robot.data.body_names:
            continue
        body_idx = robot.data.body_names.index(body_name)
        pos_w = tensor_first_row(robot.data.body_pos_w[:, body_idx, :])
        quat_w = tensor_first_row(robot.data.body_quat_w[:, body_idx, :])
        body_positions[body_name] = pos_w
        print(
            "[runner] camera-debug body "
            f"{body_name}: pos_w={format_vec(pos_w)} quat_wxyz={format_vec(quat_w)}",
            flush=True,
        )

    wrist_sensor_name = (camera_mapping or {}).get("wrist")
    if wrist_sensor_name not in scene_sensors and "wrist" in scene_sensors:
        wrist_sensor_name = "wrist"
    if wrist_sensor_name in scene_sensors:
        wrist_sensor = scene_sensors[wrist_sensor_name]
        wrist_paths = sensor_prim_paths(wrist_sensor)
        wrist_path = wrist_paths[0] if wrist_paths else getattr(wrist_sensor.cfg, "prim_path", "")
        wrist_pose = usd_world_pose(stage, wrist_path)
        hand_pos = body_positions.get("panda_hand")
        if hand_pos is None:
            hand_pos = body_positions.get("gripper")
        if wrist_pose is not None and hand_pos is not None:
            wrist_pos, _ = wrist_pose
            delta = wrist_pos - hand_pos
            print(
                "[runner] camera-debug wrist_to_hand "
                f"sensor={wrist_sensor_name} wrist_path={wrist_path} "
                f"delta_pos_w={format_vec(delta)} distance={float(np.linalg.norm(delta)):.4f}",
                flush=True,
            )

    print(f"[runner] camera-debug scene rigid objects: {rigid_object_names(env)}", flush=True)
    if target_object_name is not None and target_object_name in rigid_object_names(env):
        lego = env.scene[target_object_name]
        lego_pos = tensor_first_row(lego.data.root_pos_w)
        print(
            f"[runner] camera-debug target object={target_object_name} pos_w={format_vec(lego_pos)}",
            flush=True,
        )

    if frame_dir is not None:
        save_policy_camera_frames(policy_obs, frame_dir)

    print("[runner] camera-debug end", flush=True)


def smart_task_metrics(env: Any, robot_kind: str, target_object_name: str) -> dict[str, float]:
    """从 Isaac scene 中读取几个用于判断 pick 进展的诊断指标。

    指标：
    - `lego_z_minus_base`：乐高块高度相对于机器人 base 的差值。
    - `jaw_to_lego`：gripper jaw frame 到乐高块 root 的距离。
    - `gripper`：SO101 gripper joint 当前值，或 Franka finger 平均开口。

    这些不是严格成功判据，只是帮助我们观察动作有没有朝正确方向发展。
    """

    lego = env.scene[target_object_name]
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


def print_metrics(env: Any, prefix: str, robot_kind: str, target_object_name: str) -> None:
    """用统一格式打印 SmartTask 诊断指标。"""

    metrics = smart_task_metrics(env, robot_kind, target_object_name)
    print(
        "[runner] "
        f"{prefix} metrics target={target_object_name} "
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


def start_viewport_video_capture(args: argparse.Namespace, total_frames: int) -> Any | None:
    """启动 Isaac/Omniverse viewport mp4 录制。

    这个函数只在 `--capture-video` 打开时调用。它直接使用 Isaac Sim 自带的
    `omni.kit.capture.viewport` extension，不引入 OBS、ffmpeg、OpenCV 或其它
    额外依赖。

    参数：
    - `args`：命令行参数，里面包含输出目录、分辨率、fps、码率等 capture 设置。
    - `total_frames`：预计要捕获的 viewport 帧数。runner 会根据 GR00T 返回的
      action chunk 长度和剩余 policy call 数估算。

    返回：
    - CaptureExtension instance；后续可用 `capture.done` 等待编码完成。
    - 如果扩展不可用或启动失败，返回 None，主仿真继续运行。

    文件大小控制：
    - 默认 960x540。
    - 默认 15 fps。
    - 默认 2 Mbps H.264 mp4。

    这几个默认值足够看清机器人动作，同时避免一分钟视频膨胀到几百 MB。
    """

    if args.dry_run:
        print("[runner] capture-video requested, but dry-run does not step the viewport; skipping capture", flush=True)
        return None

    if args.headless:
        print("[runner] warning: capture-video is intended for --no-headless viewport runs", flush=True)

    try:
        import omni.kit.app

        # mp4 编码需要 omni.videoencoding；viewport 抓帧需要 omni.kit.capture.viewport。
        # 这里用 Isaac Sim extension manager 只在当前 app 进程启用，不修改环境。
        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_manager.set_extension_enabled_immediate("omni.videoencoding", True)
        extension_manager.set_extension_enabled_immediate("omni.kit.capture.viewport", True)

        from omni.kit.capture.viewport import CaptureExtension, CaptureOptions, CaptureRangeType, CaptureRenderPreset
        import omni.kit.viewport.utility as viewport_utils

        viewport = viewport_utils.get_active_viewport()
        if viewport is None:
            print("[runner] warning: no active viewport found; skipping capture", flush=True)
            return None

        capture_dir = Path(args.capture_dir).expanduser()
        if not capture_dir.is_absolute():
            capture_dir = SCRIPT_DIR / capture_dir
        capture_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        capture_name = args.capture_name or f"{args.robot}_{args.control_mode}_{timestamp}"

        options = CaptureOptions()
        # 当前 viewport 使用的 camera path。通常是用户在 viewport 中正在看的相机。
        options.camera = viewport.camera_path.pathString
        options.output_folder = str(capture_dir)
        options.file_name = capture_name
        options.file_type = ".mp4"
        options.range_type = CaptureRangeType.FRAMES
        options.start_frame = 1
        options.end_frame = max(1, int(total_frames))
        options.capture_every_Nth_frames = max(1, int(args.capture_every_nth_frames))
        options.fps = float(args.capture_fps)
        options.res_width = int(args.capture_width)
        options.res_height = int(args.capture_height)
        # RAY_TRACE 比默认 PATH_TRACE 快得多，更适合记录实验过程。
        options.render_preset = CaptureRenderPreset.RAY_TRACE
        options.spp_per_iteration = 1
        options.path_trace_spp = 1
        options.real_time_settle_latency_frames = 1
        options.overwrite_existing_frames = True
        # CaptureOptions 的 bitrate 单位是 bits/s。2 Mbps 通常已经够看机械臂动作。
        options.mp4_encoding_bitrate = int(float(args.capture_bitrate_mbps) * 1024 * 1024)
        options.mp4_encoding_iframe_interval = max(1, int(float(args.capture_fps) * 2))
        options.mp4_encoding_preset = "PRESET_DEFAULT"
        options.mp4_encoding_profile = "H264_PROFILE_HIGH"
        options.mp4_encoding_rc_mode = "RC_VBR"

        capture = CaptureExtension.get_instance()
        capture.show_default_progress_window = False
        capture.options = options

        if not capture.start():
            print("[runner] warning: viewport capture failed to start; continuing without video", flush=True)
            return None

        print(
            "[runner] viewport capture started "
            f"output={capture_dir / (capture_name + '.mp4')} "
            f"frames={options.end_frame} fps={options.fps:g} "
            f"resolution={options.res_width}x{options.res_height} "
            f"bitrate={args.capture_bitrate_mbps:g}Mbps",
            flush=True,
        )
        return capture
    except Exception as exc:  # noqa: BLE001 - capture must not break the robot run.
        print(f"[runner] warning: viewport capture unavailable: {exc}", flush=True)
        return None


def wait_for_viewport_video_capture(capture: Any | None, simulation_app: Any, env: Any, timeout_s: float) -> None:
    """等待 viewport capture 编码完成。

    Omniverse 的 mp4 capture 会先写临时帧，再异步编码。主动控制结束后如果立刻
    `simulation_app.close()`，视频文件可能还没来得及落盘。因此这里用 render
    loop 等一下 capture.done。
    """

    if capture is None:
        return

    deadline = time.time() + max(0.0, timeout_s)
    print("[runner] waiting for viewport capture to finish encoding", flush=True)

    while simulation_app.is_running() and not capture.done and time.time() < deadline:
        env.sim.render()
        time.sleep(1.0 / 30.0)

    if capture.done:
        print(f"[runner] viewport capture outputs: {capture.get_outputs()}", flush=True)
    else:
        print("[runner] warning: viewport capture did not finish before timeout", flush=True)


def parse_args() -> argparse.Namespace:
    """解析 Isaac runner 的命令行参数。"""

    parser = argparse.ArgumentParser()

    # 选择实验机器人。so101 是原 LeIsaac SmartTask；franka 是本 experiments
    # 目录新增注册的 Groot-Franka-SmartTask-v0。
    parser.add_argument("--robot", choices=("so101", "franka"), default="so101")

    parser.add_argument(
        "--deployment-mode",
        choices=("zero-shot-oxe", "so101-finetuned"),
        default="zero-shot-oxe",
        help=(
            "zero-shot-oxe expects the base GR00T OXE/DROID bridge; "
            "so101-finetuned expects a NEW_EMBODIMENT SO101 checkpoint bridge."
        ),
    )

    parser.add_argument(
        "--policy-schema",
        choices=("oxe", "so101-new-embodiment"),
        default=None,
        help="Observation/action schema expected by the GR00T bridge.",
    )

    parser.add_argument(
        "--camera-profile",
        choices=("auto", *CAMERA_PROFILE_DEFAULTS.keys()),
        default="auto",
        help=(
            "Resolve LeIsaac policy camera keys. auto detects current leisaac vs leisaac2 legacy; "
            "explicit per-role camera keys below override the profile."
        ),
    )
    parser.add_argument("--exterior-camera-key", default=None, help="Policy camera key for OXE/DROID exterior_image_1_left.")
    parser.add_argument("--top-camera-key", default=None, help="Policy camera key for SO101 NEW_EMBODIMENT video.top.")
    parser.add_argument("--wrist-camera-key", default=None, help="Policy camera key for wrist video input.")

    # IsaacLab task id。None 表示按 --robot 自动选择默认 task。
    parser.add_argument("--task", default=None)

    # GR00T bridge 的地址和端口。默认对应 README 中 bridge 的启动参数。
    parser.add_argument("--bridge-host", default="127.0.0.1")
    parser.add_argument("--bridge-port", type=int, default=5577)

    # 传给 GR00T 的自然语言任务指令。
    parser.add_argument("--instruction", default="Pick up the red 2x4 lego brick.")

    parser.add_argument(
        "--target-object-key",
        default="auto",
        help=(
            "Scene rigid-object key used for SmartTask metrics/debug. "
            "auto supports current red_2x4_lego_brick_pick and legacy red_2x4_lego_brick."
        ),
    )
    parser.add_argument(
        "--smart-scene-usd",
        default="auto",
        help=(
            "LeIsaac SmartTask scene USD for this runner process. "
            "auto uses scene_portable.usda when present, otherwise scene.usd."
        ),
    )
    parser.add_argument(
        "--smart-target-asset",
        default="cuboid",
        help=(
            "SmartTask target object source for LeIsaac SO101 runs: "
            "'cuboid' creates a red 2x4-sized primitive fallback, "
            "'scene' relies on the LeIsaac scene parser, "
            "or pass a complete USD path."
        ),
    )
    parser.add_argument(
        "--smart-target-prim-path",
        default=SMART_TARGET_MANAGED_PRIM_PATH,
        help="Prim path used when --smart-target-asset is cuboid or a USD path.",
    )

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
    parser.add_argument(
        "--so101-arm-delta-scale",
        type=float,
        default=1.0,
        help="Scale finetuned SO101 relative arm deltas before adding them to current joints.",
    )
    parser.add_argument(
        "--so101-max-arm-delta",
        type=float,
        default=None,
        help="Optional absolute clamp for each finetuned SO101 arm joint delta before execution.",
    )
    parser.add_argument(
        "--so101-gripper-min",
        type=float,
        default=None,
        help="Optional lower clamp for finetuned SO101 absolute gripper target.",
    )
    parser.add_argument(
        "--so101-gripper-max",
        type=float,
        default=None,
        help="Optional upper clamp for finetuned SO101 absolute gripper target.",
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

    # 直接调用 Isaac/Omniverse viewport capture extension，把当前 viewport 录成 mp4。
    parser.add_argument(
        "--capture-video",
        action="store_true",
        help="Record the active Isaac viewport to a compact mp4 during active robot control.",
    )
    parser.add_argument(
        "--capture-dir",
        default="runs/captures",
        help="Directory for captured mp4 files. Relative paths are resolved under this experiment folder.",
    )
    parser.add_argument("--capture-name", default=None, help="Output mp4 stem. Default: robot_controlmode_timestamp.")
    parser.add_argument("--capture-width", type=int, default=960)
    parser.add_argument("--capture-height", type=int, default=540)
    parser.add_argument("--capture-fps", type=float, default=15.0)
    parser.add_argument("--capture-bitrate-mbps", type=float, default=2.0)
    parser.add_argument(
        "--capture-every-nth-frames",
        type=int,
        default=1,
        help="Capture every Nth viewport frame. Increase to reduce file size further.",
    )
    parser.add_argument(
        "--capture-wait-timeout-s",
        type=float,
        default=60.0,
        help="How long to wait for mp4 encoding before closing Isaac Sim.",
    )
    parser.add_argument(
        "--debug-cameras",
        action="store_true",
        help="Print policy camera tensors, scene camera sensors, USD camera prims, and wrist/body poses after reset.",
    )
    parser.add_argument(
        "--debug-cameras-only",
        action="store_true",
        help="Run only the camera diagnostics after env reset, without connecting to the GR00T bridge.",
    )
    parser.add_argument(
        "--debug-camera-frame-dir",
        default=None,
        help="Optional directory for saving current policy camera1/camera2/camera3 frames as png.",
    )

    args = parser.parse_args()
    if args.deployment_mode == "so101-finetuned":
        if args.robot != "so101":
            raise ValueError("--deployment-mode so101-finetuned requires --robot so101")
        if args.control_mode != "joint":
            raise ValueError("--deployment-mode so101-finetuned currently requires --control-mode joint")
        if args.policy_schema not in (None, "so101-new-embodiment"):
            raise ValueError("--deployment-mode so101-finetuned requires --policy-schema so101-new-embodiment")
        args.policy_schema = "so101-new-embodiment"
    elif args.policy_schema is None:
        args.policy_schema = "oxe"

    if args.task is None:
        args.task = "Groot-Franka-SmartTask-v0" if args.robot == "franka" else "LeIsaac-SO101-SmartTask-v0"
    if args.policy_schema == "so101-new-embodiment":
        if args.robot != "so101":
            raise ValueError("--policy-schema so101-new-embodiment requires --robot so101")
        if args.control_mode != "joint":
            raise ValueError("--policy-schema so101-new-embodiment currently requires --control-mode joint")
    if args.debug_cameras_only:
        args.debug_cameras = True
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

    for key in ("joint_position", "eef_9d", "gripper_position", "single_arm", "gripper"):
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

    smart_scene_usd_path = install_smart_task_asset_patch(args)

    # 读取 task 默认配置，并指定 device / num_envs。
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    if smart_scene_usd_path is not None and hasattr(env_cfg.scene, "scene"):
        env_cfg.scene.scene.spawn.usd_path = str(smart_scene_usd_path)

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
    # gym.make() 期间也可能因为 scene/asset 配置错误失败；这时必须主动关闭
    # Isaac app，否则进程会停在 Omniverse 清理阶段占着 GPU。
    try:
        env = gym.make(args.task, cfg=env_cfg).unwrapped
    except Exception:
        simulation_app.close()
        raise

    # 相机历史缓存，用于构造 GR00T 的两帧 video 输入。
    history = FrameHistory(horizon=max(args.warmup_frames, 2))
    capture_instance = None

    try:
        print(
            f"[runner] robot={args.robot} task={args.task} control mode={args.control_mode} "
            f"teleop_device={teleop_device}",
            flush=True,
        )

        # reset Isaac env，拿到第一帧 observation。相机 debug 也必须放在 reset 后，
        # 因为此时 sensor/view/prim 才完整实例化。
        obs, _ = env.reset()
        policy_obs = obs["policy"]
        resolved_camera_profile, camera_mapping = resolve_camera_mapping(args, env)
        target_object_name = resolve_target_object_name(env, args.target_object_key)
        print(
            f"[runner] camera profile={args.camera_profile} resolved={resolved_camera_profile} "
            f"mapping={camera_mapping}",
            flush=True,
        )
        print(f"[runner] target object={target_object_name}", flush=True)

        if args.debug_cameras:
            debug_frame_dir = args.debug_camera_frame_dir
            if debug_frame_dir is None and args.debug_cameras_only:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_frame_dir = f"runs/camera_debug/{args.robot}_{args.control_mode}_{timestamp}"
            print_camera_debug(env, policy_obs, args.robot, debug_frame_dir, camera_mapping, target_object_name)

        if args.debug_cameras_only:
            keep_open(simulation_app, env, args.keep_open_s)
            return

        # 先 ping bridge，确认 GR00T 进程已经启动，并打印 modality schema。
        print("[runner] ping bridge", flush=True)
        ping = request(args.bridge_host, args.bridge_port, {"endpoint": "ping"}, args.timeout_s)
        if not ping.get("ok"):
            raise RuntimeError(ping)

        print(f"[runner] bridge modality: {ping.get('modality')}", flush=True)

        print(f"[runner] initial joint_pos values={tensor_values(policy_obs['joint_pos'])}", flush=True)
        print_metrics(env, "initial", args.robot, target_object_name)

        build_observation = (
            build_so101_new_embodiment_observation
            if args.policy_schema == "so101-new-embodiment"
            else build_oxe_observation
        )

        # warmup：重复把当前 observation 放入 history，确保 history 至少有两帧。
        for _ in range(args.warmup_frames):
            build_observation(policy_obs, history, args.instruction, args.robot, camera_mapping)

        stop_run = False

        # 外层循环：每次向 GR00T 请求一个 action chunk。
        for call_idx in range(args.max_policy_calls):
            groot_obs = build_observation(
                policy_obs,
                history,
                args.instruction,
                args.robot,
                camera_mapping,
            )

            print(f"[runner] request action {call_idx + 1}/{args.max_policy_calls}", flush=True)

            # 向 bridge 发起 get_action 请求。这里的 observation 会被标准库 pickle
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
                print_metrics(env, "dry-run", args.robot, target_object_name)
                continue

            # 决定这个 chunk 执行多少步。
            # 默认 action_horizon=0，表示完整执行 GR00T 返回的 40 步。
            num_action_steps = action_chunk_len(action) if args.action_horizon <= 0 else args.action_horizon
            print(f"[runner] executing {num_action_steps} action steps from this chunk", flush=True)

            # 第一次真正执行动作前启动 viewport capture。这样不会把等待 GR00T
            # 推理的空窗录进去，也能根据第一段 action chunk 估算总帧数。
            if args.capture_video and capture_instance is None:
                remaining_policy_calls = args.max_policy_calls - call_idx
                estimated_total_frames = max(1, int(num_action_steps * remaining_policy_calls))
                capture_instance = start_viewport_video_capture(args, estimated_total_frames)

            # 内层循环：逐步执行 action chunk。
            for step_idx in range(num_action_steps):
                if args.policy_schema == "so101-new-embodiment":
                    step_action = {}
                    if "single_arm" in action and action["single_arm"].shape[1] > step_idx:
                        step_action["single_arm"] = action["single_arm"][:, step_idx : step_idx + 1, :]
                    if "gripper" in action and action["gripper"].shape[1] > step_idx:
                        step_action["gripper"] = action["gripper"][:, step_idx : step_idx + 1, :]
                    if not step_action:
                        step_action = action
                    command = so101_new_embodiment_action_to_leisaac_tensor(
                        step_action,
                        env.device,
                        policy_obs["joint_pos"],
                        arm_delta_scale=args.so101_arm_delta_scale,
                        max_arm_delta=args.so101_max_arm_delta,
                        gripper_min=args.so101_gripper_min,
                        gripper_max=args.so101_gripper_max,
                    )
                elif args.control_mode == "eef":
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
                build_observation(policy_obs, history, args.instruction, args.robot, camera_mapping)

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
            print_metrics(env, f"after policy call {call_idx + 1}", args.robot, target_object_name)

            if stop_run:
                break

        wait_for_viewport_video_capture(capture_instance, simulation_app, env, args.capture_wait_timeout_s)
        keep_open(simulation_app, env, args.keep_open_s)

    finally:
        # 无论正常结束还是异常退出，都关闭 env 和 Isaac app。
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
