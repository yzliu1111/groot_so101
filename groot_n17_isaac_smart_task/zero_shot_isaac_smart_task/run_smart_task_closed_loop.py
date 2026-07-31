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
from functools import partial
import json
import os
import re
import sys
import time
import traceback
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

# These profiles are convenience presets, not required camera-number semantics.
# Explicit --isaac-*-camera-key arguments may map any distinct live keys to the
# model roles. The current single-camera LeIsaac main must first expose enough
# sensors and policy observations as documented in README_ZH.md.
# - leisaac-current preset: camera1=left, camera2=wrist, camera3=front/top.
# - leisaac2-legacy: camera1=front/top, camera2=left, camera3=template wrist.
# The runner prints the resolved mapping so the log shows what GR00T saw.
CAMERA_PROFILE_DEFAULTS = {
    "leisaac-current": {
        "exterior": "camera3",
        "top": "camera3",
        "left": "camera1",
        "wrist": "camera2",
    },
    "leisaac2-legacy": {
        "exterior": "camera1",
        "top": "camera1",
        "left": "camera2",
        "wrist": "camera3",
    },
    "franka-current": {
        "exterior": "camera1",
        "top": "camera1",
        "left": "camera3",
        "wrist": "camera2",
    },
}

SO101_VIDEO_KEYS = {
    "wrist-only": ("wrist",),
    "dual": ("top", "wrist"),
    "triple": ("top", "left", "wrist"),
}

# The process-local cuboid below is a compatibility path for the legacy base
# SmartTask only. Other tasks must keep the scene and target declared by their
# own LeIsaac env config.
SMART_TARGET_OBJECT_KEY = "red_2x4_lego_brick_pick"
SMART_TARGET_CUBOID_SIZE = (0.0318, 0.0158, 0.0096)
SMART_TARGET_CUBOID_MASS = 0.02
SMART_TARGET_MANAGED_PRIM_PATH = f"{{ENV_REGEX_NS}}/Scene/{SMART_TARGET_OBJECT_KEY}_managed"
SMART_TARGET_CUBOID_ASSET_COLOR = (0.78, 0.05, 0.02)
SMART_TARGET_CUBOID_ROUGHNESS = 0.55

SO101_ROBOT_MATERIAL_AUTO = "auto"
SO101_ROBOT_MATERIAL_ASSET = "asset"
SO101_ROBOT_MATERIAL_REAL_WHITE = "real-white"
SO101_ROBOT_MATERIAL_CHOICES = (
    SO101_ROBOT_MATERIAL_AUTO,
    SO101_ROBOT_MATERIAL_ASSET,
    SO101_ROBOT_MATERIAL_REAL_WHITE,
)
SO101_REAL_WHITE_DIFFUSE_COLOR = (1.0, 1.0, 1.0)
SO101_PRINTED_MATERIAL_SHADER_PATH_RE = re.compile(
    r"^/World/envs/env_[^/]+/Robot/Looks/material_a_3d_printed/Shader$"
)
SO101_PRINTED_MATERIAL_DIFFUSE_INPUT = "inputs:diffuse_color_constant"

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
from action_chunk import action_chunk_length, slice_action_step  # noqa: E402
from action_timing import resolve_env_steps_per_policy_action  # noqa: E402
import contact_probe  # noqa: E402
from pose_math import compose_pose_delta, quat_wxyz_to_rot6d  # noqa: E402
import scene_profiles  # noqa: E402
from so101_eval_cameras import configure_so101_eval_cameras  # noqa: E402
from so101_joint_units import (  # noqa: E402
    SO101_ARM_UNITS_DEGREES,
    SO101_ARM_UNITS_LEROBOT_MOTOR,
    SO101_CHECKPOINT_ARM_UNIT_CHOICES,
    SO101_JOINT_NAMES,
    SO101_LEROBOT_MOTOR_LIMITS,
    SO101_USD_JOINT_LIMITS_RAD,
    isaac_rad_to_so101_dataset,
    safe_absolute_dataset_target_to_isaac_rad,
    so101_dataset_to_isaac_rad,
    validate_runtime_joint_limits_match_converter,
)
from task_selection import (  # noqa: E402
    default_task_for_robot,
    is_leisaac_smart_task,
    resolve_smart_target_asset,
    resolve_target_name,
    resolve_task_instruction,
    target_name_from_env_cfg,
    uses_legacy_smart_task_asset_patch,
    validate_robot_task,
)
from wire import request  # noqa: E402

# AppLauncher 要在 sys.path 调整后再 import，这样 isaaclab/isaaclab_assets 会优先
# 来自 copied LeIsaac dependencies，而不是机器上其它源码副本。
from isaaclab.app import AppLauncher  # noqa: E402


# These deterministic reset presets are coordinate-wise medians across all 50
# episodes at an explicit dataset-start reference frame:
# - outputs/sim_2.parquet frame_index=5 (~0.17 s): LeRobot motor units for the
#   first five joints.  This is slightly later than the reset-transient frame 0,
#   but still precedes the settled wrist pose around frame 10.
# - outputs/real_1.parquet frame_index=0: degrees for the first five joints.
# The gripper remains in LeRobot [0, 100] coordinates in both datasets.
#
# A median is used because the real episode-0 pose is an outlier and the first
# three sim joints also vary between episodes.  These are dataset-start
# proprioception references, not whole-dataset means.
SO101_DATASET_START_MEDIAN_STATE = {
    SO101_ARM_UNITS_LEROBOT_MOTOR: (
        -2.288494110107422,
        -6.753181457519531,
        10.01395034790039,
        90.93464660644531,
        -53.41120147705078,
        1.1864696741104126,
    ),
    SO101_ARM_UNITS_DEGREES: (
        1.4505494832992554,
        0.7472527623176575,
        6.4175825119018555,
        91.78022003173828,
        -94.15384674072266,
        2.023319721221924,
    ),
}

SO101_DATASET_START_MEDIAN_SOURCE = {
    SO101_ARM_UNITS_LEROBOT_MOTOR: "sim_2.parquet:frame-5-median",
    SO101_ARM_UNITS_DEGREES: "real_1.parquet:frame-0-median",
}

SO101_INITIAL_POSE_AUTO = "auto"
SO101_INITIAL_POSE_DATASET_START = "dataset-start"
# Backward-compatible alias for commands copied before the sim preset moved
# from frame 0 to frame 5.  It resolves to the canonical dataset-start mode.
SO101_INITIAL_POSE_DATASET_FIRST_FRAME = "dataset-first-frame"
SO101_INITIAL_POSE_USD_DEFAULT = "usd-default"
SO101_INITIAL_POSE_CHOICES = (
    SO101_INITIAL_POSE_AUTO,
    SO101_INITIAL_POSE_DATASET_START,
    SO101_INITIAL_POSE_DATASET_FIRST_FRAME,
    SO101_INITIAL_POSE_USD_DEFAULT,
)
SO101_INITIAL_POSE_VERIFY_ATOL_RAD = 1e-4


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
        raise KeyError(
            f"{role} observation key {key!r} is not in policy obs; "
            f"available policy observation keys are {sorted(policy_obs)}"
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

    # The historical leisaac-current preset names the robot-mounted wrist sensor camera2.
    camera2_prim = _sensor_cfg_prim_path(scene_sensors.get("camera2"))
    if "wrist" in camera2_prim or "/Robot/" in camera2_prim:
        return "leisaac-current"

    # Prefer the current three-camera layout when detection is inconclusive. The
    # post-reset live-key validation will still fail closed if sensors are missing.
    return "leisaac-current"


def resolve_camera_mapping(args: argparse.Namespace, env: Any) -> tuple[str, dict[str, str]]:
    """Resolve camera profile defaults plus explicit per-role overrides."""

    profile = args.camera_profile
    if profile == "auto":
        profile = "franka-current" if args.robot == "franka" else detect_camera_profile(env)

    defaults = CAMERA_PROFILE_DEFAULTS[profile]
    mapping = {
        "exterior": args.isaac_exterior_camera_key or defaults["exterior"],
        "top": args.isaac_front_camera_key or defaults["top"],
        "left": args.isaac_left_camera_key or defaults["left"],
        "wrist": args.isaac_wrist_camera_key or defaults["wrist"],
    }
    return profile, mapping


def apply_injected_camera_mapping(
    args: argparse.Namespace,
    physical_mapping: dict[str, str],
) -> dict[str, str]:
    """Apply injected-camera defaults while allowing an explicit slot permutation.

    ``physical_mapping`` records what each live Isaac camera physically is.  An
    explicitly supplied ``--isaac-*-camera-key`` may permute those live keys to
    reproduce a checkpoint whose prepared dataset used mislabeled semantic
    slots.  It may not select a camera that the injection did not create, nor
    reuse one live camera for multiple model inputs.
    """

    camera_arg_names = {
        "top": "isaac_front_camera_key",
        "left": "isaac_left_camera_key",
        "wrist": "isaac_wrist_camera_key",
    }
    available_keys = set(physical_mapping.values())
    selected_mapping: dict[str, str] = {}

    for role, default_observation_key in physical_mapping.items():
        arg_name = camera_arg_names[role]
        configured_key = getattr(args, arg_name)
        observation_key = configured_key or default_observation_key
        if observation_key not in available_keys:
            raise ValueError(
                f"Injected SO101 eval camera role {role!r} requested {observation_key!r}, "
                f"but the injected layout only exposes {sorted(available_keys)!r}"
            )
        setattr(args, arg_name, observation_key)
        selected_mapping[role] = observation_key

    if len(set(selected_mapping.values())) != len(selected_mapping):
        raise ValueError(
            "Injected SO101 eval camera roles must use distinct live observation keys; "
            f"selected mapping={selected_mapping}"
        )

    return selected_mapping


def active_camera_mapping(
    policy_schema: str,
    camera_layout: str,
    camera_mapping: dict[str, str],
) -> dict[str, str]:
    """Keep only the live observation keys consumed by the selected policy schema."""

    roles = ("exterior", "wrist") if policy_schema == "oxe" else SO101_VIDEO_KEYS[camera_layout]
    return {role: camera_mapping[role] for role in roles}


def validate_camera_mapping(
    policy_obs: dict[str, torch.Tensor],
    camera_mapping: dict[str, str],
) -> None:
    """Validate selected live keys and reject one camera reused for multiple roles."""

    roles_by_observation_key: dict[str, list[str]] = {}
    for role, observation_key in camera_mapping.items():
        roles_by_observation_key.setdefault(observation_key, []).append(role)

    duplicate_keys = {
        observation_key: roles
        for observation_key, roles in roles_by_observation_key.items()
        if len(roles) > 1
    }
    if duplicate_keys:
        raise ValueError(
            "Each selected camera role must map to a different Isaac policy observation key; "
            f"duplicate mappings={duplicate_keys}. Do not reuse one live camera for dual/triple inputs."
        )

    for role, observation_key in camera_mapping.items():
        get_policy_camera(policy_obs, observation_key, role)


def validate_bridge_camera_layout(args: argparse.Namespace, ping: dict[str, Any]) -> None:
    """Ensure the runner layout matches the modality actually served by the bridge."""

    if args.policy_schema != "so101-new-embodiment":
        return

    bridge_layout = ping.get("camera_layout")
    if bridge_layout is not None and bridge_layout != args.camera_layout:
        raise ValueError(
            "Runner and bridge camera layouts differ: "
            f"runner={args.camera_layout!r} bridge={bridge_layout!r}"
        )

    actual_video_keys = tuple(ping.get("modality", {}).get("video", {}).get("modality_keys", ()))
    expected_video_keys = SO101_VIDEO_KEYS[args.camera_layout]
    if actual_video_keys != expected_video_keys:
        raise ValueError(
            "Runner camera layout does not match bridge video schema: "
            f"layout={args.camera_layout!r} expected={list(expected_video_keys)} "
            f"actual={list(actual_video_keys)}"
        )

    action_decoding = ping.get("action_decoding")
    if not isinstance(action_decoding, dict):
        raise ValueError(
            "Bridge did not report GR00T action-decoding semantics. Restart the bridge with "
            "this updated groot_bridge_server.py before executing SO101 actions."
        )
    expected_action_contract = {
        "policy_api_output": "decoded_dataset_action",
        "dataset_action_semantics": "absolute_joint_position_targets",
        "dataset_action_units": "checkpoint_dataset_coordinates",
    }
    contract_mismatch = {
        key: (expected, action_decoding.get(key))
        for key, expected in expected_action_contract.items()
        if action_decoding.get(key) != expected
    }
    if contract_mismatch:
        raise ValueError(
            "Bridge does not confirm the SO101 runner's executable action contract. "
            f"mismatch={contract_mismatch} bridge_report={action_decoding}"
        )
    if action_decoding.get("processor_use_relative_action") is not True:
        print(
            "[runner] warning: checkpoint processor_use_relative_action is not true. "
            "This does not turn the Policy API output into a delta, but it differs from "
            "the expected SO101 fine-tune recipe; verify the checkpoint/config pairing.",
            flush=True,
        )


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


def resolve_target_object_name(
    env: Any,
    requested_name: str,
    configured_name: str | None = None,
) -> str:
    """Resolve the selected task's target object for metrics and camera debug."""

    return resolve_target_name(rigid_object_names(env), requested_name, configured_name)


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


def smart_target_pose_from_scene(
    scene_usd_path: Path,
    override_pos: tuple[float, float, float] | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Read the target pose from the SmartTask scene, falling back to a table-top pose."""

    if override_pos is not None:
        return override_pos, (1.0, 0.0, 0.0, 0.0)

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

    return (0.0, 0.25, 0.07), (1.0, 0.0, 0.0, 0.0)


def add_smart_target_cfg(
    env_cfg: Any,
    target_asset: str,
    target_prim_path: str,
    scene_usd_path: Path,
    target_pos: tuple[float, float, float] | None,
    cuboid_size: tuple[float, float, float],
    red24_material: str = scene_profiles.RED24_MATERIAL_ASSET,
) -> None:
    """Register SmartTask's target object without modifying the LeIsaac source tree."""

    if target_asset == "scene":
        if red24_material == scene_profiles.RED24_MATERIAL_REAL_RED:
            raise ValueError(
                "Cannot apply --red24-material real-red when --smart-target-asset scene: "
                "the runner does not own that scene target's spawn material. Use an explicit "
                "--scene-profile or a runner-owned cuboid/USD target."
            )
        return
    if red24_material not in scene_profiles.RED24_MATERIAL_CHOICES:
        raise ValueError(
            f"unknown red 2x4 material {red24_material!r}; "
            f"expected one of {scene_profiles.RED24_MATERIAL_CHOICES}"
        )

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    if target_asset == "cuboid":
        diffuse_color = SMART_TARGET_CUBOID_ASSET_COLOR
        if red24_material == scene_profiles.RED24_MATERIAL_REAL_RED:
            diffuse_color = scene_profiles.REAL_RED24_DIFFUSE_COLOR
        spawn_cfg = sim_utils.CuboidCfg(
            size=cuboid_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=SMART_TARGET_CUBOID_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.7),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=diffuse_color,
                roughness=SMART_TARGET_CUBOID_ROUGHNESS,
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
        spawn_kwargs: dict[str, Any] = {}
        if red24_material == scene_profiles.RED24_MATERIAL_REAL_RED:
            spawn_kwargs["visual_material"] = sim_utils.PreviewSurfaceCfg(
                diffuse_color=scene_profiles.REAL_RED24_DIFFUSE_COLOR,
                roughness=scene_profiles.REAL_RED24_ROUGHNESS,
            )
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=str(target_usd_path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=SMART_TARGET_CUBOID_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            **spawn_kwargs,
        )

    pos, rot = smart_target_pose_from_scene(scene_usd_path, target_pos)
    setattr(
        env_cfg.scene,
        SMART_TARGET_OBJECT_KEY,
        RigidObjectCfg(
            prim_path=target_prim_path,
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
        ),
    )
    print(
        "[runner] SmartTask target cfg "
        f"key={SMART_TARGET_OBJECT_KEY} source={target_asset} prim_path={target_prim_path} "
        f"init_pos={tuple(round(float(v), 4) for v in pos)} "
        f"cuboid_size={tuple(round(float(v), 4) for v in cuboid_size)} "
        f"red24_material={red24_material}",
        flush=True,
    )


def validate_scene_profile_asset_args(args: argparse.Namespace) -> None:
    """Reject legacy/custom asset controls when an explicit profile owns composition."""

    if args.scene_profile == scene_profiles.TASK_DEFAULT_SCENE_PROFILE:
        return
    if args.smart_scene_usd != "auto":
        raise ValueError(
            "An explicit --scene-profile uses the current LeIsaac SmartTask scene; "
            "do not combine it with --smart-scene-usd"
        )
    uses_legacy_target_override = (
        args.smart_target_asset != "auto"
        or args.smart_target_pos is not None
        or args.smart_target_prim_path != SMART_TARGET_MANAGED_PRIM_PATH
        or tuple(args.smart_target_cuboid_size) != SMART_TARGET_CUBOID_SIZE
    )
    if uses_legacy_target_override:
        raise ValueError(
            "--scene-profile owns its LEGO assets and poses; do not combine an explicit profile "
            "with --smart-target-* legacy overrides"
        )


RED24_MATERIAL_AUTO = "auto"


def resolve_so101_robot_material(
    requested_material: str,
    checkpoint_arm_units: str,
) -> str:
    """Match SO101 robot appearance to real- versus sim-data checkpoints."""

    if requested_material == SO101_ROBOT_MATERIAL_AUTO:
        if checkpoint_arm_units == SO101_ARM_UNITS_DEGREES:
            return SO101_ROBOT_MATERIAL_REAL_WHITE
        return SO101_ROBOT_MATERIAL_ASSET
    if requested_material not in SO101_ROBOT_MATERIAL_CHOICES:
        raise ValueError(
            f"unknown SO101 robot material {requested_material!r}; expected "
            f"{SO101_ROBOT_MATERIAL_CHOICES}"
        )
    return requested_material


def resolve_red24_material(
    requested_material: str,
    checkpoint_arm_units: str,
) -> str:
    """Resolve the real-data red 2x4 override without changing sim checkpoints."""

    if requested_material == RED24_MATERIAL_AUTO:
        if checkpoint_arm_units == SO101_ARM_UNITS_DEGREES:
            return scene_profiles.RED24_MATERIAL_REAL_RED
        return scene_profiles.RED24_MATERIAL_ASSET
    if requested_material not in scene_profiles.RED24_MATERIAL_CHOICES:
        raise ValueError(
            f"unknown red 2x4 material {requested_material!r}; expected "
            f"{(RED24_MATERIAL_AUTO, *scene_profiles.RED24_MATERIAL_CHOICES)}"
        )
    return requested_material


def install_scene_profile_patch(args: argparse.Namespace) -> Path:
    """Compose one deployment scene profile without adding another Gym task."""

    scene_profiles.validate_scene_profile_task(args.task, args.scene_profile)
    validate_scene_profile_asset_args(args)

    import leisaac.assets.scenes.smart_scene as smart_scene
    import leisaac.tasks.smart_task.smart_task_env_cfg as smart_task_cfg

    # Follow the current LeIsaac module's source scene. Do not silently prefer a
    # stale runner-generated scene_portable.usda from an older workflow.
    source_scene_usd = Path(smart_scene.SMART_SCENE_USD_PATH).expanduser().resolve()
    if not source_scene_usd.is_file():
        raise FileNotFoundError(f"LeIsaac SmartTask scene USD does not exist: {source_scene_usd}")
    wrapper_path = scene_profiles.write_scene_wrapper(
        source_scene_usd,
        args.scene_profile,
        SCRIPT_DIR / "runs" / "scene_profiles",
    )
    lego_asset_dir = LEISAAC_ROOT / "assets" / "scenes" / "smart_scene" / "assets"

    original_parse = smart_task_cfg.parse_usd_and_create_subassets

    def parse_usd_and_create_subassets_with_profile(usd_path, env_cfg, *parse_args, **parse_kwargs):
        result = original_parse(str(wrapper_path), env_cfg, *parse_args, **parse_kwargs)
        added_keys = scene_profiles.add_scene_profile_objects(
            env_cfg,
            args.scene_profile,
            lego_asset_dir,
            red24_material=args.red24_material_resolved,
        )
        print(
            f"[runner] scene profile objects={list(added_keys)} tray_active="
            f"{scene_profiles.get_scene_profile(args.scene_profile).tray_active} "
            f"red24_material={args.red24_material_resolved}",
            flush=True,
        )
        return result

    # LeIsaac's parser and the live AssetBaseCfg must see the same file-backed layer.
    smart_scene.SMART_SCENE_USD_PATH = str(wrapper_path)
    smart_scene.SMART_SCENE_CFG.spawn.usd_path = str(wrapper_path)
    smart_task_cfg.SMART_SCENE_USD_PATH = str(wrapper_path)
    smart_task_cfg.SMART_SCENE_CFG.spawn.usd_path = str(wrapper_path)
    smart_task_cfg.parse_usd_and_create_subassets = parse_usd_and_create_subassets_with_profile

    print(
        f"[runner] scene profile={args.scene_profile} source_scene={source_scene_usd} "
        f"wrapper={wrapper_path}",
        flush=True,
    )
    return wrapper_path


def install_smart_task_asset_patch(args: argparse.Namespace) -> Path | None:
    """Install the legacy base-task asset workaround without altering other tasks."""

    if getattr(args, "scene_profile", scene_profiles.TASK_DEFAULT_SCENE_PROFILE) != (
        scene_profiles.TASK_DEFAULT_SCENE_PROFILE
    ):
        return install_scene_profile_patch(args)

    if not is_leisaac_smart_task(args.task):
        return None

    target_asset = resolve_smart_target_asset(args.task, args.smart_target_asset)
    if not uses_legacy_smart_task_asset_patch(args.task):
        uses_legacy_override = (
            args.smart_target_pos is not None
            or args.smart_target_prim_path != SMART_TARGET_MANAGED_PRIM_PATH
            or tuple(args.smart_target_cuboid_size) != SMART_TARGET_CUBOID_SIZE
        )
        if args.smart_scene_usd != "auto" or target_asset != "scene" or uses_legacy_override:
            raise ValueError(
                "--smart-scene-usd and --smart-target-* overrides are only supported for the "
                "legacy LeIsaac-SO101-SmartTask-v0 scene. The selected task must own its scene "
                "and target assets."
            )
        if args.red24_material_resolved == scene_profiles.RED24_MATERIAL_REAL_RED:
            raise ValueError(
                "Cannot apply --red24-material real-red to a task-owned task-default scene. "
                "Use an explicit --scene-profile whose red 2x4 material the runner controls."
            )
        print(
            f"[runner] selected task owns its scene and target assets: task={args.task!r}",
            flush=True,
        )
        return None

    scene_usd_path = resolve_smart_scene_usd(args.smart_scene_usd)

    import leisaac.assets.scenes.smart_scene as smart_scene
    import leisaac.tasks.smart_task.smart_task_env_cfg as smart_task_cfg

    original_parse = smart_task_cfg.parse_usd_and_create_subassets
    target_pos = tuple(args.smart_target_pos) if args.smart_target_pos is not None else None
    cuboid_size = tuple(args.smart_target_cuboid_size)

    def parse_usd_and_create_subassets_with_target(usd_path, env_cfg, *parse_args, **parse_kwargs):
        result = original_parse(str(scene_usd_path), env_cfg, *parse_args, **parse_kwargs)
        add_smart_target_cfg(
            env_cfg,
            target_asset,
            args.smart_target_prim_path,
            scene_usd_path,
            target_pos,
            cuboid_size,
            args.red24_material_resolved,
        )
        return result

    # Keep the imported module constants consistent for __post_init__ and debug logs.
    smart_scene.SMART_SCENE_USD_PATH = str(scene_usd_path)
    smart_scene.SMART_SCENE_CFG.spawn.usd_path = str(scene_usd_path)
    smart_task_cfg.SMART_SCENE_USD_PATH = str(scene_usd_path)
    smart_task_cfg.SMART_SCENE_CFG.spawn.usd_path = str(scene_usd_path)
    smart_task_cfg.parse_usd_and_create_subassets = parse_usd_and_create_subassets_with_target

    print(
        "[runner] legacy SmartTask asset patch "
        f"scene_usd={scene_usd_path} target_asset={target_asset} "
        f"target_prim_path={args.smart_target_prim_path} "
        f"red24_material={args.red24_material_resolved}",
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
    camera_layout: str,
    checkpoint_arm_units: str = SO101_ARM_UNITS_LEROBOT_MOTOR,
) -> dict[str, Any]:
    """Map LeIsaac SO101 observations to the trained NEW_EMBODIMENT schema."""

    if robot != "so101":
        raise ValueError("SO101 NEW_EMBODIMENT schema only supports --robot so101")

    video: dict[str, np.ndarray] = {}
    for role in SO101_VIDEO_KEYS[camera_layout]:
        frame = image_to_numpy(get_policy_camera(policy_obs, camera_mapping[role], role))
        history.push(role, frame)
        video[role] = history.latest(role)

    joint_rad = policy_obs["joint_pos"].detach().cpu().numpy()[0].astype(np.float32)
    joint_dataset = isaac_rad_to_so101_dataset(joint_rad, checkpoint_arm_units)
    single_arm = np.zeros((1, 1, 5), dtype=np.float32)
    single_arm[0, 0, : min(5, joint_dataset.shape[0])] = joint_dataset[:5]
    gripper_value = float(joint_dataset[5]) if joint_dataset.shape[0] > 5 else 0.0
    gripper = np.array([[[gripper_value]]], dtype=np.float32)

    return {
        "video": video,
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
    arm_target_scale: float = 1.0,
    max_arm_step_rad: float | None = None,
    max_gripper_step_rad: float | None = None,
    gripper_min: float | None = None,
    gripper_max: float | None = None,
    runtime_joint_limits: torch.Tensor | np.ndarray | None = None,
    checkpoint_arm_units: str = SO101_ARM_UNITS_LEROBOT_MOTOR,
    diagnostics_out: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Convert decoded absolute dataset actions to safe Isaac radian targets."""

    current_rad = fallback_joint.detach().cpu().numpy()[0].astype(np.float32)
    absolute_dataset_target = isaac_rad_to_so101_dataset(current_rad, checkpoint_arm_units)

    if "single_arm" in action:
        arm = np.asarray(action["single_arm"], dtype=np.float32)
        if arm.ndim != 3:
            raise ValueError(f"expected single_arm shape (B,T,D), got {arm.shape}")
        usable = min(5, arm.shape[-1])
        # Gr00tPolicy.decode_action() has already undone normalization and
        # converted configured relative actions back to absolute dataset units.
        absolute_dataset_target[:usable] = arm[0, 0, :usable]
    else:
        print("[runner] warning: GR00T action lacks single_arm; holding arm joints", flush=True)

    if "gripper" in action:
        gripper = np.asarray(action["gripper"], dtype=np.float32)
        if gripper.ndim != 3:
            raise ValueError(f"expected gripper shape (B,T,D), got {gripper.shape}")
        gripper_value = float(gripper[0, 0, 0])
        if gripper_min is not None or gripper_max is not None:
            default_lower, default_upper = SO101_LEROBOT_MOTOR_LIMITS[5]
            lower = float(default_lower) if gripper_min is None else float(gripper_min)
            upper = float(default_upper) if gripper_max is None else float(gripper_max)
            gripper_value = float(np.clip(gripper_value, lower, upper))
        absolute_dataset_target[5] = gripper_value
    else:
        print("[runner] warning: GR00T action lacks gripper; holding gripper", flush=True)

    runtime_limits_np = None
    if runtime_joint_limits is not None:
        if isinstance(runtime_joint_limits, torch.Tensor):
            runtime_limits_np = runtime_joint_limits.detach().cpu().numpy()
        else:
            runtime_limits_np = np.asarray(runtime_joint_limits, dtype=np.float32)

    command, diagnostics = safe_absolute_dataset_target_to_isaac_rad(
        current_rad,
        absolute_dataset_target,
        arm_units=checkpoint_arm_units,
        arm_target_scale=arm_target_scale,
        max_arm_step_rad=max_arm_step_rad,
        max_gripper_step_rad=max_gripper_step_rad,
        runtime_joint_limits_rad=runtime_limits_np,
    )
    if diagnostics["dataset_limit_clipped"]:
        print(
            "[runner] SO101 safety clipped GR00T absolute dataset target "
            f"arm_units={checkpoint_arm_units} "
            f"raw={diagnostics['raw_dataset_target'].round(4).tolist()} "
            f"clipped={diagnostics['clipped_dataset_target'].round(4).tolist()}",
            flush=True,
        )
    if diagnostics["arm_step_clipped"]:
        print(
            "[runner] SO101 safety limited per-policy-action arm radians "
            f"requested={diagnostics['requested_arm_delta_rad'].round(4).tolist()} "
            f"applied={diagnostics['applied_arm_delta_rad'].round(4).tolist()}",
            flush=True,
        )
    if diagnostics["gripper_step_clipped"]:
        print(
            "[runner] SO101 safety limited per-policy-action gripper radians "
            f"requested={diagnostics['requested_gripper_delta_rad']:.4f} "
            f"applied={diagnostics['applied_gripper_delta_rad']:.4f}",
            flush=True,
        )
    if diagnostics["runtime_limit_clipped"]:
        print("[runner] SO101 safety clipped command to Isaac runtime joint limits", flush=True)
    if diagnostics_out is not None:
        diagnostics_out.update(
            {
                "gripper_step_clipped": bool(diagnostics["gripper_step_clipped"]),
                "current_gripper_rad": float(current_rad[5]),
                "absolute_gripper_target_rad": float(
                    diagnostics["absolute_target_rad"][5]
                ),
                "requested_gripper_delta_rad": float(
                    diagnostics["requested_gripper_delta_rad"]
                ),
                "applied_gripper_delta_rad": float(
                    diagnostics["applied_gripper_delta_rad"]
                ),
            }
        )

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


def action_step_to_leisaac_tensor(
    args: argparse.Namespace,
    action: dict[str, np.ndarray],
    step_index: int,
    env_device: str,
    policy_obs: dict[str, torch.Tensor],
    so101_runtime_joint_limits: torch.Tensor | None,
    diagnostics_out: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Slice and convert one policy action according to the selected route."""

    if args.policy_schema == "so101-new-embodiment":
        step_action = slice_action_step(action, step_index, ("single_arm", "gripper"))
        return so101_new_embodiment_action_to_leisaac_tensor(
            step_action,
            env_device,
            policy_obs["joint_pos"],
            checkpoint_arm_units=args.so101_checkpoint_arm_units,
            arm_target_scale=args.so101_arm_target_scale,
            max_arm_step_rad=args.so101_max_arm_step_rad,
            max_gripper_step_rad=args.so101_max_gripper_step_rad,
            gripper_min=args.so101_gripper_min,
            gripper_max=args.so101_gripper_max,
            runtime_joint_limits=so101_runtime_joint_limits,
            diagnostics_out=diagnostics_out,
        )

    if args.control_mode == "eef":
        step_action = slice_action_step(action, step_index, ("eef_9d", "gripper_position"))
        return eef_action_to_leisaac_tensor(step_action, env_device, policy_obs, args.robot)

    step_action = slice_action_step(action, step_index, ("joint_position", "gripper_position"))
    return joint_action_to_leisaac_tensor(step_action, env_device, policy_obs["joint_pos"], args.robot)


def log_first_leisaac_command(
    args: argparse.Namespace,
    command: torch.Tensor,
    policy_obs: dict[str, torch.Tensor],
) -> None:
    """Log the first command in a chunk against its route-specific current value."""

    command_np = command.detach().cpu().numpy()
    joint_np = policy_obs["joint_pos"].detach().cpu().numpy()
    if args.control_mode == "eef":
        gripper_np = np.ones((joint_np.shape[0], 1), dtype=np.float32) if args.robot == "franka" else joint_np[:, 5:6]
        current_np = np.concatenate(
            [policy_obs["ee_frame_state"].detach().cpu().numpy()[:, :7], gripper_np],
            axis=1,
        )
        units = "EEF xyz=m, quaternion=unitless, gripper=route-specific"
    elif args.robot == "franka":
        current_np = np.concatenate(
            [joint_np[:, :7], np.ones((joint_np.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        units = "arm joints/delta=rad, gripper=binary"
    else:
        current_np = joint_np
        units = "joint values/delta=rad"

    delta_np = command_np - current_np
    print(
        f"[runner] first LeIsaac command ({units}) "
        f"shape={tuple(command_np.shape)} min={float(command_np.min()):.4f} "
        f"max={float(command_np.max()):.4f} values={command_np[0].round(4).tolist()} "
        f"delta={delta_np[0].round(4).tolist()}",
        flush=True,
    )


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


def apply_so101_robot_material(
    env: Any,
    resolved_material: str,
    *,
    requested_material: str | None = None,
    stage: Any | None = None,
    expected_shader_count: int | None = None,
) -> dict[str, Any]:
    """Apply the real-robot white plastic color as a process-local USD opinion.

    The SO101 asset uses a dedicated ``material_a_3d_printed`` OmniPBR shader
    for its yellow printed structure and a separate ``material_sts3215`` shader
    for the dark motors.  Editing only the former's diffuse input keeps motor
    appearance, geometry, collision, mass, and articulation properties intact.

    All target shaders are validated before any input is authored so a missing
    or unexpectedly duplicated material fails without leaving a partial
    override in the live stage.
    """

    if requested_material is None:
        requested_material = resolved_material
    summary: dict[str, Any] = {
        "requested": requested_material,
        "resolved": resolved_material,
        "applied": False,
        "expected_shader_count": 0,
        "matched_shader_paths": [],
        "diffuse_color_before": {},
        "diffuse_color_after": {},
        "rerender_on_reset": False,
    }

    if resolved_material == SO101_ROBOT_MATERIAL_ASSET:
        print(
            "[runner] SO101 robot material "
            f"requested={requested_material} resolved={resolved_material}; "
            "preserving authored USD appearance",
            flush=True,
        )
        return summary
    if resolved_material != SO101_ROBOT_MATERIAL_REAL_WHITE:
        raise ValueError(
            "SO101 robot material must be resolved before stage application; "
            f"got {resolved_material!r}"
        )

    if stage is None:
        stage = get_usd_stage(env)
    if stage is None:
        raise RuntimeError(
            "--so101-robot-material real-white could not resolve the live USD stage"
        )

    if expected_shader_count is None:
        expected_shader_count = int(getattr(env, "num_envs", 1))
    expected_shader_count = int(expected_shader_count)
    if expected_shader_count < 1:
        raise ValueError(
            "expected SO101 printed-material shader count must be positive; "
            f"got {expected_shader_count}"
        )
    summary["expected_shader_count"] = expected_shader_count

    matched_prims: list[tuple[str, Any]] = []
    for prim in stage.Traverse():
        prim_path_value = prim.GetPath()
        prim_path = getattr(prim_path_value, "pathString", str(prim_path_value))
        if SO101_PRINTED_MATERIAL_SHADER_PATH_RE.fullmatch(prim_path):
            matched_prims.append((prim_path, prim))

    matched_paths = [path for path, _ in matched_prims]
    summary["matched_shader_paths"] = matched_paths
    if len(matched_prims) != expected_shader_count:
        raise RuntimeError(
            "--so101-robot-material real-white expected exactly one printed-material "
            f"shader per environment: expected={expected_shader_count} "
            f"matched={len(matched_prims)} paths={matched_paths}"
        )

    validated_inputs: list[tuple[str, Any, tuple[float, float, float]]] = []
    for prim_path, prim in matched_prims:
        if prim.GetTypeName() != "Shader":
            raise RuntimeError(
                "SO101 printed-material path is not a USD Shader: "
                f"path={prim_path} type={prim.GetTypeName()!r}"
            )
        diffuse_attr = prim.GetAttribute(SO101_PRINTED_MATERIAL_DIFFUSE_INPUT)
        attr_is_valid = diffuse_attr is not None and (
            not hasattr(diffuse_attr, "IsValid") or diffuse_attr.IsValid()
        )
        if not attr_is_valid:
            raise RuntimeError(
                "SO101 printed-material shader is missing OmniPBR input "
                f"{SO101_PRINTED_MATERIAL_DIFFUSE_INPUT!r}: path={prim_path}"
            )
        raw_color = diffuse_attr.Get()
        try:
            before_color = tuple(float(component) for component in raw_color)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "SO101 printed-material diffuse input is not a numeric color3: "
                f"path={prim_path} value={raw_color!r}"
            ) from exc
        if len(before_color) != 3 or not all(np.isfinite(before_color)):
            raise RuntimeError(
                "SO101 printed-material diffuse input is not a finite color3: "
                f"path={prim_path} value={before_color!r}"
            )
        validated_inputs.append((prim_path, diffuse_attr, before_color))

    env_cfg = getattr(env, "cfg", None)
    if env_cfg is None:
        raise RuntimeError(
            "--so101-robot-material real-white requires env.cfg so the first "
            "policy camera frame can be rerendered"
        )
    # ManagerBasedEnv.reset() renders RTX sensors before computing its returned
    # observation when this flag is true.  Since this material opinion is
    # authored before the runner's first reset(), the first policy frame cannot
    # reuse the yellow frame rendered during environment construction.
    env_cfg.rerender_on_reset = True

    for prim_path, diffuse_attr, _ in validated_inputs:
        if diffuse_attr.Set(SO101_REAL_WHITE_DIFFUSE_COLOR) is False:
            raise RuntimeError(
                f"failed to author SO101 real-white diffuse input at {prim_path}"
            )

    after_colors: dict[str, list[float]] = {}
    before_colors: dict[str, list[float]] = {}
    changed = False
    for prim_path, diffuse_attr, before_color in validated_inputs:
        after_color = tuple(float(component) for component in diffuse_attr.Get())
        if len(after_color) != 3 or not np.allclose(
            after_color,
            SO101_REAL_WHITE_DIFFUSE_COLOR,
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError(
                "SO101 real-white diffuse input did not persist: "
                f"path={prim_path} after={after_color!r}"
            )
        before_colors[prim_path] = list(before_color)
        after_colors[prim_path] = list(after_color)
        changed = changed or not np.allclose(
            before_color,
            after_color,
            rtol=0.0,
            atol=1e-6,
        )

    summary.update(
        {
            "applied": True,
            "changed": bool(changed),
            "diffuse_color_before": before_colors,
            "diffuse_color_after": after_colors,
            "rerender_on_reset": True,
        }
    )
    print(
        "[runner] SO101 robot material "
        f"requested={requested_material} resolved={resolved_material} "
        f"shader_paths={matched_paths} before={before_colors} after={after_colors}; "
        "material_sts3215 and physics unchanged; rerender_on_reset=True",
        flush=True,
    )
    return summary


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
    这里保留原来的三个纯标量字段，避免破坏既有 JSONL 分析脚本。
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


def smart_task_spatial_metrics(env: Any, target_object_name: str) -> dict[str, Any]:
    """Return additive XYZ diagnostics without changing legacy metric fields.

    `jaw_minus_lego_w_m` uses the Isaac world frame and is signed: positive Z
    means the jaw frame is above the LEGO root.  Recording both absolute
    positions distinguishes jaw motion from an object that was pushed away.
    """

    lego_pos = env.scene[target_object_name].data.root_pos_w[0]
    jaw_pos = env.scene["ee_frame"].data.target_pos_w[0, 1]
    jaw_minus_lego = jaw_pos - lego_pos
    return {
        "lego_pos_w_m": lego_pos.detach().cpu().tolist(),
        "jaw_pos_w_m": jaw_pos.detach().cpu().tolist(),
        "jaw_minus_lego_w_m": jaw_minus_lego.detach().cpu().tolist(),
        "jaw_to_lego_xy_m": float(
            torch.linalg.vector_norm(jaw_minus_lego[:2]).detach().cpu()
        ),
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


def write_action_trace_record(trace_file: Any | None, record: dict[str, Any]) -> None:
    """Write one flush-safe JSONL record when action tracing is enabled."""

    if trace_file is None:
        return
    trace_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    trace_file.flush()


def read_contact_probe_sample(
    env: Any,
    sensor_names: tuple[str, ...],
) -> dict[str, Any]:
    """Return pairwise PhysX contact data plus mesh-derived fingertip keypoints."""

    sample = contact_probe.read_contact_probe(env, sensor_names)
    sample["tip_points"] = contact_probe.read_so101_tip_points(
        env.scene["robot"]
    )
    return sample


def settle_environment_at_reset_pose(
    env: Any,
    policy_obs: dict[str, Any],
    *,
    env_steps: int,
    robot_kind: str,
    target_object_name: str,
    contact_sensor_names: tuple[str, ...] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Advance physics before the first policy observation while holding reset joints.

    ``--warmup-frames`` only fills the temporal image buffer; it does not call
    ``env.step()``.  Dynamic scene objects therefore need an explicit settling
    phase when their authored root pose starts above the support surface.
    The returned records make that state transition visible in the JSONL trace.
    """

    if env_steps <= 0:
        return policy_obs, []
    if "joint_pos" not in policy_obs:
        raise KeyError("settling requires policy observation key 'joint_pos'")

    hold_command = policy_obs["joint_pos"].detach().clone()
    records: list[dict[str, Any]] = []
    for env_step_index in range(1, env_steps + 1):
        obs, _, terminated, timed_out, _ = env.step(hold_command)
        policy_obs = obs["policy"]
        record: dict[str, Any] = {
            "env_step_index": env_step_index,
            "joint_pos_rad": tensor_first_row(policy_obs["joint_pos"]).tolist(),
            "metrics_after": smart_task_metrics(
                env,
                robot_kind,
                target_object_name,
            ),
            "spatial_metrics_after": smart_task_spatial_metrics(
                env,
                target_object_name,
            ),
            "terminated": bool(terminated[0]),
            "timed_out": bool(timed_out[0]),
        }
        if contact_sensor_names:
            record["contact_probe_after"] = read_contact_probe_sample(
                env,
                contact_sensor_names,
            )
        records.append(record)
    return policy_obs, records


def restore_robot_frame0_after_settle(
    env: Any,
    reset_joint_pos: Any,
    *,
    target_object_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Restore the exact reset robot state without advancing settled objects.

    Holding the reset action while the target falls can still leave a small
    gravity-induced articulation error.  Teleport only the robot back to the
    saved frame-0 joint state, synchronize the action/actuator targets, then
    forward and render without a physics integration step.  The target rigid
    body must remain unchanged across this operation.
    """

    robot = env.scene["robot"]
    target = env.scene[target_object_name]
    target_state_before = target.data.root_state_w.detach().clone()
    joint_pos_before = robot.data.joint_pos.detach().clone()
    zero_joint_vel = reset_joint_pos.new_zeros(reset_joint_pos.shape)

    robot.write_joint_state_to_sim(reset_joint_pos, zero_joint_vel)
    env.action_manager.process_action(reset_joint_pos)
    env.action_manager.apply_action()
    robot.set_joint_velocity_target(zero_joint_vel)
    env.scene.write_data_to_sim()

    # The simulation timestamp does not advance during ``forward()``.  Reset
    # sensor caches so the next policy observation reflects the teleported
    # robot pose rather than the final settle-step render.
    reset_sensor_names: list[str] = []
    for sensor_name, sensor in env.scene.sensors.items():
        sensor.reset()
        reset_sensor_names.append(sensor_name)

    env.sim.forward()
    if env.sim.has_rtx_sensors():
        env.sim.render()
    observations = env.observation_manager.compute(update_history=True)
    policy_obs = observations["policy"]

    target_state_after = target.data.root_state_w.detach().clone()
    target_state_delta = float(
        (target_state_after - target_state_before).abs().max().item()
    )
    joint_error = float(
        (robot.data.joint_pos - reset_joint_pos).abs().max().item()
    )
    joint_velocity_max = float(robot.data.joint_vel.abs().max().item())
    if target_state_delta > 1.0e-7:
        raise RuntimeError(
            "Restoring the robot frame-0 pose changed the settled target state: "
            f"max_abs_delta={target_state_delta:.9g}"
        )
    if joint_error > 1.0e-6:
        raise RuntimeError(
            "Robot frame-0 restoration did not reach the requested joint state: "
            f"max_abs_error_rad={joint_error:.9g}"
        )

    return policy_obs, {
        "method": "robot_joint_state_teleport_without_physics_step",
        "joint_pos_before_restore_rad": tensor_first_row(joint_pos_before).tolist(),
        "joint_pos_requested_rad": tensor_first_row(reset_joint_pos).tolist(),
        "joint_pos_after_restore_rad": tensor_first_row(robot.data.joint_pos).tolist(),
        "joint_max_abs_error_rad": joint_error,
        "joint_velocity_max_abs_rad_s": joint_velocity_max,
        "target_root_state_before_restore": tensor_first_row(
            target_state_before
        ).tolist(),
        "target_root_state_after_restore": tensor_first_row(
            target_state_after
        ).tolist(),
        "target_root_state_max_abs_delta": target_state_delta,
        "sensors_reset": reset_sensor_names,
    }


def set_human_viewport_camera(
    camera_path: str,
    *,
    stage: Any | None = None,
    viewport: Any | None = None,
) -> str:
    """Select one live USD camera only for the interactive human viewport.

    This function deliberately receives neither policy observations nor the
    semantic camera mapping.  It does not enable viewport capture.  Optional
    ``stage`` and ``viewport`` arguments keep the validation independently
    testable without booting Isaac Sim.

    Returns:
        The previously active viewport camera path.
    """

    if stage is None:
        import omni.usd

        usd_context = omni.usd.get_context()
        stage = usd_context.get_stage() if usd_context is not None else None
    if stage is None:
        raise RuntimeError(
            f"--viewport-camera-path cannot resolve the live USD stage: {camera_path}"
        )

    camera_prim = stage.GetPrimAtPath(camera_path)
    if camera_prim is None or not camera_prim.IsValid():
        raise ValueError(
            f"--viewport-camera-path does not resolve to a live USD prim: {camera_path}"
        )
    if camera_prim.GetTypeName() != "Camera":
        raise ValueError(
            "--viewport-camera-path must resolve to a USD Camera prim: "
            f"path={camera_path} type={camera_prim.GetTypeName()!r}"
        )

    if viewport is None:
        import omni.kit.viewport.utility as viewport_utils

        viewport = viewport_utils.get_active_viewport()
    if viewport is None:
        raise RuntimeError(
            "--viewport-camera-path requires an active interactive viewport; "
            "run with --no-headless"
        )

    previous_path_value = getattr(viewport, "camera_path", "")
    previous_path = getattr(previous_path_value, "pathString", str(previous_path_value))
    viewport.set_active_camera(camera_path)
    active_path_value = getattr(viewport, "camera_path", "")
    active_path = getattr(active_path_value, "pathString", str(active_path_value))
    if active_path != camera_path:
        raise RuntimeError(
            "active viewport rejected --viewport-camera-path: "
            f"requested={camera_path} active={active_path}"
        )

    print(
        "[runner] human viewport camera changed "
        f"from={previous_path} to={active_path}; "
        "policy camera mapping unchanged; video capture not enabled",
        flush=True,
    )
    return previous_path


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
        # By default record the active human-facing viewport.  An explicit
        # camera path makes evidence reproducible and lets multi-object runs
        # use the same top camera that is fed to the policy, rather than a
        # viewport angle that may crop some assets.
        capture_camera_path = args.capture_camera_path or viewport.camera_path.pathString
        if args.capture_camera_path is not None:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            camera_prim = stage.GetPrimAtPath(capture_camera_path) if stage is not None else None
            if camera_prim is None or not camera_prim.IsValid():
                raise ValueError(
                    f"--capture-camera-path does not resolve to a live USD prim: {capture_camera_path}"
                )
        options.camera = capture_camera_path
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
            f"camera={capture_camera_path} "
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
        "--camera-layout",
        choices=("wrist-only", "dual", "triple"),
        default="dual",
        help=(
            "SO101 finetuned video schema: wrist-only=[wrist], "
            "dual=[top,wrist], triple=[top,left,wrist]."
        ),
    )

    parser.add_argument(
        "--policy-schema",
        choices=("oxe", "so101-new-embodiment"),
        default=None,
        help="Observation/action schema expected by the GR00T bridge.",
    )

    parser.add_argument(
        "--so101-checkpoint-joint-units",
        dest="so101_checkpoint_arm_units",
        choices=SO101_CHECKPOINT_ARM_UNIT_CHOICES,
        default=SO101_ARM_UNITS_LEROBOT_MOTOR,
        help=(
            "Dataset-space units used by the SO101 checkpoint's first five state/action joints. "
            "Use lerobot_motor_units for LeIsaac/sim data or degrees for real-robot data. "
            "The gripper always remains in the LeRobot [0,100] range."
        ),
    )
    parser.add_argument(
        "--so101-initial-pose",
        choices=SO101_INITIAL_POSE_CHOICES,
        default=SO101_INITIAL_POSE_AUTO,
        help=(
            "SO101 reset pose. auto selects the unit-matched dataset-start reference for "
            "so101-finetuned runs and keeps the authored USD default for other routes; "
            "dataset-start explicitly selects that preset; dataset-first-frame is a legacy "
            "alias; usd-default disables it."
        ),
    )
    parser.add_argument(
        "--so101-initial-dataset-state",
        nargs=6,
        type=float,
        default=None,
        metavar=("PAN", "LIFT", "ELBOW", "WRIST_FLEX", "WRIST_ROLL", "GRIPPER"),
        help=(
            "Override the six-value SO101 reset preset in checkpoint dataset coordinates. "
            "The first five values use --so101-checkpoint-joint-units; gripper remains "
            "LeRobot [0,100]. Only valid for so101-finetuned joint control."
        ),
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
    parser.add_argument(
        "--inject-so101-eval-cameras",
        action="store_true",
        help=(
            "Process-locally add the fixed top/left cameras required by dual/triple "
            "SO101 checkpoint evaluation without modifying the LeIsaac checkout."
        ),
    )
    parser.add_argument(
        "--isaac-exterior-camera-key",
        default=None,
        help="Isaac obs['policy'] key for OXE/DROID exterior_image_1_left.",
    )
    parser.add_argument(
        "--isaac-front-camera-key",
        default=None,
        help="Isaac obs['policy'] key mapped to SO101 GR00T video.top.",
    )
    parser.add_argument(
        "--isaac-left-camera-key",
        default=None,
        help="Isaac obs['policy'] key mapped to SO101 GR00T video.left in triple mode.",
    )
    parser.add_argument(
        "--isaac-wrist-camera-key",
        default=None,
        help="Isaac obs['policy'] key mapped to the GR00T wrist video input.",
    )

    # Gym task chooses the base environment implementation. The scene profile
    # owns deployment-time tray/LEGO composition and may also rebind the env
    # cfg success criterion. The runner disables that termination by default.
    parser.add_argument(
        "--task",
        default=None,
        help=(
            "Registered Gym task ID for the base robot/action/observation environment. "
            "Defaults to LeIsaac-SO101-SmartTask-v0 for SO101."
        ),
    )
    parser.add_argument(
        "--scene-profile",
        choices=scene_profiles.SCENE_PROFILE_CHOICES,
        default=scene_profiles.TASK_DEFAULT_SCENE_PROFILE,
        help=(
            "Deployment-time object layout and profile-owned target/configured-success behavior, independent of "
            "Gym task registration. See scene_profiles.SCENE_PROFILE_CHOICES for the complete "
            "supported set; task-default keeps the selected task scene."
        ),
    )
    parser.add_argument(
        "--red24-material",
        choices=(RED24_MATERIAL_AUTO, *scene_profiles.RED24_MATERIAL_CHOICES),
        default=RED24_MATERIAL_AUTO,
        help=(
            "Visual material for managed red 2x4 LEGO assets. auto selects pure "
            "real-red for degree-unit real-data checkpoints and preserves the "
            "authored asset material for LeRobot-motor-unit sim checkpoints."
        ),
    )
    parser.add_argument(
        "--so101-robot-material",
        choices=SO101_ROBOT_MATERIAL_CHOICES,
        default=SO101_ROBOT_MATERIAL_AUTO,
        help=(
            "SO101 printed-structure appearance. auto selects real-white for "
            "degree-unit real-data checkpoints and preserves the authored yellow "
            "asset for LeRobot-motor-unit sim checkpoints. real-white changes only "
            "material_a_3d_printed; motor material and physics remain unchanged."
        ),
    )

    # GR00T bridge 的地址和端口。默认对应 README 中 bridge 的启动参数。
    parser.add_argument("--bridge-host", default="127.0.0.1")
    parser.add_argument("--bridge-port", type=int, default=5577)

    # 传给 GR00T 的自然语言任务指令。None 表示读取所选 env cfg 的 task_description。
    parser.add_argument(
        "--instruction",
        default=None,
        help=(
            "Language instruction used by the checkpoint. "
            "Defaults to the legacy instruction for the base SO101/Franka tasks, otherwise "
            "to the selected env config's task_description."
        ),
    )

    parser.add_argument(
        "--target-object-key",
        default="auto",
        help=(
            "Scene rigid-object key used for SmartTask metrics/debug. "
            "Single-object scene profiles resolve auto; multi-lego-tray requires one explicit object key. "
            "task-default reads the selected task configuration."
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
        default="auto",
        help=(
            "Target object source for the legacy LeIsaac base SmartTask: "
            "'auto' uses its red 2x4 cuboid workaround but leaves every other task's scene unchanged; "
            "'cuboid' explicitly selects that workaround, "
            "'scene' relies on the LeIsaac scene parser, "
            "or pass a complete USD path."
        ),
    )
    parser.add_argument(
        "--smart-target-prim-path",
        default=SMART_TARGET_MANAGED_PRIM_PATH,
        help="Prim path used when --smart-target-asset is cuboid or a USD path.",
    )
    parser.add_argument(
        "--smart-target-pos",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional SmartTask target center position override in scene-local meters.",
    )
    parser.add_argument(
        "--smart-target-cuboid-size",
        nargs=3,
        type=float,
        default=SMART_TARGET_CUBOID_SIZE,
        metavar=("X", "Y", "Z"),
        help="Fallback cuboid size in meters when --smart-target-asset cuboid is used.",
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
        "--so101-arm-target-scale",
        "--so101-arm-delta-scale",
        dest="so101_arm_target_scale",
        type=float,
        default=1.0,
        help=(
            "Interpolate from current Isaac radians toward the finetuned checkpoint's decoded "
            "absolute arm target. 1.0 uses the full target. "
            "--so101-arm-delta-scale is a deprecated compatibility alias."
        ),
    )
    parser.add_argument(
        "--so101-max-arm-step-rad",
        "--so101-max-arm-delta",
        dest="so101_max_arm_step_rad",
        type=float,
        default=0.08,
        help=(
            "Maximum change in Isaac radians for each arm joint per policy action. "
            "The command may be held for multiple env steps. Defaults to the provisional "
            "0.08-rad smoke-test guard; use 0 to disable. "
            "--so101-max-arm-delta is a deprecated compatibility alias."
        ),
    )
    parser.add_argument(
        "--so101-max-gripper-step-rad",
        type=float,
        default=0.0,
        help=(
            "Maximum gripper-joint change in Isaac radians per policy action, "
            "measured from the current simulated joint. The command may be held "
            "for multiple env steps. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--so101-gripper-effort-limit-sim",
        type=float,
        default=None,
        help=(
            "Optional fixed PhysX effort limit for the SO101 gripper joint. "
            "Setting this disables LeIsaac's mass-based dynamic effort reset so "
            "the limit remains explicit and reproducible."
        ),
    )
    parser.add_argument(
        "--so101-gripper-min",
        type=float,
        default=None,
        help="Optional lower clamp in the checkpoint gripper's LeRobot [0,100] range.",
    )
    parser.add_argument(
        "--so101-gripper-max",
        type=float,
        default=None,
        help="Optional upper clamp in the checkpoint gripper's LeRobot [0,100] range.",
    )
    parser.add_argument(
        "--dynamic-reset-gripper-effort-limit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override LeIsaac's dynamic SO101 gripper effort-limit reset. "
            "Use --no-dynamic-reset-gripper-effort-limit to test whether contact-time "
            "command tracking is being suppressed; omit to preserve the task config."
        ),
    )

    # 请求 GR00T 的次数。每次请求会返回一个 action chunk。
    parser.add_argument(
        "--max-policy-calls",
        type=int,
        default=1,
        help="Maximum number of action chunks to request from the GR00T bridge.",
    )

    # 每个 action chunk 执行多少步。0 表示执行完整 chunk。
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=0,
        help="Number of policy actions to execute from each GR00T chunk. 0 means execute the full returned chunk.",
    )
    parser.add_argument(
        "--policy-action-hz",
        type=float,
        default=None,
        help=(
            "Time base of consecutive actions in the returned chunk. SO101 finetuned mode "
            "defaults to the prepared dataset's 30 Hz; zero-shot modes default to one action "
            "per env step to preserve their previous behavior."
        ),
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
        help=(
            "Wall-clock sleep after each Isaac environment step so motion is visible in viewport mode; "
            "this does not change simulated time."
        ),
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
    parser.add_argument(
        "--viewport-camera-path",
        default=None,
        help=(
            "Live USD Camera prim shown only in the interactive human viewport after reset. "
            "This does not change policy observation camera keys and does not record video. "
            "Requires --no-headless; example physical-left path: "
            "/World/envs/env_0/Scene/camera_left_xform/camera_left."
        ),
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
    parser.add_argument(
        "--capture-camera-path",
        default=None,
        help=(
            "Optional live USD camera prim used only for the evidence video. "
            "The policy observation mapping is unchanged. Default: active viewport camera."
        ),
    )
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
    parser.add_argument(
        "--action-trace-jsonl",
        default=None,
        help=(
            "Optional JSONL output path for decoded action chunks, applied commands, "
            "post-call joint states, and SmartTask metrics."
        ),
    )
    parser.add_argument(
        "--settle-env-steps",
        type=int,
        default=0,
        help=(
            "Advance this many physics/environment steps after reset while holding the "
            "reset joint pose, then build the first policy observation. Defaults to 0 "
            "for backward compatibility; use this when spawned dynamic objects must "
            "fall onto their support surface before evaluation."
        ),
    )
    parser.add_argument(
        "--contact-probe",
        action="store_true",
        help=(
            "Enable opt-in gripper/jaw contact sensors against the selected target and "
            "record per-physics-substep force, point, normal, and separation data in "
            "--action-trace-jsonl."
        ),
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
    if (
        args.deployment_mode != "so101-finetuned"
        and args.so101_checkpoint_arm_units != SO101_ARM_UNITS_LEROBOT_MOTOR
    ):
        raise ValueError(
            "--so101-checkpoint-joint-units degrees requires --deployment-mode so101-finetuned"
        )
    dataset_start_pose_requested = args.so101_initial_pose in (
        SO101_INITIAL_POSE_DATASET_START,
        SO101_INITIAL_POSE_DATASET_FIRST_FRAME,
    )
    if dataset_start_pose_requested and args.deployment_mode != "so101-finetuned":
        raise ValueError(
            "--so101-initial-pose dataset-start/dataset-first-frame requires "
            "--deployment-mode so101-finetuned"
        )
    if args.so101_initial_dataset_state is not None:
        if args.deployment_mode != "so101-finetuned":
            raise ValueError(
                "--so101-initial-dataset-state requires "
                "--deployment-mode so101-finetuned"
            )
        if args.so101_initial_pose == SO101_INITIAL_POSE_USD_DEFAULT:
            raise ValueError(
                "--so101-initial-dataset-state cannot be combined with "
                "--so101-initial-pose usd-default"
            )
        initial_dataset_state = np.asarray(
            args.so101_initial_dataset_state,
            dtype=np.float32,
        )
        if not np.all(np.isfinite(initial_dataset_state)):
            raise ValueError("--so101-initial-dataset-state values must be finite")
    if args.deployment_mode != "so101-finetuned" and args.camera_layout != "dual":
        raise ValueError("--camera-layout wrist-only/triple requires --deployment-mode so101-finetuned")
    if args.max_policy_calls < 1:
        raise ValueError("--max-policy-calls must be at least 1")
    if args.action_horizon < 0:
        raise ValueError("--action-horizon must be 0 or a positive integer")
    if not np.isfinite(args.so101_arm_target_scale) or not 0.0 <= args.so101_arm_target_scale <= 1.0:
        raise ValueError("--so101-arm-target-scale must be finite and within [0, 1]")
    if not np.isfinite(args.so101_max_arm_step_rad) or args.so101_max_arm_step_rad < 0.0:
        raise ValueError("--so101-max-arm-step-rad must be finite and non-negative")
    if (
        not np.isfinite(args.so101_max_gripper_step_rad)
        or args.so101_max_gripper_step_rad < 0.0
    ):
        raise ValueError(
            "--so101-max-gripper-step-rad must be finite and non-negative"
        )
    if args.so101_gripper_effort_limit_sim is not None and (
        not np.isfinite(args.so101_gripper_effort_limit_sim)
        or args.so101_gripper_effort_limit_sim <= 0.0
    ):
        raise ValueError(
            "--so101-gripper-effort-limit-sim must be finite and positive"
        )
    if (
        args.so101_gripper_effort_limit_sim is not None
        and args.dynamic_reset_gripper_effort_limit is True
    ):
        raise ValueError(
            "--so101-gripper-effort-limit-sim cannot be combined with "
            "--dynamic-reset-gripper-effort-limit"
        )
    if (
        args.so101_max_gripper_step_rad > 0.0
        or args.so101_gripper_effort_limit_sim is not None
    ) and not is_so101_finetuned_joint_route(args):
        raise ValueError(
            "SO101 gripper step/effort limits require the finetuned SO101 joint route"
        )
    for flag, value in (
        ("--so101-gripper-min", args.so101_gripper_min),
        ("--so101-gripper-max", args.so101_gripper_max),
    ):
        if value is not None and not np.isfinite(value):
            raise ValueError(f"{flag} must be finite when provided")
    if (
        args.so101_gripper_min is not None
        and args.so101_gripper_max is not None
        and args.so101_gripper_min > args.so101_gripper_max
    ):
        raise ValueError("--so101-gripper-min cannot exceed --so101-gripper-max")
    if args.policy_action_hz is not None and (
        not np.isfinite(args.policy_action_hz) or args.policy_action_hz <= 0.0
    ):
        raise ValueError("--policy-action-hz must be finite and positive")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be non-negative")
    if args.settle_env_steps < 0:
        raise ValueError("--settle-env-steps must be non-negative")
    if args.settle_env_steps and (
        args.deployment_mode != "so101-finetuned"
        or args.robot != "so101"
        or args.control_mode != "joint"
    ):
        raise ValueError(
            "--settle-env-steps currently requires SO101 finetuned joint control"
        )
    if args.contact_probe and args.action_trace_jsonl is None:
        raise ValueError("--contact-probe requires --action-trace-jsonl")
    if args.contact_probe and (
        args.deployment_mode != "so101-finetuned" or args.robot != "so101"
    ):
        raise ValueError(
            "--contact-probe currently requires the SO101 finetuned deployment route"
        )
    if not np.isfinite(args.timeout_s) or args.timeout_s <= 0.0:
        raise ValueError("--timeout-s must be finite and positive")
    if not np.isfinite(args.render_sleep_s) or args.render_sleep_s < 0.0:
        raise ValueError("--render-sleep-s must be finite and non-negative")
    if not np.isfinite(args.keep_open_s) or args.keep_open_s < 0.0:
        raise ValueError("--keep-open-s must be finite and non-negative")
    if args.viewport_camera_path is not None and args.headless:
        raise ValueError("--viewport-camera-path requires --no-headless")
    if args.viewport_camera_path is not None and args.capture_video:
        raise ValueError(
            "--viewport-camera-path is human-view-only and cannot be combined with "
            "--capture-video"
        )
    if args.capture_width < 1 or args.capture_height < 1:
        raise ValueError("--capture-width and --capture-height must be positive")
    if not np.isfinite(args.capture_fps) or args.capture_fps <= 0.0:
        raise ValueError("--capture-fps must be finite and positive")
    if not np.isfinite(args.capture_bitrate_mbps) or args.capture_bitrate_mbps <= 0.0:
        raise ValueError("--capture-bitrate-mbps must be finite and positive")
    if args.capture_every_nth_frames < 1:
        raise ValueError("--capture-every-nth-frames must be at least 1")
    if not np.isfinite(args.capture_wait_timeout_s) or args.capture_wait_timeout_s <= 0.0:
        raise ValueError("--capture-wait-timeout-s must be finite and positive")

    if args.task is None:
        args.task = default_task_for_robot(args.robot)
    validate_robot_task(args.robot, args.task)
    scene_profiles.validate_scene_profile_task(args.task, args.scene_profile)
    validate_scene_profile_asset_args(args)
    scene_profile_target = scene_profiles.resolve_scene_profile_target(
        args.scene_profile,
        args.target_object_key,
    )
    if args.scene_profile != scene_profiles.TASK_DEFAULT_SCENE_PROFILE:
        profile = scene_profiles.get_scene_profile(args.scene_profile)
        if profile.requires_explicit_instruction and args.instruction is None:
            raise ValueError(
                f"--scene-profile {args.scene_profile} requires --instruction to match the checkpoint task"
            )
        if scene_profile_target is None:
            raise RuntimeError(f"scene profile {args.scene_profile!r} did not resolve a target")
    if args.policy_schema == "so101-new-embodiment":
        if args.robot != "so101":
            raise ValueError("--policy-schema so101-new-embodiment requires --robot so101")
        if args.control_mode != "joint":
            raise ValueError("--policy-schema so101-new-embodiment currently requires --control-mode joint")
    if args.debug_cameras_only:
        args.debug_cameras = True
    args.red24_material_resolved = resolve_red24_material(
        args.red24_material,
        args.so101_checkpoint_arm_units,
    )
    args.so101_robot_material_resolved = resolve_so101_robot_material(
        args.so101_robot_material,
        args.so101_checkpoint_arm_units,
    )
    if (
        args.so101_robot_material_resolved == SO101_ROBOT_MATERIAL_REAL_WHITE
        and args.robot != "so101"
    ):
        raise ValueError("--so101-robot-material real-white requires --robot so101")
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


def is_so101_finetuned_joint_route(args: argparse.Namespace) -> bool:
    """Return whether the runner uses the SO101 absolute-joint checkpoint contract."""

    return (
        args.deployment_mode == "so101-finetuned"
        and args.robot == "so101"
        and args.control_mode == "joint"
        and args.policy_schema == "so101-new-embodiment"
    )


def resolve_so101_initial_pose(args: argparse.Namespace) -> dict[str, Any] | None:
    """Resolve an optional unit-matched SO101 reset pose.

    ``auto`` aligns finetuned SO101 runs to the unit-matched dataset-start
    reference.  Other routes keep the task's authored USD reset pose.  A
    six-value CLI override is interpreted in the selected checkpoint dataset
    coordinates, including a LeRobot [0, 100] gripper value.
    """

    requested_mode = args.so101_initial_pose
    finetuned_so101_joint = is_so101_finetuned_joint_route(args)

    if requested_mode == SO101_INITIAL_POSE_AUTO:
        resolved_mode = (
            SO101_INITIAL_POSE_DATASET_START
            if finetuned_so101_joint
            else SO101_INITIAL_POSE_USD_DEFAULT
        )
    elif requested_mode == SO101_INITIAL_POSE_DATASET_FIRST_FRAME:
        resolved_mode = SO101_INITIAL_POSE_DATASET_START
    else:
        resolved_mode = requested_mode

    custom_state = args.so101_initial_dataset_state
    if resolved_mode == SO101_INITIAL_POSE_USD_DEFAULT:
        if custom_state is not None:
            raise ValueError(
                "--so101-initial-dataset-state cannot be combined with "
                "--so101-initial-pose usd-default"
            )
        return None

    if not finetuned_so101_joint:
        raise ValueError(
            "--so101-initial-pose dataset-start/dataset-first-frame and "
            "--so101-initial-dataset-state require "
            "--deployment-mode so101-finetuned --robot so101 --control-mode joint"
        )

    arm_units = args.so101_checkpoint_arm_units
    if custom_state is None:
        dataset_state = np.asarray(
            SO101_DATASET_START_MEDIAN_STATE[arm_units],
            dtype=np.float32,
        )
        source = SO101_DATASET_START_MEDIAN_SOURCE[arm_units]
    else:
        dataset_state = np.asarray(custom_state, dtype=np.float32)
        source = "cli:--so101-initial-dataset-state"

    joint_rad = so101_dataset_to_isaac_rad(dataset_state, arm_units).reshape(6)
    lower = SO101_USD_JOINT_LIMITS_RAD[:, 0]
    upper = SO101_USD_JOINT_LIMITS_RAD[:, 1]
    if np.any(joint_rad < lower - 1e-6) or np.any(joint_rad > upper + 1e-6):
        raise ValueError(
            "SO101 initial pose falls outside the converter's USD joint limits: "
            f"dataset_state={dataset_state.tolist()} arm_units={arm_units} "
            f"joint_rad={joint_rad.tolist()}"
        )

    return {
        "mode": resolved_mode,
        "source": source,
        "arm_units": arm_units,
        "dataset_state": dataset_state,
        "joint_rad": joint_rad,
    }


def apply_so101_initial_pose(
    env_cfg: Any,
    pose: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Apply an already-resolved pose to this env config before ``gym.make()``."""

    if pose is None:
        return None

    try:
        init_state = env_cfg.scene.robot.init_state
        authored_joint_pos = init_state.joint_pos
    except AttributeError as exc:
        raise ValueError("Selected env cfg does not expose scene.robot.init_state.joint_pos") from exc
    if not isinstance(authored_joint_pos, dict):
        raise TypeError(
            "env_cfg.scene.robot.init_state.joint_pos must be a joint-name dictionary, "
            f"got {type(authored_joint_pos).__name__}"
        )

    updated_joint_pos = dict(authored_joint_pos)
    updated_joint_pos.update(
        {
            joint_name: float(joint_value)
            for joint_name, joint_value in zip(SO101_JOINT_NAMES, pose["joint_rad"])
        }
    )
    init_state.joint_pos = updated_joint_pos

    # The wrist camera moves with the arm.  Ask IsaacLab to render the reset
    # state before it computes the first RTX-camera observation returned to the
    # policy; otherwise the image may still show the authored USD pose.
    env_cfg.rerender_on_reset = True

    print(
        "[runner] SO101 initial pose configured "
        f"mode={pose['mode']} source={pose['source']} arm_units={pose['arm_units']} "
        f"dataset_values={format_vec(pose['dataset_state'], precision=6)} "
        f"Isaac_radians={format_vec(pose['joint_rad'], precision=6)} "
        "rerender_on_reset=True",
        flush=True,
    )
    return pose


def configure_so101_absolute_joint_actions(
    env_cfg: Any,
    args: argparse.Namespace,
) -> bool:
    """Disable IsaacLab's default-position offsets for absolute SO101 commands."""

    if not is_so101_finetuned_joint_route(args):
        return False

    try:
        action_cfgs = {
            "arm_action": env_cfg.actions.arm_action,
            "gripper_action": env_cfg.actions.gripper_action,
        }
    except AttributeError as exc:
        raise ValueError(
            "SO101 finetuned joint control requires arm_action and gripper_action configs"
        ) from exc

    for name, action_cfg in action_cfgs.items():
        if not hasattr(action_cfg, "use_default_offset"):
            raise TypeError(
                f"SO101 {name} must expose use_default_offset for absolute joint control"
            )
        action_cfg.use_default_offset = False

    print(
        "[runner] SO101 action semantics configured "
        "absolute_radians=True arm_action.use_default_offset=False "
        "gripper_action.use_default_offset=False",
        flush=True,
    )
    return True


def configure_so101_gripper_effort_limit(
    env_cfg: Any,
    args: argparse.Namespace,
) -> dict[str, float] | None:
    """Apply one explicit gripper effort limit before ``gym.make()``."""

    requested = args.so101_gripper_effort_limit_sim
    if requested is None:
        return None
    if not is_so101_finetuned_joint_route(args):
        raise ValueError(
            "--so101-gripper-effort-limit-sim requires the finetuned SO101 joint route"
        )

    try:
        actuator_cfg = env_cfg.scene.robot.actuators["sts3215-gripper"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(
            "Selected SO101 env cfg does not expose the sts3215-gripper actuator"
        ) from exc
    if not hasattr(actuator_cfg, "effort_limit_sim"):
        raise TypeError(
            "SO101 sts3215-gripper actuator must expose effort_limit_sim"
        )

    previous = actuator_cfg.effort_limit_sim
    actuator_cfg.effort_limit_sim = float(requested)
    env_cfg.dynamic_reset_gripper_effort_limit = False
    summary = {
        "previous_effort_limit_sim": float(previous),
        "requested_effort_limit_sim": float(requested),
    }
    print(
        "[runner] SO101 explicit gripper effort limit "
        f"previous={summary['previous_effort_limit_sim']:.6g} "
        f"requested={summary['requested_effort_limit_sim']:.6g} "
        "dynamic_reset=False",
        flush=True,
    )
    return summary


def validate_so101_gripper_effort_limit(
    env: Any,
    args: argparse.Namespace,
) -> float | None:
    """Verify the explicit gripper effort limit reached the live articulation."""

    requested = args.so101_gripper_effort_limit_sim
    if requested is None:
        return None
    if env.cfg.dynamic_reset_gripper_effort_limit:
        raise RuntimeError(
            "Explicit SO101 gripper effort limit requires dynamic reset to remain disabled"
        )

    robot_data = env.scene["robot"].data
    joint_names = tuple(robot_data.joint_names)
    try:
        gripper_joint_id = joint_names.index("gripper")
    except ValueError as exc:
        raise RuntimeError(
            f"Live SO101 articulation has no gripper joint: names={joint_names}"
        ) from exc

    limits = robot_data.joint_effort_limits
    if (
        limits.ndim != 2
        or limits.shape[0] < 1
        or limits.shape[1] <= gripper_joint_id
    ):
        raise RuntimeError(
            "SO101 joint_effort_limits does not cover the gripper joint: "
            f"gripper_joint_id={gripper_joint_id} "
            f"got {tuple(limits.shape)}"
        )
    actual = float(limits[0, gripper_joint_id].detach().cpu())
    if not np.isclose(actual, requested, atol=1e-6, rtol=0.0):
        raise RuntimeError(
            "SO101 gripper effort limit did not reach the live articulation: "
            f"requested={requested} actual={actual}"
        )
    print(
        f"[runner] SO101 live gripper effort limit verified={actual:.6g}",
        flush=True,
    )
    return actual


def validate_so101_absolute_joint_actions(
    env: Any,
    args: argparse.Namespace,
) -> None:
    """Fail closed if instantiated action terms would add the reset pose again."""

    if not is_so101_finetuned_joint_route(args):
        return

    offsets: dict[str, list[float]] = {}
    for name in ("arm_action", "gripper_action"):
        action_term = env.action_manager.get_term(name)
        if getattr(action_term.cfg, "use_default_offset", None) is not False:
            raise RuntimeError(
                f"SO101 {name} did not preserve use_default_offset=False at runtime"
            )
        if not hasattr(action_term, "_offset"):
            raise RuntimeError(f"SO101 {name} does not expose a verifiable runtime offset")

        offset = action_term._offset
        if isinstance(offset, torch.Tensor):
            offset_array = offset.detach().cpu().numpy()
        else:
            offset_array = np.asarray(offset, dtype=np.float32)
        if not np.all(np.isfinite(offset_array)) or not np.allclose(
            offset_array,
            0.0,
            atol=1e-8,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"SO101 {name} runtime offset must be zero for absolute commands, "
                f"got {np.asarray(offset_array).tolist()}"
            )
        offsets[name] = np.asarray(offset_array, dtype=np.float64).round(8).reshape(-1).tolist()

    print(
        "[runner] SO101 action semantics verified "
        f"absolute_radians=True runtime_offsets={offsets}",
        flush=True,
    )


def log_so101_initial_joint_state(
    policy_obs: dict[str, torch.Tensor],
    args: argparse.Namespace,
    runtime_joint_limits: torch.Tensor | None,
    configured_pose: dict[str, Any] | None,
) -> None:
    """Log the post-reset SO101 state before camera debug or bridge access."""

    print(f"[runner] initial joint_pos values={tensor_values(policy_obs['joint_pos'])}", flush=True)
    if runtime_joint_limits is None:
        return

    initial_joint_rad = policy_obs["joint_pos"].detach().cpu().numpy()[0].astype(np.float32)
    initial_joint_dataset = isaac_rad_to_so101_dataset(
        initial_joint_rad,
        args.so101_checkpoint_arm_units,
    )
    print(
        "[runner] SO101 state conversion "
        f"Isaac radians={initial_joint_rad.round(4).tolist()} -> "
        f"dataset values={initial_joint_dataset.round(4).tolist()} "
        f"arm_units={args.so101_checkpoint_arm_units}",
        flush=True,
    )
    if configured_pose is not None:
        max_abs_error_rad = float(
            np.max(np.abs(initial_joint_rad - configured_pose["joint_rad"]))
        )
        print(
            "[runner] SO101 initial pose verification "
            f"source={configured_pose['source']} "
            f"max_abs_error_rad={max_abs_error_rad:.8f}",
            flush=True,
        )
        if max_abs_error_rad > SO101_INITIAL_POSE_VERIFY_ATOL_RAD:
            raise RuntimeError(
                "SO101 reset did not reach the configured initial pose: "
                f"max_abs_error_rad={max_abs_error_rad:.8f} "
                f"tolerance_rad={SO101_INITIAL_POSE_VERIFY_ATOL_RAD:.8f}"
            )
    print(
        "[runner] SO101 Isaac runtime soft joint limits radians="
        f"{runtime_joint_limits.detach().cpu().numpy().round(4).tolist()}",
        flush=True,
    )


def get_so101_runtime_joint_limits(env: Any, policy_schema: str) -> torch.Tensor | None:
    """Validate the loaded SO101 joint order/ranges and return its soft limits."""

    if policy_schema != "so101-new-embodiment":
        return None

    robot_data = env.scene["robot"].data
    actual_joint_names = tuple(robot_data.joint_names[:6])
    if actual_joint_names != SO101_JOINT_NAMES:
        raise ValueError(
            "SO101 joint order differs from the deployment converter: "
            f"expected={list(SO101_JOINT_NAMES)} actual={list(actual_joint_names)}"
        )

    soft_limits = robot_data.soft_joint_pos_limits
    if soft_limits.ndim != 3 or soft_limits.shape[1] < 6 or soft_limits.shape[2] != 2:
        raise ValueError(
            "Expected Isaac robot soft_joint_pos_limits shape (num_envs, >=6, 2), "
            f"got {tuple(soft_limits.shape)}"
        )
    limits = soft_limits[0, :6].detach().clone()
    validate_runtime_joint_limits_match_converter(limits.detach().cpu().numpy())
    return limits


def main() -> None:
    """Isaac runner 主流程。"""

    args = parse_args()
    # Resolve dataset coordinates and reject invalid custom poses before
    # AppLauncher starts Kit, so a CLI/config error cannot leave a simulator
    # process behind.
    configured_initial_pose = resolve_so101_initial_pose(args)

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

    # 读取所选 task 的 env config。Gym ID 只代表注册存在；entry-point 缺类时
    # parse_env_cfg 仍会失败，所以这里先打印完整异常并正常关闭 SimulationApp。
    contact_sensor_names: tuple[str, ...] = ()
    try:
        smart_scene_usd_path = install_smart_task_asset_patch(args)
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        if args.inject_so101_eval_cameras:
            if args.deployment_mode != "so101-finetuned" or args.robot != "so101":
                raise ValueError(
                    "--inject-so101-eval-cameras requires the SO101 finetuned deployment route"
                )
            physical_camera_mapping = configure_so101_eval_cameras(env_cfg, args.camera_layout)
            selected_camera_mapping = apply_injected_camera_mapping(
                args,
                physical_camera_mapping,
            )
            print(
                "[runner] injected SO101 eval cameras "
                f"physical_mapping={physical_camera_mapping} "
                f"model_slot_mapping={selected_camera_mapping}",
                flush=True,
            )
        scene_profile_target = scene_profiles.resolve_scene_profile_target(
            args.scene_profile,
            args.target_object_key,
        )
        if scene_profile_target is not None:
            scene_profiles.apply_scene_profile_target(
                env_cfg,
                args.scene_profile,
                scene_profile_target,
            )
            configured_target_object_name = scene_profile_target
        else:
            configured_target_object_name = target_name_from_env_cfg(env_cfg)
        args.instruction = resolve_task_instruction(args.instruction, env_cfg, args.task)
        if args.contact_probe:
            contact_sensor_names = (
                contact_probe.configure_so101_target_contact_probe(
                    env_cfg,
                    configured_target_object_name,
                )
            )
            print(
                "[runner] contact probe configured "
                f"target={configured_target_object_name} "
                f"sensors={contact_sensor_names}",
                flush=True,
            )
    except Exception as exc:
        print(
            f"[runner] task setup failed for {args.task!r}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        simulation_app.close()
        raise

    gripper_effort_cfg_summary: dict[str, float] | None = None
    try:
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
        configure_so101_absolute_joint_actions(env_cfg, args)
        apply_so101_initial_pose(env_cfg, configured_initial_pose)
        gripper_effort_cfg_summary = configure_so101_gripper_effort_limit(
            env_cfg,
            args,
        )
        if args.dynamic_reset_gripper_effort_limit is not None:
            env_cfg.dynamic_reset_gripper_effort_limit = (
                args.dynamic_reset_gripper_effort_limit
            )
            print(
                "[runner] dynamic gripper effort-limit reset override="
                f"{env_cfg.dynamic_reset_gripper_effort_limit}",
                flush=True,
            )

        env_cfg.seed = args.seed

        # 关闭 recorder，避免这个实验无意中写 LeIsaac 数据集。
        env_cfg.recorders = None

        # 关闭 time_out termination，方便我们自己控制运行长度。
        env_cfg.terminations.time_out = None

        # SmartTask 当前 success 判断太松，会出现几乎没动就成功的问题。
        # 默认禁用它；用户显式加 --use-env-success-termination 时才恢复。
        if not args.use_env_success_termination and hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None
    except Exception as exc:
        print(
            f"[runner] env config failed for {args.task!r}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        simulation_app.close()
        raise

    # 创建 gymnasium env，并拿 unwrapped 环境方便访问 scene、sim、cfg 等属性。
    # gym.make() 期间也可能因为 scene/asset 配置错误失败。必须在关闭
    # SimulationApp 之前打印 traceback；Isaac/Kit 的完整 shutdown 会卸载日志
    # 和运行框架，如果先 close()，真正的 Python 异常可能在终端里消失，只剩
    # 下一个 shell prompt，看起来像“窗口无报错自动关闭”。
    print(f"[runner] creating Isaac env via gym.make(task={args.task!r})", flush=True)
    env = None
    so101_robot_material_summary: dict[str, Any] = {
        "requested": args.so101_robot_material,
        "resolved": args.so101_robot_material_resolved,
        "applied": False,
        "reason": f"robot={args.robot}",
    }
    live_gripper_effort_limit_sim = None
    try:
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        validate_so101_absolute_joint_actions(env, args)
        live_gripper_effort_limit_sim = validate_so101_gripper_effort_limit(
            env,
            args,
        )
        if args.robot == "so101":
            so101_robot_material_summary = apply_so101_robot_material(
                env,
                args.so101_robot_material_resolved,
                requested_material=args.so101_robot_material,
                expected_shader_count=int(getattr(env, "num_envs", 1)),
            )
    except Exception as exc:
        print(
            f"[runner] env creation/validation failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        simulation_app.close()
        raise
    print("[runner] Isaac env created successfully", flush=True)

    # 相机历史缓存，用于构造 GR00T 的两帧 video 输入。
    history = FrameHistory(horizon=max(args.warmup_frames, 2))
    capture_instance = None
    action_trace_file = None

    try:
        print(
            f"[runner] robot={args.robot} task={args.task} scene_profile={args.scene_profile} "
            f"control mode={args.control_mode} "
            f"teleop_device={teleop_device}",
            flush=True,
        )
        print(f"[runner] instruction={args.instruction!r}", flush=True)

        # reset Isaac env，拿到第一帧 observation。相机 debug 也必须放在 reset 后，
        # 因为此时 sensor/view/prim 才完整实例化。
        obs, _ = env.reset()
        policy_obs = obs["policy"]
        so101_runtime_joint_limits = get_so101_runtime_joint_limits(env, args.policy_schema)
        log_so101_initial_joint_state(
            policy_obs,
            args,
            so101_runtime_joint_limits,
            configured_initial_pose,
        )
        resolved_camera_profile, camera_mapping = resolve_camera_mapping(args, env)
        camera_mapping = active_camera_mapping(args.policy_schema, args.camera_layout, camera_mapping)
        validate_camera_mapping(policy_obs, camera_mapping)
        target_object_name = resolve_target_object_name(
            env,
            args.target_object_key,
            configured_target_object_name,
        )
        print(
            f"[runner] camera layout={args.camera_layout} profile={args.camera_profile} "
            f"resolved={resolved_camera_profile}",
            flush=True,
        )
        oxe_video_keys = {
            "exterior": "exterior_image_1_left",
            "wrist": "wrist_image_left",
        }
        for role, observation_key in camera_mapping.items():
            groot_key = oxe_video_keys[role] if args.policy_schema == "oxe" else role
            print(f"[runner] Isaac obs['policy'][{observation_key!r}] -> GR00T video.{groot_key}", flush=True)
        if args.viewport_camera_path is not None:
            set_human_viewport_camera(args.viewport_camera_path)
        print(f"[runner] target object={target_object_name}", flush=True)
        target_asset = env.scene[target_object_name]
        target_cfg = getattr(target_asset, "cfg", None)
        target_prim_path = getattr(target_cfg, "prim_path", "<unknown>")
        print(
            "[runner] target object state "
            f"prim_path={target_prim_path} "
            f"root_pos_w={format_vec(tensor_first_row(target_asset.data.root_pos_w))}",
            flush=True,
        )

        pre_settle_metrics = smart_task_metrics(
            env,
            args.robot,
            target_object_name,
        )
        pre_settle_spatial_metrics = smart_task_spatial_metrics(
            env,
            target_object_name,
        )
        settle_step_metrics: list[dict[str, Any]] = []
        settle_pose_restore: dict[str, Any] | None = None
        if args.settle_env_steps:
            reset_joint_pos = policy_obs["joint_pos"].detach().clone()
            print(
                "[runner] settling dynamic scene before first policy observation "
                f"env_steps={args.settle_env_steps} physics_dt_s={float(env.physics_dt):.8g}",
                flush=True,
            )
            policy_obs, settle_step_metrics = settle_environment_at_reset_pose(
                env,
                policy_obs,
                env_steps=args.settle_env_steps,
                robot_kind=args.robot,
                target_object_name=target_object_name,
                contact_sensor_names=contact_sensor_names,
            )
            policy_obs, settle_pose_restore = restore_robot_frame0_after_settle(
                env,
                reset_joint_pos,
                target_object_name=target_object_name,
            )
            print(
                "[runner] settle complete "
                f"target_root_pos_w={format_vec(tensor_first_row(target_asset.data.root_pos_w))} "
                "robot_frame0_restore_max_error_rad="
                f"{settle_pose_restore['joint_max_abs_error_rad']:.3g}",
                flush=True,
            )

        if args.action_trace_jsonl is not None:
            action_trace_path = Path(args.action_trace_jsonl).expanduser()
            if not action_trace_path.is_absolute():
                action_trace_path = SCRIPT_DIR / action_trace_path
            action_trace_path = action_trace_path.resolve()
            action_trace_path.parent.mkdir(parents=True, exist_ok=True)
            action_trace_file = action_trace_path.open("w", encoding="utf-8")
            print(f"[runner] action trace JSONL={action_trace_path}", flush=True)
            write_action_trace_record(
                action_trace_file,
                {
                    "type": "run_config",
                    "instruction": args.instruction,
                    "camera_layout": args.camera_layout,
                    "camera_mapping": camera_mapping,
                    "checkpoint_arm_units": args.so101_checkpoint_arm_units,
                    "red24_material_requested": args.red24_material,
                    "red24_material_resolved": args.red24_material_resolved,
                    "so101_robot_material_requested": args.so101_robot_material,
                    "so101_robot_material_resolved": args.so101_robot_material_resolved,
                    "so101_robot_material_application": so101_robot_material_summary,
                    "seed": args.seed,
                    "action_horizon": args.action_horizon,
                    "max_policy_calls": args.max_policy_calls,
                    "arm_target_scale": args.so101_arm_target_scale,
                    "max_arm_step_rad": args.so101_max_arm_step_rad,
                    "max_gripper_step_rad": args.so101_max_gripper_step_rad,
                    "gripper_effort_limit_sim_requested": (
                        args.so101_gripper_effort_limit_sim
                    ),
                    "gripper_effort_limit_cfg": gripper_effort_cfg_summary,
                    "gripper_effort_limit_sim_live": (
                        live_gripper_effort_limit_sim
                    ),
                    "settle_env_steps": args.settle_env_steps,
                    "physics_dt_s": float(env.physics_dt),
                    "contact_probe_enabled": bool(args.contact_probe),
                    "dynamic_reset_gripper_effort_limit": (
                        env.cfg.dynamic_reset_gripper_effort_limit
                    ),
                    "pre_settle_metrics": pre_settle_metrics,
                    "pre_settle_spatial_metrics": pre_settle_spatial_metrics,
                    "settle_step_metrics": settle_step_metrics,
                    "settle_pose_restore": settle_pose_restore,
                    "initial_joint_pos_rad": tensor_first_row(policy_obs["joint_pos"]).tolist(),
                    "initial_metrics": smart_task_metrics(env, args.robot, target_object_name),
                    "initial_spatial_metrics": smart_task_spatial_metrics(
                        env,
                        target_object_name,
                    ),
                    "initial_contact_probe": (
                        read_contact_probe_sample(
                            env,
                            contact_sensor_names,
                        )
                        if contact_sensor_names
                        else None
                    ),
                },
            )

        if args.debug_cameras:
            debug_frame_dir = args.debug_camera_frame_dir
            if debug_frame_dir is None and args.debug_cameras_only:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_frame_dir = f"runs/camera_debug/{args.robot}_{args.control_mode}_{timestamp}"
            print_camera_debug(env, policy_obs, args.robot, debug_frame_dir, camera_mapping, target_object_name)

        if args.debug_cameras_only:
            keep_open(simulation_app, env, args.keep_open_s)
            return

        env_step_dt_s = float(env.step_dt)
        policy_action_hz = args.policy_action_hz
        if policy_action_hz is None:
            policy_action_hz = 30.0 if args.policy_schema == "so101-new-embodiment" else 1.0 / env_step_dt_s
        env_steps_per_policy_action, effective_policy_action_hz = resolve_env_steps_per_policy_action(
            policy_action_hz,
            env_step_dt_s,
        )
        print(
            "[runner] action timing "
            f"requested_policy_hz={policy_action_hz:.6g} "
            f"env_step_dt_s={env_step_dt_s:.8g} env_step_hz={1.0 / env_step_dt_s:.6g} "
            f"env_steps_per_policy_action={env_steps_per_policy_action} "
            f"effective_policy_hz={effective_policy_action_hz:.6g}",
            flush=True,
        )

        # 先 ping bridge，确认 GR00T 进程已经启动，并打印 modality schema。
        print("[runner] ping bridge", flush=True)
        ping = request(args.bridge_host, args.bridge_port, {"endpoint": "ping"}, args.timeout_s)
        if not ping.get("ok"):
            raise RuntimeError(ping)

        print(f"[runner] bridge modality: {ping.get('modality')}", flush=True)
        print(f"[runner] bridge action decoding: {ping.get('action_decoding')}", flush=True)
        validate_bridge_camera_layout(args, ping)

        print_metrics(env, "initial", args.robot, target_object_name)

        build_observation = build_oxe_observation
        if args.policy_schema == "so101-new-embodiment":
            build_observation = partial(
                build_so101_new_embodiment_observation,
                camera_layout=args.camera_layout,
                checkpoint_arm_units=args.so101_checkpoint_arm_units,
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
            if args.policy_schema == "so101-new-embodiment":
                print(
                    "[runner] SO101 action values below are decoded absolute dataset targets "
                    f"arm_units={args.so101_checkpoint_arm_units}",
                    flush=True,
                )
            print(f"[runner] action summary: {summarize_action(action)}", flush=True)

            trace_record = {
                "type": "policy_call",
                "call_index": call_idx + 1,
                "joint_pos_before_rad": tensor_first_row(policy_obs["joint_pos"]).tolist(),
                "metrics_before": smart_task_metrics(env, args.robot, target_object_name),
                "spatial_metrics_before": smart_task_spatial_metrics(
                    env,
                    target_object_name,
                ),
                "decoded_dataset_action": {
                    key: np.asarray(value).tolist() for key, value in sorted(action.items())
                },
                "applied_commands_rad": [],
                "executed_step_metrics": [],
            }

            if args.dry_run:
                print_metrics(env, "dry-run", args.robot, target_object_name)
                trace_record["executed_action_steps"] = 0
                trace_record["joint_pos_after_rad"] = tensor_first_row(policy_obs["joint_pos"]).tolist()
                trace_record["metrics_after"] = smart_task_metrics(env, args.robot, target_object_name)
                trace_record["spatial_metrics_after"] = smart_task_spatial_metrics(
                    env,
                    target_object_name,
                )
                write_action_trace_record(action_trace_file, trace_record)
                continue

            # 决定这个 chunk 执行多少步。
            # 默认 action_horizon=0，表示执行 GR00T 返回的完整 chunk；长度由 checkpoint schema 决定。
            returned_chunk_len = action_chunk_length(action)
            num_action_steps = (
                returned_chunk_len if args.action_horizon <= 0 else min(args.action_horizon, returned_chunk_len)
            )
            if args.action_horizon > returned_chunk_len:
                print(
                    "[runner] requested action horizon exceeds returned chunk; "
                    f"requested={args.action_horizon} returned={returned_chunk_len} "
                    f"executing={num_action_steps}",
                    flush=True,
                )
            print(f"[runner] executing {num_action_steps} action steps from this chunk", flush=True)

            # 第一次真正执行动作前启动 viewport capture。这样不会把等待 GR00T
            # 推理的空窗录进去，也能根据第一段 action chunk 估算总帧数。
            if args.capture_video and capture_instance is None:
                remaining_policy_calls = args.max_policy_calls - call_idx
                estimated_total_frames = max(
                    1,
                    int(num_action_steps * remaining_policy_calls * env_steps_per_policy_action),
                )
                capture_instance = start_viewport_video_capture(args, estimated_total_frames)

            # 内层循环：逐步执行 action chunk。
            for step_idx in range(num_action_steps):
                conversion_diagnostics: dict[str, Any] = {}
                command = action_step_to_leisaac_tensor(
                    args,
                    action,
                    step_idx,
                    env.device,
                    policy_obs,
                    so101_runtime_joint_limits,
                    conversion_diagnostics,
                )
                trace_record["applied_commands_rad"].append(
                    command.detach().cpu().numpy()[0].tolist()
                )

                # 只在每个 chunk 的第一步打印 command，避免日志过大。
                if step_idx == 0:
                    log_first_leisaac_command(args, command, policy_obs)

                # Zero-order hold: one policy action represents one policy period.
                # For the 30 Hz SO101 data and 60 Hz env this sends the same target
                # through two env.step() calls instead of time-compressing the chunk.
                env_steps_executed = 0
                env_substep_contact_metrics: list[dict[str, Any]] = []
                for env_substep_index in range(1, env_steps_per_policy_action + 1):
                    # LeIsaac 原本会根据 gripper 距离物体动态调整 effort limit。
                    # 因为我们绕开了 LeIsaac 的 teleop loop，所以这里手动调用一次。
                    if env.cfg.dynamic_reset_gripper_effort_limit:
                        dynamic_reset_gripper_effort_limit_sim(env, teleop_device)

                    # 真正把保持中的 action target 送进 IsaacLab env。
                    # 返回值：obs, reward, terminated, timed_out, info。
                    obs, _, terminated, timed_out, _ = env.step(command)
                    env_steps_executed += 1
                    policy_obs = obs["policy"]

                    if contact_sensor_names:
                        env_substep_contact_metrics.append(
                            {
                                "env_substep_index": env_substep_index,
                                "contact_probe_after": read_contact_probe_sample(
                                    env,
                                    contact_sensor_names,
                                ),
                                "spatial_metrics_after": smart_task_spatial_metrics(
                                    env,
                                    target_object_name,
                                ),
                                "gripper_rad_after": float(
                                    env.scene["robot"].data.joint_pos[0, -1]
                                    .detach()
                                    .cpu()
                                ),
                            }
                        )

                    # 用每个 env step 的新 observation 更新相机历史。
                    build_observation(policy_obs, history, args.instruction, args.robot, camera_mapping)

                    if terminated[0] or timed_out[0]:
                        print(
                            f"[runner] episode ended: terminated={terminated[0]} timed_out={timed_out[0]}",
                            flush=True,
                        )
                        if not args.ignore_terminations:
                            stop_run = True
                            break

                    if not simulation_app.is_running():
                        stop_run = True
                        break

                    if args.render_sleep_s > 0:
                        time.sleep(args.render_sleep_s)

                if action_trace_file is not None:
                    trace_record["executed_step_metrics"].append(
                        {
                            "action_step_index": step_idx + 1,
                            "env_steps_executed": env_steps_executed,
                            "metrics_after": smart_task_metrics(
                                env,
                                args.robot,
                                target_object_name,
                            ),
                            "spatial_metrics_after": smart_task_spatial_metrics(
                                env,
                                target_object_name,
                            ),
                            "conversion_diagnostics": conversion_diagnostics,
                            "env_substep_contact_metrics": env_substep_contact_metrics,
                        }
                    )

                if stop_run:
                    break

            print(f"[runner] latest joint_pos values={tensor_values(policy_obs['joint_pos'])}", flush=True)
            print_metrics(env, f"after policy call {call_idx + 1}", args.robot, target_object_name)
            trace_record["executed_action_steps"] = len(trace_record["applied_commands_rad"])
            trace_record["joint_pos_after_rad"] = tensor_first_row(policy_obs["joint_pos"]).tolist()
            trace_record["metrics_after"] = smart_task_metrics(env, args.robot, target_object_name)
            trace_record["spatial_metrics_after"] = smart_task_spatial_metrics(
                env,
                target_object_name,
            )
            write_action_trace_record(action_trace_file, trace_record)

            if stop_run:
                break

        wait_for_viewport_video_capture(capture_instance, simulation_app, env, args.capture_wait_timeout_s)
        keep_open(simulation_app, env, args.keep_open_s)

    except Exception:
        # Isaac SimulationApp.close() may terminate the process before Python
        # prints an uncaught exception. Emit it while the app is still alive so
        # bridge/protocol failures remain visible in the rollout log.
        print("[runner] closed-loop execution failed:", flush=True)
        traceback.print_exc()
        raise
    finally:
        try:
            if action_trace_file is not None:
                action_trace_file.close()
        finally:
            # Trace finalization errors must not strand an Isaac env or
            # SimulationApp process.
            env.close()
            simulation_app.close()


if __name__ == "__main__":
    main()
