"""SO101 joint-coordinate conversion and command safety helpers.

LeIsaac simulates SO101 joints in radians and records the corresponding dataset
state/action through its explicit USD-range <-> LeRobot motor-range mapping.
GR00T checkpoints trained on those datasets therefore consume and return that
dataset motor-coordinate representation.

This is the simulator/data-coordinate mapping used by LeIsaac.  It is distinct
from a physical robot's encoder calibration JSON, and it does not assume that
an episode reset pose, LeRobot ``u=0``, and USD/URDF ``q=0`` are the same pose.

Keep this module independent from Isaac/LeIsaac imports so the critical unit
math can be tested without starting SimulationApp.
"""

from __future__ import annotations

from typing import Any

import numpy as np


SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Mirrors leisaac.assets.robots.lerobot.SO101_FOLLOWER_USD_JOINT_LIMLITS.
# Values are degrees; Isaac itself exposes the corresponding positions in radians.
SO101_USD_JOINT_LIMITS_DEG = np.asarray(
    [
        (-110.0, 110.0),
        (-100.0, 100.0),
        (-100.0, 90.0),
        (-95.0, 95.0),
        (-160.0, 160.0),
        (-10.0, 100.0),
    ],
    dtype=np.float32,
)

# Mirrors leisaac.assets.robots.lerobot.SO101_FOLLOWER_MOTOR_LIMITS.
SO101_LEROBOT_MOTOR_LIMITS = np.asarray(
    [
        (-100.0, 100.0),
        (-100.0, 100.0),
        (-100.0, 100.0),
        (-100.0, 100.0),
        (-100.0, 100.0),
        (0.0, 100.0),
    ],
    dtype=np.float32,
)

SO101_USD_JOINT_LIMITS_RAD = np.deg2rad(SO101_USD_JOINT_LIMITS_DEG).astype(np.float32)


def _joint_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape[-1:] != (len(SO101_JOINT_NAMES),):
        raise ValueError(f"{name} must have last dimension 6, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def isaac_rad_to_lerobot_motor(joint_rad: Any) -> np.ndarray:
    """Convert SO101 Isaac joint radians to LeRobot motor-domain values."""

    joint_rad_array = _joint_array(joint_rad, "joint_rad")
    joint_deg = np.rad2deg(joint_rad_array)
    usd_low = SO101_USD_JOINT_LIMITS_DEG[:, 0]
    usd_range = SO101_USD_JOINT_LIMITS_DEG[:, 1] - usd_low
    motor_low = SO101_LEROBOT_MOTOR_LIMITS[:, 0]
    motor_range = SO101_LEROBOT_MOTOR_LIMITS[:, 1] - motor_low
    return ((joint_deg - usd_low) / usd_range * motor_range + motor_low).astype(np.float32)


def lerobot_motor_to_isaac_rad(motor_target: Any) -> np.ndarray:
    """Convert SO101 LeRobot motor-domain values to Isaac joint radians."""

    motor_array = _joint_array(motor_target, "motor_target")
    motor_low = SO101_LEROBOT_MOTOR_LIMITS[:, 0]
    motor_range = SO101_LEROBOT_MOTOR_LIMITS[:, 1] - motor_low
    usd_low = SO101_USD_JOINT_LIMITS_DEG[:, 0]
    usd_range = SO101_USD_JOINT_LIMITS_DEG[:, 1] - usd_low
    joint_deg = (motor_array - motor_low) / motor_range * usd_range + usd_low
    return np.deg2rad(joint_deg).astype(np.float32)


def validate_runtime_joint_limits_match_converter(
    runtime_joint_limits_rad: Any,
    *,
    atol: float = 1e-4,
) -> None:
    """Fail closed if the loaded Isaac asset does not match this converter.

    A different USD, joint order, sign convention, or joint range requires a
    different mapping.  Silently using this converter in that case would make
    otherwise valid absolute motor targets point to the wrong physical poses.
    """

    limits = np.asarray(runtime_joint_limits_rad, dtype=np.float32)
    if limits.shape != (6, 2):
        raise ValueError(f"runtime_joint_limits_rad must have shape (6, 2), got {limits.shape}")
    if not np.all(np.isfinite(limits)):
        raise ValueError("runtime_joint_limits_rad contains NaN or infinity")
    if not np.allclose(limits, SO101_USD_JOINT_LIMITS_RAD, atol=float(atol), rtol=0.0):
        raise ValueError(
            "Loaded Isaac SO101 joint limits do not match the LeIsaac motor-unit converter: "
            f"expected_rad={SO101_USD_JOINT_LIMITS_RAD.round(5).tolist()} "
            f"actual_rad={limits.round(5).tolist()}"
        )


def safe_absolute_motor_target_to_isaac_rad(
    current_joint_rad: Any,
    absolute_motor_target: Any,
    *,
    arm_target_scale: float = 1.0,
    max_arm_step_rad: float | None = 0.08,
    runtime_joint_limits_rad: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Safely convert one decoded absolute GR00T action to an Isaac target.

    ``Gr00tPolicy.get_action()`` returns the result after ``decode_action`` in
    the raw dataset action space.  This project's raw SO101 actions are absolute
    motor-domain targets, even though the processor may use a relative arm
    representation internally.  This helper therefore never adds the returned
    action to the current radians.
    It converts the absolute motor-domain target first, then optionally limits
    how far the five arm joints may move during one policy action.  The runner
    may hold that command for multiple simulator steps to preserve policy time.
    """

    current = _joint_array(current_joint_rad, "current_joint_rad").reshape(6)
    raw_motor = _joint_array(absolute_motor_target, "absolute_motor_target").reshape(6)

    motor_low = SO101_LEROBOT_MOTOR_LIMITS[:, 0]
    motor_high = SO101_LEROBOT_MOTOR_LIMITS[:, 1]
    clipped_motor = np.clip(raw_motor, motor_low, motor_high).astype(np.float32)
    absolute_target_rad = lerobot_motor_to_isaac_rad(clipped_motor).reshape(6)

    if not np.isfinite(arm_target_scale) or not 0.0 <= arm_target_scale <= 1.0:
        raise ValueError(f"arm_target_scale must be within [0, 1], got {arm_target_scale}")

    requested_arm_delta = (absolute_target_rad[:5] - current[:5]) * float(arm_target_scale)
    applied_arm_delta = requested_arm_delta.copy()
    if max_arm_step_rad is not None:
        bound = float(max_arm_step_rad)
        if not np.isfinite(bound):
            raise ValueError(f"max_arm_step_rad must be finite or None, got {max_arm_step_rad}")
        if bound < 0.0:
            raise ValueError(f"max_arm_step_rad must be non-negative or None, got {max_arm_step_rad}")
        if bound > 0.0:
            applied_arm_delta = np.clip(applied_arm_delta, -bound, bound)

    command = absolute_target_rad.copy()
    command[:5] = current[:5] + applied_arm_delta

    runtime_limit_clipped = False
    if runtime_joint_limits_rad is not None:
        limits = np.asarray(runtime_joint_limits_rad, dtype=np.float32)
        if limits.shape != (6, 2):
            raise ValueError(f"runtime_joint_limits_rad must have shape (6, 2), got {limits.shape}")
        if not np.all(np.isfinite(limits)):
            raise ValueError("runtime_joint_limits_rad contains NaN or infinity")
        limited_command = np.clip(command, limits[:, 0], limits[:, 1])
        runtime_limit_clipped = not np.allclose(limited_command, command, atol=1e-7, rtol=0.0)
        command = limited_command.astype(np.float32)

    if not np.all(np.isfinite(command)):
        raise ValueError("safe SO101 command contains NaN or infinity")

    diagnostics = {
        "motor_limit_clipped": not np.allclose(clipped_motor, raw_motor, atol=1e-7, rtol=0.0),
        "arm_step_clipped": not np.allclose(applied_arm_delta, requested_arm_delta, atol=1e-7, rtol=0.0),
        "runtime_limit_clipped": runtime_limit_clipped,
        "raw_motor_target": raw_motor,
        "clipped_motor_target": clipped_motor,
        "absolute_target_rad": absolute_target_rad,
        "requested_arm_delta_rad": requested_arm_delta,
        "applied_arm_delta_rad": applied_arm_delta,
    }
    return command.astype(np.float32), diagnostics
