#!/usr/bin/env python
"""Deploy a wrist-only GR00T N1.7 checkpoint on a real SO101 via LeRobot.

Safety defaults:

* inference-only unless ``--execute`` is supplied;
* one policy call unless explicitly increased;
* per-tick arm/gripper target clipping;
* existing LeRobot calibration is required unless ``--calibrate`` is supplied;
* Ctrl+C disconnects the robot and follows LeRobot's torque-disable setting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SIM_BRIDGE_DIR = SCRIPT_DIR.parent / "zero_shot_isaac_smart_task"
if str(SIM_BRIDGE_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_BRIDGE_DIR))

from lerobot_groot_bridge_server import LeRobotGrootPolicy  # noqa: E402

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
POSITION_KEYS = tuple(f"{name}.pos" for name in JOINT_NAMES)


def parse_camera_device(value: str) -> int | Path:
    """Accept either an OpenCV integer index or a stable /dev/v4l/by-id path."""

    try:
        return int(value)
    except ValueError:
        return Path(value).expanduser()


def state_from_observation(observation: dict[str, Any]) -> np.ndarray:
    missing = [key for key in POSITION_KEYS if key not in observation]
    if missing:
        raise KeyError(f"SO101 observation is missing motor positions: {missing}")
    state = np.asarray([observation[key] for key in POSITION_KEYS], dtype=np.float32)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError(f"invalid SO101 state: shape={state.shape} values={state.tolist()}")
    return state


def build_policy_observation(
    observation: dict[str, Any],
    *,
    wrist_camera_key: str,
    instruction: str,
) -> dict[str, Any]:
    if wrist_camera_key not in observation:
        raise KeyError(
            f"SO101 observation has no camera {wrist_camera_key!r}; "
            f"available keys={sorted(observation)}"
        )
    image = np.asarray(observation[wrist_camera_key])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected wrist RGB HWC image, got {image.shape}")

    state = state_from_observation(observation)
    return {
        "video": {"wrist": image[None, None, ...]},
        "state": {
            "single_arm": state[None, None, :5],
            "gripper": state[None, None, 5:6],
        },
        "language": {
            "annotation.human.task_description": [[instruction]],
        },
    }


def safe_target(
    current: np.ndarray,
    requested: np.ndarray,
    *,
    max_arm_delta: float,
    max_gripper_delta: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clip one absolute motor target relative to the latest measured state."""

    current = np.asarray(current, dtype=np.float32)
    requested = np.asarray(requested, dtype=np.float32)
    if current.shape != (6,) or requested.shape != (6,):
        raise ValueError(f"expected current/requested shape (6,), got {current.shape}/{requested.shape}")
    if not np.isfinite(requested).all():
        raise ValueError(f"policy produced non-finite target: {requested.tolist()}")

    limits = np.asarray(
        [max_arm_delta] * 5 + [max_gripper_delta],
        dtype=np.float32,
    )
    delta = requested - current
    applied_delta = np.clip(delta, -limits, limits)
    applied = current + applied_delta
    # LeRobot's SO101 gripper normalization is always RANGE_0_100.
    applied[5] = np.clip(applied[5], 0.0, 100.0)
    return applied, {
        "requested_delta": delta.tolist(),
        "applied_delta": applied_delta.tolist(),
        "clipped": bool(not np.allclose(delta, applied_delta)),
    }


def action_dict(target: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float32)
    if target.shape != (6,):
        raise ValueError(f"expected SO101 target shape (6,), got {target.shape}")
    return {key: float(target[index]) for index, key in enumerate(POSITION_KEYS)}


def write_trace(handle, record: dict[str, Any]) -> None:
    if handle is None:
        return
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def build_robot(args: argparse.Namespace):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    camera = OpenCVCameraConfig(
        index_or_path=parse_camera_device(args.wrist_camera),
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        fourcc=args.camera_fourcc,
    )
    config = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        calibration_dir=args.calibration_dir,
        cameras={args.wrist_camera_key: camera},
        use_degrees=args.checkpoint_arm_units == "degrees",
        # Hardware-layer second guard. The runner applies the same limits and
        # records requested/applied values before calling send_action().
        max_relative_target={
            **{name: args.max_arm_delta for name in JOINT_NAMES[:5]},
            "gripper": args.max_gripper_delta,
        },
        disable_torque_on_disconnect=not args.keep_torque_on_disconnect,
    )
    return SO101Follower(config)


def confirm_execution(args: argparse.Namespace) -> None:
    if not args.execute:
        return
    if args.yes:
        return
    answer = input(
        "\nReal motor execution requested. Clear the workspace, keep an emergency-stop path ready, "
        "then type EXECUTE: "
    )
    if answer.strip() != "EXECUTE":
        raise RuntimeError("execution confirmation rejected")


def run(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint}")
    if args.action_horizon < 1 or args.action_horizon > 16:
        raise ValueError("--action-horizon must be in [1, 16]")

    print(f"[real] loading LeRobot GR00T checkpoint in {args.parameter_dtype}: {checkpoint}", flush=True)
    policy = LeRobotGrootPolicy(checkpoint, args.device, args.parameter_dtype)
    print("[real] policy loaded; hardware has not been connected yet", flush=True)

    robot = build_robot(args)
    if not args.calibrate and not robot.calibration_fpath.is_file():
        raise FileNotFoundError(
            f"calibration file not found: {robot.calibration_fpath}. "
            "Run lerobot-calibrate first or pass --calibrate for an interactive calibration."
        )

    confirm_execution(args)
    trace_handle = None
    if args.trace_jsonl:
        trace_path = Path(args.trace_jsonl).expanduser().resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_handle = trace_path.open("x", encoding="utf-8")

    print(
        f"[real] mode={'EXECUTE' if args.execute else 'DRY-RUN'} "
        f"port={args.robot_port} camera={args.wrist_camera}",
        flush=True,
    )
    connected = False
    try:
        robot.connect(calibrate=args.calibrate)
        connected = True
        write_trace(
            trace_handle,
            {
                "type": "run_config",
                "checkpoint": str(checkpoint),
                "instruction": args.instruction,
                "execute": args.execute,
                "parameter_dtype": args.parameter_dtype,
                "checkpoint_arm_units": args.checkpoint_arm_units,
                "fps": args.fps,
                "action_horizon": args.action_horizon,
                "max_policy_calls": args.max_policy_calls,
                "max_arm_delta": args.max_arm_delta,
                "max_gripper_delta": args.max_gripper_delta,
            },
        )
        for call_index in range(1, args.max_policy_calls + 1):
            observation = robot.get_observation()
            state_before = state_from_observation(observation)
            nested_observation = build_policy_observation(
                observation,
                wrist_camera_key=args.wrist_camera_key,
                instruction=args.instruction,
            )
            action, info = policy.get_action(nested_observation)
            chunk = np.concatenate((action["single_arm"], action["gripper"]), axis=-1)[0]
            horizon = min(args.action_horizon, chunk.shape[0])
            print(
                f"[real] policy call {call_index}/{args.max_policy_calls} "
                f"chunk={chunk.shape} inference={info['inference_elapsed_s']:.3f}s",
                flush=True,
            )

            record = {
                "type": "policy_call",
                "call_index": call_index,
                "state_before": state_before.tolist(),
                "decoded_absolute_chunk": chunk.tolist(),
                "inference_elapsed_s": info["inference_elapsed_s"],
                "steps": [],
            }
            if not args.execute:
                print(
                    f"[real] dry-run first target={np.round(chunk[0], 4).tolist()}",
                    flush=True,
                )
                write_trace(trace_handle, record)
                continue

            next_deadline = time.perf_counter()
            for step_index in range(horizon):
                latest = robot.get_observation()
                current = state_from_observation(latest)
                applied, diagnostics = safe_target(
                    current,
                    chunk[step_index],
                    max_arm_delta=args.max_arm_delta,
                    max_gripper_delta=args.max_gripper_delta,
                )
                sent = robot.send_action(action_dict(applied))
                record["steps"].append(
                    {
                        "step_index": step_index,
                        "current": current.tolist(),
                        "requested": chunk[step_index].tolist(),
                        "applied": applied.tolist(),
                        "sent": sent,
                        **diagnostics,
                    }
                )
                next_deadline += 1.0 / args.fps
                remaining = next_deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
            write_trace(trace_handle, record)
    except KeyboardInterrupt:
        print("[real] Ctrl+C received; stopping rollout", flush=True)
    finally:
        if trace_handle is not None:
            trace_handle.close()
        if connected and robot.is_connected:
            robot.disconnect()
            print("[real] robot disconnected", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--robot-port", required=True)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--wrist-camera", required=True, help="OpenCV index or /dev/v4l/by-id path")
    parser.add_argument("--wrist-camera-key", default="wrist")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-fourcc", default="MJPG")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--parameter-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--checkpoint-arm-units",
        choices=("degrees", "lerobot_motor_units"),
        default="degrees",
        help="Must match the dataset coordinates used to train the checkpoint.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-policy-calls", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument(
        "--max-arm-delta",
        type=float,
        default=5.0,
        help="Checkpoint arm units per action step (degrees or normalized motor units)",
    )
    parser.add_argument("--max-gripper-delta", type=float, default=10.0, help="0-100 units per action step")
    parser.add_argument("--calibration-dir", type=Path, default=None)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip the EXECUTE confirmation prompt")
    parser.add_argument("--keep-torque-on-disconnect", action="store_true")
    parser.add_argument("--trace-jsonl", default=None)
    args = parser.parse_args()
    if args.max_policy_calls < 1:
        parser.error("--max-policy-calls must be >= 1")
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive")
    if args.max_arm_delta <= 0 or args.max_gripper_delta <= 0:
        parser.error("per-step limits must be positive")
    if args.yes and not args.execute:
        parser.error("--yes is only valid with --execute")
    return args


if __name__ == "__main__":
    run(parse_args())
