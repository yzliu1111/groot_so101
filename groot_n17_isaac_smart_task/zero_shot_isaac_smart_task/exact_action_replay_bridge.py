"""Serve exact SO101 commands recorded in a runner action trace.

This bridge is intentionally independent from GR00T and Isaac.  It reads the
``applied_commands_rad`` recorded for each ``policy_call``, converts those
absolute Isaac-radian commands back to the checkpoint dataset coordinates, and
serves the original call chunks through the existing ``wire.py`` protocol.

For an exact replay, run the Isaac runner with:

``--so101-arm-target-scale 1 --so101-max-arm-step-rad 0
--so101-max-gripper-step-rad 0``

so its normal safety conversion does not rescale or re-limit the recorded
absolute commands.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import socket
import traceback
from typing import Any, Sequence

import numpy as np

from so101_joint_units import (
    SO101_CHECKPOINT_ARM_UNIT_CHOICES,
    isaac_rad_to_so101_dataset,
    so101_dataset_to_isaac_rad,
)
from wire import recv_message, send_message


SO101_VIDEO_KEYS = {
    "wrist-only": ("wrist",),
    "dual": ("top", "wrist"),
    "triple": ("top", "left", "wrist"),
}

ACTION_DECODING_CONTRACT = {
    "processor_use_relative_action": True,
    "policy_api_output": "decoded_dataset_action",
    "dataset_action_semantics": "absolute_joint_position_targets",
    "dataset_action_units": "checkpoint_dataset_coordinates",
}


@dataclass(frozen=True)
class ReplayChunk:
    """One source ``policy_call`` converted to the runner's action schema."""

    call_index: int
    commands_rad: np.ndarray
    action: dict[str, np.ndarray]


@dataclass(frozen=True)
class ReplayTrace:
    """Validated replay data loaded from one runner JSONL trace."""

    source_path: Path
    run_config: dict[str, Any]
    checkpoint_arm_units: str
    chunks: tuple[ReplayChunk, ...]


def _require_int(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _finite_vector(value: Any, name: str, *, size: int = 6) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector of length {size}") from exc
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _validate_run_config(record: dict[str, Any], *, line_number: int) -> dict[str, Any]:
    prefix = f"line {line_number} run_config"
    arm_units = record.get("checkpoint_arm_units")
    if arm_units not in SO101_CHECKPOINT_ARM_UNIT_CHOICES:
        raise ValueError(
            f"{prefix}.checkpoint_arm_units must be one of "
            f"{SO101_CHECKPOINT_ARM_UNIT_CHOICES}, got {arm_units!r}"
        )

    camera_layout = record.get("camera_layout")
    if camera_layout not in SO101_VIDEO_KEYS:
        raise ValueError(
            f"{prefix}.camera_layout must be one of {tuple(SO101_VIDEO_KEYS)}, "
            f"got {camera_layout!r}"
        )

    _require_int(record.get("max_policy_calls"), f"{prefix}.max_policy_calls", minimum=1)
    _require_int(record.get("action_horizon"), f"{prefix}.action_horizon", minimum=0)
    _finite_vector(record.get("initial_joint_pos_rad"), f"{prefix}.initial_joint_pos_rad")
    return dict(record)


def _load_json_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"action trace is not valid UTF-8: {path}") from exc

    if not lines:
        raise ValueError(f"action trace is empty: {path}")

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL record at {path}:{line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"JSONL record at {path}:{line_number} must be an object, "
                f"got {type(record).__name__}"
            )
        records.append((line_number, record))
    return records


def load_replay_trace(path_value: str | Path) -> ReplayTrace:
    """Load and strictly validate one runner action trace."""

    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"action trace does not exist or is not a file: {path}")

    records = _load_json_records(path)
    first_line, first_record = records[0]
    if first_record.get("type") != "run_config":
        raise ValueError(
            f"first JSONL record must be type='run_config', got "
            f"{first_record.get('type')!r} at {path}:{first_line}"
        )
    run_config = _validate_run_config(first_record, line_number=first_line)
    checkpoint_arm_units = str(run_config["checkpoint_arm_units"])
    max_policy_calls = int(run_config["max_policy_calls"])

    chunks: list[ReplayChunk] = []
    expected_call_index = 1
    for line_number, record in records[1:]:
        record_type = record.get("type")
        if record_type == "run_config":
            raise ValueError(f"duplicate run_config at {path}:{line_number}")
        if record_type != "policy_call":
            raise ValueError(
                f"unsupported record type {record_type!r} at {path}:{line_number}; "
                "expected policy_call"
            )

        call_index = _require_int(
            record.get("call_index"),
            f"line {line_number} policy_call.call_index",
            minimum=1,
        )
        if call_index != expected_call_index:
            raise ValueError(
                f"policy_call indices must be contiguous from 1: "
                f"expected {expected_call_index}, got {call_index} at {path}:{line_number}"
            )

        try:
            commands_rad = np.asarray(record.get("applied_commands_rad"), dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"policy_call {call_index} applied_commands_rad must be a numeric Nx6 array"
            ) from exc
        if commands_rad.ndim != 2 or commands_rad.shape[0] < 1 or commands_rad.shape[1] != 6:
            raise ValueError(
                f"policy_call {call_index} applied_commands_rad must have shape Nx6 "
                f"with N >= 1, got {commands_rad.shape}"
            )
        if not np.all(np.isfinite(commands_rad)):
            raise ValueError(
                f"policy_call {call_index} applied_commands_rad contains NaN or infinity"
            )

        executed_steps = _require_int(
            record.get("executed_action_steps"),
            f"policy_call {call_index}.executed_action_steps",
            minimum=1,
        )
        if executed_steps != commands_rad.shape[0]:
            raise ValueError(
                f"policy_call {call_index} executed_action_steps={executed_steps} "
                f"does not match {commands_rad.shape[0]} applied commands"
            )

        dataset_targets = isaac_rad_to_so101_dataset(
            commands_rad,
            checkpoint_arm_units,
        )
        reconstructed_rad = so101_dataset_to_isaac_rad(
            dataset_targets,
            checkpoint_arm_units,
        )
        if not np.allclose(reconstructed_rad, commands_rad, atol=1e-6, rtol=0.0):
            max_error = float(np.max(np.abs(reconstructed_rad - commands_rad)))
            raise ValueError(
                f"policy_call {call_index} cannot round-trip through "
                f"{checkpoint_arm_units!r}; max_error_rad={max_error:.9g}"
            )

        action = {
            "single_arm": dataset_targets[None, :, :5].astype(np.float32),
            "gripper": dataset_targets[None, :, 5:6].astype(np.float32),
        }
        chunks.append(
            ReplayChunk(
                call_index=call_index,
                commands_rad=commands_rad.copy(),
                action=action,
            )
        )
        expected_call_index += 1

    if not chunks:
        raise ValueError(f"action trace contains no executable policy_call records: {path}")
    if len(chunks) > max_policy_calls:
        raise ValueError(
            f"trace contains {len(chunks)} policy calls but run_config.max_policy_calls="
            f"{max_policy_calls}"
        )

    return ReplayTrace(
        source_path=path,
        run_config=run_config,
        checkpoint_arm_units=checkpoint_arm_units,
        chunks=tuple(chunks),
    )


def modality_summary(camera_layout: str) -> dict[str, dict[str, Any]]:
    """Return the SO101 finetuned modality contract expected by the runner."""

    if camera_layout not in SO101_VIDEO_KEYS:
        raise ValueError(f"unsupported camera layout: {camera_layout!r}")
    return {
        "video": {
            "delta_indices": [0],
            "modality_keys": list(SO101_VIDEO_KEYS[camera_layout]),
        },
        "state": {
            "delta_indices": [0],
            "modality_keys": ["single_arm", "gripper"],
        },
        "action": {
            "delta_indices": list(range(16)),
            "modality_keys": ["single_arm", "gripper"],
        },
        "language": {
            "delta_indices": [0],
            "modality_keys": ["annotation.human.task_description"],
        },
    }


class ExactActionReplayBridge:
    """Stateful request handler for one validated replay trace."""

    def __init__(
        self,
        trace: ReplayTrace,
        *,
        camera_layout: str = "dual",
        exit_after_replay: bool = False,
    ) -> None:
        if camera_layout not in SO101_VIDEO_KEYS:
            raise ValueError(f"unsupported camera layout: {camera_layout!r}")
        self.trace = trace
        self.camera_layout = camera_layout
        self.exit_after_replay = bool(exit_after_replay)
        self.cursor = 0

    def _replay_metadata(self) -> dict[str, Any]:
        return {
            "source_trace": str(self.trace.source_path),
            "checkpoint_arm_units": self.trace.checkpoint_arm_units,
            "source_camera_layout": self.trace.run_config["camera_layout"],
            "served_camera_layout": self.camera_layout,
            "total_chunks": len(self.trace.chunks),
            "next_chunk_index": self.cursor + 1,
            "remaining_chunks": len(self.trace.chunks) - self.cursor,
        }

    def handle_request(self, request: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Handle one wire request and return ``(response, should_exit)``."""

        endpoint = request.get("endpoint", "get_action")
        if endpoint == "ping":
            return (
                {
                    "ok": True,
                    "camera_layout": self.camera_layout,
                    "modality": modality_summary(self.camera_layout),
                    "action_decoding": dict(ACTION_DECODING_CONTRACT),
                    "replay": self._replay_metadata(),
                },
                False,
            )

        if endpoint == "reset":
            self.cursor = 0
            return (
                {
                    "ok": True,
                    "info": {
                        "reset": True,
                        "replay": self._replay_metadata(),
                    },
                },
                False,
            )

        if endpoint == "shutdown":
            return ({"ok": True}, True)

        if endpoint != "get_action":
            return ({"ok": False, "error": f"unknown endpoint: {endpoint}"}, False)

        if "observation" not in request:
            return (
                {
                    "ok": False,
                    "error": "get_action requires an observation field (content is ignored for exact replay)",
                },
                False,
            )
        if self.cursor >= len(self.trace.chunks):
            return (
                {
                    "ok": False,
                    "error": (
                        "exact-action replay exhausted; send reset to replay from call 1 "
                        "or shutdown to stop the bridge"
                    ),
                    "replay": self._replay_metadata(),
                },
                False,
            )

        chunk = self.trace.chunks[self.cursor]
        self.cursor += 1
        should_exit = self.exit_after_replay and self.cursor == len(self.trace.chunks)
        response = {
            "ok": True,
            "action": {key: value.copy() for key, value in chunk.action.items()},
            "info": {
                "exact_action_replay": True,
                "source_call_index": chunk.call_index,
                "command_count": int(chunk.commands_rad.shape[0]),
                "replay": self._replay_metadata(),
            },
        }
        return response, should_exit


def serve(
    bridge: ExactActionReplayBridge,
    *,
    host: str,
    port: int,
) -> None:
    """Serve replay chunks through the same one-request-per-connection protocol."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        print(f"[replay-bridge] listening on {host}:{port}", flush=True)

        while True:
            conn, addr = server.accept()
            should_exit = False
            with conn:
                try:
                    request = recv_message(conn)
                    response, should_exit = bridge.handle_request(request)
                except Exception as exc:
                    traceback.print_exc()
                    response = {
                        "ok": False,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                send_message(conn, response)
            print(f"[replay-bridge] handled request from {addr}", flush=True)
            if should_exit:
                print("[replay-bridge] graceful exit requested", flush=True)
                return


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay exact SO101 applied commands from a runner action trace."
    )
    parser.add_argument("--action-trace-jsonl", required=True)
    parser.add_argument(
        "--camera-layout",
        choices=tuple(SO101_VIDEO_KEYS),
        default="dual",
        help="Contract advertised to the runner; dual is the default A/B route.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5577)
    parser.add_argument(
        "--exit-after-replay",
        action="store_true",
        help="Exit after sending the final recorded policy-call chunk.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    trace = load_replay_trace(args.action_trace_jsonl)
    bridge = ExactActionReplayBridge(
        trace,
        camera_layout=args.camera_layout,
        exit_after_replay=args.exit_after_replay,
    )
    print(
        "[replay-bridge] loaded "
        f"trace={trace.source_path} chunks={len(trace.chunks)} "
        f"checkpoint_arm_units={trace.checkpoint_arm_units} "
        f"source_camera_layout={trace.run_config['camera_layout']} "
        f"served_camera_layout={args.camera_layout}",
        flush=True,
    )
    print(
        "[replay-bridge] exact runner settings required: "
        "--so101-arm-target-scale 1 --so101-max-arm-step-rad 0 "
        "--so101-max-gripper-step-rad 0",
        flush=True,
    )
    serve(bridge, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
