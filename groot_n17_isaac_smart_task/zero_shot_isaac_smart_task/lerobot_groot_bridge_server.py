#!/usr/bin/env python
"""Serve a raw GR00T N1.7 checkpoint through LeRobot's policy implementation.

This is a drop-in replacement for ``groot_bridge_server.py`` for the existing
SO101 Isaac runner.  The wire contract intentionally stays unchanged:

* input: NVIDIA-style NEW_EMBODIMENT observation dictionaries;
* output: decoded absolute ``single_arm`` and ``gripper`` action chunks.

Keeping the runner, scene, camera mapping, controller and safety envelope
unchanged makes the known-good sim002 rollout a useful A/B validation of the
LeRobot inference path.
"""

from __future__ import annotations

import argparse
import os
import random
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from wire import recv_message, send_message  # noqa: E402

CAMERA_LAYOUTS = {
    "wrist-only": ("wrist",),
    "dual": ("top", "wrist"),
    "triple": ("top", "left", "wrist"),
}


def _seed_inference(seed: int | None) -> None:
    if seed is None:
        print("[lerobot-bridge] inference seed: <unseeded>", flush=True)
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[lerobot-bridge] inference seed: {seed}", flush=True)


def _flatten_observation(
    observation: dict[str, Any],
    video_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Translate the existing bridge schema to LeRobot's flat observation keys."""

    video = observation["video"]
    state = observation["state"]
    language = observation["language"]
    if set(video) != set(video_keys):
        raise ValueError(f"expected video keys {list(video_keys)}, got {list(video)}")

    arm = np.asarray(state["single_arm"], dtype=np.float32)
    gripper = np.asarray(state["gripper"], dtype=np.float32)
    if arm.shape != (1, 1, 5) or gripper.shape != (1, 1, 1):
        raise ValueError(
            "expected state.single_arm=(1,1,5) and state.gripper=(1,1,1), "
            f"got {arm.shape} and {gripper.shape}"
        )

    task = language["annotation.human.task_description"]
    if not isinstance(task, list) or not task or not isinstance(task[0], list) or not task[0]:
        raise ValueError(f"unexpected task format: {task!r}")

    # The existing runner already adds a batch/time axis for NVIDIA GR00T.
    # LeRobot's online preprocessor owns the batch axis, so pass one HWC frame
    # and one 6-D state vector here.
    result = {
        "observation.state": torch.from_numpy(
            np.concatenate((arm[0, 0], gripper[0, 0]), axis=0)
        ),
        "task": str(task[0][0]),
    }
    for key in video_keys:
        image = np.asarray(video[key])
        if image.ndim == 5 and image.shape[:2] == (1, 1):
            image = image[0, 0]
        elif image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"expected {key} HWC RGB image, got {image.shape}")
        result[f"observation.images.{key}"] = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
    return result


class LeRobotGrootPolicy:
    def __init__(
        self,
        checkpoint: Path,
        device: str,
        parameter_dtype: str,
        camera_layout: str = "wrist-only",
    ) -> None:
        from lerobot.configs import FeatureType, PolicyFeature
        from lerobot.policies.groot.configuration_groot import GrootConfig
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        from lerobot.policies.groot.processor_groot import (
            make_groot_pre_post_processors_from_pretrained,
        )
        from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

        self.video_keys = CAMERA_LAYOUTS[camera_layout]
        config = GrootConfig(
            base_model_path=str(checkpoint),
            embodiment_tag="new_embodiment",
            input_features={
                **{
                    f"{OBS_IMAGES}.{key}": PolicyFeature(
                        type=FeatureType.VISUAL, shape=(3, 480, 640)
                    )
                    for key in self.video_keys
                },
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            },
            output_features={
                ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
            },
            device=device,
            use_bf16=device.startswith("cuda"),
            model_params_fp32=True,
            chunk_size=16,
            n_action_steps=16,
            use_relative_actions=True,
            relative_exclude_joints=["gripper"],
            action_decode_transform=None,
        )
        self.policy = GrootPolicy.from_pretrained(
            checkpoint,
            config=config,
            local_files_only=True,
        )
        dtype = torch.bfloat16 if parameter_dtype == "bf16" else torch.float32
        self.policy.to(device=device, dtype=dtype)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_groot_pre_post_processors_from_pretrained(
            self.policy.config,
            str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    def reset(self, _options: Any = None) -> dict[str, Any]:
        self.policy.reset()
        return {}

    @torch.inference_mode()
    def get_action(
        self, observation: dict[str, Any], _options: Any = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        started = time.perf_counter()
        batch = self.preprocessor(_flatten_observation(observation, self.video_keys))
        raw_chunk = self.policy.predict_action_chunk(batch)
        # Relative arm actions must be decoded as one chunk against the state
        # captured by this preprocessor call. Never queue raw relative actions.
        decoded_chunk = self.postprocessor(raw_chunk)
        if isinstance(decoded_chunk, torch.Tensor):
            decoded = decoded_chunk.detach().cpu().numpy()
        else:
            decoded = np.asarray(decoded_chunk)
        if decoded.ndim != 3 or decoded.shape[0] != 1 or decoded.shape[-1] != 6:
            raise ValueError(f"expected decoded action chunk (1,T,6), got {decoded.shape}")
        decoded = decoded.astype(np.float32, copy=False)
        elapsed_s = time.perf_counter() - started
        print(f"[lerobot-bridge] inference elapsed_s={elapsed_s:.3f}", flush=True)
        return {
            "single_arm": decoded[:, :, :5],
            "gripper": decoded[:, :, 5:6],
        }, {
            "backend": "lerobot",
            "decoded_shape": list(decoded.shape),
            "inference_elapsed_s": elapsed_s,
        }


def serve(args: argparse.Namespace) -> None:
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    _seed_inference(args.seed)
    print(f"[lerobot-bridge] loading checkpoint: {args.model_path}", flush=True)
    policy = LeRobotGrootPolicy(
        Path(args.model_path).expanduser().resolve(),
        args.device,
        args.parameter_dtype,
        args.camera_layout,
    )
    modality = {
        "video": {
            "delta_indices": [0],
            "modality_keys": list(CAMERA_LAYOUTS[args.camera_layout]),
        },
        "state": {"delta_indices": [0], "modality_keys": ["single_arm", "gripper"]},
        "action": {
            "delta_indices": list(range(16)),
            "modality_keys": ["single_arm", "gripper"],
        },
        "language": {
            "delta_indices": [0],
            "modality_keys": ["annotation.human.task_description"],
        },
    }
    action_decoding = {
        "processor_use_relative_action": True,
        "policy_api_output": "decoded_dataset_action",
        "dataset_action_semantics": "absolute_joint_position_targets",
        "dataset_action_units": "checkpoint_dataset_coordinates",
        "backend": "lerobot",
    }
    print("[lerobot-bridge] policy loaded", flush=True)
    print(f"[lerobot-bridge] camera layout: {args.camera_layout}", flush=True)
    print(f"[lerobot-bridge] modality: {modality}", flush=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"[lerobot-bridge] listening on {args.host}:{args.port}", flush=True)
        while True:
            conn, addr = server.accept()
            with conn:
                try:
                    request = recv_message(conn)
                    endpoint = request.get("endpoint", "get_action")
                    if endpoint == "ping":
                        response = {
                            "ok": True,
                            "camera_layout": args.camera_layout,
                            "modality": modality,
                            "action_decoding": action_decoding,
                        }
                    elif endpoint == "reset":
                        response = {"ok": True, "info": policy.reset(request.get("options"))}
                    elif endpoint == "get_action":
                        action, info = policy.get_action(
                            request["observation"], request.get("options")
                        )
                        # The LeRobot env currently has NumPy 2 while the Isaac
                        # env has NumPy 1. Pickling ndarray objects across that
                        # boundary imports ``numpy._core`` on the receiver and
                        # fails. Plain nested lists keep the wire contract
                        # version-independent; the runner already calls
                        # np.asarray before using an action.
                        response = {
                            "ok": True,
                            "action": {key: value.tolist() for key, value in action.items()},
                            "info": info,
                        }
                    elif endpoint == "shutdown":
                        send_message(conn, {"ok": True})
                        print("[lerobot-bridge] shutdown requested", flush=True)
                        return
                    else:
                        response = {"ok": False, "error": f"unknown endpoint: {endpoint}"}
                    send_message(conn, response)
                except Exception as exc:
                    traceback.print_exc()
                    send_message(
                        conn,
                        {
                            "ok": False,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                finally:
                    print(f"[lerobot-bridge] handled request from {addr}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera-layout", choices=tuple(CAMERA_LAYOUTS), default="wrist-only")
    parser.add_argument(
        "--parameter-dtype",
        choices=("fp32", "bf16"),
        default="fp32",
        help="Cast model parameters after loading. bf16 is the 16 GB single-GPU deployment mode.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5577)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    checkpoint = Path(args.model_path).expanduser()
    if not checkpoint.is_dir():
        parser.error(f"--model-path is not a checkpoint directory: {checkpoint}")
    return args


if __name__ == "__main__":
    serve(parse_args())
