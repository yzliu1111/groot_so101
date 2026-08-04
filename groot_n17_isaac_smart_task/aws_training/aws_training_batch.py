#!/usr/bin/env python3
"""Manifest-driven audit, preparation, smoke, and training for the eight AWS runs.

Each manifest item is always launched as its own GR00T run.  The script never
turns ``--all`` into one mixed dataset, and it never overrides source task text.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
FULL_FINETUNE_DIR = EXPERIMENT_ROOT / "full_finetune_so101"
TRAIN_WRAPPER = FULL_FINETUNE_DIR / "train_so101_synthetic_groot.py"
DEFAULT_MANIFEST = SCRIPT_DIR / "aws_tuning_8_manifest.json"
MODALITY_CONFIGS = {
    "wrist-only": FULL_FINETUNE_DIR / "so101_synthetic_groot_wrist_only_config.py",
    "dual": FULL_FINETUNE_DIR / "so101_synthetic_groot_config.py",
    "triple": FULL_FINETUNE_DIR / "so101_synthetic_groot_triple_config.py",
}
V21_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
SO101_JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
ROLE_FLAGS = {
    "top": "--dataset-front-camera-key",
    "left": "--dataset-left-camera-key",
    "wrist": "--dataset-wrist-camera-key",
}
LAYOUT_ROLES = {
    "wrist-only": {"wrist"},
    "dual": {"top", "wrist"},
    "triple": {"top", "left", "wrist"},
}
SAFE_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    domain: str
    raw_source: Path
    training_source: Path
    prepared_root: Path
    prepared_dataset: Path
    expected_prepared_content_sha256: str
    camera_layout: str
    camera_roles: dict[str, str]
    max_steps: int
    save_steps: int
    save_total_limit: int
    removed_raw_episodes: tuple[int, ...]
    known_risks: tuple[str, ...]
    instruction_override: str | None
    instruction_override_reason: str | None
    expected_raw: dict[str, Any]
    expected_training: dict[str, Any]


@dataclass(frozen=True)
class TrainingDefaults:
    base_model_path: str
    global_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    dataloader_num_workers: int
    state_dropout_prob: float
    color_jitter: bool


def _project_path(value: str, *, field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a project-relative path without '..': {value!r}")
    return PROJECT_ROOT / path


def _absolute_executable(path: Path) -> Path:
    """Make an executable path absolute without dereferencing venv symlinks."""

    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.absolute()


def _positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return value


def _expected_summary(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    episodes = _positive_int(value.get("episodes"), field=f"{field}.episodes")
    frames = _positive_int(value.get("frames"), field=f"{field}.frames")
    task_counts_value = value.get("task_episode_counts")
    if not isinstance(task_counts_value, dict) or not task_counts_value:
        raise ValueError(f"{field}.task_episode_counts must be a non-empty object")
    task_counts: dict[str, int] = {}
    for task_text, count in task_counts_value.items():
        normalized_text = str(task_text).strip()
        if not normalized_text or normalized_text in task_counts:
            raise ValueError(f"{field}: empty or duplicate task text {task_text!r}")
        task_counts[normalized_text] = _positive_int(
            count, field=f"{field}.task_episode_counts[{normalized_text!r}]"
        )
    if sum(task_counts.values()) != episodes:
        raise ValueError(
            f"{field}: task episode counts sum to {sum(task_counts.values())}, "
            f"expected {episodes}"
        )
    return {
        "episodes": episodes,
        "frames": frames,
        "task_episode_counts": task_counts,
    }


def load_manifest(path: Path) -> tuple[dict[str, Any], TrainingDefaults, list[DatasetSpec]]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema_version: {payload.get('schema_version')!r}")
    base_model_revision = str(payload.get("base_model_revision", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", base_model_revision):
        raise ValueError(
            f"Manifest base_model_revision must be a full SHA: {base_model_revision!r}"
        )
    rows = payload.get("datasets")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest datasets must be a non-empty list")
    expected_count = _positive_int(
        payload.get("expected_dataset_count"), field="expected_dataset_count"
    )
    if len(rows) != expected_count:
        raise ValueError(f"Manifest expected {expected_count} datasets, found {len(rows)}")

    default_values = payload.get("defaults")
    if not isinstance(default_values, dict):
        raise ValueError("Manifest defaults must be an object")
    color_jitter_value = default_values.get("color_jitter")
    if not isinstance(color_jitter_value, bool):
        raise ValueError("defaults.color_jitter must be a JSON boolean")
    defaults = TrainingDefaults(
        base_model_path=str(payload["base_model_path"]),
        global_batch_size=_positive_int(
            default_values.get("global_batch_size"), field="global_batch_size"
        ),
        gradient_accumulation_steps=_positive_int(
            default_values.get("gradient_accumulation_steps"),
            field="gradient_accumulation_steps",
        ),
        learning_rate=float(default_values.get("learning_rate")),
        dataloader_num_workers=int(default_values.get("dataloader_num_workers")),
        state_dropout_prob=float(default_values.get("state_dropout_prob")),
        color_jitter=color_jitter_value,
    )
    if not math.isfinite(defaults.learning_rate) or defaults.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive and finite: {defaults.learning_rate}")
    if defaults.dataloader_num_workers < 0:
        raise ValueError("dataloader_num_workers must be non-negative")
    if not 0 <= defaults.state_dropout_prob <= 1:
        raise ValueError("state_dropout_prob must be between 0 and 1")

    specs: list[DatasetSpec] = []
    seen_ids: set[str] = set()
    seen_prepared: set[Path] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Dataset row must be an object: {row!r}")
        dataset_id = str(row.get("id", ""))
        if not dataset_id or dataset_id in seen_ids:
            raise ValueError(f"Dataset id is empty or duplicated: {dataset_id!r}")
        if not SAFE_SLUG.fullmatch(dataset_id):
            raise ValueError(f"Dataset id must be a safe slug: {dataset_id!r}")
        seen_ids.add(dataset_id)
        instruction_override_value = row.get("instruction_override")
        instruction_override = (
            str(instruction_override_value).strip()
            if instruction_override_value is not None
            else None
        )
        instruction_override_reason_value = row.get("instruction_override_reason")
        instruction_override_reason = (
            str(instruction_override_reason_value).strip()
            if instruction_override_reason_value is not None
            else None
        )
        if bool(instruction_override) != bool(instruction_override_reason):
            raise ValueError(
                f"{dataset_id}: instruction_override and instruction_override_reason "
                "must be supplied together"
            )

        camera_layout = str(row.get("camera_layout"))
        camera_roles = row.get("camera_roles")
        if camera_layout not in LAYOUT_ROLES or not isinstance(camera_roles, dict):
            raise ValueError(f"{dataset_id}: invalid camera layout/roles")
        camera_roles = {str(role): str(key) for role, key in camera_roles.items()}
        if set(camera_roles) != LAYOUT_ROLES[camera_layout]:
            raise ValueError(
                f"{dataset_id}: {camera_layout} requires roles "
                f"{sorted(LAYOUT_ROLES[camera_layout])}, got {sorted(camera_roles)}"
            )
        if len(set(camera_roles.values())) != len(camera_roles):
            raise ValueError(f"{dataset_id}: camera roles must use distinct feature keys")

        prepared_dataset = _project_path(
            str(row["prepared_dataset"]), field=f"{dataset_id}.prepared_dataset"
        )
        expected_prepared_content_sha256 = str(
            row.get("expected_prepared_content_sha256", "")
        )
        if not re.fullmatch(r"[0-9a-f]{64}", expected_prepared_content_sha256):
            raise ValueError(
                f"{dataset_id}: expected_prepared_content_sha256 must be a SHA256"
            )
        if prepared_dataset in seen_prepared:
            raise ValueError(f"Duplicate prepared dataset path: {prepared_dataset}")
        seen_prepared.add(prepared_dataset)
        prepared_root = _project_path(
            str(row["prepared_root"]), field=f"{dataset_id}.prepared_root"
        )
        if prepared_root not in prepared_dataset.parents:
            raise ValueError(
                f"{dataset_id}: prepared_dataset must be inside prepared_root: "
                f"{prepared_dataset} vs {prepared_root}"
            )

        removed = tuple(int(index) for index in row.get("removed_raw_episodes", []))
        if len(set(removed)) != len(removed) or any(index < 0 for index in removed):
            raise ValueError(f"{dataset_id}: invalid removed_raw_episodes={removed}")
        expected_raw = _expected_summary(
            row.get("expected_raw"), field=f"{dataset_id}.expected_raw"
        )
        expected_training = _expected_summary(
            row.get("expected_training", row.get("expected_raw")),
            field=f"{dataset_id}.expected_training",
        )
        specs.append(
            DatasetSpec(
                dataset_id=dataset_id,
                domain=str(row["domain"]),
                raw_source=_project_path(
                    str(row["raw_source"]), field=f"{dataset_id}.raw_source"
                ),
                training_source=_project_path(
                    str(row["training_source"]), field=f"{dataset_id}.training_source"
                ),
                prepared_root=prepared_root,
                prepared_dataset=prepared_dataset,
                expected_prepared_content_sha256=expected_prepared_content_sha256,
                camera_layout=camera_layout,
                camera_roles=camera_roles,
                max_steps=_positive_int(row.get("max_steps"), field=f"{dataset_id}.max_steps"),
                save_steps=_positive_int(row.get("save_steps"), field=f"{dataset_id}.save_steps"),
                save_total_limit=_positive_int(
                    row.get("save_total_limit"), field=f"{dataset_id}.save_total_limit"
                ),
                removed_raw_episodes=removed,
                known_risks=tuple(str(item) for item in row.get("known_risks", [])),
                instruction_override=instruction_override,
                instruction_override_reason=instruction_override_reason,
                expected_raw=expected_raw,
                expected_training=expected_training,
            )
        )
    return payload, defaults, specs


def select_specs(specs: list[DatasetSpec], requested: list[str]) -> list[DatasetSpec]:
    if not requested or requested == ["all"]:
        return specs
    if "all" in requested:
        raise ValueError("Use either 'all' or explicit dataset ids, not both")
    by_id = {spec.dataset_id: spec for spec in specs}
    missing = [dataset_id for dataset_id in requested if dataset_id not in by_id]
    if missing:
        raise ValueError(f"Unknown dataset ids: {missing}; available={sorted(by_id)}")
    if len(set(requested)) != len(requested):
        raise ValueError(f"Dataset ids are duplicated: {requested}")
    return [by_id[dataset_id] for dataset_id in requested]


def _task_text(row: dict[str, Any]) -> str:
    if "task" in row:
        value = row["task"]
    elif "__index_level_0__" in row:
        value = row["__index_level_0__"]
    else:
        values = [value for value in row.values() if isinstance(value, str)]
        if not values:
            raise ValueError(f"Cannot infer task text from {row!r}")
        value = values[0]
    text = str(value).strip()
    if not text:
        raise ValueError(f"Task text is empty: {row!r}")
    return text


def _assert_expected_summary(
    report: dict[str, Any], expected: dict[str, Any], *, context: str
) -> None:
    actual = {
        "episodes": report["total_episodes"],
        "frames": report["total_frames"],
        "task_episode_counts": report["task_text_episode_counts"],
    }
    if actual != expected:
        raise ValueError(
            f"{context}: dataset identity mismatch; expected={expected!r} actual={actual!r}. "
            "Do not train a similarly named or stale directory."
        )


def audit_v3_dataset(root: Path, spec: DatasetSpec, *, label: str) -> dict[str, Any]:
    try:
        import numpy as np
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Dataset audit requires numpy and pyarrow; run it with the AWS data Python"
        ) from exc

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"{spec.dataset_id}/{label}: missing {info_path}")
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"{spec.dataset_id}/{label}: expected v3.0, got {info.get('codebase_version')}")
    features = info.get("features", {})
    for key in ("action", "observation.state"):
        feature = features.get(key)
        if not isinstance(feature, dict) or feature.get("shape") != [6]:
            raise ValueError(f"{spec.dataset_id}/{label}: invalid {key} feature")
        if feature.get("names") != SO101_JOINT_NAMES:
            raise ValueError(f"{spec.dataset_id}/{label}: invalid {key} joint order")
    video_keys = {
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    }
    selected_video_keys = set(spec.camera_roles.values())
    if video_keys != selected_video_keys:
        raise ValueError(
            f"{spec.dataset_id}/{label}: camera mapping does not consume all source videos: "
            f"available={sorted(video_keys)} selected={sorted(selected_video_keys)}"
        )
    expected_layout = {1: "wrist-only", 2: "dual", 3: "triple"}.get(len(video_keys))
    if spec.camera_layout != expected_layout:
        raise ValueError(
            f"{spec.dataset_id}/{label}: {len(video_keys)} cameras require {expected_layout}, "
            f"got {spec.camera_layout}"
        )

    task_rows = pq.read_table(root / "meta" / "tasks.parquet").to_pylist()
    task_map = {int(row["task_index"]): _task_text(row) for row in task_rows}
    if len(task_map) != len(task_rows) or not task_map:
        raise ValueError(f"{spec.dataset_id}/{label}: duplicated or empty task table")
    if spec.instruction_override and len(task_map) != 1:
        raise ValueError(
            f"{spec.dataset_id}/{label}: instruction override is only allowed for a "
            f"single-task dataset, found {len(task_map)} tasks"
        )

    episode_paths = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"{spec.dataset_id}/{label}: no episode metadata")
    episodes: list[dict[str, Any]] = []
    for path in episode_paths:
        episodes.extend(pq.read_table(path).to_pylist())
    episodes.sort(key=lambda row: int(row["episode_index"]))
    episode_indices = [int(row["episode_index"]) for row in episodes]
    if episode_indices != list(range(len(episodes))):
        raise ValueError(f"{spec.dataset_id}/{label}: episode_index is not contiguous")
    lengths = {int(row["episode_index"]): int(row["length"]) for row in episodes}
    if any(length <= 0 for length in lengths.values()):
        raise ValueError(f"{spec.dataset_id}/{label}: non-positive episode length")
    if int(info.get("total_episodes", -1)) != len(episodes):
        raise ValueError(f"{spec.dataset_id}/{label}: total_episodes mismatch")
    if int(info.get("total_frames", -1)) != sum(lengths.values()):
        raise ValueError(f"{spec.dataset_id}/{label}: total_frames mismatch")

    expected_tasks_by_episode: dict[int, set[int]] = {}
    text_to_task_index = {text: index for index, text in task_map.items()}
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        task_texts = {str(text) for text in episode.get("tasks", [])}
        unknown = task_texts - set(text_to_task_index)
        if unknown:
            raise ValueError(
                f"{spec.dataset_id}/{label}: episode {episode_index} has unknown tasks {unknown}"
            )
        expected_tasks_by_episode[episode_index] = {
            text_to_task_index[text] for text in task_texts
        }

    counts = {episode_index: 0 for episode_index in lengths}
    frames: dict[int, list[int]] = {episode_index: [] for episode_index in lengths}
    task_ids: dict[int, set[int]] = {episode_index: set() for episode_index in lengths}
    states: dict[int, list[Any]] = {episode_index: [] for episode_index in lengths}
    actions: dict[int, list[Any]] = {episode_index: [] for episode_index in lengths}
    data_paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"{spec.dataset_id}/{label}: no data parquet")
    required_columns = {
        "action",
        "observation.state",
        "timestamp",
        "frame_index",
        "episode_index",
        "task_index",
    }
    for path in data_paths:
        table = pq.read_table(path)
        missing = required_columns - set(table.column_names)
        if missing:
            raise ValueError(f"{spec.dataset_id}/{label}: {path} missing columns {missing}")
        episode_values = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        frame_values = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
        task_values = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
        timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        state_values = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        action_values = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        if state_values.shape != (table.num_rows, 6) or action_values.shape != (table.num_rows, 6):
            raise ValueError(f"{spec.dataset_id}/{label}: invalid action/state matrix shape in {path}")
        if not np.isfinite(state_values).all() or not np.isfinite(action_values).all():
            raise ValueError(f"{spec.dataset_id}/{label}: non-finite action/state in {path}")
        if not np.isfinite(timestamps).all():
            raise ValueError(f"{spec.dataset_id}/{label}: non-finite timestamp in {path}")
        fps = float(info["fps"])
        if not np.allclose(timestamps, frame_values / fps, atol=1e-5, rtol=0):
            raise ValueError(f"{spec.dataset_id}/{label}: timestamp != frame_index/fps in {path}")
        for row_index, episode_index_value in enumerate(episode_values):
            episode_index = int(episode_index_value)
            if episode_index not in counts:
                raise ValueError(f"{spec.dataset_id}/{label}: unknown episode {episode_index}")
            counts[episode_index] += 1
            frames[episode_index].append(int(frame_values[row_index]))
            task_ids[episode_index].add(int(task_values[row_index]))
            states[episode_index].append(state_values[row_index])
            actions[episode_index].append(action_values[row_index])

    anomaly_rows: list[dict[str, Any]] = []
    for episode_index, expected_length in lengths.items():
        if counts[episode_index] != expected_length:
            raise ValueError(
                f"{spec.dataset_id}/{label}: episode {episode_index} rows "
                f"{counts[episode_index]} != {expected_length}"
            )
        if frames[episode_index] != list(range(expected_length)):
            raise ValueError(f"{spec.dataset_id}/{label}: frame_index gap in episode {episode_index}")
        if task_ids[episode_index] != expected_tasks_by_episode[episode_index]:
            raise ValueError(
                f"{spec.dataset_id}/{label}: episode {episode_index} task mismatch "
                f"frames={task_ids[episode_index]} metadata={expected_tasks_by_episode[episode_index]}"
            )
        episode_states = np.asarray(states[episode_index])
        episode_actions = np.asarray(actions[episode_index])
        state_steps = np.linalg.norm(np.diff(episode_states[:, :5], axis=0), axis=1)
        anomaly_rows.append(
            {
                "episode_index": episode_index,
                "initial_action_state_l2": float(
                    np.linalg.norm(episode_actions[0, :5] - episode_states[0, :5])
                ),
                "max_state_step_l2": float(state_steps.max()) if len(state_steps) else 0.0,
                "state_span_l2": float(
                    np.linalg.norm(
                        episode_states[:, :5].max(axis=0) - episode_states[:, :5].min(axis=0)
                    )
                ),
            }
        )

    for episode in episodes:
        for video_key in video_keys:
            chunk_index = int(episode[f"videos/{video_key}/chunk_index"])
            file_index = int(episode[f"videos/{video_key}/file_index"])
            video_path = (
                root
                / "videos"
                / video_key
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4"
            )
            if not video_path.is_file() or video_path.stat().st_size == 0:
                raise FileNotFoundError(
                    f"{spec.dataset_id}/{label}: missing/empty source video {video_path}"
                )

    top_initial = sorted(
        anomaly_rows, key=lambda row: row["initial_action_state_l2"], reverse=True
    )[:10]
    top_steps = sorted(anomaly_rows, key=lambda row: row["max_state_step_l2"], reverse=True)[:10]
    lowest_motion = sorted(anomaly_rows, key=lambda row: row["state_span_l2"])[:10]
    task_episode_counts = {
        str(task_index): sum(task_ids[index] == {task_index} for index in task_ids)
        for task_index in task_map
    }
    return {
        "path": str(root),
        "codebase_version": info["codebase_version"],
        "fps": info["fps"],
        "total_episodes": len(episodes),
        "total_frames": sum(lengths.values()),
        "tasks": task_map,
        "task_episode_counts": task_episode_counts,
        "task_text_episode_counts": {
            task_map[task_index]: task_episode_counts[str(task_index)]
            for task_index in task_map
        },
        "video_keys": sorted(video_keys),
        "camera_layout": spec.camera_layout,
        "camera_roles": spec.camera_roles,
        "instruction_override": spec.instruction_override,
        "instruction_override_reason": spec.instruction_override_reason,
        "top_initial_action_state_gaps": top_initial,
        "top_state_steps": top_steps,
        "lowest_motion_episodes": lowest_motion,
    }


def deep_video_audit(root: Path, video_keys: set[str], expected_frames: int) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for video_key in sorted(video_keys):
        files = sorted((root / "videos" / video_key).glob("chunk-*/*.mp4"))
        if not files:
            raise FileNotFoundError(f"No videos found for {video_key}: {root}")
        total_frames = 0
        streams: list[dict[str, Any]] = []
        for path in files:
            command = [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration",
                "-of",
                "json",
                str(path),
            ]
            payload = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
            stream = payload["streams"][0]
            frame_count = int(stream["nb_read_frames"])
            total_frames += frame_count
            streams.append({"path": str(path), **stream})
        if total_frames != expected_frames:
            raise ValueError(
                f"Video frame total mismatch for {root}/{video_key}: "
                f"decoded={total_frames} expected={expected_frames}"
            )
        results[video_key] = {"total_frames": total_frames, "files": streams}
    return results


def _sample_episode_indices(total_episodes: int) -> list[int]:
    if total_episodes <= 0:
        raise ValueError(f"total_episodes must be positive, got {total_episodes}")
    return sorted({0, total_episodes // 2, total_episodes - 1})


def verify_prepared_identity(spec: DatasetSpec) -> dict[str, Any]:
    """Match a prepared copy to the manifest, not merely to its own metadata."""

    root = spec.prepared_dataset
    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.jsonl"
    episodes_path = root / "meta" / "episodes.jsonl"
    for path in (info_path, tasks_path, episodes_path):
        if not path.is_file():
            raise FileNotFoundError(f"{spec.dataset_id}: missing prepared metadata {path}")

    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError(
            f"{spec.dataset_id}: expected prepared v2.1 data, "
            f"got {info.get('codebase_version')!r}"
        )
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"{spec.dataset_id}: prepared features must be an object")
    video_keys = {
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    }
    expected_video_keys = set(spec.camera_roles.values())
    if video_keys != expected_video_keys:
        raise ValueError(
            f"{spec.dataset_id}: prepared cameras do not match manifest: "
            f"actual={sorted(video_keys)} expected={sorted(expected_video_keys)}"
        )

    task_rows = [
        json.loads(line)
        for line in tasks_path.read_text().splitlines()
        if line.strip()
    ]
    task_map: dict[int, str] = {}
    for row in task_rows:
        task_index = int(row["task_index"])
        task_text = str(row["task"]).strip()
        if task_index in task_map or not task_text:
            raise ValueError(f"{spec.dataset_id}: invalid prepared task row {row!r}")
        task_map[task_index] = task_text
    if not task_map or len(set(task_map.values())) != len(task_map):
        raise ValueError(f"{spec.dataset_id}: prepared task text is empty or duplicated")

    episode_rows = [
        json.loads(line)
        for line in episodes_path.read_text().splitlines()
        if line.strip()
    ]
    episode_rows.sort(key=lambda row: int(row["episode_index"]))
    episode_indices = [int(row["episode_index"]) for row in episode_rows]
    if episode_indices != list(range(len(episode_rows))):
        raise ValueError(f"{spec.dataset_id}: prepared episode_index is not contiguous")
    task_episode_counts: dict[str, int] = {}
    total_frames = 0
    known_tasks = set(task_map.values())
    for row in episode_rows:
        length = int(row["length"])
        if length <= 0:
            raise ValueError(f"{spec.dataset_id}: prepared episode has invalid length {row!r}")
        total_frames += length
        episode_tasks = {str(text).strip() for text in row.get("tasks", [])}
        if len(episode_tasks) != 1 or not episode_tasks <= known_tasks:
            raise ValueError(
                f"{spec.dataset_id}: prepared episode must reference exactly one known task: "
                f"{row!r}"
            )
        task_text = next(iter(episode_tasks))
        task_episode_counts[task_text] = task_episode_counts.get(task_text, 0) + 1

    expected = dict(spec.expected_training)
    if spec.instruction_override:
        expected = {
            "episodes": expected["episodes"],
            "frames": expected["frames"],
            "task_episode_counts": {
                spec.instruction_override: expected["episodes"],
            },
        }
    report = {
        "total_episodes": len(episode_rows),
        "total_frames": total_frames,
        "task_text_episode_counts": task_episode_counts,
    }
    _assert_expected_summary(report, expected, context=f"{spec.dataset_id}/prepared")
    if int(info.get("total_episodes", -1)) != report["total_episodes"]:
        raise ValueError(f"{spec.dataset_id}: prepared info total_episodes mismatch")
    if int(info.get("total_frames", -1)) != report["total_frames"]:
        raise ValueError(f"{spec.dataset_id}: prepared info total_frames mismatch")
    if int(info.get("total_tasks", -1)) != len(task_map):
        raise ValueError(f"{spec.dataset_id}: prepared info total_tasks mismatch")

    print(
        f"[verify] {spec.dataset_id}: prepared identity PASS "
        f"episodes={report['total_episodes']} frames={report['total_frames']} "
        f"tasks={task_episode_counts}"
    )
    return report


def verify_prepared_video_samples(spec: DatasetSpec) -> dict[str, Any]:
    """Decode-count first/middle/last prepared videos for every camera role."""

    root = spec.prepared_dataset
    info = json.loads((root / "meta" / "info.json").read_text())
    episode_rows = [
        json.loads(line)
        for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    episode_map = {int(row["episode_index"]): row for row in episode_rows}
    total_episodes = int(info["total_episodes"])
    if len(episode_map) != total_episodes:
        raise ValueError(
            f"{spec.dataset_id}: prepared episode metadata count mismatch: "
            f"rows={len(episode_map)} info={total_episodes}"
        )
    chunks_size = int(info.get("chunks_size", 1000))
    fps = Fraction(str(info["fps"]))
    sample_indices = _sample_episode_indices(total_episodes)
    results: dict[str, Any] = {}
    for episode_index in sample_indices:
        episode = episode_map[episode_index]
        expected_frames = int(episode["length"])
        episode_results: dict[str, Any] = {}
        for video_key in sorted(set(spec.camera_roles.values())):
            feature = info["features"][video_key]
            height, width = (int(value) for value in feature["shape"][:2])
            video_path = root / V21_VIDEO_PATH.format(
                episode_chunk=episode_index // chunks_size,
                video_key=video_key,
                episode_index=episode_index,
            )
            command = [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
                "-of",
                "json",
                str(video_path),
            ]
            payload = json.loads(
                subprocess.run(command, check=True, capture_output=True, text=True).stdout
            )
            stream = payload["streams"][0]
            actual_frames = int(stream["nb_read_frames"])
            actual_rate = Fraction(stream["avg_frame_rate"])
            actual_shape = (int(stream["height"]), int(stream["width"]))
            if actual_frames != expected_frames:
                raise ValueError(
                    f"{spec.dataset_id}: prepared video frame mismatch: path={video_path} "
                    f"decoded={actual_frames} expected={expected_frames}"
                )
            if actual_rate != fps or actual_shape != (height, width):
                raise ValueError(
                    f"{spec.dataset_id}: prepared video stream mismatch: path={video_path} "
                    f"fps={actual_rate}/{fps} shape={actual_shape}/{(height, width)}"
                )
            episode_results[video_key] = {
                "path": str(video_path),
                "frames": actual_frames,
                "fps": str(actual_rate),
                "shape": [height, width],
                "codec": stream["codec_name"],
            }
        results[str(episode_index)] = episode_results
    print(
        f"[verify] {spec.dataset_id}: prepared video samples PASS "
        f"episodes={sample_indices} cameras={sorted(set(spec.camera_roles.values()))}"
    )
    return results


def _common_wrapper_args(spec: DatasetSpec, groot_root: Path) -> list[str]:
    args = [
        "--groot-root",
        str(groot_root),
        "--camera-layout",
        spec.camera_layout,
    ]
    for role in ("top", "left", "wrist"):
        if role in spec.camera_roles:
            args.extend([ROLE_FLAGS[role], spec.camera_roles[role]])
    return args


def _training_args(defaults: TrainingDefaults, spec: DatasetSpec) -> list[str]:
    args = [
        "--base-model-path",
        defaults.base_model_path,
        "--global-batch-size",
        str(defaults.global_batch_size),
        "--gradient-accumulation-steps",
        str(defaults.gradient_accumulation_steps),
        "--learning-rate",
        str(defaults.learning_rate),
        "--dataloader-num-workers",
        str(defaults.dataloader_num_workers),
        "--max-steps",
        str(spec.max_steps),
        "--save-steps",
        str(spec.save_steps),
        "--save-total-limit",
        str(spec.save_total_limit),
        "--state-dropout-prob",
        str(defaults.state_dropout_prob),
    ]
    args.append("--color-jitter" if defaults.color_jitter else "--no-color-jitter")
    return args


def build_command(
    action: str,
    spec: DatasetSpec,
    defaults: TrainingDefaults,
    *,
    data_python: Path,
    groot_python: Path,
    groot_root: Path,
    output_root: Path,
    run_tag: str,
    force_prepare: bool,
) -> list[str]:
    common = _common_wrapper_args(spec, groot_root)
    if action == "prepare":
        command = [
            str(data_python),
            str(TRAIN_WRAPPER),
            "--source-dataset",
            str(spec.training_source),
            "--prepared-root",
            str(spec.prepared_root),
            *common,
            "--skip-stats",
            "--prepare-only",
        ]
        if force_prepare:
            command.append("--force-prepare")
        if spec.instruction_override:
            command.extend(["--instruction", spec.instruction_override])
        return command

    command = [
        str(groot_python),
        str(TRAIN_WRAPPER),
        "--prepared-dataset",
        str(spec.prepared_dataset),
        *common,
        "--skip-prepare",
    ]
    if action == "stats":
        return [*command, "--force-stats", "--prepare-only"]
    command.extend(["--skip-stats", *_training_args(defaults, spec)])
    if action == "dry-run":
        return [
            *command,
            "--output-dir",
            str(output_root / f"{run_tag}-dry-run"),
            "--experiment-name",
            spec.dataset_id,
            "--dry-run",
        ]
    if action == "smoke":
        command.extend(
            [
                "--output-dir",
                str(output_root / f"{run_tag}-smoke"),
                "--experiment-name",
                spec.dataset_id,
                "--global-batch-size",
                "1",
                "--gradient-accumulation-steps",
                "1",
                "--dataloader-num-workers",
                "0",
                "--max-steps",
                "1",
                "--save-steps",
                "1",
                "--save-total-limit",
                "1",
            ]
        )
        return command
    if action == "train":
        return [
            *command,
            "--output-dir",
            str(output_root / run_tag),
            "--experiment-name",
            spec.dataset_id,
        ]
    raise ValueError(f"Unsupported command action: {action}")


def _print_command(command: list[str]) -> None:
    print(f"[batch] {shlex.join(command)}", flush=True)


def _require_groot_commit(groot_root: Path, expected_commit: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(groot_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual_commit = result.stdout.strip()
    if actual_commit != expected_commit:
        raise ValueError(
            f"Formal smoke/train requires Isaac-GR00T commit {expected_commit}, "
            f"got {actual_commit} at {groot_root}"
        )
    dirty = subprocess.run(
        ["git", "-C", str(groot_root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError(
            f"Formal stats/smoke/train requires a clean Isaac-GR00T worktree: "
            f"{groot_root}\n{dirty}"
        )
    return actual_commit


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_prepared_dataset(root: Path) -> dict[str, Any]:
    """Hash every prepared artifact so same-count stale copies cannot share lineage."""

    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Prepared dataset not found for fingerprint: {root}")
    tree_digest = hashlib.sha256()
    content_digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    stats_sha256: dict[str, str] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(f"Prepared dataset contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_sha256 = _sha256_file(path)
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(str(size).encode("ascii"))
        tree_digest.update(b"\0")
        tree_digest.update(bytes.fromhex(file_sha256))
        if relative not in {"meta/stats.json", "meta/relative_stats.json"}:
            content_digest.update(relative.encode("utf-8"))
            content_digest.update(b"\0")
            content_digest.update(str(size).encode("ascii"))
            content_digest.update(b"\0")
            content_digest.update(bytes.fromhex(file_sha256))
        total_bytes += size
        file_count += 1
        if relative in {"meta/stats.json", "meta/relative_stats.json"}:
            stats_sha256[relative] = file_sha256
    required_stats = {"meta/stats.json", "meta/relative_stats.json"}
    if set(stats_sha256) != required_stats:
        raise FileNotFoundError(
            f"Prepared fingerprint requires fresh stats files: root={root} "
            f"found={sorted(stats_sha256)}"
        )
    result = {
        "sha256": tree_digest.hexdigest(),
        "content_sha256_excluding_stats": content_digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "stats_sha256": stats_sha256,
    }
    print(
        f"[lineage] prepared fingerprint root={root} files={file_count} "
        f"bytes={total_bytes} sha256={result['sha256']}"
    )
    return result


def verify_prepared_fingerprint(
    spec: DatasetSpec,
    fingerprint: dict[str, Any],
) -> None:
    actual = fingerprint.get("content_sha256_excluding_stats")
    expected = spec.expected_prepared_content_sha256
    if actual != expected:
        raise ValueError(
            f"{spec.dataset_id}: prepared content SHA256 does not match the local golden "
            f"manifest value: expected={expected} actual={actual}. Do not train this upload."
        )
    print(f"[lineage] {spec.dataset_id}: prepared content matches manifest SHA256")


def _pipeline_code_sha256(spec: DatasetSpec) -> dict[str, str]:
    paths = {
        "aws_training_batch.py": Path(__file__).resolve(),
        "aws_training_pipeline.sh": SCRIPT_DIR / "aws_training_pipeline.sh",
        "version_contract.py": SCRIPT_DIR / "version_contract.py",
        "train_so101_synthetic_groot.py": TRAIN_WRAPPER,
        "modality_config": MODALITY_CONFIGS[spec.camera_layout],
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


def _write_run_lineage(
    path: Path,
    *,
    manifest_path: Path,
    manifest_payload: dict[str, Any],
    spec: DatasetSpec,
    command: list[str],
    action: str,
    run_tag: str,
    actual_groot_commit: str,
    prepared_fingerprint: dict[str, Any],
) -> None:
    manifest_bytes = manifest_path.read_bytes()
    stable_payload = {
        "schema_version": 1,
        "action": action,
        "run_tag": run_tag,
        "dataset_id": spec.dataset_id,
        "manifest_name": manifest_payload.get("name"),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "declared_groot_commit": manifest_payload.get("groot_commit"),
        "actual_groot_commit": actual_groot_commit,
        "base_model_revision": manifest_payload.get("base_model_revision"),
        "pipeline_code_sha256": _pipeline_code_sha256(spec),
        "prepared_fingerprint": prepared_fingerprint,
        "expected_prepared_content_sha256": spec.expected_prepared_content_sha256,
        "training_source": str(spec.training_source),
        "prepared_dataset": str(spec.prepared_dataset),
        "camera_layout": spec.camera_layout,
        "camera_roles": spec.camera_roles,
        "instruction_override": spec.instruction_override,
        "max_steps": spec.max_steps,
        "save_steps": spec.save_steps,
        "save_total_limit": spec.save_total_limit,
        "command": command,
    }
    if path.is_file():
        existing = json.loads(path.read_text())
        existing.pop("created_at_utc", None)
        if existing != stable_payload:
            raise FileExistsError(
                f"Run lineage already exists with different inputs: {path}. "
                "Use a new --run-tag."
            )
        print(f"[lineage] reuse matching {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **stable_payload,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[lineage] wrote {path}")


def _require_matching_stats_lineage(
    path: Path,
    *,
    manifest_path: Path,
    manifest_payload: dict[str, Any],
    spec: DatasetSpec,
    actual_groot_commit: str,
    prepared_fingerprint: dict[str, Any],
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Fresh stats lineage is missing for {spec.dataset_id}: {path}. "
            "Run stats with the same --run-tag before smoke/train."
        )
    payload = json.loads(path.read_text())
    expected = {
        "action": "stats",
        "run_tag": path.parents[1].name.removesuffix("-stats"),
        "dataset_id": spec.dataset_id,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "actual_groot_commit": actual_groot_commit,
        "base_model_revision": manifest_payload.get("base_model_revision"),
        "pipeline_code_sha256": _pipeline_code_sha256(spec),
        "prepared_fingerprint": prepared_fingerprint,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Stats lineage does not match current training input for {spec.dataset_id}: "
            f"{mismatches}. Rerun stats with this --run-tag."
        )
    print(f"[lineage] matching fresh stats lineage: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("audit", "prepare", "verify", "stats", "dry-run", "smoke", "train"),
    )
    parser.add_argument("datasets", nargs="*", default=["all"])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--data-python",
        type=Path,
        default=Path(os.environ.get("AWS_DATA_PYTHON", sys.executable)),
    )
    parser.add_argument(
        "--groot-root",
        type=Path,
        default=Path(os.environ.get("GROOT_ROOT", str(Path.home() / "Isaac-GR00T"))),
    )
    parser.add_argument("--groot-python", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get(
                "TRAIN_OUTPUT_ROOT",
                str(PROJECT_ROOT / "outputs" / "groot_so101_synthetic_finetune"),
            )
        ),
    )
    parser.add_argument("--run-tag", default="aws-third-20260804")
    parser.add_argument(
        "--base-model-path",
        type=Path,
        default=None,
        help="Resolved pinned base-model snapshot used by formal AWS smoke/train.",
    )
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--deep-video", action="store_true")
    parser.add_argument("--audit-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not SAFE_SLUG.fullmatch(args.run_tag):
        raise ValueError(f"--run-tag must be a safe slug, got {args.run_tag!r}")
    manifest_path = args.manifest.expanduser().resolve()
    payload, defaults, all_specs = load_manifest(manifest_path)
    if args.action in {"smoke", "train"} and args.base_model_path is None:
        raise ValueError(
            "Formal smoke/train requires --base-model-path pointing to the pinned "
            "Hugging Face snapshot prepared by aws_training_pipeline.sh preflight."
        )
    if args.base_model_path is not None:
        resolved_base_model = args.base_model_path.expanduser().resolve()
        expected_model_revision = str(payload["base_model_revision"])
        if not resolved_base_model.is_dir():
            raise FileNotFoundError(f"Base-model snapshot not found: {resolved_base_model}")
        if args.action in {"smoke", "train"} and resolved_base_model.name != expected_model_revision:
            raise ValueError(
                "Base-model snapshot path does not end in the pinned revision: "
                f"path={resolved_base_model} expected={expected_model_revision}"
            )
        defaults = replace(
            defaults,
            base_model_path=str(resolved_base_model),
        )
    specs = select_specs(all_specs, args.datasets)
    groot_root = args.groot_root.expanduser().resolve()
    groot_python = (
        _absolute_executable(args.groot_python)
        if args.groot_python
        else groot_root / ".venv" / "bin" / "python"
    )
    data_python = _absolute_executable(args.data_python)
    output_root = args.output_root.expanduser().resolve()

    print(
        f"[batch] manifest={manifest_path} action={args.action} "
        f"datasets={[spec.dataset_id for spec in specs]}"
    )
    if args.action == "audit":
        report: dict[str, Any] = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest_path),
            "manifest_name": payload.get("name"),
            "datasets": {},
        }
        for spec in specs:
            print(f"[audit] {spec.dataset_id}: raw={spec.raw_source}", flush=True)
            raw_report = audit_v3_dataset(spec.raw_source, spec, label="raw")
            _assert_expected_summary(
                raw_report,
                spec.expected_raw,
                context=f"{spec.dataset_id}/raw",
            )
            dataset_report: dict[str, Any] = {
                "raw": raw_report,
                "removed_raw_episodes": list(spec.removed_raw_episodes),
                "known_risks": list(spec.known_risks),
                "instruction_override": spec.instruction_override,
                "instruction_override_reason": spec.instruction_override_reason,
            }
            if spec.training_source != spec.raw_source:
                print(
                    f"[audit] {spec.dataset_id}: training_source={spec.training_source}",
                    flush=True,
                )
                training_report = audit_v3_dataset(
                    spec.training_source, spec, label="training_source"
                )
                _assert_expected_summary(
                    training_report,
                    spec.expected_training,
                    context=f"{spec.dataset_id}/training_source",
                )
                if (
                    raw_report["total_episodes"] - len(spec.removed_raw_episodes)
                    != training_report["total_episodes"]
                ):
                    raise ValueError(
                        f"{spec.dataset_id}: clean episode count does not match removed_raw_episodes"
                    )
                dataset_report["training_source"] = training_report
            if args.deep_video:
                target_report = dataset_report.get("training_source", raw_report)
                dataset_report["deep_video"] = deep_video_audit(
                    spec.training_source,
                    set(spec.camera_roles.values()),
                    int(target_report["total_frames"]),
                )
            report["datasets"][spec.dataset_id] = dataset_report

        audit_output = (
            args.audit_output.expanduser().resolve()
            if args.audit_output
            else PROJECT_ROOT
            / "outputs"
            / "aws_training_audits"
            / f"{payload.get('name', manifest_path.stem)}.json"
        )
        audit_output.parent.mkdir(parents=True, exist_ok=True)
        audit_output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"[audit] PASS report={audit_output}")
        return

    if args.action == "verify":
        for spec in specs:
            verify_prepared_identity(spec)
            verify_prepared_video_samples(spec)
        print(f"[batch] PASS action=verify count={len(specs)}")
        return

    required_python = data_python if args.action == "prepare" else groot_python
    if not required_python.is_file():
        raise FileNotFoundError(f"Required Python not found: {required_python}")
    actual_groot_commit: str | None = None
    if args.action in {"stats", "smoke", "train"}:
        expected_commit = str(payload.get("groot_commit", ""))
        if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
            raise ValueError(f"Manifest groot_commit is not a full SHA: {expected_commit!r}")
        actual_groot_commit = _require_groot_commit(groot_root, expected_commit)
    for spec in specs:
        command = build_command(
            args.action,
            spec,
            defaults,
            data_python=data_python,
            groot_python=groot_python,
            groot_root=groot_root,
            output_root=output_root,
            run_tag=args.run_tag,
            force_prepare=args.force_prepare,
        )
        prepared_fingerprint: dict[str, Any] | None = None
        if args.action in {"smoke", "train"}:
            prepared_fingerprint = fingerprint_prepared_dataset(spec.prepared_dataset)
            verify_prepared_fingerprint(spec, prepared_fingerprint)
            _require_matching_stats_lineage(
                output_root
                / f"{args.run_tag}-stats"
                / "_lineage"
                / f"{spec.dataset_id}.json",
                manifest_path=manifest_path,
                manifest_payload=payload,
                spec=spec,
                actual_groot_commit=actual_groot_commit,
                prepared_fingerprint=prepared_fingerprint,
            )
        if actual_groot_commit is not None and args.action in {"smoke", "train"}:
            run_root_name = f"{args.run_tag}-smoke" if args.action == "smoke" else args.run_tag
            _write_run_lineage(
                output_root / run_root_name / "_lineage" / f"{spec.dataset_id}.json",
                manifest_path=manifest_path,
                manifest_payload=payload,
                spec=spec,
                command=command,
                action=args.action,
                run_tag=args.run_tag,
                actual_groot_commit=actual_groot_commit,
                prepared_fingerprint=prepared_fingerprint,
            )
        _print_command(command)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        if args.action == "stats":
            prepared_fingerprint = fingerprint_prepared_dataset(spec.prepared_dataset)
            verify_prepared_fingerprint(spec, prepared_fingerprint)
            _write_run_lineage(
                output_root / f"{args.run_tag}-stats" / "_lineage" / f"{spec.dataset_id}.json",
                manifest_path=manifest_path,
                manifest_payload=payload,
                spec=spec,
                command=command,
                action=args.action,
                run_tag=args.run_tag,
                actual_groot_commit=actual_groot_commit,
                prepared_fingerprint=prepared_fingerprint,
            )
        if args.action == "prepare" and not spec.prepared_dataset.is_dir():
            raise FileNotFoundError(
                f"Prepared output did not appear at manifest path: {spec.prepared_dataset}"
            )
        if args.action == "prepare":
            verify_prepared_identity(spec)
            verify_prepared_video_samples(spec)
    print(f"[batch] PASS action={args.action} count={len(specs)}")


if __name__ == "__main__":
    main()
