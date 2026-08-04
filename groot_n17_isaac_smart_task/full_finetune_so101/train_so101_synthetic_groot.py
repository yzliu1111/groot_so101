#!/usr/bin/env python3
"""Prepare LeIsaac SO101 synthetic data and launch GR00T N1.7 fine-tuning.

This helper is intentionally split-friendly because this workstation keeps
LeRobot and Isaac-GR00T in separate Python environments.

Recommended local flow:

1. Run dataset conversion/preparation in the LeRobot conda environment:

    conda run -n lerobot python \
      experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
        --force-prepare --skip-stats --prepare-only

2. Run GR00T stats/fine-tuning from the Isaac-GR00T virtualenv:

    ${GROOT_ROOT:-$HOME/Isaac-GR00T-py312}/.venv/bin/python \
      experiments/groot_n17_isaac_smart_task/full_finetune_so101/train_so101_synthetic_groot.py \
        --skip-prepare

By default, the script writes prepared GR00T-flavored LeRobot v2.1 copies under
``outputs/`` and does not modify the source ``dataset/`` folders.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any

import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_GROOT_ROOT = Path(
    os.environ.get("GROOT_ROOT", str(Path.home() / "Isaac-GR00T-py312"))
).expanduser()
DEFAULT_DATASETS = (
    REPO_ROOT / "dataset" / "so101_lego_pick_0609_1722",
    REPO_ROOT / "dataset" / "so101_lego_pick_0609_1722_mimic",
)
DUAL_MODALITY_CONFIG_PATH = SCRIPT_DIR / "so101_synthetic_groot_config.py"
WRIST_ONLY_MODALITY_CONFIG_PATH = SCRIPT_DIR / "so101_synthetic_groot_wrist_only_config.py"
TRIPLE_MODALITY_CONFIG_PATH = SCRIPT_DIR / "so101_synthetic_groot_triple_config.py"
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"

V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
SO101_JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
LOW_DIM_MODALITY = {
    "single_arm": {"start": 0, "end": 5},
    "gripper": {"start": 5, "end": 6},
}


def _is_lerobot_v3_dataset(path: Path) -> bool:
    info_path = path / "meta" / "info.json"
    if not info_path.exists():
        return False
    try:
        with info_path.open("r") as f:
            info = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return info.get("codebase_version") == "v3.0"


def _safe_dataset_stem(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return safe.strip("._-") or "dataset"


def _discover_lerobot_v3_datasets(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Source root does not exist: {root}")
    if _is_lerobot_v3_dataset(root):
        return [root]

    datasets: list[Path] = []
    for dirpath, _, filenames in os.walk(root, followlinks=False):
        path = Path(dirpath)
        if path.name == "meta" and "info.json" in filenames:
            dataset_path = path.parent
            if _is_lerobot_v3_dataset(dataset_path):
                datasets.append(dataset_path)
    return sorted(set(datasets))


def _camera_layout_leaf_name(name: str, camera_layout: str) -> str:
    if camera_layout == "dual":
        return name
    return f"{name}_{camera_layout.replace('-', '_')}"


def _prepared_relative_path(
    source_path: Path,
    root: Path | None,
    camera_layout: str,
) -> Path:
    if root is None:
        return Path(_camera_layout_leaf_name(_safe_dataset_stem(source_path.name), camera_layout))

    rel = source_path.relative_to(root)
    parts = [_safe_dataset_stem(part) for part in rel.parts]
    if not parts:
        parts = [_safe_dataset_stem(source_path.name)]
    parts[-1] = _camera_layout_leaf_name(parts[-1], camera_layout)
    return Path(*parts)


def _format_relative_path(path: Path) -> str:
    return path.as_posix()


def _resolve_source_datasets(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    source_specs: list[tuple[Path, Path | None]] = []

    if args.source_dataset:
        source_specs.extend((path.expanduser().resolve(), None) for path in args.source_dataset)

    if args.source_root:
        for root_value in args.source_root:
            root = root_value.expanduser().resolve()
            discovered = _discover_lerobot_v3_datasets(root)
            if not discovered:
                raise FileNotFoundError(f"No LeRobot v3 datasets found under source root: {root}")
            source_specs.extend((path, root) for path in discovered)

    if not source_specs:
        source_specs.extend((path.resolve(), None) for path in DEFAULT_DATASETS)

    seen_sources: set[Path] = set()
    resolved: list[tuple[Path, Path]] = []
    for source_path, root in source_specs:
        source_path = source_path.expanduser()
        resolved_source_path = source_path.resolve()
        if resolved_source_path in seen_sources:
            continue
        seen_sources.add(resolved_source_path)
        if not _is_lerobot_v3_dataset(source_path):
            raise ValueError(f"Expected a LeRobot v3 dataset with meta/info.json: {source_path}")
        _validate_so101_schema(_load_info(resolved_source_path), context=resolved_source_path)
        prepared_relative_path = _prepared_relative_path(source_path, root, args.camera_layout)
        resolved.append((resolved_source_path, prepared_relative_path))

    names: dict[Path, Path] = {}
    for source_path, prepared_relative_path in resolved:
        previous = names.get(prepared_relative_path)
        if previous is not None:
            raise ValueError(
                "Prepared dataset path collision: "
                f"{_format_relative_path(prepared_relative_path)!r} from {previous} and {source_path}. "
                "Use separate --prepared-root values or rename one source dataset."
            )
        names[prepared_relative_path] = source_path

    return resolved


def _json_default(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return value.as_py()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=4, default=_json_default)
        f.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default))
            f.write("\n")


def _load_info(dataset_path: Path) -> dict[str, Any]:
    with (dataset_path / "meta" / "info.json").open("r") as f:
        return json.load(f)


def _validate_so101_schema(info: dict[str, Any], *, context: Path) -> None:
    """Reject datasets whose low-dimensional columns do not match this experiment."""

    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Dataset has no feature mapping in meta/info.json: {context}")

    for feature_key in ("observation.state", "action"):
        feature = features.get(feature_key)
        if not isinstance(feature, dict):
            raise ValueError(f"Required feature {feature_key!r} is missing: {context}")
        if feature.get("shape") != [6]:
            raise ValueError(
                f"Expected {feature_key} shape [6], got {feature.get('shape')!r}: {context}"
            )
        if feature.get("names") != SO101_JOINT_NAMES:
            raise ValueError(
                f"Unexpected {feature_key} joint order in {context}: "
                f"expected={SO101_JOINT_NAMES!r} actual={feature.get('names')!r}"
            )
        dtype = feature.get("dtype")
        if not isinstance(dtype, str) or not dtype.startswith("float"):
            raise ValueError(f"Expected floating-point {feature_key}, got {dtype!r}: {context}")

    fps = info.get("fps")
    if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Expected a positive finite dataset fps, got {fps!r}: {context}")


def _validate_camera_layout_matches_source(
    info: dict[str, Any],
    *,
    video_key_map: dict[str, str],
    camera_layout: str,
    context: Path,
) -> None:
    """Require the training layout to consume every source video camera exactly once."""

    features = info.get("features", {})
    source_video_keys = {
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    }
    expected_layout_by_count = {
        1: "wrist-only",
        2: "dual",
        3: "triple",
    }
    expected_layout = expected_layout_by_count.get(len(source_video_keys))
    if expected_layout is None:
        raise ValueError(
            f"Expected 1, 2, or 3 video cameras, got {len(source_video_keys)} "
            f"({sorted(source_video_keys)!r}): {context}"
        )
    if camera_layout != expected_layout:
        raise ValueError(
            "Camera layout must match source camera count so training cannot silently "
            f"drop a view: source={context} cameras={sorted(source_video_keys)!r} "
            f"requires --camera-layout {expected_layout}, got {camera_layout!r}"
        )

    selected_video_keys = set(video_key_map.values())
    if selected_video_keys != source_video_keys:
        raise ValueError(
            "Camera role mapping must use every source video exactly once: "
            f"source={context} available={sorted(source_video_keys)!r} "
            f"selected={sorted(selected_video_keys)!r}"
        )


def _load_groot_lerobot_converter(groot_root: Path) -> Any:
    """Load Isaac-GR00T's official LeRobot v3 -> v2 conversion helpers."""

    converter_path = groot_root / "scripts" / "lerobot_conversion" / "convert_v3_to_v2.py"
    if not converter_path.exists():
        raise FileNotFoundError(f"GR00T LeRobot converter not found: {converter_path}")

    if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
        sys.path.insert(0, str(LEROBOT_SRC))

    spec = importlib.util.spec_from_file_location("groot_convert_v3_to_v2", converter_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load GR00T LeRobot converter: {converter_path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (ImportError, ModuleNotFoundError) as exc:
        print(
            "[prepare] official GR00T LeRobot converter is present but its "
            f"dependencies/API are not compatible in this Python env ({exc}); "
            "using the local non-destructive fallback converter."
        )
        return _LightweightV3ToV2Converter()
    print(f"[prepare] using official GR00T LeRobot converter: {converter_path}")
    return module


def _load_episode_records(dataset_path: Path) -> list[dict[str, Any]]:
    episode_paths = sorted((dataset_path / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"No v3 episode metadata parquet files found under {dataset_path}")

    records: list[dict[str, Any]] = []
    for path in episode_paths:
        records.extend(pq.read_table(path).to_pylist())
    return sorted(records, key=lambda item: int(item["episode_index"]))


def _load_tasks(dataset_path: Path, instruction_override: str | None) -> list[dict[str, Any]]:
    table = pq.read_table(dataset_path / "meta" / "tasks.parquet")
    rows = table.to_pylist()
    if instruction_override and len(rows) > 1:
        raise ValueError(
            f"--instruction would collapse {len(rows)} distinct task rows into one label for "
            f"multi-task dataset {dataset_path}. Preserve the source task text instead."
        )
    tasks: list[dict[str, Any]] = []
    seen_task_indices: set[int] = set()

    for row in rows:
        task_index = int(row["task_index"])
        if task_index in seen_task_indices:
            raise ValueError(f"Duplicate task_index {task_index} in {dataset_path}")
        seen_task_indices.add(task_index)
        if instruction_override:
            task = instruction_override
        elif "task" in row:
            task = str(row["task"])
        elif "__index_level_0__" in row:
            task = str(row["__index_level_0__"])
        else:
            string_values = [value for value in row.values() if isinstance(value, str)]
            if not string_values:
                raise ValueError(f"Could not infer task text from row: {row}")
            task = string_values[0]
        if not task.strip():
            raise ValueError(f"Empty task text for task_index {task_index} in {dataset_path}")
        tasks.append({"task_index": task_index, "task": task})

    return sorted(tasks, key=lambda item: item["task_index"])


def _safe_remove_prepared_dir(path: Path, source_path: Path) -> None:
    path = path.resolve()
    source_path = source_path.resolve()
    if path == source_path or source_path in path.parents or path in source_path.parents:
        raise ValueError(
            "Refusing to remove a prepared path that is equal to, inside, or contains "
            f"the source dataset: prepared={path} source={source_path}"
        )
    shutil.rmtree(path)


def _validate_prepared_output_path(path: Path, prepared_root: Path) -> None:
    """Prevent a prepared path or nested symlink from escaping its output root."""

    prepared_root = prepared_root.resolve()
    lexical_path = Path(os.path.abspath(path))
    try:
        relative_path = lexical_path.relative_to(prepared_root)
    except ValueError as exc:
        raise ValueError(
            f"Prepared output is outside --prepared-root: path={path} root={prepared_root}"
        ) from exc
    if not relative_path.parts:
        raise ValueError(f"Refusing to use --prepared-root itself as a dataset output: {path}")

    current = prepared_root
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Prepared output path contains a symlink: {current}")

    resolved_path = lexical_path.resolve()
    if prepared_root not in resolved_path.parents:
        raise ValueError(
            f"Prepared output resolves outside --prepared-root: path={path} root={prepared_root}"
        )


def _feature_subset(info: dict[str, Any], video_keys: list[str]) -> dict[str, Any]:
    keep_keys = {
        "action",
        "observation.state",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        *video_keys,
    }
    return {key: value for key, value in info["features"].items() if key in keep_keys}


def _camera_layout_settings(
    camera_layout: str,
    top_camera_key: str,
    left_camera_key: str,
    wrist_camera_key: str,
) -> tuple[dict[str, str], Path]:
    if camera_layout == "dual":
        if not top_camera_key or not wrist_camera_key:
            raise ValueError("dual camera layout requires both top/front and wrist camera keys")
        if top_camera_key == wrist_camera_key:
            raise ValueError("dual camera layout requires different top/front and wrist camera keys")
        return {"top": top_camera_key, "wrist": wrist_camera_key}, DUAL_MODALITY_CONFIG_PATH
    if camera_layout == "wrist-only":
        if not wrist_camera_key:
            raise ValueError("wrist-only camera layout requires --dataset-wrist-camera-key")
        return {"wrist": wrist_camera_key}, WRIST_ONLY_MODALITY_CONFIG_PATH
    if camera_layout == "triple":
        camera_keys = {
            "top/front": top_camera_key,
            "left": left_camera_key,
            "wrist": wrist_camera_key,
        }
        missing_roles = [role for role, key in camera_keys.items() if not key]
        if missing_roles:
            raise ValueError(f"triple camera layout requires camera keys for: {missing_roles}")
        if len(set(camera_keys.values())) != len(camera_keys):
            raise ValueError(f"triple camera layout requires three different camera keys: {camera_keys}")
        return {
            "top": top_camera_key,
            "left": left_camera_key,
            "wrist": wrist_camera_key,
        }, TRIPLE_MODALITY_CONFIG_PATH
    raise ValueError(f"Unsupported camera layout: {camera_layout}")


def _expected_modality_payload(video_key_map: dict[str, str]) -> dict[str, Any]:
    return {
        "state": LOW_DIM_MODALITY,
        "action": LOW_DIM_MODALITY,
        "video": {
            config_key: {"original_key": original_key}
            for config_key, original_key in video_key_map.items()
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }


def _write_modality_json(output_path: Path, video_key_map: dict[str, str]) -> None:
    _write_json(output_path / "meta" / "modality.json", _expected_modality_payload(video_key_map))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            rows.append(row)
    return rows


def _validate_prepared_dataset(dataset_path: Path, video_key_map: dict[str, str]) -> None:
    """Validate prepared metadata and every episode artifact before stats/training."""

    dataset_path = dataset_path.resolve()
    info_path = dataset_path / "meta" / "info.json"
    modality_path = dataset_path / "meta" / "modality.json"
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    for required_path in (info_path, modality_path, tasks_path, episodes_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Prepared dataset file not found: {required_path}")

    info = _load_info(dataset_path)
    if info.get("codebase_version") != "v2.1":
        raise ValueError(
            f"Expected prepared LeRobot v2.1 data, got {info.get('codebase_version')!r}: "
            f"{dataset_path}"
        )
    _validate_so101_schema(info, context=dataset_path)

    features = info["features"]
    for video_key in video_key_map.values():
        feature = features.get(video_key)
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            raise ValueError(
                f"Selected camera feature {video_key!r} is missing or is not video: {dataset_path}"
            )

    with modality_path.open("r") as f:
        modality = json.load(f)
    expected_modality = _expected_modality_payload(video_key_map)
    if modality != expected_modality:
        raise ValueError(
            "Prepared dataset modality does not exactly match the selected SO101 layout: "
            f"dataset={dataset_path} expected={expected_modality!r} actual={modality!r}. "
            "Use a different --prepared-root or rebuild it with --force-prepare."
        )

    tasks = _load_jsonl(tasks_path)
    if not tasks:
        raise ValueError(f"Prepared dataset has no tasks: {tasks_path}")
    task_by_index: dict[int, str] = {}
    for task in tasks:
        try:
            task_index = int(task["task_index"])
            task_text = str(task["task"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid task metadata in {tasks_path}: {task!r}") from exc
        if task_index < 0 or not task_text or task_index in task_by_index:
            raise ValueError(f"Invalid or duplicate task metadata in {tasks_path}: {task!r}")
        if task_text in task_by_index.values():
            raise ValueError(f"Duplicate task text in {tasks_path}: {task_text!r}")
        task_by_index[task_index] = task_text
    if info.get("total_tasks") != len(tasks):
        raise ValueError(
            f"Prepared total_tasks mismatch in {info_path}: "
            f"info={info.get('total_tasks')!r} tasks={len(tasks)}"
        )
    episodes = _load_jsonl(episodes_path)
    if not episodes:
        raise ValueError(f"Prepared dataset has no episodes: {episodes_path}")

    episode_indices: set[int] = set()
    total_frames = 0
    chunks_size = int(info.get("chunks_size", 1000))
    if chunks_size <= 0:
        raise ValueError(f"Prepared dataset chunks_size must be positive: {dataset_path}")

    for episode in episodes:
        try:
            episode_index = int(episode["episode_index"])
            episode_length = int(episode["length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid episode metadata in {episodes_path}: {episode!r}") from exc
        if episode_index < 0 or episode_length <= 0:
            raise ValueError(f"Invalid episode index/length in {episodes_path}: {episode!r}")
        if episode_index in episode_indices:
            raise ValueError(f"Duplicate episode_index {episode_index} in {episodes_path}")
        episode_indices.add(episode_index)
        total_frames += episode_length
        episode_tasks = {str(task).strip() for task in episode.get("tasks", [])}
        if not episode_tasks or "" in episode_tasks:
            raise ValueError(f"Prepared episode has no valid tasks: {episode!r}")
        unknown_episode_tasks = episode_tasks - set(task_by_index.values())
        if unknown_episode_tasks:
            raise ValueError(
                f"Prepared episode {episode_index} references unknown tasks: "
                f"{sorted(unknown_episode_tasks)!r}"
            )

        episode_chunk = episode_index // chunks_size
        data_path = dataset_path / V21_DATA_PATH.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        if not data_path.is_file() or data_path.stat().st_size == 0:
            raise FileNotFoundError(f"Prepared episode parquet is missing or empty: {data_path}")
        episode_table = pq.read_table(
            data_path,
            columns=["episode_index", "frame_index", "task_index"],
        )
        parquet_rows = episode_table.num_rows
        if parquet_rows != episode_length:
            raise ValueError(
                f"Prepared episode row count mismatch: {data_path} "
                f"metadata_length={episode_length} parquet_rows={parquet_rows}"
            )
        parquet_episode_indices = {
            int(value) for value in episode_table["episode_index"].to_pylist()
        }
        parquet_frame_indices = [
            int(value) for value in episode_table["frame_index"].to_pylist()
        ]
        parquet_task_indices = {
            int(value) for value in episode_table["task_index"].to_pylist()
        }
        unknown_task_indices = parquet_task_indices - set(task_by_index)
        if parquet_episode_indices != {episode_index}:
            raise ValueError(
                f"Prepared episode_index mismatch in {data_path}: "
                f"{sorted(parquet_episode_indices)!r}"
            )
        if parquet_frame_indices != list(range(episode_length)):
            raise ValueError(f"Prepared frame_index gap in {data_path}")
        if unknown_task_indices:
            raise ValueError(
                f"Prepared task_index is unknown in {data_path}: "
                f"{sorted(unknown_task_indices)!r}"
            )
        parquet_task_texts = {task_by_index[index] for index in parquet_task_indices}
        if parquet_task_texts != episode_tasks:
            raise ValueError(
                f"Prepared task text/index mismatch in {data_path}: "
                f"parquet={sorted(parquet_task_texts)!r} "
                f"episode={sorted(episode_tasks)!r}"
            )

        for video_key in dict.fromkeys(video_key_map.values()):
            video_path = dataset_path / V21_VIDEO_PATH.format(
                episode_chunk=episode_chunk,
                video_key=video_key,
                episode_index=episode_index,
            )
            if not video_path.is_file() or video_path.stat().st_size == 0:
                raise FileNotFoundError(f"Prepared episode video is missing or empty: {video_path}")

    if info.get("total_episodes") != len(episodes):
        raise ValueError(
            f"Prepared total_episodes mismatch in {info_path}: "
            f"info={info.get('total_episodes')!r} episodes={len(episodes)}"
        )
    if episode_indices != set(range(len(episodes))):
        raise ValueError(
            f"Prepared episode_index must be contiguous from zero: {episodes_path}"
        )
    if info.get("total_frames") != total_frames:
        raise ValueError(
            f"Prepared total_frames mismatch in {info_path}: "
            f"info={info.get('total_frames')!r} episodes_sum={total_frames}"
        )
    expected_total_videos = len(episodes) * len(set(video_key_map.values()))
    if info.get("total_videos") != expected_total_videos:
        raise ValueError(
            f"Prepared total_videos mismatch in {info_path}: "
            f"info={info.get('total_videos')!r} expected={expected_total_videos}"
        )


def _convert_data_files(source_path: Path, output_path: Path, records: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (int(record["data/chunk_index"]), int(record["data/file_index"]))
        grouped.setdefault(key, []).append(record)

    for (chunk_index, file_index), group_records in grouped.items():
        source_file = source_path / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        if not source_file.is_file():
            raise FileNotFoundError(f"Expected source parquet file not found: {source_file}")
        table = pq.read_table(source_file)
        group_records = sorted(group_records, key=lambda item: int(item["dataset_from_index"]))
        if table.num_rows == 0:
            raise ValueError(f"Source parquet file is empty: {source_file}")
        if "index" in table.column_names:
            file_offset = int(table.column("index")[0].as_py())
        else:
            file_offset = int(group_records[0]["dataset_from_index"])

        for record in group_records:
            episode_index = int(record["episode_index"])
            start = int(record["dataset_from_index"]) - file_offset
            stop = int(record["dataset_to_index"]) - file_offset
            length = stop - start
            if start < 0 or length <= 0 or stop > table.num_rows:
                raise ValueError(
                    "Invalid episode parquet slice: "
                    f"source={source_file} episode_index={episode_index} "
                    f"start={start} stop={stop} rows={table.num_rows}"
                )
            episode_table = table.slice(start, length)
            out_chunk = episode_index // 1000
            out_file = output_path / V21_DATA_PATH.format(
                episode_chunk=out_chunk,
                episode_index=episode_index,
            )
            out_file.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(episode_table, out_file)


def _split_video(
    source_file: Path,
    output_file: Path,
    start_s: float,
    end_s: float,
) -> None:
    if not source_file.is_file():
        raise FileNotFoundError(f"Expected source video file not found: {source_file}")
    if not math.isfinite(start_s) or not math.isfinite(end_s) or start_s < 0 or start_s >= end_s:
        raise ValueError(
            f"Invalid video segment timestamps: source={source_file} start={start_s} end={end_s}"
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    duration_s = end_s - start_s
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.6f}",
        "-i",
        str(source_file),
        "-t",
        f"{duration_s:.6f}",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "1",
        "-y",
        str(output_file),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=300,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for LeRobot v3 video conversion") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg timed out while splitting {source_file} -> {output_file}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() if exc.stderr else "no ffmpeg stderr"
        raise RuntimeError(
            f"ffmpeg failed while splitting {source_file} -> {output_file}: {details}"
        ) from exc


def _convert_video_files(
    source_path: Path,
    output_path: Path,
    records: list[dict[str, Any]],
    video_keys: list[str],
) -> None:
    for video_key in video_keys:
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
        chunk_column = f"videos/{video_key}/chunk_index"
        file_column = f"videos/{video_key}/file_index"

        for record in records:
            if chunk_column not in record or file_column not in record:
                raise KeyError(f"Episode metadata does not contain video key {video_key!r}")
            key = (int(record[chunk_column]), int(record[file_column]))
            grouped.setdefault(key, []).append(record)

        for (chunk_index, file_index), group_records in grouped.items():
            source_file = (
                source_path
                / "videos"
                / video_key
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4"
            )
            for record in sorted(
                group_records,
                key=lambda item: float(item[f"videos/{video_key}/from_timestamp"]),
            ):
                episode_index = int(record["episode_index"])
                out_chunk = episode_index // 1000
                output_file = output_path / V21_VIDEO_PATH.format(
                    episode_chunk=out_chunk,
                    video_key=video_key,
                    episode_index=episode_index,
                )
                _split_video(
                    source_file=source_file,
                    output_file=output_file,
                    start_s=float(record[f"videos/{video_key}/from_timestamp"]),
                    end_s=float(record[f"videos/{video_key}/to_timestamp"]),
                )


class _LightweightV3ToV2Converter:
    """Small fallback matching the official converter calls used by this script."""

    @staticmethod
    def load_episode_records(root: Path) -> list[dict[str, Any]]:
        return _load_episode_records(root)

    @staticmethod
    def convert_info(
        root: Path,
        new_root: Path,
        episode_records: list[dict[str, Any]],
        video_keys: list[str],
    ) -> None:
        info = _load_info(root)
        info["codebase_version"] = "v2.1"
        info["data_path"] = V21_DATA_PATH
        info["video_path"] = V21_VIDEO_PATH if video_keys else None
        info.pop("data_files_size_in_mb", None)
        info.pop("video_files_size_in_mb", None)
        _write_json(new_root / "meta" / "info.json", info)

    @staticmethod
    def convert_tasks(root: Path, new_root: Path) -> None:
        _write_jsonl(new_root / "meta" / "tasks.jsonl", _load_tasks(root, None))

    @staticmethod
    def convert_data(
        root: Path,
        new_root: Path,
        episode_records: list[dict[str, Any]],
        chunks_size: int,
    ) -> None:
        del chunks_size
        _convert_data_files(root, new_root, episode_records)

    @staticmethod
    def convert_videos(
        root: Path,
        new_root: Path,
        episode_records: list[dict[str, Any]],
        video_keys: list[str],
        chunks_size: int,
    ) -> None:
        del chunks_size
        _convert_video_files(root, new_root, episode_records, video_keys)

    @staticmethod
    def convert_episodes_metadata(new_root: Path, episode_records: list[dict[str, Any]]) -> None:
        _write_jsonl(
            new_root / "meta" / "episodes.jsonl",
            [
                {
                    "episode_index": int(record["episode_index"]),
                    "tasks": list(record["tasks"]),
                    "length": int(record["length"]),
                }
                for record in episode_records
            ],
        )


def _rewrite_info_for_groot(
    output_path: Path,
    records: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    video_keys: list[str],
) -> None:
    info = _load_info(output_path)
    chunks_size = int(info.get("chunks_size", 1000))
    total_episodes = len(records)

    info["codebase_version"] = "v2.1"
    info["total_episodes"] = total_episodes
    info["total_frames"] = sum(int(record["length"]) for record in records)
    info["total_tasks"] = len(tasks)
    info["data_path"] = V21_DATA_PATH
    info["video_path"] = V21_VIDEO_PATH
    info["features"] = _feature_subset(info, video_keys)
    info["splits"] = {"train": f"0:{total_episodes}"}
    info["total_chunks"] = math.ceil(total_episodes / chunks_size) if total_episodes else 0
    info["total_videos"] = total_episodes * len(video_keys)
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)
    _write_json(output_path / "meta" / "info.json", info)


def prepare_dataset(
    source_path: Path,
    output_path: Path,
    *,
    groot_root: Path,
    video_key_map: dict[str, str],
    instruction_override: str | None,
    force: bool,
    max_episodes: int | None,
) -> Path:
    source_path = source_path.resolve()
    output_path = output_path.resolve()

    if output_path.exists():
        if not force:
            print(f"[prepare] reuse existing dataset: {output_path}")
            return output_path
        print(f"[prepare] removing previous prepared dataset: {output_path}")
        _safe_remove_prepared_dir(output_path, source_path)

    print(f"[prepare] converting {source_path} -> {output_path}")
    converter = _load_groot_lerobot_converter(groot_root)
    info = _load_info(source_path)
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected LeRobot v3.0 dataset, got {info.get('codebase_version')}")
    _validate_so101_schema(info, context=source_path)

    records = converter.load_episode_records(source_path)
    if max_episodes is not None:
        records = records[:max_episodes]
    if not records:
        raise ValueError(f"No episodes selected for {source_path}")

    video_keys = list(dict.fromkeys(video_key_map.values()))
    missing_video_keys = [key for key in video_keys if key not in info["features"]]
    if missing_video_keys:
        raise KeyError(f"Video keys not found in {source_path}: {missing_video_keys}")

    chunks_size = int(info.get("chunks_size", 1000))
    tasks = _load_tasks(source_path, instruction_override)

    # Reuse NVIDIA's LeRobot v3 -> v2 helpers, but do not call convert_dataset()
    # because that entry point moves the source dataset aside and writes v2 back
    # into the original path. This experiment always writes a separate copy.
    converter.convert_info(source_path, output_path, records, video_keys)
    converter.convert_tasks(source_path, output_path)
    converter.convert_data(source_path, output_path, records, chunks_size)
    converter.convert_videos(source_path, output_path, records, video_keys, chunks_size)
    converter.convert_episodes_metadata(output_path, records)

    _rewrite_info_for_groot(output_path, records, tasks, video_keys)
    _write_jsonl(output_path / "meta" / "tasks.jsonl", tasks)
    _write_jsonl(
        output_path / "meta" / "episodes.jsonl",
        [
            {
                "episode_index": int(record["episode_index"]),
                "tasks": [instruction_override] if instruction_override else list(record["tasks"]),
                "length": int(record["length"]),
            }
            for record in records
        ],
    )
    _write_modality_json(output_path, video_key_map)
    return output_path


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    printable = shlex.join(cmd)
    print(f"[run] cd {cwd} && {printable}")
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def generate_stats(
    dataset_path: Path,
    groot_root: Path,
    modality_config_path: Path,
    dry_run: bool,
    force: bool,
) -> None:
    stats_paths = [
        dataset_path / "meta" / "stats.json",
        dataset_path / "meta" / "relative_stats.json",
    ]
    backups: list[tuple[Path, Path]] = []
    if force:
        if dry_run:
            raise ValueError("--force-stats cannot be combined with --dry-run")
        for stats_path in stats_paths:
            backup_path = stats_path.with_name(f".{stats_path.name}.aws-recompute-backup")
            if backup_path.exists():
                raise FileExistsError(
                    f"Unresolved stats backup exists: {backup_path}. Restore or remove it "
                    "after checking the previous stats failure."
                )
            if stats_path.exists():
                if not stats_path.is_file() or stats_path.is_symlink():
                    raise ValueError(f"Stats cache must be a regular file: {stats_path}")
                stats_path.replace(backup_path)
                backups.append((stats_path, backup_path))
                print(f"[stats] isolated stale cache: {stats_path} -> {backup_path}")

    command = [
        sys.executable,
        "gr00t/data/stats.py",
        "--dataset-path",
        str(dataset_path),
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--modality-config-path",
        str(modality_config_path),
    ]
    try:
        _run(command, cwd=groot_root, dry_run=dry_run)
        if not dry_run:
            missing = [path for path in stats_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Stats generation did not create: {missing}")
    except BaseException:
        for stats_path, backup_path in backups:
            if stats_path.exists():
                stats_path.unlink()
            if backup_path.exists():
                backup_path.replace(stats_path)
        raise
    else:
        for _, backup_path in backups:
            backup_path.unlink()


def _relative_stats_worst_span_ratio(dataset_path: Path) -> dict[str, Any]:
    """Return the relative-action entry most distorted by full-range outliers."""
    stats_path = dataset_path / "meta" / "relative_stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"Relative-action statistics are missing: {stats_path}. "
            "Run stats generation before training."
        )

    payload = json.loads(stats_path.read_text())
    worst: dict[str, Any] | None = None
    required_keys = ("min", "max", "q01", "q99")
    for group_name, group_stats in payload.items():
        if not isinstance(group_stats, dict) or not all(
            key in group_stats for key in required_keys
        ):
            continue
        rows = zip(*(group_stats[key] for key in required_keys), strict=True)
        for horizon_index, (mins, maxs, q01s, q99s) in enumerate(rows):
            values = zip(mins, maxs, q01s, q99s, strict=True)
            for joint_index, (minimum, maximum, q01, q99) in enumerate(values):
                full_span = float(maximum) - float(minimum)
                central_span = float(q99) - float(q01)
                if full_span < 0 or central_span < 0:
                    raise ValueError(
                        f"Invalid relative stats ordering in {stats_path}: "
                        f"group={group_name} horizon={horizon_index} joint={joint_index}"
                    )
                ratio = full_span / max(central_span, 1e-8)
                if group_name == "single_arm" and joint_index < 5:
                    joint_name = SO101_JOINT_NAMES[joint_index]
                elif group_name == "gripper" and joint_index == 0:
                    joint_name = SO101_JOINT_NAMES[5]
                else:
                    joint_name = f"{group_name}[{joint_index}]"
                candidate = {
                    "group": group_name,
                    "horizon_index": horizon_index,
                    "joint_index": joint_index,
                    "joint_name": joint_name,
                    "minimum": float(minimum),
                    "maximum": float(maximum),
                    "q01": float(q01),
                    "q99": float(q99),
                    "ratio": ratio,
                    "stats_path": stats_path,
                }
                if worst is None or ratio > worst["ratio"]:
                    worst = candidate

    if worst is None:
        raise ValueError(f"No usable relative-action groups found in {stats_path}")
    return worst


def audit_relative_stats(
    dataset_path: Path,
    *,
    max_span_ratio: float,
    allow_outliers: bool,
) -> None:
    worst = _relative_stats_worst_span_ratio(dataset_path)
    print(
        "[preflight] relative stats "
        f"dataset={dataset_path} group={worst['group']} "
        f"horizon={worst['horizon_index']} joint={worst['joint_name']} "
        f"full=[{worst['minimum']:.6g}, {worst['maximum']:.6g}] "
        f"q01/q99=[{worst['q01']:.6g}, {worst['q99']:.6g}] "
        f"span_ratio={worst['ratio']:.3f} limit={max_span_ratio:.3f}"
    )
    if worst["ratio"] <= max_span_ratio:
        return

    message = (
        "Relative-action full min/max are dominated by a small tail: "
        f"span ratio {worst['ratio']:.3f} exceeds {max_span_ratio:.3f} at "
        f"{worst['stats_path']} group={worst['group']} "
        f"horizon={worst['horizon_index']} joint={worst['joint_name']}. "
        "GR00T N1.7 currently replaces percentile action bounds with these full "
        "relative min/max values, so a low normalized loss can hide poor physical "
        "precision. Inspect reset-boundary action/state gaps before spending a full run."
    )
    if not allow_outliers:
        raise ValueError(
            message
            + " Pass --allow-relative-stats-outliers only after explicitly accepting "
            "the full-range normalization."
        )
    print(f"[warning] {message}")


def _color_jitter_cli_args(enabled: bool) -> list[str]:
    # Passing nothing (or None) does not disable augmentation: the N1.7 processor
    # loader then inherits the base checkpoint's non-zero ColorJitter defaults.
    # An explicit all-zero mapping is the portable no-op override.
    params = {
        "brightness": 0.3 if enabled else 0.0,
        "contrast": 0.4 if enabled else 0.0,
        "saturation": 0.5 if enabled else 0.0,
        "hue": 0.08 if enabled else 0.0,
    }
    args = ["--color-jitter-params"]
    for name, value in params.items():
        args.extend([name, str(value)])
    return args


def _validate_new_run_dir(
    output_dir: Path,
    experiment_name: str | None,
    *,
    allow_existing: bool,
) -> Path:
    run_dir = output_dir.resolve()
    if experiment_name:
        run_dir /= experiment_name
    if run_dir.exists() and any(run_dir.iterdir()) and not allow_existing:
        raise FileExistsError(
            f"Training run directory is not empty: {run_dir}. Use a unique --output-dir "
            "and --experiment-name so checkpoint lineage cannot be overwritten. Pass "
            "--allow-existing-run-dir only for a deliberately reviewed reuse."
        )
    return run_dir


def _should_audit_relative_stats(args: argparse.Namespace) -> bool:
    return not args.dry_run and (not args.prepare_only or not args.skip_stats)


def launch_finetune(
    args: argparse.Namespace,
    prepared_paths: list[Path],
    modality_config_path: Path,
) -> None:
    cmd = [
        sys.executable,
        "gr00t/experiment/launch_finetune.py",
        "--base-model-path",
        args.base_model_path,
        "--dataset-path",
        os.pathsep.join(str(path) for path in prepared_paths),
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--modality-config-path",
        str(modality_config_path),
        "--num-gpus",
        str(args.num_gpus),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--save-total-limit",
        str(args.save_total_limit),
        "--save-steps",
        str(args.save_steps),
        "--max-steps",
        str(args.max_steps),
        "--global-batch-size",
        str(args.global_batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--dataloader-num-workers",
        str(args.dataloader_num_workers),
        "--state-dropout-prob",
        str(args.state_dropout_prob),
    ]

    if args.experiment_name:
        cmd.extend(["--experiment-name", args.experiment_name])
    if args.use_wandb:
        cmd.append("--use-wandb")
    cmd.extend(_color_jitter_cli_args(args.color_jitter))

    _run(cmd, cwd=args.groot_root.resolve(), dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dataset",
        action="append",
        type=Path,
        default=None,
        help="LeRobot v3 source dataset. Repeat to mix multiple datasets.",
    )
    parser.add_argument(
        "--source-root",
        action="append",
        type=Path,
        default=None,
        help=(
            "Directory to recursively scan for LeRobot v3 datasets. Repeat to mix "
            "task folders such as dataset/custom/pick_only/* and dataset/custom/place_only/*."
        ),
    )
    parser.add_argument(
        "--prepared-dataset",
        action="append",
        type=Path,
        default=None,
        help=(
            "Exact prepared LeRobot v2.1 dataset path. Repeat for a homogeneous mixture. "
            "This is the preferred AWS training input after prepared data is uploaded; "
            "it requires --skip-prepare and cannot be combined with source selectors."
        ),
    )
    parser.add_argument(
        "--allow-multiple-datasets",
        action="store_true",
        help=(
            "Acknowledge that every discovered dataset uses the same action/state coordinate "
            "system. Required when more than one v3 dataset is selected."
        ),
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "groot_so101_synthetic_datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "groot_so101_synthetic_finetune",
    )
    parser.add_argument("--groot-root", type=Path, default=DEFAULT_GROOT_ROOT)
    parser.add_argument("--base-model-path", default="nvidia/GR00T-N1.7-3B")
    parser.add_argument(
        "--dataset-front-camera-key",
        "--top-camera-key",
        "--front-camera-key",
        dest="top_camera_key",
        default="observation.images.camera1",
        help="LeRobot dataset feature key for the front/top view.",
    )
    parser.add_argument(
        "--dataset-left-camera-key",
        "--left-camera-key",
        dest="left_camera_key",
        default="observation.images.camera2",
        help="LeRobot dataset feature key for the left-side view; used by --camera-layout triple.",
    )
    parser.add_argument(
        "--dataset-wrist-camera-key",
        "--wrist-camera-key",
        dest="wrist_camera_key",
        default="observation.images.camera3",
        help="LeRobot dataset feature key for the wrist view.",
    )
    parser.add_argument(
        "--camera-layout",
        choices=("wrist-only", "dual", "triple"),
        required=True,
        help=(
            "wrist-only maps only the wrist camera; dual maps front/top + wrist; "
            "triple maps front/top + left + wrist. The layout must consume all source "
            "video cameras: one camera requires wrist-only, two require dual, and three "
            "require triple. Non-dual prepared datasets get a layout suffix."
        ),
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Optional language override, e.g. 'Pick up the red 2x4 lego brick.'.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument(
        "--force-stats",
        action="store_true",
        help=(
            "Temporarily isolate stats.json and relative_stats.json so GR00T must "
            "recompute them; restore the old files automatically if generation fails."
        ),
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--experiment-name", default="so101_smart_task_synthetic")
    parser.add_argument("--use-wandb", action="store_true")
    jitter_group = parser.add_mutually_exclusive_group()
    jitter_group.add_argument(
        "--color-jitter",
        dest="color_jitter",
        action="store_true",
        help="Enable N1.7's strong non-zero ColorJitter augmentation.",
    )
    jitter_group.add_argument(
        "--no-color-jitter",
        dest="color_jitter",
        action="store_false",
        help=(
            "Disable ColorJitter with an explicit all-zero override instead of "
            "silently inheriting the base checkpoint defaults."
        ),
    )
    parser.set_defaults(color_jitter=False)
    parser.add_argument(
        "--state-dropout-prob",
        type=float,
        default=0.0,
        help=(
            "State dropout probability passed to N1.7. The upstream processor and "
            "model both apply it independently, so the SO101-safe default is 0."
        ),
    )
    parser.add_argument(
        "--max-relative-stats-span-ratio",
        type=float,
        default=5.0,
        help=(
            "Fail when full relative-action range divided by q01-q99 range exceeds "
            "this value. This catches reset-boundary outliers that compress control."
        ),
    )
    parser.add_argument(
        "--allow-relative-stats-outliers",
        action="store_true",
        help="Acknowledge and allow a relative-stats span-ratio preflight failure.",
    )
    parser.add_argument(
        "--allow-existing-run-dir",
        action="store_true",
        help="Allow a non-empty output_dir/experiment_name after manually reviewing lineage.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "num_gpus",
        "global_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "save_steps",
        "save_total_limit",
    )
    for field in positive_integer_fields:
        value = getattr(args, field)
        if value <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive, got {value}")
    if args.dataloader_num_workers < 0:
        raise ValueError(
            "--dataloader-num-workers must be zero or positive, "
            f"got {args.dataloader_num_workers}"
        )
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError(f"--learning-rate must be positive and finite, got {args.learning_rate}")
    if not math.isfinite(args.state_dropout_prob) or not 0 <= args.state_dropout_prob <= 1:
        raise ValueError(
            "--state-dropout-prob must be finite and between 0 and 1, "
            f"got {args.state_dropout_prob}"
        )
    if (
        not math.isfinite(args.max_relative_stats_span_ratio)
        or args.max_relative_stats_span_ratio <= 1
    ):
        raise ValueError(
            "--max-relative-stats-span-ratio must be finite and greater than 1, "
            f"got {args.max_relative_stats_span_ratio}"
        )
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError(f"--max-episodes must be positive, got {args.max_episodes}")
    if args.instruction is not None and not args.instruction.strip():
        raise ValueError("--instruction cannot be empty or whitespace")
    if args.global_batch_size % args.num_gpus != 0:
        raise ValueError(
            "--global-batch-size must be divisible by --num-gpus, "
            f"got global_batch_size={args.global_batch_size} num_gpus={args.num_gpus}"
        )
    if args.force_prepare and args.skip_prepare:
        raise ValueError("--force-prepare and --skip-prepare cannot be used together")
    if args.force_stats and args.skip_stats:
        raise ValueError("--force-stats and --skip-stats cannot be used together")
    if args.force_stats and args.dry_run:
        raise ValueError("--force-stats cannot be combined with --dry-run")
    if args.prepared_dataset:
        if args.source_dataset or args.source_root:
            raise ValueError(
                "--prepared-dataset cannot be combined with --source-dataset or --source-root"
            )
        if not args.skip_prepare:
            raise ValueError("--prepared-dataset requires --skip-prepare")
    if args.skip_prepare and args.max_episodes is not None:
        raise ValueError("--max-episodes has no effect with --skip-prepare")
    if args.skip_prepare and args.instruction is not None:
        raise ValueError("--instruction has no effect with --skip-prepare")
    if args.dry_run and not args.skip_prepare:
        raise ValueError(
            "--dry-run does not simulate conversion; add --skip-prepare so it cannot write data"
        )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if not args.prepare_only and not args.dry_run:
        run_dir = _validate_new_run_dir(
            args.output_dir,
            args.experiment_name,
            allow_existing=args.allow_existing_run_dir,
        )
        print(f"[preflight] new training run directory: {run_dir}")
    video_key_map, modality_config_path = _camera_layout_settings(
        args.camera_layout,
        args.top_camera_key,
        args.left_camera_key,
        args.wrist_camera_key,
    )
    modality_config_path = modality_config_path.resolve()
    print(f"[config] camera_layout: {args.camera_layout}")
    for groot_key, dataset_key in video_key_map.items():
        print(f"[config] dataset {dataset_key} -> GR00T video.{groot_key}")
    prepared_paths: list[Path]
    if args.prepared_dataset:
        prepared_paths = [path.expanduser().resolve() for path in args.prepared_dataset]
        if len(set(prepared_paths)) != len(prepared_paths):
            raise ValueError(f"Duplicate --prepared-dataset paths: {prepared_paths}")
        if len(prepared_paths) > 1 and not args.allow_multiple_datasets:
            selected = "\n".join(f"  - {path}" for path in prepared_paths)
            raise ValueError(
                "Multiple prepared datasets were selected. Verify that all of them use the "
                "same action/state coordinate system, then rerun with "
                f"--allow-multiple-datasets. Selected datasets:\n{selected}"
            )
    else:
        source_specs = _resolve_source_datasets(args)
        for source_path, _ in source_specs:
            _validate_camera_layout_matches_source(
                _load_info(source_path),
                video_key_map=video_key_map,
                camera_layout=args.camera_layout,
                context=source_path,
            )
        if len(source_specs) > 1 and not args.allow_multiple_datasets:
            selected = "\n".join(f"  - {source_path}" for source_path, _ in source_specs)
            raise ValueError(
                "Multiple source datasets were selected. The wrapper cannot infer whether their "
                "action/state values are degrees, LeRobot motor units, or another coordinate system. "
                "Verify that all selected datasets use the same coordinate system, then rerun with "
                f"--allow-multiple-datasets. Selected datasets:\n{selected}"
            )

        prepared_paths = []
        prepared_root = args.prepared_root.resolve()
        for source_path, prepared_relative_path in source_specs:
            prepared_path = prepared_root / prepared_relative_path
            _validate_prepared_output_path(prepared_path, prepared_root)
            if not args.skip_prepare:
                prepare_dataset(
                    source_path,
                    prepared_path,
                    groot_root=args.groot_root.resolve(),
                    video_key_map=video_key_map,
                    instruction_override=args.instruction,
                    force=args.force_prepare,
                    max_episodes=args.max_episodes,
                )
            prepared_paths.append(prepared_path)

    for prepared_path in prepared_paths:
        _validate_prepared_dataset(prepared_path, video_key_map)

    if not args.skip_stats:
        for prepared_path in prepared_paths:
            generate_stats(
                prepared_path,
                args.groot_root.resolve(),
                modality_config_path,
                args.dry_run,
                args.force_stats,
            )

    # A stats-only run is also a preflight: freshly generated relative stats must
    # pass before the dataset is considered ready. A conversion-only run uses
    # --prepare-only --skip-stats and therefore cannot audit stats yet.
    if _should_audit_relative_stats(args):
        for prepared_path in prepared_paths:
            audit_relative_stats(
                prepared_path,
                max_span_ratio=args.max_relative_stats_span_ratio,
                allow_outliers=args.allow_relative_stats_outliers,
            )

    if args.prepare_only:
        print("[done] prepared datasets:")
        for path in prepared_paths:
            print(f"  {path}")
        return

    launch_finetune(args, prepared_paths, modality_config_path)


if __name__ == "__main__":
    main()
