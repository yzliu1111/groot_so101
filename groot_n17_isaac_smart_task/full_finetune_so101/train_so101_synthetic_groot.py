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
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"

V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


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
    for dirpath, _, filenames in os.walk(root, followlinks=True):
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
    tasks: list[dict[str, Any]] = []

    for row in rows:
        task_index = int(row["task_index"])
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
        tasks.append({"task_index": task_index, "task": task})

    return sorted(tasks, key=lambda item: item["task_index"])


def _safe_remove_prepared_dir(path: Path, source_path: Path) -> None:
    path = path.resolve()
    source_path = source_path.resolve()
    if path == source_path or source_path in path.parents:
        raise ValueError(f"Refusing to remove source dataset path: {path}")
    shutil.rmtree(path)


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
    wrist_camera_key: str,
) -> tuple[dict[str, str], Path]:
    if camera_layout == "dual":
        if not top_camera_key or not wrist_camera_key:
            raise ValueError("dual camera layout requires both top/front and wrist camera keys")
        return {"top": top_camera_key, "wrist": wrist_camera_key}, DUAL_MODALITY_CONFIG_PATH
    if camera_layout == "wrist-only":
        if not wrist_camera_key:
            raise ValueError("wrist-only camera layout requires --wrist-camera-key")
        return {"wrist": wrist_camera_key}, WRIST_ONLY_MODALITY_CONFIG_PATH
    raise ValueError(f"Unsupported camera layout: {camera_layout}")


def _write_modality_json(output_path: Path, video_key_map: dict[str, str]) -> None:
    _write_json(
        output_path / "meta" / "modality.json",
        {
            "state": {
                "single_arm": {"start": 0, "end": 5},
                "gripper": {"start": 5, "end": 6},
            },
            "action": {
                "single_arm": {"start": 0, "end": 5},
                "gripper": {"start": 5, "end": 6},
            },
            "video": {
                config_key: {"original_key": original_key}
                for config_key, original_key in video_key_map.items()
            },
            "annotation": {
                "human.task_description": {"original_key": "task_index"},
            },
        },
    )


def _convert_data_files(source_path: Path, output_path: Path, records: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (int(record["data/chunk_index"]), int(record["data/file_index"]))
        grouped.setdefault(key, []).append(record)

    for (chunk_index, file_index), group_records in grouped.items():
        source_file = source_path / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        table = pq.read_table(source_file)
        group_records = sorted(group_records, key=lambda item: int(item["dataset_from_index"]))
        file_offset = int(group_records[0]["dataset_from_index"])

        for record in group_records:
            episode_index = int(record["episode_index"])
            start = int(record["dataset_from_index"]) - file_offset
            stop = int(record["dataset_to_index"]) - file_offset
            episode_table = table.slice(start, stop - start)
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
    output_file.parent.mkdir(parents=True, exist_ok=True)
    duration_s = max(end_s - start_s, 1e-6)
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
    subprocess.run(cmd, check=True)


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
    printable = " ".join(cmd)
    print(f"[run] cd {cwd} && {printable}")
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def generate_stats(
    dataset_path: Path,
    groot_root: Path,
    modality_config_path: Path,
    dry_run: bool,
) -> None:
    _run(
        [
            sys.executable,
            "gr00t/data/stats.py",
            "--dataset-path",
            str(dataset_path),
            "--embodiment-tag",
            "NEW_EMBODIMENT",
            "--modality-config-path",
            str(modality_config_path),
        ],
        cwd=groot_root,
        dry_run=dry_run,
    )


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
    ]

    if args.experiment_name:
        cmd.extend(["--experiment-name", args.experiment_name])
    if args.use_wandb:
        cmd.append("--use-wandb")
    if args.color_jitter:
        cmd.extend(
            [
                "--color-jitter-params",
                "brightness",
                "0.3",
                "contrast",
                "0.4",
                "saturation",
                "0.5",
                "hue",
                "0.08",
            ]
        )

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
        "--top-camera-key",
        "--front-camera-key",
        dest="top_camera_key",
        default="observation.images.camera1",
        help="Top/global camera feature key. --front-camera-key is kept as a deprecated alias.",
    )
    parser.add_argument("--wrist-camera-key", default="observation.images.camera3")
    parser.add_argument(
        "--camera-layout",
        choices=("dual", "wrist-only"),
        default="dual",
        help=(
            "dual maps top/front + wrist cameras; wrist-only trains with only "
            "--wrist-camera-key and writes prepared datasets with a _wrist_only suffix."
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
    parser.add_argument("--no-color-jitter", dest="color_jitter", action="store_false")
    parser.set_defaults(color_jitter=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_key_map, modality_config_path = _camera_layout_settings(
        args.camera_layout,
        args.top_camera_key,
        args.wrist_camera_key,
    )
    modality_config_path = modality_config_path.resolve()
    source_specs = _resolve_source_datasets(args)

    prepared_paths: list[Path] = []
    for source_path, prepared_relative_path in source_specs:
        prepared_path = args.prepared_root.resolve() / prepared_relative_path
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

    if not args.skip_stats:
        for prepared_path in prepared_paths:
            generate_stats(
                prepared_path,
                args.groot_root.resolve(),
                modality_config_path,
                args.dry_run,
            )

    if args.prepare_only:
        print("[done] prepared datasets:")
        for path in prepared_paths:
            print(f"  {path}")
        return

    launch_finetune(args, prepared_paths, modality_config_path)


if __name__ == "__main__":
    main()
