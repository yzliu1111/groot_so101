from __future__ import annotations

from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "train_so101_synthetic_groot.py"
SPEC = importlib.util.spec_from_file_location("train_so101_synthetic_groot", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)


class SchemaValidationTests(unittest.TestCase):
    def _info(self) -> dict:
        low_dim = {
            "dtype": "float32",
            "shape": [6],
            "names": list(train.SO101_JOINT_NAMES),
        }
        return {
            "fps": 30,
            "features": {
                "observation.state": dict(low_dim),
                "action": dict(low_dim),
            },
        }

    def test_accepts_expected_so101_schema(self) -> None:
        train._validate_so101_schema(self._info(), context=Path("dataset"))

    def test_rejects_wrong_joint_order(self) -> None:
        info = self._info()
        info["features"]["action"]["names"] = list(reversed(train.SO101_JOINT_NAMES))
        with self.assertRaisesRegex(ValueError, "joint order"):
            train._validate_so101_schema(info, context=Path("dataset"))

    def test_rejects_wrong_action_shape(self) -> None:
        info = self._info()
        info["features"]["action"]["shape"] = [5]
        with self.assertRaisesRegex(ValueError, r"shape \[6\]"):
            train._validate_so101_schema(info, context=Path("dataset"))


class FilesystemSafetyTests(unittest.TestCase):
    def test_refuses_source_or_any_ancestor_relationship(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "prepared-child").mkdir()
            with self.assertRaises(ValueError):
                train._safe_remove_prepared_dir(source, source)
            with self.assertRaises(ValueError):
                train._safe_remove_prepared_dir(source / "prepared-child", source)
            with self.assertRaises(ValueError):
                train._safe_remove_prepared_dir(root, source)

    def test_removes_only_safe_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            prepared = root / "prepared"
            source.mkdir()
            prepared.mkdir()
            train._safe_remove_prepared_dir(prepared, source)
            self.assertTrue(source.is_dir())
            self.assertFalse(prepared.exists())

    def test_prepared_output_must_be_a_non_symlinked_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared_root = root / "prepared-root"
            prepared_root.mkdir()
            train._validate_prepared_output_path(prepared_root / "dataset", prepared_root)
            with self.assertRaisesRegex(ValueError, "itself"):
                train._validate_prepared_output_path(prepared_root, prepared_root)
            with self.assertRaisesRegex(ValueError, "outside"):
                train._validate_prepared_output_path(root / "outside", prepared_root)

    def test_prepared_output_rejects_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared_root = root / "prepared-root"
            outside = root / "outside"
            prepared_root.mkdir()
            outside.mkdir()
            (prepared_root / "linked").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                train._validate_prepared_output_path(
                    prepared_root / "linked" / "dataset",
                    prepared_root,
                )


class PreparedDatasetValidationTests(unittest.TestCase):
    def _create_dataset(self, root: Path) -> dict[str, str]:
        video_key_map = {"wrist": "observation.images.camera1"}
        low_dim = {
            "dtype": "float32",
            "shape": [6],
            "names": list(train.SO101_JOINT_NAMES),
        }
        info = {
            "codebase_version": "v2.1",
            "fps": 30,
            "chunks_size": 1000,
            "total_episodes": 1,
            "total_frames": 2,
            "total_tasks": 1,
            "total_videos": 1,
            "features": {
                "observation.state": dict(low_dim),
                "action": dict(low_dim),
                "observation.images.camera1": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channels"],
                },
            },
        }
        train._write_json(root / "meta" / "info.json", info)
        train._write_json(
            root / "meta" / "modality.json",
            train._expected_modality_payload(video_key_map),
        )
        train._write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "pick"}])
        train._write_jsonl(
            root / "meta" / "episodes.jsonl",
            [{"episode_index": 0, "tasks": ["pick"], "length": 2}],
        )
        data_path = root / "data" / "chunk-000" / "episode_000000.parquet"
        data_path.parent.mkdir(parents=True)
        pq.write_table(pa.table({"frame_index": [0, 1]}), data_path)
        video_path = (
            root
            / "videos"
            / "chunk-000"
            / "observation.images.camera1"
            / "episode_000000.mp4"
        )
        video_path.parent.mkdir(parents=True)
        video_path.write_bytes(b"non-empty test placeholder")
        return video_key_map

    def test_accepts_complete_prepared_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_key_map = self._create_dataset(root)
            train._validate_prepared_dataset(root, video_key_map)

    def test_rejects_missing_episode_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_key_map = self._create_dataset(root)
            next((root / "videos").rglob("*.mp4")).unlink()
            with self.assertRaisesRegex(FileNotFoundError, "video is missing"):
                train._validate_prepared_dataset(root, video_key_map)

    def test_rejects_inexact_modality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_key_map = self._create_dataset(root)
            modality_path = root / "meta" / "modality.json"
            modality = json.loads(modality_path.read_text())
            modality["state"]["single_arm"]["end"] = 4
            modality_path.write_text(json.dumps(modality))
            with self.assertRaisesRegex(ValueError, "modality does not exactly match"):
                train._validate_prepared_dataset(root, video_key_map)


class FallbackConverterTests(unittest.TestCase):
    def test_data_split_uses_parquet_index_for_file_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            source_file = source / "data" / "chunk-000" / "file-000.parquet"
            source_file.parent.mkdir(parents=True)
            pq.write_table(
                pa.table({"index": [100, 101, 102, 103], "value": [0, 1, 2, 3]}),
                source_file,
            )
            train._convert_data_files(
                source,
                output,
                [
                    {
                        "data/chunk_index": 0,
                        "data/file_index": 0,
                        "dataset_from_index": 101,
                        "dataset_to_index": 103,
                        "episode_index": 5,
                    }
                ],
            )
            converted = pq.read_table(
                output / "data" / "chunk-000" / "episode_000005.parquet"
            )
            self.assertEqual(converted.column("index").to_pylist(), [101, 102])

    def test_video_split_rejects_missing_source_before_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(FileNotFoundError, "source video"):
                train._split_video(root / "missing.mp4", root / "out.mp4", 0.0, 1.0)


class TaskMetadataTests(unittest.TestCase):
    def test_loads_sim_index_column_without_instruction_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_path = root / "meta" / "tasks.parquet"
            tasks_path.parent.mkdir(parents=True)
            pq.write_table(
                pa.table({"task_index": [0], "__index_level_0__": ["pick up block"]}),
                tasks_path,
            )
            self.assertEqual(
                train._load_tasks(root, None),
                [{"task_index": 0, "task": "pick up block"}],
            )

    def test_rejects_duplicate_task_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_path = root / "meta" / "tasks.parquet"
            tasks_path.parent.mkdir(parents=True)
            pq.write_table(
                pa.table({"task_index": [0, 0], "task": ["pick", "place"]}),
                tasks_path,
            )
            with self.assertRaisesRegex(ValueError, "Duplicate task_index"):
                train._load_tasks(root, None)


class ArgumentValidationTests(unittest.TestCase):
    def _args(self, **overrides: object) -> Namespace:
        values = {
            "num_gpus": 1,
            "global_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "max_steps": 10,
            "save_steps": 5,
            "save_total_limit": 2,
            "dataloader_num_workers": 0,
            "learning_rate": 1e-4,
            "max_episodes": None,
            "instruction": None,
            "force_prepare": False,
            "skip_prepare": False,
            "dry_run": False,
        }
        values.update(overrides)
        return Namespace(**values)

    def test_rejects_dry_run_that_would_prepare(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not simulate conversion"):
            train._validate_args(self._args(dry_run=True))

    def test_rejects_force_and_skip_prepare(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            train._validate_args(self._args(force_prepare=True, skip_prepare=True))

    def test_rejects_non_divisible_global_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            train._validate_args(self._args(num_gpus=2, global_batch_size=3))

    def test_accepts_training_arguments(self) -> None:
        train._validate_args(self._args())


if __name__ == "__main__":
    unittest.main()
