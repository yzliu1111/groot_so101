from __future__ import annotations

from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

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
        pq.write_table(
            pa.table(
                {
                    "episode_index": [0, 0],
                    "frame_index": [0, 1],
                    "task_index": [0, 0],
                }
            ),
            data_path,
        )
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

    def test_rejects_task_text_index_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_key_map = self._create_dataset(root)
            train._write_jsonl(
                root / "meta" / "episodes.jsonl",
                [{"episode_index": 0, "tasks": ["different task"], "length": 2}],
            )
            with self.assertRaisesRegex(ValueError, "unknown tasks"):
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

    def test_rejects_instruction_override_for_multitask_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_path = root / "meta" / "tasks.parquet"
            tasks_path.parent.mkdir(parents=True)
            pq.write_table(
                pa.table(
                    {
                        "task_index": [0, 1],
                        "task": ["pick blue", "pick red"],
                    }
                ),
                tasks_path,
            )
            with self.assertRaisesRegex(ValueError, "collapse 2 distinct task rows"):
                train._load_tasks(root, "pick blue")


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
            "state_dropout_prob": 0.0,
            "max_relative_stats_span_ratio": 5.0,
            "max_episodes": None,
            "instruction": None,
            "force_prepare": False,
            "force_stats": False,
            "skip_stats": False,
            "skip_prepare": False,
            "dry_run": False,
            "prepared_dataset": None,
            "source_dataset": None,
            "source_root": None,
        }
        values.update(overrides)
        return Namespace(**values)

    def test_rejects_dry_run_that_would_prepare(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not simulate conversion"):
            train._validate_args(self._args(dry_run=True))

    def test_rejects_force_and_skip_prepare(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            train._validate_args(self._args(force_prepare=True, skip_prepare=True))

    def test_rejects_force_and_skip_stats(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            train._validate_args(self._args(force_stats=True, skip_stats=True))

    def test_rejects_non_divisible_global_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            train._validate_args(self._args(num_gpus=2, global_batch_size=3))

    def test_accepts_training_arguments(self) -> None:
        train._validate_args(self._args())

    def test_rejects_invalid_state_dropout(self) -> None:
        with self.assertRaisesRegex(ValueError, "state-dropout-prob"):
            train._validate_args(self._args(state_dropout_prob=1.1))

    def test_exact_prepared_dataset_requires_skip_prepare(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --skip-prepare"):
            train._validate_args(self._args(prepared_dataset=[Path("prepared")]))

    def test_exact_prepared_dataset_rejects_source_selector(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            train._validate_args(
                self._args(
                    prepared_dataset=[Path("prepared")],
                    source_dataset=[Path("raw")],
                    skip_prepare=True,
                )
            )


class TrainingSafetyPreflightTests(unittest.TestCase):
    def _info_with_video_keys(self, *video_keys: str) -> dict[str, object]:
        return {
            "features": {
                key: {"dtype": "video"}
                for key in video_keys
            }
        }

    def test_single_camera_requires_wrist_only(self) -> None:
        train._validate_camera_layout_matches_source(
            self._info_with_video_keys("observation.images.camera1"),
            video_key_map={"wrist": "observation.images.camera1"},
            camera_layout="wrist-only",
            context=Path("single"),
        )
        with self.assertRaisesRegex(ValueError, "requires --camera-layout wrist-only"):
            train._validate_camera_layout_matches_source(
                self._info_with_video_keys("observation.images.camera1"),
                video_key_map={
                    "top": "observation.images.camera2",
                    "wrist": "observation.images.camera1",
                },
                camera_layout="dual",
                context=Path("single"),
            )

    def test_three_cameras_require_triple_and_all_roles(self) -> None:
        info = self._info_with_video_keys(
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.camera3",
        )
        train._validate_camera_layout_matches_source(
            info,
            video_key_map={
                "top": "observation.images.camera3",
                "left": "observation.images.camera1",
                "wrist": "observation.images.camera2",
            },
            camera_layout="triple",
            context=Path("triple"),
        )
        with self.assertRaisesRegex(ValueError, "requires --camera-layout triple"):
            train._validate_camera_layout_matches_source(
                info,
                video_key_map={
                    "top": "observation.images.camera3",
                    "wrist": "observation.images.camera2",
                },
                camera_layout="dual",
                context=Path("triple"),
            )

    def test_camera_roles_cannot_duplicate_or_omit_source_video(self) -> None:
        info = self._info_with_video_keys(
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.camera3",
        )
        with self.assertRaisesRegex(ValueError, "use every source video"):
            train._validate_camera_layout_matches_source(
                info,
                video_key_map={
                    "top": "observation.images.camera3",
                    "left": "observation.images.camera1",
                    "wrist": "observation.images.unexpected",
                },
                camera_layout="triple",
                context=Path("triple"),
            )

    def test_no_color_jitter_is_an_explicit_zero_override(self) -> None:
        args = train._color_jitter_cli_args(False)
        self.assertEqual(args[0], "--color-jitter-params")
        self.assertEqual(args[2::2], ["0.0", "0.0", "0.0", "0.0"])

    def test_color_jitter_keeps_the_intended_nonzero_values(self) -> None:
        args = train._color_jitter_cli_args(True)
        self.assertEqual(args[2::2], ["0.3", "0.4", "0.5", "0.08"])

    def _write_relative_stats(self, root: Path, maximum: float) -> Path:
        dataset = root / "dataset"
        train._write_json(
            dataset / "meta" / "relative_stats.json",
            {
                "single_arm": {
                    "min": [[-1.0, -1.0]],
                    "max": [[1.0, maximum]],
                    "q01": [[-0.5, -0.5]],
                    "q99": [[0.5, 0.5]],
                    "mean": [[0.0, 0.0]],
                    "std": [[1.0, 1.0]],
                }
            },
        )
        return dataset

    def test_force_stats_restores_old_cache_when_generation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            meta = dataset / "meta"
            meta.mkdir(parents=True)
            stats_path = meta / "stats.json"
            relative_path = meta / "relative_stats.json"
            stats_path.write_text('{"old": "stats"}\n')
            relative_path.write_text('{"old": "relative"}\n')
            with mock.patch.object(train, "_run", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    train.generate_stats(
                        dataset,
                        Path(tmp),
                        Path(tmp) / "modality.py",
                        dry_run=False,
                        force=True,
                    )
            self.assertEqual(stats_path.read_text(), '{"old": "stats"}\n')
            self.assertEqual(relative_path.read_text(), '{"old": "relative"}\n')

    def test_relative_stats_preflight_identifies_worst_joint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._write_relative_stats(Path(tmp), maximum=12.0)
            worst = train._relative_stats_worst_span_ratio(dataset)
            self.assertEqual(worst["joint_name"], "shoulder_lift.pos")
            self.assertEqual(worst["ratio"], 13.0)

    def test_relative_stats_preflight_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._write_relative_stats(Path(tmp), maximum=12.0)
            with self.assertRaisesRegex(ValueError, "13.000 exceeds 5.000"):
                train.audit_relative_stats(
                    dataset,
                    max_span_ratio=5.0,
                    allow_outliers=False,
                )

    def test_rejects_reused_nonempty_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            run_dir = output_dir / "real007"
            run_dir.mkdir(parents=True)
            (run_dir / "experiment_cfg").mkdir()
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                train._validate_new_run_dir(
                    output_dir,
                    "real007",
                    allow_existing=False,
                )

    def test_accepts_unique_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            expected = output_dir / "real007_v2"
            self.assertEqual(
                train._validate_new_run_dir(
                    output_dir,
                    "real007_v2",
                    allow_existing=False,
                ),
                expected,
            )

    def test_stats_only_run_still_audits_relative_stats(self) -> None:
        args = Namespace(dry_run=False, prepare_only=True, skip_stats=False)
        self.assertTrue(train._should_audit_relative_stats(args))

    def test_conversion_only_run_defers_relative_stats_audit(self) -> None:
        args = Namespace(dry_run=False, prepare_only=True, skip_stats=True)
        self.assertFalse(train._should_audit_relative_stats(args))


if __name__ == "__main__":
    unittest.main()
