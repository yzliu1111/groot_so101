from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = SCRIPT_DIR / "aws_training_batch.py"
MANIFEST_PATH = SCRIPT_DIR / "aws_tuning_8_manifest.json"
SPEC = importlib.util.spec_from_file_location("aws_training_batch", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
batch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = batch
SPEC.loader.exec_module(batch)


class AwsTuningManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload, cls.defaults, cls.specs = batch.load_manifest(MANIFEST_PATH)
        cls.by_id = {spec.dataset_id: spec for spec in cls.specs}

    def test_manifest_has_exactly_the_eight_expected_datasets(self) -> None:
        self.assertEqual(self.payload["expected_dataset_count"], 8)
        self.assertEqual(len(self.specs), 8)
        self.assertEqual(
            [spec.dataset_id for spec in self.specs],
            [
                "real001",
                "sim002",
                "real003",
                "sim004",
                "sim005",
                "real006",
                "real007",
                "sim008",
            ],
        )

    def test_dedicated_aws_directory_reuses_shared_full_finetune_code(self) -> None:
        self.assertEqual(batch.SCRIPT_DIR.name, "aws_training")
        self.assertEqual(
            batch.FULL_FINETUNE_DIR,
            batch.EXPERIMENT_ROOT / "full_finetune_so101",
        )
        self.assertEqual(
            batch.TRAIN_WRAPPER,
            batch.FULL_FINETUNE_DIR / "train_so101_synthetic_groot.py",
        )
        self.assertTrue(batch.TRAIN_WRAPPER.is_file())
        self.assertTrue(all(path.is_file() for path in batch.MODALITY_CONFIGS.values()))

    def test_camera_layout_counts_and_roles_are_exact(self) -> None:
        wrist_only = [spec for spec in self.specs if spec.camera_layout == "wrist-only"]
        triple = [spec for spec in self.specs if spec.camera_layout == "triple"]
        self.assertEqual({spec.dataset_id for spec in wrist_only}, {"real001", "sim002"})
        self.assertEqual(len(wrist_only), 2)
        self.assertEqual(len(triple), 6)

        for spec in self.specs:
            with self.subTest(dataset=spec.dataset_id):
                self.assertEqual(
                    set(spec.camera_roles),
                    batch.LAYOUT_ROLES[spec.camera_layout],
                )
                self.assertEqual(
                    len(set(spec.camera_roles.values())),
                    len(spec.camera_roles),
                )
                self.assertTrue(
                    all(
                        feature_key.startswith("observation.images.")
                        for feature_key in spec.camera_roles.values()
                    )
                )

    def test_only_real003_real006_real007_use_clean_training_sources(self) -> None:
        cleaned_ids = {
            spec.dataset_id
            for spec in self.specs
            if spec.training_source != spec.raw_source
        }
        self.assertEqual(cleaned_ids, {"real003", "real006", "real007"})

        expected_removed = {
            "real003": (0,),
            "real006": (0, 100),
            "real007": (0, 53, 100, 121, 200, 211),
        }
        for dataset_id, removed in expected_removed.items():
            with self.subTest(dataset=dataset_id):
                spec = self.by_id[dataset_id]
                self.assertIn(
                    "outputs/groot_so101_cleaned_source_datasets",
                    spec.training_source.as_posix(),
                )
                self.assertEqual(spec.removed_raw_episodes, removed)

    def test_sim004_is_the_only_instruction_override_and_has_a_reason(self) -> None:
        overrides = [spec for spec in self.specs if spec.instruction_override]
        self.assertEqual([spec.dataset_id for spec in overrides], ["sim004"])
        self.assertEqual(overrides[0].instruction_override, "pick up block")
        self.assertTrue(overrides[0].instruction_override_reason)
        self.assertIn("raw task", overrides[0].instruction_override_reason)

    def test_manifest_pins_exact_dataset_identity_counts(self) -> None:
        self.assertEqual(
            self.by_id["real001"].expected_raw,
            {
                "episodes": 50,
                "frames": 22492,
                "task_episode_counts": {"pick up block": 50},
            },
        )
        self.assertEqual(self.by_id["real003"].expected_training["episodes"], 49)
        self.assertEqual(self.by_id["real006"].expected_training["frames"], 177764)
        self.assertEqual(
            set(self.by_id["real007"].expected_training["task_episode_counts"].values()),
            {98},
        )
        self.assertEqual(self.by_id["sim008"].expected_raw["episodes"], 287)

    def test_dataset_identity_check_fails_closed(self) -> None:
        expected = {
            "episodes": 50,
            "frames": 1000,
            "task_episode_counts": {"pick up block": 50},
        }
        report = {
            "total_episodes": 50,
            "total_frames": 999,
            "task_text_episode_counts": {"pick up block": 50},
        }
        with self.assertRaisesRegex(ValueError, "dataset identity mismatch"):
            batch._assert_expected_summary(report, expected, context="real001/raw")


class AwsBatchCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload, cls.defaults, cls.specs = batch.load_manifest(MANIFEST_PATH)
        cls.selected = batch.select_specs(cls.specs, ["all"])
        cls.command_kwargs = {
            "data_python": Path("/opt/data-python"),
            "groot_python": Path("/opt/groot-python"),
            "groot_root": Path("/opt/Isaac-GR00T"),
            "output_root": Path("/opt/training-output"),
            "run_tag": "aws-test-run",
            "force_prepare": False,
        }

    def _commands(self, action: str) -> list[tuple[batch.DatasetSpec, list[str]]]:
        return [
            (
                spec,
                batch.build_command(
                    action,
                    spec,
                    self.defaults,
                    **self.command_kwargs,
                ),
            )
            for spec in self.selected
        ]

    def test_all_builds_eight_independent_training_commands(self) -> None:
        commands = self._commands("train")
        self.assertEqual(len(commands), 8)
        self.assertEqual(len({tuple(command) for _, command in commands}), 8)

        selected_paths: set[str] = set()
        for spec, command in commands:
            with self.subTest(dataset=spec.dataset_id):
                self.assertEqual(command.count("--prepared-dataset"), 1)
                prepared_index = command.index("--prepared-dataset")
                selected_path = command[prepared_index + 1]
                self.assertEqual(selected_path, str(spec.prepared_dataset))
                selected_paths.add(selected_path)

                self.assertNotIn("--source-root", command)
                self.assertNotIn("--source-dataset", command)
                self.assertNotIn("--allow-multiple-datasets", command)
                self.assertEqual(command.count("--output-dir"), 1)
                self.assertEqual(command.count("--experiment-name"), 1)
                experiment_index = command.index("--experiment-name")
                self.assertEqual(command[experiment_index + 1], spec.dataset_id)

        self.assertEqual(len(selected_paths), 8)

    def test_prepare_uses_one_exact_source_per_command_without_recursive_mix(self) -> None:
        commands = self._commands("prepare")
        self.assertEqual(len(commands), 8)

        selected_sources: set[str] = set()
        for spec, command in commands:
            with self.subTest(dataset=spec.dataset_id):
                self.assertEqual(command.count("--source-dataset"), 1)
                source_index = command.index("--source-dataset")
                selected_source = command[source_index + 1]
                self.assertEqual(selected_source, str(spec.training_source))
                selected_sources.add(selected_source)

                self.assertNotIn("--source-root", command)
                self.assertNotIn("--allow-multiple-datasets", command)
                self.assertNotIn("--prepared-dataset", command)
                self.assertIn("--prepare-only", command)

        self.assertEqual(len(selected_sources), 8)

    def test_prepare_never_overwrites_multitask_instruction_text(self) -> None:
        commands = self._commands("prepare")
        commands_with_instruction = [
            (spec, command)
            for spec, command in commands
            if "--instruction" in command
        ]
        self.assertEqual(
            [spec.dataset_id for spec, _ in commands_with_instruction],
            ["sim004"],
        )

        sim004, command = commands_with_instruction[0]
        instruction_index = command.index("--instruction")
        self.assertEqual(command[instruction_index + 1], sim004.instruction_override)

        for dataset_id in ("real006", "real007", "sim008"):
            command = next(
                command
                for spec, command in commands
                if spec.dataset_id == dataset_id
            )
            with self.subTest(dataset=dataset_id):
                self.assertNotIn("--instruction", command)

    def test_stats_forces_recomputation(self) -> None:
        for spec, command in self._commands("stats"):
            with self.subTest(dataset=spec.dataset_id):
                self.assertIn("--force-stats", command)
                self.assertIn("--prepare-only", command)

    def test_venv_python_path_is_not_dereferenced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_python = root / "base-python"
            base_python.touch()
            venv_python = root / "venv-python"
            venv_python.symlink_to(base_python)
            self.assertEqual(batch._absolute_executable(venv_python), venv_python.absolute())
            self.assertNotEqual(batch._absolute_executable(venv_python), venv_python.resolve())

    def test_prepared_video_sampling_uses_first_middle_last(self) -> None:
        self.assertEqual(batch._sample_episode_indices(1), [0])
        self.assertEqual(batch._sample_episode_indices(2), [0, 1])
        self.assertEqual(batch._sample_episode_indices(50), [0, 25, 49])

    def test_prepared_identity_is_checked_against_manifest_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "codebase_version": "v2.1",
                        "total_episodes": 1,
                        "total_frames": 2,
                        "total_tasks": 1,
                        "features": {
                            "observation.images.camera1": {"dtype": "video"},
                        },
                    }
                )
            )
            (root / "meta" / "tasks.jsonl").write_text(
                json.dumps({"task_index": 0, "task": "pick blue"}) + "\n"
            )
            (root / "meta" / "episodes.jsonl").write_text(
                json.dumps(
                    {"episode_index": 0, "length": 2, "tasks": ["pick blue"]}
                )
                + "\n"
            )
            spec = replace(
                self.specs[0],
                prepared_dataset=root,
                expected_training={
                    "episodes": 1,
                    "frames": 2,
                    "task_episode_counts": {"pick red": 1},
                },
            )
            with self.assertRaisesRegex(ValueError, "dataset identity mismatch"):
                batch.verify_prepared_identity(spec)

    def test_prepared_fingerprint_changes_with_file_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            (root / "data").mkdir()
            (root / "meta" / "stats.json").write_text("{}\n")
            (root / "meta" / "relative_stats.json").write_text("{}\n")
            data_path = root / "data" / "episode.parquet"
            data_path.write_bytes(b"first")
            first = batch.fingerprint_prepared_dataset(root)
            data_path.write_bytes(b"second")
            second = batch.fingerprint_prepared_dataset(root)
            self.assertNotEqual(first["sha256"], second["sha256"])

    def test_run_lineage_is_idempotent_but_rejects_changed_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "_lineage" / "real001.json"
            kwargs = {
                "manifest_path": MANIFEST_PATH,
                "manifest_payload": self.payload,
                "spec": self.specs[0],
                "command": ["python", "train.py", "--dataset", "real001"],
                "action": "train",
                "run_tag": "aws-test-run",
                "actual_groot_commit": self.payload["groot_commit"],
                "prepared_fingerprint": {
                    "sha256": "a" * 64,
                    "content_sha256_excluding_stats": "d" * 64,
                    "file_count": 1,
                    "total_bytes": 1,
                    "stats_sha256": {
                        "meta/stats.json": "b" * 64,
                        "meta/relative_stats.json": "c" * 64,
                    },
                },
            }
            batch._write_run_lineage(path, **kwargs)
            batch._write_run_lineage(path, **kwargs)
            changed = {**kwargs, "command": ["python", "train.py", "--dataset", "other"]}
            with self.assertRaisesRegex(FileExistsError, "different inputs"):
                batch._write_run_lineage(path, **changed)

    def test_smoke_requires_matching_stats_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aws-test-run-stats" / "_lineage" / "real001.json"
            fingerprint = {
                "sha256": "a" * 64,
                "content_sha256_excluding_stats": self.specs[0].expected_prepared_content_sha256,
                "file_count": 1,
                "total_bytes": 1,
                "stats_sha256": {
                    "meta/stats.json": "b" * 64,
                    "meta/relative_stats.json": "c" * 64,
                },
            }
            batch._write_run_lineage(
                path,
                manifest_path=MANIFEST_PATH,
                manifest_payload=self.payload,
                spec=self.specs[0],
                command=["stats"],
                action="stats",
                run_tag="aws-test-run",
                actual_groot_commit=self.payload["groot_commit"],
                prepared_fingerprint=fingerprint,
            )
            batch._require_matching_stats_lineage(
                path,
                manifest_path=MANIFEST_PATH,
                manifest_payload=self.payload,
                spec=self.specs[0],
                actual_groot_commit=self.payload["groot_commit"],
                prepared_fingerprint=fingerprint,
            )
            changed = {**fingerprint, "sha256": "e" * 64}
            with self.assertRaisesRegex(ValueError, "does not match"):
                batch._require_matching_stats_lineage(
                    path,
                    manifest_path=MANIFEST_PATH,
                    manifest_payload=self.payload,
                    spec=self.specs[0],
                    actual_groot_commit=self.payload["groot_commit"],
                    prepared_fingerprint=changed,
                )


if __name__ == "__main__":
    unittest.main()
