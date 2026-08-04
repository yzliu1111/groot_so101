from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


AWS_DIR = Path(__file__).resolve().parents[1]
PIPELINE = AWS_DIR / "aws_training_pipeline.sh"
PROJECT_ROOT = AWS_DIR.parents[2]
AWS_GUIDE = AWS_DIR.parent / "full_finetune_so101" / "AWS_UBUNTU_FULL_FINETUNE_ZH.md"
RUNTIME_ENV_KEYS = (
    "AWS_MIN_FREE_GIB",
    "AWS_OUTPUTS_TARGET",
    "AWS_STORAGE_ROOT",
    "SMART_PROJECT",
    "GROOT_ROOT",
    "TRAIN_OUTPUT_ROOT",
    "HF_HOME",
    "UV_CACHE_DIR",
    "UV_PYTHON_INSTALL_DIR",
    "XDG_CACHE_HOME",
    "TORCH_HOME",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TMPDIR",
)


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in RUNTIME_ENV_KEYS:
        env.pop(key, None)
    return env


def _run_pipeline(
    pipeline: Path,
    *args: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(pipeline), *args],
        cwd="/",
        env=env or _clean_env(),
        text=True,
        capture_output=True,
        check=False,
    )


def _write_fake_storage_tools(root: Path) -> Path:
    fake_bin = root / "fake-bin"
    fake_bin.mkdir()
    findmnt = fake_bin / "findmnt"
    findmnt.write_text(
        """#!/usr/bin/env bash
set -eu
field=""
target=""
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -o) field="$2"; shift 2 ;;
        -T) target="$2"; shift 2 ;;
        *) shift ;;
    esac
done
if [[ "$field" == "MAJ:MIN" ]]; then
    if [[ "$target" == "/" ]]; then printf '259:0\\n'; else printf '259:1\\n'; fi
elif [[ "$field" == "TARGET" ]]; then
    if [[ "$target" == "/" ]]; then printf '/\\n'; else printf '/mock-data\\n'; fi
else
    exit 2
fi
"""
    )
    findmnt.chmod(0o755)

    df = fake_bin / "df"
    df.write_text(
        """#!/usr/bin/env bash
printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'
printf '/dev/mock 4000000000 1 3999999999 1%% /mock-data\\n'
"""
    )
    df.chmod(0o755)
    return fake_bin


def _make_project_copy(root: Path, project: Path | None = None) -> tuple[Path, Path]:
    project = project or root / "arbitrary-checkout" / "smart_project"
    aws_dir = project / "experiments" / "groot_n17_isaac_smart_task" / "aws_training"
    full_dir = project / "experiments" / "groot_n17_isaac_smart_task" / "full_finetune_so101"
    aws_dir.mkdir(parents=True)
    full_dir.mkdir(parents=True)
    pipeline = aws_dir / PIPELINE.name
    shutil.copy2(PIPELINE, pipeline)
    (aws_dir / "aws_training_batch.py").write_text("# test fixture\n")
    (aws_dir / "aws_tuning_8_manifest.json").write_text("{}\n")
    (full_dir / "train_so101_synthetic_groot.py").write_text("# test fixture\n")
    return project, pipeline


class AwsTrainingPipelinePathTests(unittest.TestCase):
    def test_paths_discovers_project_from_script_not_local_cwd(self) -> None:
        result = _run_pipeline(PIPELINE, "paths")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"SMART_PROJECT={PROJECT_ROOT}", result.stdout)
        self.assertIn("AWS_OUTPUTS_LINK=", result.stdout)
        self.assertIn("AWS_OUTPUTS_TARGET=", result.stdout)
        self.assertIn("AWS_TUNING_PREPARED_TARGET=", result.stdout)
        self.assertNotIn("AWS_RUNTIME_LINK=", result.stdout)

    def test_conflicting_smart_project_fails_before_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _clean_env()
            env["SMART_PROJECT"] = tmp
            result = _run_pipeline(PIPELINE, "paths", env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must resolve to the project containing this script", result.stderr)

    def test_storage_link_rejects_root_filesystem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _clean_env()
            env["AWS_STORAGE_ROOT"] = tmp
            env["AWS_MIN_FREE_GIB"] = "1"
            result = _run_pipeline(PIPELINE, "storage-link", env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("still on the root filesystem", result.stderr)

    def test_storage_link_is_exact_and_idempotent_for_arbitrary_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, pipeline = _make_project_copy(root)
            storage = root / "large-volume"
            storage.mkdir()
            fake_bin = _write_fake_storage_tools(root)
            env = _clean_env()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["AWS_STORAGE_ROOT"] = str(storage)
            env["AWS_MIN_FREE_GIB"] = "1"

            first = _run_pipeline(pipeline, "storage-link", env=env)
            second = _run_pipeline(pipeline, "storage-link", env=env)

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(
                (project / "outputs").resolve(),
                (storage / "smart_project_outputs").resolve(),
            )
            self.assertFalse((project / ".aws_runtime").exists())

    def test_storage_link_supports_checkout_already_on_large_volume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = root / "large-volume"
            storage.mkdir()
            project, pipeline = _make_project_copy(root, storage / "smart_project")
            fake_bin = _write_fake_storage_tools(root)
            env = _clean_env()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["AWS_STORAGE_ROOT"] = str(storage)
            env["AWS_MIN_FREE_GIB"] = "1"

            result = _run_pipeline(pipeline, "storage-link", env=env)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                (project / "outputs").resolve(),
                (storage / "smart_project_outputs").resolve(),
            )

    def test_storage_link_refuses_existing_directory_without_modifying_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, pipeline = _make_project_copy(root)
            marker = project / "outputs" / "keep.txt"
            marker.parent.mkdir()
            marker.write_text("keep\n")
            storage = root / "large-volume"
            storage.mkdir()
            fake_bin = _write_fake_storage_tools(root)
            env = _clean_env()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["AWS_STORAGE_ROOT"] = str(storage)
            env["AWS_MIN_FREE_GIB"] = "1"

            result = _run_pipeline(pipeline, "storage-link", env=env)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already exists and is not a symlink", result.stderr)
            self.assertEqual(marker.read_text(), "keep\n")

    def test_storage_link_refuses_wrong_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, pipeline = _make_project_copy(root)
            storage = root / "large-volume"
            expected = storage / "smart_project_outputs"
            wrong = storage / "wrong-output"
            expected.mkdir(parents=True)
            wrong.mkdir()
            (project / "outputs").symlink_to(wrong)
            fake_bin = _write_fake_storage_tools(root)
            env = _clean_env()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["AWS_STORAGE_ROOT"] = str(storage)
            env["AWS_MIN_FREE_GIB"] = "1"

            result = _run_pipeline(pipeline, "storage-link", env=env)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("points to the wrong target", result.stderr)
            self.assertEqual((project / "outputs").resolve(), wrong.resolve())

    def test_storage_link_rejects_target_escape_before_creating_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, pipeline = _make_project_copy(root)
            storage = root / "large-volume"
            escaped = root / "root-filesystem-escape"
            storage.mkdir()
            escaped.mkdir()
            (storage / "smart_project_outputs").symlink_to(
                escaped, target_is_directory=True
            )
            fake_bin = _write_fake_storage_tools(root)
            env = _clean_env()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["AWS_STORAGE_ROOT"] = str(storage)
            env["AWS_MIN_FREE_GIB"] = "1"

            result = _run_pipeline(pipeline, "storage-link", env=env)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("would resolve outside AWS_STORAGE_ROOT", result.stderr)
            outputs = root / "arbitrary-checkout" / "smart_project" / "outputs"
            self.assertFalse(outputs.exists())

    def test_lightweight_batch_actions_require_outputs_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, pipeline = _make_project_copy(root)
            storage = root / "large-volume"
            storage.mkdir()
            fake_bin = _write_fake_storage_tools(root)
            env = _clean_env()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["AWS_STORAGE_ROOT"] = str(storage)
            env["AWS_MIN_FREE_GIB"] = "1"

            for action in ("audit", "prepare", "verify", "dry-run"):
                with self.subTest(action=action):
                    result = _run_pipeline(pipeline, action, "real001", env=env)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("outputs symlink is missing", result.stderr)

    def test_pipeline_has_no_device_format_mount_or_data_migration_commands(self) -> None:
        command_pattern = re.compile(r"^(?:sudo\s+)?(?:mkfs(?:\.[a-z0-9]+)?|mount|rm|mv)(?:\s|$)")
        for line in PIPELINE.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            self.assertIsNone(command_pattern.match(stripped), stripped)

    def test_guide_has_three_individual_commands_for_every_dataset(self) -> None:
        guide = AWS_GUIDE.read_text()
        for dataset_id in (
            "real001",
            "sim002",
            "real003",
            "sim004",
            "sim005",
            "real006",
            "real007",
            "sim008",
        ):
            run_tag = f"aws-third-20260804-{dataset_id}"
            for action in ("stats", "smoke", "train"):
                with self.subTest(dataset=dataset_id, action=action):
                    self.assertIn(
                        f"./aws_training_pipeline.sh {action} {dataset_id} "
                        f"--run-tag {run_tag}",
                        guide,
                    )

    def test_guide_does_not_bind_project_checkout_to_dlami_path(self) -> None:
        guide = AWS_GUIDE.read_text()
        self.assertNotIn(
            "/opt/dlami/nvme/smart_project/experiments/groot_n17_isaac_smart_task",
            guide,
        )
        self.assertIn("./aws_training_pipeline.sh storage-link", guide)
        self.assertIn(
            "/opt/dlami/nvme/smart_project_outputs/"
            "groot_so101_synthetic_datasets/aws_third_training_20260804/",
            guide,
        )


if __name__ == "__main__":
    unittest.main()
