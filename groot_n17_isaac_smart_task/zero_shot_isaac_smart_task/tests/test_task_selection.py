from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from task_selection import (  # noqa: E402
    DEFAULT_FRANKA_TASK,
    DEFAULT_SO101_TASK,
    default_task_for_robot,
    resolve_smart_target_asset,
    resolve_target_name,
    resolve_task_instruction,
    target_name_from_env_cfg,
    validate_robot_task,
)


class TaskSelectionTest(unittest.TestCase):
    def test_so101_default_task_is_backwards_compatible(self) -> None:
        self.assertEqual(default_task_for_robot("so101"), DEFAULT_SO101_TASK)

    def test_instruction_defaults_to_selected_task_description(self) -> None:
        env_cfg = SimpleNamespace(task_description="Pick up the blue block and place it on the tray")
        self.assertEqual(
            resolve_task_instruction(None, env_cfg, "LeIsaac-SO101-SmartTask-Blue-v0"),
            env_cfg.task_description,
        )

    def test_base_instruction_preserves_checkpoint_training_text(self) -> None:
        env_cfg = SimpleNamespace(task_description="pick up block")
        for task in (DEFAULT_SO101_TASK, DEFAULT_FRANKA_TASK):
            with self.subTest(task=task):
                self.assertEqual(
                    resolve_task_instruction(None, env_cfg, task),
                    "Pick up the red 2x4 lego brick.",
                )

    def test_explicit_checkpoint_instruction_takes_precedence(self) -> None:
        env_cfg = SimpleNamespace(task_description="different task text")
        self.assertEqual(
            resolve_task_instruction("  checkpoint instruction  ", env_cfg, DEFAULT_SO101_TASK),
            "checkpoint instruction",
        )

    def test_unknown_task_without_description_requires_instruction(self) -> None:
        with self.assertRaisesRegex(ValueError, "pass the checkpoint instruction"):
            resolve_task_instruction(None, SimpleNamespace(), "LeIsaac-SO101-Custom-v0")

    def test_robot_task_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --robot so101"):
            validate_robot_task("franka", DEFAULT_SO101_TASK)
        with self.assertRaisesRegex(ValueError, "requires --robot franka"):
            validate_robot_task("so101", "Groot-Franka-SmartTask-v0")

    def test_mimic_task_is_not_a_deployment_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "data-generation Mimic"):
            validate_robot_task("so101", "LeIsaac-SO101-SmartTask-Mimic-v0")

    def test_target_comes_from_selected_task_success_config(self) -> None:
        object_cfg = SimpleNamespace(name="blue_2x4_lego_brick")
        env_cfg = SimpleNamespace(
            terminations=SimpleNamespace(success=SimpleNamespace(params={"object_cfg": object_cfg}))
        )
        self.assertEqual(target_name_from_env_cfg(env_cfg), "blue_2x4_lego_brick")
        self.assertEqual(
            resolve_target_name(
                ["red_2x4_lego_brick", "blue_2x4_lego_brick"],
                "auto",
                target_name_from_env_cfg(env_cfg),
            ),
            "blue_2x4_lego_brick",
        )

    def test_multi_lego_scene_without_task_target_fails_closed(self) -> None:
        with self.assertRaisesRegex(KeyError, "multi-LEGO"):
            resolve_target_name(
                ["red_2x4_lego_brick", "blue_2x4_lego_brick"],
                "auto",
            )

    def test_explicit_target_is_validated_against_scene(self) -> None:
        self.assertEqual(
            resolve_target_name(
                ["red_2x4_lego_brick", "blue_2x4_lego_brick"],
                "blue_2x4_lego_brick",
            ),
            "blue_2x4_lego_brick",
        )

    def test_auto_cuboid_is_limited_to_legacy_base_task(self) -> None:
        self.assertEqual(resolve_smart_target_asset(DEFAULT_SO101_TASK, "auto"), "cuboid")
        self.assertEqual(
            resolve_smart_target_asset("LeIsaac-SO101-SmartTask-Blue-v0", "auto"),
            "scene",
        )
        self.assertEqual(
            resolve_smart_target_asset("LeIsaac-SO101-SmartTask-Mimic-v0", "auto"),
            "scene",
        )

    def test_explicit_asset_choice_is_preserved(self) -> None:
        self.assertEqual(
            resolve_smart_target_asset(DEFAULT_SO101_TASK, "/tmp/complete.usd"),
            "/tmp/complete.usd",
        )


if __name__ == "__main__":
    unittest.main()
