from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from run_so101_groot_real import (
    AWS_PROFILES,
    POSITION_KEYS,
    action_dict,
    build_robot,
    build_policy_observation,
    resolve_camera_devices,
    resolve_run,
    safe_target,
    state_from_observation,
    validate_checkpoint_layout,
)


class RealSO101GrootHelpersTest(unittest.TestCase):
    def observation(self):
        obs = {key: float(index) for index, key in enumerate(POSITION_KEYS)}
        obs["top"] = np.zeros((480, 640, 3), dtype=np.uint8)
        obs["left"] = np.zeros((480, 640, 3), dtype=np.uint8)
        obs["wrist"] = np.zeros((480, 640, 3), dtype=np.uint8)
        return obs

    def test_builds_wrist_only_new_embodiment_observation(self):
        nested = build_policy_observation(
            self.observation(),
            video_keys=("wrist",),
            instruction="pick up block",
        )
        self.assertEqual(nested["video"]["wrist"].shape, (1, 1, 480, 640, 3))
        self.assertEqual(nested["state"]["single_arm"].shape, (1, 1, 5))
        self.assertEqual(nested["state"]["gripper"].shape, (1, 1, 1))
        self.assertEqual(
            nested["language"]["annotation.human.task_description"],
            [["pick up block"]],
        )

    def test_builds_triple_observation(self):
        nested = build_policy_observation(
            self.observation(),
            video_keys=("top", "left", "wrist"),
            instruction="pick up block",
        )
        self.assertEqual(tuple(nested["video"]), ("top", "left", "wrist"))
        self.assertTrue(all(value.shape == (1, 1, 480, 640, 3) for value in nested["video"].values()))

    def test_aws_profiles_cover_all_five_checkpoints_and_historical_mapping(self):
        self.assertEqual(set(AWS_PROFILES), {"real001", "real003", "real007", "sim002", "sim004"})
        self.assertEqual(
            AWS_PROFILES["real003"].role_to_slot,
            {"top": "camera2", "left": "camera1", "wrist": "camera3"},
        )
        self.assertEqual(AWS_PROFILES["sim004"].camera_layout, "triple")

    def test_profile_resolves_checkpoint_layout_units_and_devices(self):
        args = Namespace(
            aws_profile="real003",
            checkpoint=None,
            smart_project="/tmp/smart_project",
            checkpoint_arm_units=None,
            instruction=None,
            camera_layout="wrist-only",
            camera1="/dev/cam-one",
            camera2="/dev/cam-two",
            camera3="/dev/cam-three",
            top_camera=None,
            left_camera=None,
            wrist_camera=None,
        )
        checkpoint, instruction, keys, mapping = resolve_run(args)
        self.assertTrue(str(checkpoint).endswith("checkpoint-10000"))
        self.assertEqual(instruction, "pick up block")
        self.assertEqual(keys, ("top", "left", "wrist"))
        self.assertEqual(args.checkpoint_arm_units, "degrees")
        self.assertEqual(
            resolve_camera_devices(args, mapping),
            {"top": "/dev/cam-two", "left": "/dev/cam-one", "wrist": "/dev/cam-three"},
        )

    def test_explicit_dual_layout_is_supported(self):
        args = Namespace(
            aws_profile=None,
            checkpoint="/tmp/checkpoint",
            smart_project="/tmp/smart_project",
            checkpoint_arm_units="lerobot_motor_units",
            instruction="pick",
            camera_layout="dual",
            top_camera="/dev/top",
            left_camera=None,
            wrist_camera="/dev/wrist",
        )
        _, _, keys, mapping = resolve_run(args)
        self.assertEqual(keys, ("top", "wrist"))
        self.assertEqual(
            resolve_camera_devices(args, mapping),
            {"top": "/dev/top", "wrist": "/dev/wrist"},
        )

    def test_checkpoint_processor_layout_is_enforced(self):
        with TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir)
            config = {
                "processor_kwargs": {
                    "modality_configs": {
                        "new_embodiment": {
                            "video": {"modality_keys": ["top", "left", "wrist"]}
                        }
                    }
                }
            }
            (checkpoint / "processor_config.json").write_text(
                __import__("json").dumps(config), encoding="utf-8"
            )
            validate_checkpoint_layout(checkpoint, ("top", "left", "wrist"))
            with self.assertRaisesRegex(ValueError, "camera layout/checkpoint mismatch"):
                validate_checkpoint_layout(checkpoint, ("wrist",))

    def test_state_order_matches_so101_motor_order(self):
        np.testing.assert_array_equal(
            state_from_observation(self.observation()),
            np.arange(6, dtype=np.float32),
        )

    def test_safe_target_clips_arm_and_gripper_per_tick(self):
        applied, diagnostics = safe_target(
            np.zeros(6, dtype=np.float32),
            np.asarray([10, -10, 4, -4, 100, 50], dtype=np.float32),
            max_arm_delta=5,
            max_gripper_delta=10,
        )
        np.testing.assert_array_equal(applied, [5, -5, 4, -4, 5, 10])
        self.assertTrue(diagnostics["clipped"])

    def test_action_dict_uses_pos_keys(self):
        result = action_dict(np.arange(6, dtype=np.float32))
        self.assertEqual(tuple(result), POSITION_KEYS)
        self.assertEqual(result["gripper.pos"], 5.0)

    def test_build_robot_uses_lerobot_so101_without_connecting_hardware(self):
        with TemporaryDirectory() as temp_dir:
            robot = build_robot(
                Namespace(
                    camera_width=640,
                    camera_height=480,
                    camera_fps=30,
                    camera_fourcc="MJPG",
                    robot_port="/dev/ttyACM-test",
                    robot_id="test_follower",
                    calibration_dir=Path(temp_dir),
                    checkpoint_arm_units="degrees",
                    max_arm_delta=5.0,
                    max_gripper_delta=10.0,
                    keep_torque_on_disconnect=False,
                ),
                {"wrist": "0"},
            )
            self.assertEqual(robot.bus.port, "/dev/ttyACM-test")
            self.assertEqual(tuple(robot.cameras), ("wrist",))
            self.assertFalse(robot.is_connected)


if __name__ == "__main__":
    unittest.main()
