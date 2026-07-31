from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from run_so101_groot_real import (
    POSITION_KEYS,
    action_dict,
    build_robot,
    build_policy_observation,
    safe_target,
    state_from_observation,
)


class RealSO101GrootHelpersTest(unittest.TestCase):
    def observation(self):
        obs = {key: float(index) for index, key in enumerate(POSITION_KEYS)}
        obs["wrist"] = np.zeros((480, 640, 3), dtype=np.uint8)
        return obs

    def test_builds_wrist_only_new_embodiment_observation(self):
        nested = build_policy_observation(
            self.observation(),
            wrist_camera_key="wrist",
            instruction="pick up block",
        )
        self.assertEqual(nested["video"]["wrist"].shape, (1, 1, 480, 640, 3))
        self.assertEqual(nested["state"]["single_arm"].shape, (1, 1, 5))
        self.assertEqual(nested["state"]["gripper"].shape, (1, 1, 1))
        self.assertEqual(
            nested["language"]["annotation.human.task_description"],
            [["pick up block"]],
        )

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
                    wrist_camera="0",
                    camera_width=640,
                    camera_height=480,
                    camera_fps=30,
                    camera_fourcc="MJPG",
                    wrist_camera_key="wrist",
                    robot_port="/dev/ttyACM-test",
                    robot_id="test_follower",
                    calibration_dir=Path(temp_dir),
                    checkpoint_arm_units="degrees",
                    max_arm_delta=5.0,
                    max_gripper_delta=10.0,
                    keep_torque_on_disconnect=False,
                )
            )
            self.assertEqual(robot.bus.port, "/dev/ttyACM-test")
            self.assertEqual(tuple(robot.cameras), ("wrist",))
            self.assertFalse(robot.is_connected)


if __name__ == "__main__":
    unittest.main()
