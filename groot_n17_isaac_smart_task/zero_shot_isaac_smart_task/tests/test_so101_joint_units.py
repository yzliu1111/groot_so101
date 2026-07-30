from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from so101_joint_units import (  # noqa: E402
    SO101_ARM_UNITS_DEGREES,
    SO101_ARM_UNITS_LEROBOT_MOTOR,
    SO101_LEROBOT_MOTOR_LIMITS,
    SO101_USD_JOINT_LIMITS_DEG,
    isaac_rad_to_lerobot_motor,
    isaac_rad_to_so101_dataset,
    lerobot_motor_to_isaac_rad,
    safe_absolute_dataset_target_to_isaac_rad,
    so101_dataset_to_isaac_rad,
    validate_runtime_joint_limits_match_converter,
)


def _safe_motor_target(current, target, **kwargs):
    return safe_absolute_dataset_target_to_isaac_rad(
        current,
        target,
        arm_units=SO101_ARM_UNITS_LEROBOT_MOTOR,
        **kwargs,
    )


class So101JointUnitsTest(unittest.TestCase):
    def test_usd_limits_map_to_motor_limits(self) -> None:
        joint_limits_rad = np.deg2rad(SO101_USD_JOINT_LIMITS_DEG)
        np.testing.assert_allclose(
            isaac_rad_to_lerobot_motor(joint_limits_rad[:, 0]),
            SO101_LEROBOT_MOTOR_LIMITS[:, 0],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            isaac_rad_to_lerobot_motor(joint_limits_rad[:, 1]),
            SO101_LEROBOT_MOTOR_LIMITS[:, 1],
            atol=1e-5,
        )

    def test_round_trip_preserves_joint_radians(self) -> None:
        joint_rad = np.asarray([0.2, -0.7, 0.8, 1.1, -1.2, 0.35], dtype=np.float32)
        np.testing.assert_allclose(
            lerobot_motor_to_isaac_rad(isaac_rad_to_lerobot_motor(joint_rad)),
            joint_rad,
            atol=2e-6,
        )

    def test_degree_dataset_round_trip_preserves_arm_and_gripper(self) -> None:
        joint_rad = np.asarray([0.2, -0.7, 0.8, 1.1, -1.2, 0.35], dtype=np.float32)
        dataset_values = isaac_rad_to_so101_dataset(joint_rad, SO101_ARM_UNITS_DEGREES)

        np.testing.assert_allclose(dataset_values[:5], np.rad2deg(joint_rad[:5]), atol=2e-5)
        self.assertAlmostEqual(
            float(dataset_values[5]),
            float(isaac_rad_to_lerobot_motor(joint_rad)[5]),
            places=5,
        )
        np.testing.assert_allclose(
            so101_dataset_to_isaac_rad(dataset_values, SO101_ARM_UNITS_DEGREES),
            joint_rad,
            atol=2e-6,
        )

    def test_degree_target_converts_arm_directly_but_gripper_by_range(self) -> None:
        current = np.zeros(6, dtype=np.float32)
        dataset_target = np.asarray([90.0, -45.0, 0.0, 80.0, -90.0, 50.0], dtype=np.float32)
        command, diagnostics = safe_absolute_dataset_target_to_isaac_rad(
            current,
            dataset_target,
            arm_units=SO101_ARM_UNITS_DEGREES,
            max_arm_step_rad=None,
        )

        np.testing.assert_allclose(command[:5], np.deg2rad(dataset_target[:5]), atol=2e-6)
        expected_gripper = lerobot_motor_to_isaac_rad(
            np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 50.0], dtype=np.float32)
        )[5]
        self.assertAlmostEqual(float(command[5]), float(expected_gripper), places=6)
        self.assertNotAlmostEqual(float(command[5]), float(np.deg2rad(50.0)), places=3)
        self.assertFalse(diagnostics["dataset_limit_clipped"])

    def test_degree_target_reuses_dataset_and_step_limits(self) -> None:
        current = np.zeros(6, dtype=np.float32)
        command, diagnostics = safe_absolute_dataset_target_to_isaac_rad(
            current,
            np.asarray([500.0, -500.0, 500.0, 500.0, -500.0, 1000.0], dtype=np.float32),
            arm_units=SO101_ARM_UNITS_DEGREES,
            max_arm_step_rad=0.08,
        )
        self.assertTrue(diagnostics["dataset_limit_clipped"])
        np.testing.assert_allclose(
            diagnostics["clipped_dataset_target"],
            np.asarray([110.0, -100.0, 90.0, 95.0, -160.0, 100.0], dtype=np.float32),
        )
        self.assertTrue(diagnostics["arm_step_clipped"])
        self.assertLessEqual(float(np.max(np.abs(command[:5] - current[:5]))), 0.080001)
        self.assertTrue(np.all(np.isfinite(command)))

    def test_gripper_step_limit_clamps_closing_from_actual_joint(self) -> None:
        current = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.279], dtype=np.float32)
        dataset_target = isaac_rad_to_so101_dataset(
            np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -0.127], dtype=np.float32),
            SO101_ARM_UNITS_DEGREES,
        )
        command, diagnostics = safe_absolute_dataset_target_to_isaac_rad(
            current,
            dataset_target,
            arm_units=SO101_ARM_UNITS_DEGREES,
            max_arm_step_rad=None,
            max_gripper_step_rad=0.04,
        )

        self.assertAlmostEqual(float(command[5]), 0.239, places=6)
        self.assertTrue(diagnostics["gripper_step_clipped"])
        self.assertAlmostEqual(
            float(diagnostics["requested_gripper_delta_rad"]),
            -0.406,
            places=5,
        )
        self.assertAlmostEqual(
            float(diagnostics["applied_gripper_delta_rad"]),
            -0.04,
            places=6,
        )

    def test_gripper_step_limit_is_symmetric_for_opening(self) -> None:
        current = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -0.14], dtype=np.float32)
        dataset_target = isaac_rad_to_so101_dataset(
            np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.35], dtype=np.float32),
            SO101_ARM_UNITS_LEROBOT_MOTOR,
        )
        command, diagnostics = _safe_motor_target(
            current,
            dataset_target,
            max_arm_step_rad=None,
            max_gripper_step_rad=0.04,
        )

        self.assertAlmostEqual(float(command[5]), -0.10, places=6)
        self.assertTrue(diagnostics["gripper_step_clipped"])
        self.assertAlmostEqual(
            float(diagnostics["applied_gripper_delta_rad"]),
            0.04,
            places=6,
        )

    def test_zero_gripper_step_limit_preserves_absolute_target(self) -> None:
        current = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.279], dtype=np.float32)
        dataset_target = isaac_rad_to_so101_dataset(
            np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -0.127], dtype=np.float32),
            SO101_ARM_UNITS_DEGREES,
        )
        command, diagnostics = safe_absolute_dataset_target_to_isaac_rad(
            current,
            dataset_target,
            arm_units=SO101_ARM_UNITS_DEGREES,
            max_arm_step_rad=None,
            max_gripper_step_rad=0.0,
        )

        self.assertAlmostEqual(float(command[5]), -0.127, places=6)
        self.assertFalse(diagnostics["gripper_step_clipped"])

    def test_unknown_checkpoint_arm_units_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "arm_units must be one of"):
            isaac_rad_to_so101_dataset(np.zeros(6, dtype=np.float32), "radians")

    def test_runtime_limits_must_match_converter_mapping(self) -> None:
        runtime_limits = np.deg2rad(SO101_USD_JOINT_LIMITS_DEG)
        validate_runtime_joint_limits_match_converter(runtime_limits)

        wrong_limits = runtime_limits.copy()
        wrong_limits[0, 0] += 0.1
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_runtime_joint_limits_match_converter(wrong_limits)

    def test_absolute_motor_target_is_not_added_as_radians(self) -> None:
        current = np.zeros(6, dtype=np.float32)
        motor_target = np.asarray([20.0, -20.0, 30.0, 50.0, 40.0, 80.0], dtype=np.float32)
        command, diagnostics = _safe_motor_target(
            current,
            motor_target,
            max_arm_step_rad=None,
        )
        expected = lerobot_motor_to_isaac_rad(motor_target)
        np.testing.assert_allclose(command, expected, atol=2e-6)
        self.assertLess(float(np.max(np.abs(command))), 3.0)
        self.assertFalse(diagnostics["dataset_limit_clipped"])

    def test_extreme_output_is_motor_and_step_clipped(self) -> None:
        current = np.zeros(6, dtype=np.float32)
        command, diagnostics = _safe_motor_target(
            current,
            np.asarray([1000.0, -1000.0, 500.0, 300.0, -400.0, 1000.0], dtype=np.float32),
            max_arm_step_rad=0.08,
        )
        self.assertTrue(diagnostics["dataset_limit_clipped"])
        self.assertTrue(diagnostics["arm_step_clipped"])
        self.assertLessEqual(float(np.max(np.abs(command[:5] - current[:5]))), 0.080001)
        self.assertTrue(np.all(np.isfinite(command)))

    def test_runtime_limits_are_final_guard(self) -> None:
        current = np.zeros(6, dtype=np.float32)
        runtime_limits = np.asarray([[-0.05, 0.05]] * 6, dtype=np.float32)
        command, diagnostics = _safe_motor_target(
            current,
            np.asarray([100.0] * 6, dtype=np.float32),
            max_arm_step_rad=None,
            runtime_joint_limits_rad=runtime_limits,
        )
        self.assertTrue(diagnostics["runtime_limit_clipped"])
        self.assertLessEqual(float(np.max(command)), 0.050001)

    def test_nonfinite_model_output_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            _safe_motor_target(
                np.zeros(6, dtype=np.float32),
                np.asarray([0.0, 0.0, np.nan, 0.0, 0.0, 0.0], dtype=np.float32),
            )

    def test_target_interpolation_scale_cannot_overshoot(self) -> None:
        with self.assertRaisesRegex(ValueError, r"within \[0, 1\]"):
            _safe_motor_target(
                np.zeros(6, dtype=np.float32),
                np.zeros(6, dtype=np.float32),
                arm_target_scale=1.1,
            )

    def test_nonfinite_step_limit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be finite"):
            _safe_motor_target(
                np.zeros(6, dtype=np.float32),
                np.zeros(6, dtype=np.float32),
                max_arm_step_rad=np.nan,
            )

    def test_negative_step_limit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _safe_motor_target(
                np.zeros(6, dtype=np.float32),
                np.zeros(6, dtype=np.float32),
                max_arm_step_rad=-0.01,
            )

    def test_nonfinite_gripper_step_limit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be finite"):
            _safe_motor_target(
                np.zeros(6, dtype=np.float32),
                np.zeros(6, dtype=np.float32),
                max_gripper_step_rad=np.nan,
            )

    def test_negative_gripper_step_limit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _safe_motor_target(
                np.zeros(6, dtype=np.float32),
                np.zeros(6, dtype=np.float32),
                max_gripper_step_rad=-0.01,
            )


if __name__ == "__main__":
    unittest.main()
