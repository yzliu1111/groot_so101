from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from so101_joint_units import (  # noqa: E402
    SO101_LEROBOT_MOTOR_LIMITS,
    SO101_USD_JOINT_LIMITS_DEG,
    isaac_rad_to_lerobot_motor,
    lerobot_motor_to_isaac_rad,
    safe_absolute_motor_target_to_isaac_rad,
    validate_runtime_joint_limits_match_converter,
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
        command, diagnostics = safe_absolute_motor_target_to_isaac_rad(
            current,
            motor_target,
            max_arm_step_rad=None,
        )
        expected = lerobot_motor_to_isaac_rad(motor_target)
        np.testing.assert_allclose(command, expected, atol=2e-6)
        self.assertLess(float(np.max(np.abs(command))), 3.0)
        self.assertFalse(diagnostics["motor_limit_clipped"])

    def test_extreme_output_is_motor_and_step_clipped(self) -> None:
        current = np.zeros(6, dtype=np.float32)
        command, diagnostics = safe_absolute_motor_target_to_isaac_rad(
            current,
            np.asarray([1000.0, -1000.0, 500.0, 300.0, -400.0, 1000.0], dtype=np.float32),
            max_arm_step_rad=0.08,
        )
        self.assertTrue(diagnostics["motor_limit_clipped"])
        self.assertTrue(diagnostics["arm_step_clipped"])
        self.assertLessEqual(float(np.max(np.abs(command[:5] - current[:5]))), 0.080001)
        self.assertTrue(np.all(np.isfinite(command)))

    def test_runtime_limits_are_final_guard(self) -> None:
        current = np.zeros(6, dtype=np.float32)
        runtime_limits = np.asarray([[-0.05, 0.05]] * 6, dtype=np.float32)
        command, diagnostics = safe_absolute_motor_target_to_isaac_rad(
            current,
            np.asarray([100.0] * 6, dtype=np.float32),
            max_arm_step_rad=None,
            runtime_joint_limits_rad=runtime_limits,
        )
        self.assertTrue(diagnostics["runtime_limit_clipped"])
        self.assertLessEqual(float(np.max(command)), 0.050001)

    def test_nonfinite_model_output_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            safe_absolute_motor_target_to_isaac_rad(
                np.zeros(6, dtype=np.float32),
                np.asarray([0.0, 0.0, np.nan, 0.0, 0.0, 0.0], dtype=np.float32),
            )

    def test_target_interpolation_scale_cannot_overshoot(self) -> None:
        with self.assertRaisesRegex(ValueError, r"within \[0, 1\]"):
            safe_absolute_motor_target_to_isaac_rad(
                np.zeros(6, dtype=np.float32),
                np.zeros(6, dtype=np.float32),
                arm_target_scale=1.1,
            )

    def test_nonfinite_step_limit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be finite"):
            safe_absolute_motor_target_to_isaac_rad(
                np.zeros(6, dtype=np.float32),
                np.zeros(6, dtype=np.float32),
                max_arm_step_rad=np.nan,
            )

    def test_negative_step_limit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            safe_absolute_motor_target_to_isaac_rad(
                np.zeros(6, dtype=np.float32),
                np.zeros(6, dtype=np.float32),
                max_arm_step_rad=-0.01,
            )


if __name__ == "__main__":
    unittest.main()
