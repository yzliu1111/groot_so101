from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pose_math import (  # noqa: E402
    compose_pose_delta,
    matrix_to_quat_wxyz,
    quat_wxyz_to_matrix,
    quat_wxyz_to_rot6d,
    rot6d_to_matrix,
)


class PoseMathTest(unittest.TestCase):
    def test_identity_quaternion_has_identity_rot6d(self) -> None:
        np.testing.assert_allclose(
            quat_wxyz_to_rot6d([1.0, 0.0, 0.0, 0.0]),
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            atol=1e-7,
        )

    def test_matrix_quaternion_round_trip_preserves_rotation(self) -> None:
        quat = np.asarray([0.5, -0.5, 0.5, 0.5], dtype=np.float32)
        matrix = quat_wxyz_to_matrix(quat)
        np.testing.assert_allclose(quat_wxyz_to_matrix(matrix_to_quat_wxyz(matrix)), matrix, atol=1e-6)

    def test_degenerate_second_rot6d_row_still_returns_rotation(self) -> None:
        matrix = rot6d_to_matrix([0.0, 1.0, 0.0, 0.0, 2.0, 0.0])
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=6)

    def test_pose_delta_is_composed_in_eef_frame(self) -> None:
        target_pos, target_quat = compose_pose_delta(
            [1.0, 2.0, 3.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.1, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        np.testing.assert_allclose(target_pos, [1.1, 1.8, 3.3], atol=1e-6)
        np.testing.assert_allclose(target_quat, [1.0, 0.0, 0.0, 0.0], atol=1e-6)

    def test_nonfinite_pose_input_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            compose_pose_delta(
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [np.nan, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            )


if __name__ == "__main__":
    unittest.main()
