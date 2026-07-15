from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from action_chunk import action_chunk_length, slice_action_step  # noqa: E402


class ActionChunkTest(unittest.TestCase):
    def test_length_requires_matching_time_dimensions(self) -> None:
        action = {
            "single_arm": np.zeros((1, 3, 5), dtype=np.float32),
            "gripper": np.zeros((1, 3, 1), dtype=np.float32),
        }
        self.assertEqual(action_chunk_length(action), 3)

    def test_mismatched_time_dimensions_are_rejected(self) -> None:
        action = {
            "single_arm": np.zeros((1, 3, 5), dtype=np.float32),
            "gripper": np.zeros((1, 2, 1), dtype=np.float32),
        }
        with self.assertRaisesRegex(ValueError, "do not match"):
            action_chunk_length(action)

    def test_known_action_arrays_must_be_rank_three(self) -> None:
        with self.assertRaisesRegex(ValueError, r"shape \(B,T,D\)"):
            action_chunk_length({"joint_position": np.zeros((3, 7), dtype=np.float32)})

    def test_nonfinite_action_is_rejected_before_execution(self) -> None:
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            action_chunk_length({"joint_position": np.asarray([[[0.0, np.nan]]], dtype=np.float32)})

    def test_slice_returns_only_requested_existing_keys(self) -> None:
        action = {
            "eef_9d": np.arange(27, dtype=np.float32).reshape(1, 3, 9),
            "gripper_position": np.arange(3, dtype=np.float32).reshape(1, 3, 1),
            "metadata": "ignored",
        }
        selected = slice_action_step(action, 1, ("eef_9d", "gripper_position", "missing"))
        self.assertEqual(set(selected), {"eef_9d", "gripper_position"})
        self.assertEqual(selected["eef_9d"].shape, (1, 1, 9))
        np.testing.assert_array_equal(selected["eef_9d"], action["eef_9d"][:, 1:2, :])

    def test_slice_rejects_out_of_range_index(self) -> None:
        with self.assertRaisesRegex(IndexError, "outside"):
            slice_action_step({"gripper": np.zeros((1, 1, 1))}, 1, ("gripper",))


if __name__ == "__main__":
    unittest.main()
