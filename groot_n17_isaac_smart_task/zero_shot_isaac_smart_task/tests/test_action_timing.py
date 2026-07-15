from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from action_timing import resolve_env_steps_per_policy_action  # noqa: E402


class ActionTimingTest(unittest.TestCase):
    def test_30_hz_policy_is_held_for_two_60_hz_env_steps(self) -> None:
        env_steps, effective_hz = resolve_env_steps_per_policy_action(30.0, 1.0 / 60.0)
        self.assertEqual(env_steps, 2)
        self.assertAlmostEqual(effective_hz, 30.0)

    def test_matching_policy_and_env_rates_use_one_step(self) -> None:
        env_steps, effective_hz = resolve_env_steps_per_policy_action(60.0, 1.0 / 60.0)
        self.assertEqual(env_steps, 1)
        self.assertAlmostEqual(effective_hz, 60.0)

    def test_non_integer_ratio_is_rejected_instead_of_rounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "not an integer"):
            resolve_env_steps_per_policy_action(25.0, 1.0 / 60.0)

    def test_policy_cannot_run_faster_than_environment(self) -> None:
        with self.assertRaisesRegex(ValueError, "faster"):
            resolve_env_steps_per_policy_action(120.0, 1.0 / 60.0)

    def test_invalid_rates_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            resolve_env_steps_per_policy_action(0.0, 1.0 / 60.0)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            resolve_env_steps_per_policy_action(30.0, float("nan"))


if __name__ == "__main__":
    unittest.main()
