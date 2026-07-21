from __future__ import annotations

from contextlib import redirect_stderr
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from groot_bridge_server import _summarize_action_decoding, parse_args  # noqa: E402


class _Processor:
    def __init__(self, use_relative_action: bool) -> None:
        self.use_relative_action = use_relative_action


class _Policy:
    def __init__(self, use_relative_action: bool) -> None:
        self.processor = _Processor(use_relative_action)


class _Wrapper:
    def __init__(self, policy: _Policy) -> None:
        self.policy = policy


class BridgeActionDecodingTest(unittest.TestCase):
    def test_reports_so101_external_contract_separately_from_internal_representation(self) -> None:
        summary = _summarize_action_decoding(_Policy(True), "so101-finetuned")
        self.assertIs(summary["processor_use_relative_action"], True)
        self.assertEqual(summary["policy_api_output"], "decoded_dataset_action")
        self.assertEqual(summary["dataset_action_semantics"], "absolute_joint_position_targets")
        self.assertEqual(summary["dataset_action_units"], "lerobot_motor_units")
        self.assertEqual(summary["dataset_arm_joint_units"], "lerobot_motor_units")
        self.assertEqual(summary["dataset_state_arm_joint_units"], "lerobot_motor_units")
        self.assertEqual(summary["dataset_gripper_units"], "lerobot_range_0_100")

    def test_reports_real_checkpoint_degree_contract(self) -> None:
        summary = _summarize_action_decoding(_Policy(True), "so101-finetuned", "degrees")
        self.assertEqual(summary["dataset_action_units"], "arm_degrees_gripper_range_0_100")
        self.assertEqual(summary["dataset_arm_joint_units"], "degrees")
        self.assertEqual(summary["dataset_state_arm_joint_units"], "degrees")
        self.assertEqual(summary["dataset_gripper_units"], "lerobot_range_0_100")

    def test_unwraps_sim_policy_wrapper(self) -> None:
        summary = _summarize_action_decoding(_Wrapper(_Policy(True)), "so101-finetuned")
        self.assertIs(summary["processor_use_relative_action"], True)

    def test_internal_relative_flag_does_not_change_external_dataset_contract(self) -> None:
        summary = _summarize_action_decoding(_Policy(False), "so101-finetuned")
        self.assertIs(summary["processor_use_relative_action"], False)
        self.assertEqual(summary["policy_api_output"], "decoded_dataset_action")
        self.assertEqual(summary["dataset_action_semantics"], "absolute_joint_position_targets")

    def test_unknown_processor_is_reported_without_relabeling_output_as_delta(self) -> None:
        summary = _summarize_action_decoding(object(), "so101-finetuned")
        self.assertIsNone(summary["processor_use_relative_action"])
        self.assertEqual(summary["policy_api_output"], "decoded_dataset_action")

    def test_zero_shot_contract_remains_embodiment_specific(self) -> None:
        summary = _summarize_action_decoding(_Policy(True), "zero-shot-oxe")
        self.assertEqual(summary["dataset_action_semantics"], "embodiment_specific")
        self.assertEqual(summary["dataset_action_units"], "embodiment_specific")

    def test_bridge_cli_defaults_to_sim_motor_units(self) -> None:
        with patch.object(sys, "argv", ["groot_bridge_server.py"]):
            args = parse_args()
        self.assertEqual(args.so101_checkpoint_arm_units, "lerobot_motor_units")

    def test_bridge_cli_accepts_degree_finetuned_checkpoint(self) -> None:
        argv = [
            "groot_bridge_server.py",
            "--deployment-mode",
            "so101-finetuned",
            "--model-path",
            "/tmp/real-checkpoint",
            "--so101-checkpoint-joint-units",
            "degrees",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.so101_checkpoint_arm_units, "degrees")
        self.assertEqual(args.embodiment_tag, "NEW_EMBODIMENT")

    def test_bridge_cli_rejects_degree_units_for_zero_shot(self) -> None:
        argv = [
            "groot_bridge_server.py",
            "--so101-checkpoint-joint-units",
            "degrees",
        ]
        with (
            patch.object(sys, "argv", argv),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            parse_args()


if __name__ == "__main__":
    unittest.main()
