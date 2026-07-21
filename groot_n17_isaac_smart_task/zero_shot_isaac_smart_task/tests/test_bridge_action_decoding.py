from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from groot_bridge_server import _summarize_action_decoding  # noqa: E402


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
        self.assertEqual(summary["dataset_action_units"], "checkpoint_dataset_coordinates")

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

if __name__ == "__main__":
    unittest.main()
