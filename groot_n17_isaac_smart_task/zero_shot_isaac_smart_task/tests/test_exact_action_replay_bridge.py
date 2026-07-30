from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exact_action_replay_bridge import (  # noqa: E402
    ACTION_DECODING_CONTRACT,
    ExactActionReplayBridge,
    load_replay_trace,
)
from so101_joint_units import so101_dataset_to_isaac_rad  # noqa: E402


COMMAND_CHUNKS = (
    np.asarray(
        [
            [0.03, -0.8, 0.9, 1.5, -1.4, -0.15],
            [0.05, -0.75, 0.85, 1.45, -1.35, 0.10],
        ],
        dtype=np.float32,
    ),
    np.asarray(
        [
            [0.08, -0.70, 0.80, 1.40, -1.30, 0.50],
        ],
        dtype=np.float32,
    ),
)


def _records(*, arm_units: str = "degrees") -> list[dict]:
    records: list[dict] = [
        {
            "type": "run_config",
            "camera_layout": "triple",
            "checkpoint_arm_units": arm_units,
            "max_policy_calls": len(COMMAND_CHUNKS),
            "action_horizon": 0,
            "initial_joint_pos_rad": [0.0, -0.9, 1.0, 1.6, -1.6, -0.1],
        }
    ]
    for call_index, commands in enumerate(COMMAND_CHUNKS, start=1):
        records.append(
            {
                "type": "policy_call",
                "call_index": call_index,
                "applied_commands_rad": commands.tolist(),
                "executed_action_steps": len(commands),
            }
        )
    return records


class _TraceFile:
    def __init__(self, records: list[dict]) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self._temporary_directory.name) / "action_trace.jsonl"
        self.path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def cleanup(self) -> None:
        self._temporary_directory.cleanup()


class ExactActionReplayBridgeTest(unittest.TestCase):
    def test_trace_conversion_round_trips_to_each_original_applied_command(self) -> None:
        for arm_units in ("degrees", "lerobot_motor_units"):
            trace_file = _TraceFile(_records(arm_units=arm_units))
            self.addCleanup(trace_file.cleanup)
            trace = load_replay_trace(trace_file.path)

            self.assertEqual(trace.checkpoint_arm_units, arm_units)
            self.assertEqual(len(trace.chunks), len(COMMAND_CHUNKS))
            for chunk, expected_rad in zip(trace.chunks, COMMAND_CHUNKS, strict=True):
                dataset_action = np.concatenate(
                    [chunk.action["single_arm"], chunk.action["gripper"]],
                    axis=-1,
                )[0]
                reconstructed_rad = so101_dataset_to_isaac_rad(dataset_action, arm_units)
                np.testing.assert_allclose(reconstructed_rad, expected_rad, atol=1e-6, rtol=0.0)

    def test_ping_exposes_dual_so101_finetuned_contract(self) -> None:
        trace_file = _TraceFile(_records())
        self.addCleanup(trace_file.cleanup)
        bridge = ExactActionReplayBridge(load_replay_trace(trace_file.path))

        response, should_exit = bridge.handle_request({"endpoint": "ping"})

        self.assertTrue(response["ok"])
        self.assertFalse(should_exit)
        self.assertEqual(response["camera_layout"], "dual")
        self.assertEqual(
            response["modality"]["video"]["modality_keys"],
            ["top", "wrist"],
        )
        self.assertEqual(response["action_decoding"], ACTION_DECODING_CONTRACT)
        self.assertEqual(response["replay"]["checkpoint_arm_units"], "degrees")
        self.assertEqual(response["replay"]["total_chunks"], 2)

    def test_get_action_preserves_call_sequence_and_reset_rewinds_it(self) -> None:
        trace_file = _TraceFile(_records())
        self.addCleanup(trace_file.cleanup)
        bridge = ExactActionReplayBridge(load_replay_trace(trace_file.path))

        first, first_exit = bridge.handle_request(
            {"endpoint": "get_action", "observation": {}}
        )
        second, second_exit = bridge.handle_request(
            {"endpoint": "get_action", "observation": {}}
        )
        exhausted, exhausted_exit = bridge.handle_request(
            {"endpoint": "get_action", "observation": {}}
        )

        self.assertFalse(first_exit)
        self.assertFalse(second_exit)
        self.assertFalse(exhausted_exit)
        self.assertEqual(first["info"]["source_call_index"], 1)
        self.assertEqual(second["info"]["source_call_index"], 2)
        self.assertEqual(first["action"]["single_arm"].shape, (1, 2, 5))
        self.assertEqual(second["action"]["gripper"].shape, (1, 1, 1))
        self.assertFalse(exhausted["ok"])
        self.assertIn("exhausted", exhausted["error"])

        reset, reset_exit = bridge.handle_request({"endpoint": "reset"})
        replayed_first, _ = bridge.handle_request(
            {"endpoint": "get_action", "observation": {}}
        )
        self.assertTrue(reset["ok"])
        self.assertFalse(reset_exit)
        self.assertEqual(replayed_first["info"]["source_call_index"], 1)
        np.testing.assert_array_equal(
            replayed_first["action"]["single_arm"],
            first["action"]["single_arm"],
        )

    def test_exit_after_replay_requests_graceful_exit_only_after_final_chunk(self) -> None:
        trace_file = _TraceFile(_records())
        self.addCleanup(trace_file.cleanup)
        bridge = ExactActionReplayBridge(
            load_replay_trace(trace_file.path),
            exit_after_replay=True,
        )

        _, first_exit = bridge.handle_request({"endpoint": "get_action", "observation": {}})
        _, final_exit = bridge.handle_request({"endpoint": "get_action", "observation": {}})

        self.assertFalse(first_exit)
        self.assertTrue(final_exit)

    def test_rejects_discontinuous_calls_nonfinite_commands_and_shape_errors(self) -> None:
        invalid_record_sets = []

        discontinuous = _records()
        discontinuous[2]["call_index"] = 3
        invalid_record_sets.append((discontinuous, "contiguous"))

        nonfinite = _records()
        nonfinite[1]["applied_commands_rad"][0][0] = float("nan")
        invalid_record_sets.append((nonfinite, "NaN or infinity"))

        wrong_shape = _records()
        wrong_shape[1]["applied_commands_rad"] = [[0.0] * 5]
        wrong_shape[1]["executed_action_steps"] = 1
        invalid_record_sets.append((wrong_shape, "shape Nx6"))

        for records, expected_message in invalid_record_sets:
            with self.subTest(expected_message=expected_message):
                trace_file = _TraceFile(records)
                try:
                    with self.assertRaisesRegex(ValueError, expected_message):
                        load_replay_trace(trace_file.path)
                finally:
                    trace_file.cleanup()

    def test_rejects_missing_or_duplicate_run_config(self) -> None:
        missing = _TraceFile(_records()[1:])
        duplicate_records = _records()
        duplicate_records.insert(1, dict(duplicate_records[0]))
        duplicate = _TraceFile(duplicate_records)
        self.addCleanup(missing.cleanup)
        self.addCleanup(duplicate.cleanup)

        with self.assertRaisesRegex(ValueError, "first JSONL record"):
            load_replay_trace(missing.path)
        with self.assertRaisesRegex(ValueError, "duplicate run_config"):
            load_replay_trace(duplicate.path)


if __name__ == "__main__":
    unittest.main()
