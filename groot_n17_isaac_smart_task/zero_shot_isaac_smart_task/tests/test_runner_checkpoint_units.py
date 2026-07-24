from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1]
RUNNER_PATH = SCRIPT_DIR / "run_smart_task_closed_loop.py"


def _load_runner_without_isaac_app():
    """Import runner logic without bootstrapping Omniverse/SimulationApp."""

    fake_isaaclab = ModuleType("isaaclab")
    fake_isaaclab.__path__ = []  # type: ignore[attr-defined]
    fake_app = ModuleType("isaaclab.app")
    fake_app.AppLauncher = object  # type: ignore[attr-defined]

    spec = importlib.util.spec_from_file_location("runner_checkpoint_units_test_module", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "isaaclab": fake_isaaclab,
            "isaaclab.app": fake_app,
            spec.name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module


runner = _load_runner_without_isaac_app()


def _finetuned_ping() -> dict:
    return {
        "camera_layout": "dual",
        "modality": {"video": {"modality_keys": ["top", "wrist"]}},
        "action_decoding": {
            "processor_use_relative_action": True,
            "policy_api_output": "decoded_dataset_action",
            "dataset_action_semantics": "absolute_joint_position_targets",
            "dataset_action_units": "checkpoint_dataset_coordinates",
        },
    }


class RunnerCheckpointUnitsTest(unittest.TestCase):
    def test_cli_defaults_to_base_so101_task(self) -> None:
        with patch.object(sys, "argv", ["run_smart_task_closed_loop.py"]):
            args = runner.parse_args()
        self.assertEqual(args.task, "LeIsaac-SO101-SmartTask-v0")
        self.assertEqual(args.scene_profile, "task-default")

    def test_cli_action_horizon_zero_is_the_full_chunk_sentinel(self) -> None:
        with patch.object(sys, "argv", ["run_smart_task_closed_loop.py"]):
            default_args = runner.parse_args()
        with patch.object(
            sys,
            "argv",
            ["run_smart_task_closed_loop.py", "--action-horizon", "0"],
        ):
            explicit_args = runner.parse_args()

        self.assertEqual(default_args.action_horizon, 0)
        self.assertEqual(explicit_args.action_horizon, 0)

    def test_cli_rejects_negative_action_horizon(self) -> None:
        argv = ["run_smart_task_closed_loop.py", "--action-horizon", "-1"]
        with patch.object(sys, "argv", argv), self.assertRaisesRegex(
            ValueError,
            "must be 0 or a positive integer",
        ):
            runner.parse_args()

    def test_cli_accepts_table_red24_without_another_task_id(self) -> None:
        argv = ["run_smart_task_closed_loop.py", "--scene-profile", "table-red24"]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()
        self.assertEqual(args.task, "LeIsaac-SO101-SmartTask-v0")
        self.assertEqual(args.scene_profile, "table-red24")
    def test_cli_accepts_tray_red24_without_another_task_id(self) -> None:
        argv = ["run_smart_task_closed_loop.py", "--scene-profile", "tray-red24"]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()
        self.assertEqual(args.task, "LeIsaac-SO101-SmartTask-v0")
        self.assertEqual(args.scene_profile, "tray-red24")

    def test_explicit_scene_profile_rejects_legacy_asset_controls(self) -> None:
        cases = (
            (["--smart-target-asset", "cuboid"], "smart-target"),
            (["--smart-target-pos", "0", "0", "0"], "smart-target"),
            (["--smart-scene-usd", "/tmp/custom.usd"], "smart-scene-usd"),
        )
        for extra_args, expected_message in cases:
            with self.subTest(extra_args=extra_args):
                argv = [
                    "run_smart_task_closed_loop.py",
                    "--scene-profile",
                    "table-red24",
                    *extra_args,
                ]
                with patch.object(sys, "argv", argv), self.assertRaisesRegex(
                    ValueError,
                    expected_message,
                ):
                    runner.parse_args()


    def test_multi_lego_profile_requires_target_and_instruction(self) -> None:
        missing_target = ["run_smart_task_closed_loop.py", "--scene-profile", "multi-lego-tray"]
        with patch.object(sys, "argv", missing_target), self.assertRaisesRegex(
            ValueError,
            "contains multiple LEGO",
        ):
            runner.parse_args()

        missing_instruction = [
            "run_smart_task_closed_loop.py",
            "--scene-profile",
            "multi-lego-tray",
            "--target-object-key",
            "blue_2x4_lego_brick",
        ]
        with patch.object(sys, "argv", missing_instruction), self.assertRaisesRegex(
            ValueError,
            "requires --instruction",
        ):
            runner.parse_args()

    def test_multi_lego_profile_accepts_explicit_target_and_instruction(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--scene-profile",
            "multi-lego-tray",
            "--target-object-key",
            "red_2x2_lego_brick",
            "--instruction",
            "Pick up the small red block and place it in the tray.",
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()
        self.assertEqual(args.scene_profile, "multi-lego-tray")
        self.assertEqual(args.target_object_key, "red_2x2_lego_brick")

    def test_cli_preserves_explicit_task_id(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--task",
            "LeIsaac-SO101-SmartTask-Blue-v0",
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()
        self.assertEqual(args.task, "LeIsaac-SO101-SmartTask-Blue-v0")

    def test_cli_rejects_robot_task_mismatch(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--robot",
            "franka",
            "--task",
            "LeIsaac-SO101-SmartTask-v0",
        ]
        with patch.object(sys, "argv", argv), self.assertRaisesRegex(
            ValueError,
            "requires --robot so101",
        ):
            runner.parse_args()

    def test_nonbase_task_rejects_legacy_asset_overrides(self) -> None:
        args = SimpleNamespace(
            task="LeIsaac-SO101-SmartTask-Blue-v0",
            smart_scene_usd="auto",
            smart_target_asset="auto",
            smart_target_pos=(0.1, 0.2, 0.3),
            smart_target_prim_path=runner.SMART_TARGET_MANAGED_PRIM_PATH,
            smart_target_cuboid_size=runner.SMART_TARGET_CUBOID_SIZE,
        )
        with self.assertRaisesRegex(ValueError, r"--smart-target-\*"):
            runner.install_smart_task_asset_patch(args)

    def test_cli_defaults_to_sim_motor_units(self) -> None:
        with patch.object(sys, "argv", ["run_smart_task_closed_loop.py"]):
            args = runner.parse_args()
        self.assertEqual(args.so101_checkpoint_arm_units, "lerobot_motor_units")

    def test_cli_accepts_degree_finetuned_checkpoint(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--deployment-mode",
            "so101-finetuned",
            "--so101-checkpoint-joint-units",
            "degrees",
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()
        self.assertEqual(args.so101_checkpoint_arm_units, "degrees")
        self.assertEqual(args.policy_schema, "so101-new-embodiment")

    def test_zero_shot_auto_initial_pose_keeps_usd_default(self) -> None:
        with patch.object(sys, "argv", ["run_smart_task_closed_loop.py"]):
            args = runner.parse_args()

        self.assertEqual(args.so101_initial_pose, "auto")
        self.assertIsNone(runner.resolve_so101_initial_pose(args))

    def test_finetuned_auto_initial_pose_selects_motor_frame_five_median(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--deployment-mode",
            "so101-finetuned",
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()

        pose = runner.resolve_so101_initial_pose(args)
        self.assertIsNotNone(pose)
        self.assertEqual(pose["mode"], "dataset-start")
        self.assertEqual(pose["source"], "sim_2.parquet:frame-5-median")
        np.testing.assert_allclose(
            pose["dataset_state"],
            runner.SO101_DATASET_START_MEDIAN_STATE["lerobot_motor_units"],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            pose["joint_rad"],
            [-0.0439360, -0.1178652, 0.0787711, 1.5077535, -1.4915221, -0.1517543],
            atol=2e-6,
        )

    def test_finetuned_auto_initial_pose_selects_degree_frame_zero_median(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--deployment-mode",
            "so101-finetuned",
            "--so101-checkpoint-joint-units",
            "degrees",
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()

        pose = runner.resolve_so101_initial_pose(args)
        self.assertIsNotNone(pose)
        self.assertEqual(pose["mode"], "dataset-start")
        self.assertEqual(pose["source"], "real_1.parquet:frame-0-median")
        np.testing.assert_allclose(
            pose["dataset_state"],
            runner.SO101_DATASET_START_MEDIAN_STATE["degrees"],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            pose["joint_rad"],
            [0.0253169, 0.0130420, 0.1120079, 1.6018670, -1.6432946, -0.1356880],
            atol=2e-6,
        )

    def test_legacy_first_frame_name_aliases_dataset_start(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--deployment-mode",
            "so101-finetuned",
            "--so101-initial-pose",
            "dataset-first-frame",
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()

        pose = runner.resolve_so101_initial_pose(args)
        self.assertEqual(pose["mode"], "dataset-start")
        self.assertEqual(pose["source"], "sim_2.parquet:frame-5-median")

    def test_configure_initial_pose_updates_only_env_cfg_instance_and_rerenders(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--deployment-mode",
            "so101-finetuned",
            "--so101-checkpoint-joint-units",
            "degrees",
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()

        authored_joint_pos = {name: 0.0 for name in runner.SO101_JOINT_NAMES}
        env_cfg = SimpleNamespace(
            scene=SimpleNamespace(
                robot=SimpleNamespace(
                    init_state=SimpleNamespace(joint_pos=authored_joint_pos),
                )
            ),
            rerender_on_reset=False,
        )
        pose = runner.configure_so101_initial_pose(env_cfg, args)

        self.assertIsNotNone(pose)
        self.assertTrue(env_cfg.rerender_on_reset)
        self.assertIsNot(env_cfg.scene.robot.init_state.joint_pos, authored_joint_pos)
        self.assertEqual(authored_joint_pos, {name: 0.0 for name in runner.SO101_JOINT_NAMES})
        np.testing.assert_allclose(
            [env_cfg.scene.robot.init_state.joint_pos[name] for name in runner.SO101_JOINT_NAMES],
            pose["joint_rad"],
            atol=1e-7,
        )

    def test_usd_default_initial_pose_does_not_modify_env_cfg(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--deployment-mode",
            "so101-finetuned",
            "--so101-initial-pose",
            "usd-default",
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()

        authored_joint_pos = {name: 0.0 for name in runner.SO101_JOINT_NAMES}
        env_cfg = SimpleNamespace(
            scene=SimpleNamespace(
                robot=SimpleNamespace(
                    init_state=SimpleNamespace(joint_pos=authored_joint_pos),
                )
            ),
            rerender_on_reset=False,
        )

        self.assertIsNone(runner.configure_so101_initial_pose(env_cfg, args))
        self.assertIs(env_cfg.scene.robot.init_state.joint_pos, authored_joint_pos)
        self.assertFalse(env_cfg.rerender_on_reset)

    def test_finetuned_joint_actions_disable_default_pose_offsets(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--deployment-mode",
            "so101-finetuned",
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()

        env_cfg = SimpleNamespace(
            actions=SimpleNamespace(
                arm_action=SimpleNamespace(use_default_offset=True),
                gripper_action=SimpleNamespace(use_default_offset=True),
            )
        )
        self.assertTrue(runner.configure_so101_absolute_joint_actions(env_cfg, args))
        self.assertFalse(env_cfg.actions.arm_action.use_default_offset)
        self.assertFalse(env_cfg.actions.gripper_action.use_default_offset)

        with patch.object(sys, "argv", ["run_smart_task_closed_loop.py"]):
            zero_shot_args = runner.parse_args()
        zero_shot_cfg = SimpleNamespace(
            actions=SimpleNamespace(
                arm_action=SimpleNamespace(use_default_offset=True),
                gripper_action=SimpleNamespace(use_default_offset=True),
            )
        )
        self.assertFalse(
            runner.configure_so101_absolute_joint_actions(zero_shot_cfg, zero_shot_args)
        )
        self.assertTrue(zero_shot_cfg.actions.arm_action.use_default_offset)
        self.assertTrue(zero_shot_cfg.actions.gripper_action.use_default_offset)

    def test_runtime_joint_action_offsets_must_stay_zero(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--deployment-mode",
            "so101-finetuned",
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()

        terms = {
            "arm_action": SimpleNamespace(
                cfg=SimpleNamespace(use_default_offset=False),
                _offset=torch.zeros((1, 5)),
            ),
            "gripper_action": SimpleNamespace(
                cfg=SimpleNamespace(use_default_offset=False),
                _offset=torch.zeros((1, 1)),
            ),
        }
        env = SimpleNamespace(
            action_manager=SimpleNamespace(get_term=lambda name: terms[name]),
        )
        runner.validate_so101_absolute_joint_actions(env, args)

        terms["arm_action"]._offset[0, 0] = 0.25
        with self.assertRaisesRegex(RuntimeError, "runtime offset must be zero"):
            runner.validate_so101_absolute_joint_actions(env, args)

    def test_initial_pose_verification_fails_closed_on_reset_mismatch(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--deployment-mode",
            "so101-finetuned",
            "--so101-checkpoint-joint-units",
            "degrees",
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()

        configured_pose = runner.resolve_so101_initial_pose(args)
        policy_obs = {"joint_pos": torch.zeros((1, 6), dtype=torch.float32)}
        runtime_limits = torch.from_numpy(runner.SO101_USD_JOINT_LIMITS_RAD.copy())
        with self.assertRaisesRegex(RuntimeError, "did not reach the configured initial pose"):
            runner.log_so101_initial_joint_state(
                policy_obs,
                args,
                runtime_limits,
                configured_pose,
            )

    def test_custom_initial_dataset_state_overrides_unit_preset(self) -> None:
        custom_state = ["1", "2", "3", "4", "5", "6"]
        argv = [
            "run_smart_task_closed_loop.py",
            "--deployment-mode",
            "so101-finetuned",
            "--so101-checkpoint-joint-units",
            "degrees",
            "--so101-initial-dataset-state",
            *custom_state,
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()

        pose = runner.resolve_so101_initial_pose(args)
        self.assertEqual(pose["source"], "cli:--so101-initial-dataset-state")
        np.testing.assert_allclose(pose["dataset_state"], [1, 2, 3, 4, 5, 6])

    def test_initial_dataset_state_rejects_wrong_route_conflict_and_limits(self) -> None:
        wrong_route = [
            "run_smart_task_closed_loop.py",
            "--so101-initial-dataset-state",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
        ]
        with patch.object(sys, "argv", wrong_route), self.assertRaisesRegex(
            ValueError,
            "requires --deployment-mode so101-finetuned",
        ):
            runner.parse_args()

        conflict = [
            "run_smart_task_closed_loop.py",
            "--deployment-mode",
            "so101-finetuned",
            "--so101-initial-pose",
            "usd-default",
            "--so101-initial-dataset-state",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
        ]
        with patch.object(sys, "argv", conflict), self.assertRaisesRegex(
            ValueError,
            "cannot be combined",
        ):
            runner.parse_args()

        outside_limits = [
            "run_smart_task_closed_loop.py",
            "--deployment-mode",
            "so101-finetuned",
            "--so101-initial-dataset-state",
            "500",
            "0",
            "0",
            "0",
            "0",
            "0",
        ]
        with patch.object(sys, "argv", outside_limits):
            args = runner.parse_args()
        with self.assertRaisesRegex(ValueError, "outside the converter's USD joint limits"):
            runner.resolve_so101_initial_pose(args)

    def test_cli_rejects_degree_units_for_zero_shot(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--so101-checkpoint-joint-units",
            "degrees",
        ]
        with patch.object(sys, "argv", argv), self.assertRaisesRegex(
            ValueError,
            "requires --deployment-mode so101-finetuned",
        ):
            runner.parse_args()

    def test_bridge_contract_is_unit_agnostic(self) -> None:
        args = SimpleNamespace(
            policy_schema="so101-new-embodiment",
            camera_layout="dual",
        )
        runner.validate_bridge_camera_layout(args, _finetuned_ping())

    def test_cli_accepts_explicit_isaac_camera_keys(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--isaac-front-camera-key",
            "live_top",
            "--isaac-left-camera-key",
            "live_left",
            "--isaac-wrist-camera-key",
            "live_wrist",
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()
        self.assertEqual(args.isaac_front_camera_key, "live_top")
        self.assertEqual(args.isaac_left_camera_key, "live_left")
        self.assertEqual(args.isaac_wrist_camera_key, "live_wrist")

    def test_camera_mapping_accepts_distinct_live_keys(self) -> None:
        policy_obs = {
            "camera2": torch.zeros((1, 2, 2, 3), dtype=torch.uint8),
            "camera3": torch.zeros((1, 2, 2, 3), dtype=torch.uint8),
        }
        runner.validate_camera_mapping(
            policy_obs,
            {"top": "camera3", "wrist": "camera2"},
        )

    def test_wrist_only_camera_mapping_accepts_one_live_key(self) -> None:
        policy_obs = {"camera2": torch.zeros((1, 2, 2, 3), dtype=torch.uint8)}
        runner.validate_camera_mapping(policy_obs, {"wrist": "camera2"})

    def test_dual_camera_mapping_rejects_reused_live_key(self) -> None:
        policy_obs = {"camera1": torch.zeros((1, 2, 2, 3), dtype=torch.uint8)}
        with self.assertRaisesRegex(ValueError, "different Isaac policy observation key"):
            runner.validate_camera_mapping(
                policy_obs,
                {"top": "camera1", "wrist": "camera1"},
            )

    def test_triple_camera_mapping_rejects_any_reused_live_key(self) -> None:
        policy_obs = {
            "camera1": torch.zeros((1, 2, 2, 3), dtype=torch.uint8),
            "camera2": torch.zeros((1, 2, 2, 3), dtype=torch.uint8),
        }
        with self.assertRaisesRegex(ValueError, "different Isaac policy observation key"):
            runner.validate_camera_mapping(
                policy_obs,
                {"top": "camera1", "left": "camera1", "wrist": "camera2"},
            )

    def test_runner_rejects_bridge_that_relabels_dataset_coordinates(self) -> None:
        ping = _finetuned_ping()
        ping["action_decoding"]["dataset_action_units"] = "radians"
        args = SimpleNamespace(
            policy_schema="so101-new-embodiment",
            camera_layout="dual",
        )
        with self.assertRaisesRegex(ValueError, "executable action contract"):
            runner.validate_bridge_camera_layout(args, ping)

    def test_degree_observation_wiring_converts_arm_but_not_gripper_as_degrees(self) -> None:
        joint_rad = torch.tensor(
            [[0.1, -0.2, 0.3, 0.4, -0.5, np.deg2rad(45.0)]],
            dtype=torch.float32,
        )
        policy_obs = {
            "joint_pos": joint_rad,
            "camera3": torch.zeros((1, 2, 2, 3), dtype=torch.uint8),
            "camera2": torch.zeros((1, 2, 2, 3), dtype=torch.uint8),
        }

        observation = runner.build_so101_new_embodiment_observation(
            policy_obs,
            runner.FrameHistory(horizon=2),
            "pick the block",
            "so101",
            {"top": "camera3", "wrist": "camera2"},
            "dual",
            checkpoint_arm_units="degrees",
        )

        np.testing.assert_allclose(
            observation["state"]["single_arm"][0, 0],
            np.rad2deg(joint_rad.numpy()[0, :5]),
            atol=2e-5,
        )
        # The SO101 gripper maps -10..100 USD degrees to the dataset's 0..100
        # range, so 45 degrees is 50 dataset units.  It is not reported as 45.
        self.assertAlmostEqual(float(observation["state"]["gripper"][0, 0, 0]), 50.0, places=5)

    def test_degree_action_wiring_converts_arm_and_range_mapped_gripper(self) -> None:
        fallback_joint = torch.zeros((1, 6), dtype=torch.float32)
        action = {
            "single_arm": np.asarray([[[10.0, -20.0, 30.0, 40.0, -50.0]]], dtype=np.float32),
            "gripper": np.asarray([[[50.0]]], dtype=np.float32),
        }

        command = runner.so101_new_embodiment_action_to_leisaac_tensor(
            action,
            "cpu",
            fallback_joint,
            max_arm_step_rad=None,
            checkpoint_arm_units="degrees",
        ).numpy()[0]

        np.testing.assert_allclose(command[:5], np.deg2rad(action["single_arm"][0, 0]), atol=2e-6)
        self.assertAlmostEqual(float(command[5]), float(np.deg2rad(45.0)), places=6)

    def test_degree_action_without_arm_holds_current_arm(self) -> None:
        fallback_joint = torch.tensor(
            [[0.1, -0.2, 0.3, 0.4, -0.5, 0.25]],
            dtype=torch.float32,
        )

        command = runner.so101_new_embodiment_action_to_leisaac_tensor(
            {"gripper": np.asarray([[[50.0]]], dtype=np.float32)},
            "cpu",
            fallback_joint,
            max_arm_step_rad=None,
            checkpoint_arm_units="degrees",
        ).numpy()[0]

        np.testing.assert_allclose(command[:5], fallback_joint.numpy()[0, :5], atol=2e-6)

if __name__ == "__main__":
    unittest.main()
