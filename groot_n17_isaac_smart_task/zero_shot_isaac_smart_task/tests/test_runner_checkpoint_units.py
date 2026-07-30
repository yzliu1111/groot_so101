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


class _FakeUsdPath:
    def __init__(self, path: str):
        self.pathString = path

    def __str__(self) -> str:
        return self.pathString


class _FakeUsdAttribute:
    def __init__(self, value=None, *, valid: bool = True):
        self.value = value
        self.valid = valid
        self.set_calls: list[tuple[float, float, float]] = []

    def IsValid(self) -> bool:
        return self.valid

    def Get(self):
        return self.value

    def Set(self, value) -> bool:
        self.value = tuple(float(component) for component in value)
        self.set_calls.append(self.value)
        return True


class _FakeUsdPrim:
    def __init__(
        self,
        path: str,
        diffuse_attr: _FakeUsdAttribute,
        *,
        type_name: str = "Shader",
    ):
        self.path = _FakeUsdPath(path)
        self.diffuse_attr = diffuse_attr
        self.type_name = type_name

    def GetPath(self) -> _FakeUsdPath:
        return self.path

    def GetTypeName(self) -> str:
        return self.type_name

    def GetAttribute(self, name: str) -> _FakeUsdAttribute:
        if name == runner.SO101_PRINTED_MATERIAL_DIFFUSE_INPUT:
            return self.diffuse_attr
        return _FakeUsdAttribute(valid=False)


class _FakeUsdStage:
    def __init__(self, prims: list[_FakeUsdPrim]):
        self.prims = prims

    def Traverse(self):
        return iter(self.prims)


class RunnerCheckpointUnitsTest(unittest.TestCase):
    def test_cli_defaults_to_base_so101_task(self) -> None:
        with patch.object(sys, "argv", ["run_smart_task_closed_loop.py"]):
            args = runner.parse_args()
        self.assertEqual(args.task, "LeIsaac-SO101-SmartTask-v0")
        self.assertEqual(args.scene_profile, "task-default")
        self.assertEqual(args.red24_material_resolved, "asset")
        self.assertEqual(args.so101_robot_material, "auto")
        self.assertEqual(args.so101_robot_material_resolved, "asset")

    def test_real_degree_checkpoint_auto_selects_pure_red_2x4_material(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--deployment-mode",
            "so101-finetuned",
            "--so101-checkpoint-joint-units",
            "degrees",
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()
        self.assertEqual(args.red24_material, "auto")
        self.assertEqual(args.red24_material_resolved, "real-red")

    def test_real_degree_checkpoint_auto_selects_white_so101_material(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--deployment-mode",
            "so101-finetuned",
            "--so101-checkpoint-joint-units",
            "degrees",
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()

        self.assertEqual(args.so101_robot_material, "auto")
        self.assertEqual(args.so101_robot_material_resolved, "real-white")

    def test_explicit_asset_so101_material_overrides_real_auto_selection(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--deployment-mode",
            "so101-finetuned",
            "--so101-checkpoint-joint-units",
            "degrees",
            "--so101-robot-material",
            "asset",
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()

        self.assertEqual(args.so101_robot_material_resolved, "asset")

    def test_real_white_so101_material_rejects_non_so101_robot(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--robot",
            "franka",
            "--so101-robot-material",
            "real-white",
        ]
        with patch.object(sys, "argv", argv), self.assertRaisesRegex(
            ValueError,
            "requires --robot so101",
        ):
            runner.parse_args()

    def test_explicit_asset_material_overrides_real_auto_selection(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--deployment-mode",
            "so101-finetuned",
            "--so101-checkpoint-joint-units",
            "degrees",
            "--red24-material",
            "asset",
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()
        self.assertEqual(args.red24_material_resolved, "asset")

    def test_sim_checkpoint_can_explicitly_request_real_red_material(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--red24-material",
            "real-red",
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()
        self.assertEqual(args.red24_material_resolved, "real-red")

    def test_real_white_changes_only_so101_printed_material_shader(self) -> None:
        printed_path = (
            "/World/envs/env_0/Robot/Looks/material_a_3d_printed/Shader"
        )
        motor_path = "/World/envs/env_0/Robot/Looks/material_sts3215/Shader"
        printed_attr = _FakeUsdAttribute((1.0, 0.82, 0.12))
        motor_attr = _FakeUsdAttribute((0.1, 0.1, 0.1))
        stage = _FakeUsdStage(
            [
                _FakeUsdPrim(printed_path, printed_attr),
                _FakeUsdPrim(motor_path, motor_attr),
            ]
        )

        env = SimpleNamespace(
            num_envs=1,
            cfg=SimpleNamespace(rerender_on_reset=False),
        )
        summary = runner.apply_so101_robot_material(
            env,
            "real-white",
            requested_material="auto",
            stage=stage,
        )

        self.assertEqual(printed_attr.value, (1.0, 1.0, 1.0))
        self.assertEqual(printed_attr.set_calls, [(1.0, 1.0, 1.0)])
        self.assertEqual(motor_attr.value, (0.1, 0.1, 0.1))
        self.assertEqual(motor_attr.set_calls, [])
        self.assertTrue(summary["applied"])
        self.assertTrue(summary["changed"])
        self.assertTrue(summary["rerender_on_reset"])
        self.assertTrue(env.cfg.rerender_on_reset)
        self.assertEqual(summary["matched_shader_paths"], [printed_path])
        self.assertEqual(
            summary["diffuse_color_before"][printed_path],
            [1.0, 0.82, 0.12],
        )
        self.assertEqual(
            summary["diffuse_color_after"][printed_path],
            [1.0, 1.0, 1.0],
        )

    def test_real_white_fails_before_editing_on_abnormal_shader_count(self) -> None:
        first_attr = _FakeUsdAttribute((1.0, 0.82, 0.12))
        second_attr = _FakeUsdAttribute((1.0, 0.82, 0.12))
        stage = _FakeUsdStage(
            [
                _FakeUsdPrim(
                    "/World/envs/env_0/Robot/Looks/material_a_3d_printed/Shader",
                    first_attr,
                ),
                _FakeUsdPrim(
                    "/World/envs/env_1/Robot/Looks/material_a_3d_printed/Shader",
                    second_attr,
                ),
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "expected=1 matched=2"):
            runner.apply_so101_robot_material(
                SimpleNamespace(
                    num_envs=1,
                    cfg=SimpleNamespace(rerender_on_reset=False),
                ),
                "real-white",
                stage=stage,
            )

        self.assertEqual(first_attr.set_calls, [])
        self.assertEqual(second_attr.set_calls, [])

    def test_real_white_fails_on_missing_shader_or_omniverse_input(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "expected=1 matched=0"):
            runner.apply_so101_robot_material(
                SimpleNamespace(
                    num_envs=1,
                    cfg=SimpleNamespace(rerender_on_reset=False),
                ),
                "real-white",
                stage=_FakeUsdStage([]),
            )

        invalid_attr = _FakeUsdAttribute(valid=False)
        stage = _FakeUsdStage(
            [
                _FakeUsdPrim(
                    "/World/envs/env_0/Robot/Looks/material_a_3d_printed/Shader",
                    invalid_attr,
                )
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "missing OmniPBR input"):
            runner.apply_so101_robot_material(
                SimpleNamespace(
                    num_envs=1,
                    cfg=SimpleNamespace(rerender_on_reset=False),
                ),
                "real-white",
                stage=stage,
            )
        self.assertEqual(invalid_attr.set_calls, [])

    def test_asset_so101_material_preserves_stage_without_lookup(self) -> None:
        with patch.object(
            runner,
            "get_usd_stage",
            side_effect=AssertionError("asset mode must not inspect the stage"),
        ):
            summary = runner.apply_so101_robot_material(
                SimpleNamespace(num_envs=1),
                "asset",
                requested_material="auto",
            )

        self.assertFalse(summary["applied"])
        self.assertEqual(summary["resolved"], "asset")

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

    def test_capture_camera_path_is_optional_and_preserved(self) -> None:
        with patch.object(sys, "argv", ["run_smart_task_closed_loop.py"]):
            default_args = runner.parse_args()
        camera_path = "/World/envs/env_0/Scene/camera_front_xform/camera_front"
        with patch.object(
            sys,
            "argv",
            ["run_smart_task_closed_loop.py", "--capture-camera-path", camera_path],
        ):
            explicit_args = runner.parse_args()

        self.assertIsNone(default_args.capture_camera_path)
        self.assertEqual(explicit_args.capture_camera_path, camera_path)

    def test_viewport_camera_path_is_human_only_and_does_not_enable_capture(self) -> None:
        camera_path = "/World/envs/env_0/Scene/camera_left_xform/camera_left"
        with patch.object(
            sys,
            "argv",
            [
                "run_smart_task_closed_loop.py",
                "--no-headless",
                "--viewport-camera-path",
                camera_path,
            ],
        ):
            args = runner.parse_args()

        self.assertEqual(args.viewport_camera_path, camera_path)
        self.assertFalse(args.capture_video)

    def test_viewport_camera_path_rejects_headless_and_video_capture(self) -> None:
        camera_path = "/World/envs/env_0/Scene/camera_left_xform/camera_left"
        with patch.object(
            sys,
            "argv",
            ["run_smart_task_closed_loop.py", "--viewport-camera-path", camera_path],
        ), self.assertRaisesRegex(ValueError, "requires --no-headless"):
            runner.parse_args()

        with patch.object(
            sys,
            "argv",
            [
                "run_smart_task_closed_loop.py",
                "--no-headless",
                "--viewport-camera-path",
                camera_path,
                "--capture-video",
            ],
        ), self.assertRaisesRegex(ValueError, "human-view-only"):
            runner.parse_args()

    def test_set_human_viewport_camera_validates_and_switches_camera(self) -> None:
        class FakePrim:
            def __init__(self, valid: bool, type_name: str):
                self._valid = valid
                self._type_name = type_name

            def IsValid(self) -> bool:
                return self._valid

            def GetTypeName(self) -> str:
                return self._type_name

        class FakeStage:
            def __init__(self, prim):
                self.prim = prim

            def GetPrimAtPath(self, _path: str):
                return self.prim

        class FakePath:
            def __init__(self, path: str):
                self.pathString = path

        class FakeViewport:
            def __init__(self):
                self.camera_path = FakePath("/OmniverseKit_Persp")

            def set_active_camera(self, path: str) -> None:
                self.camera_path = FakePath(path)

        camera_path = "/World/envs/env_0/Scene/camera_left_xform/camera_left"
        viewport = FakeViewport()
        previous_path = runner.set_human_viewport_camera(
            camera_path,
            stage=FakeStage(FakePrim(True, "Camera")),
            viewport=viewport,
        )

        self.assertEqual(previous_path, "/OmniverseKit_Persp")
        self.assertEqual(viewport.camera_path.pathString, camera_path)

        with self.assertRaisesRegex(ValueError, "live USD prim"):
            runner.set_human_viewport_camera(
                camera_path,
                stage=FakeStage(FakePrim(False, "Camera")),
                viewport=viewport,
            )
        with self.assertRaisesRegex(ValueError, "USD Camera prim"):
            runner.set_human_viewport_camera(
                camera_path,
                stage=FakeStage(FakePrim(True, "Xform")),
                viewport=viewport,
            )

    def test_dynamic_gripper_effort_override_is_optional_and_can_be_disabled(self) -> None:
        with patch.object(sys, "argv", ["run_smart_task_closed_loop.py"]):
            default_args = runner.parse_args()
        with patch.object(
            sys,
            "argv",
            [
                "run_smart_task_closed_loop.py",
                "--no-dynamic-reset-gripper-effort-limit",
            ],
        ):
            disabled_args = runner.parse_args()

        self.assertIsNone(default_args.dynamic_reset_gripper_effort_limit)
        self.assertIs(disabled_args.dynamic_reset_gripper_effort_limit, False)

    def test_explicit_gripper_control_limits_parse_on_finetuned_route(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--deployment-mode",
            "so101-finetuned",
            "--so101-max-gripper-step-rad",
            "0.04",
            "--so101-gripper-effort-limit-sim",
            "0.1",
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()

        self.assertAlmostEqual(args.so101_max_gripper_step_rad, 0.04)
        self.assertAlmostEqual(args.so101_gripper_effort_limit_sim, 0.1)

    def test_explicit_gripper_effort_rejects_dynamic_reset(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--deployment-mode",
            "so101-finetuned",
            "--so101-gripper-effort-limit-sim",
            "0.1",
            "--dynamic-reset-gripper-effort-limit",
        ]
        with patch.object(sys, "argv", argv), self.assertRaisesRegex(
            ValueError,
            "cannot be combined",
        ):
            runner.parse_args()

    def test_explicit_gripper_control_limits_require_finetuned_route(self) -> None:
        argv = [
            "run_smart_task_closed_loop.py",
            "--so101-max-gripper-step-rad",
            "0.04",
        ]
        with patch.object(sys, "argv", argv), self.assertRaisesRegex(
            ValueError,
            "require the finetuned SO101 joint route",
        ):
            runner.parse_args()

    def test_configure_and_validate_explicit_gripper_effort_limit(self) -> None:
        actuator_cfg = SimpleNamespace(effort_limit_sim=10.0)
        env_cfg = SimpleNamespace(
            scene=SimpleNamespace(
                robot=SimpleNamespace(
                    actuators={"sts3215-gripper": actuator_cfg},
                )
            ),
            dynamic_reset_gripper_effort_limit=True,
        )
        args = SimpleNamespace(
            deployment_mode="so101-finetuned",
            robot="so101",
            control_mode="joint",
            policy_schema="so101-new-embodiment",
            so101_gripper_effort_limit_sim=0.1,
        )

        summary = runner.configure_so101_gripper_effort_limit(env_cfg, args)

        self.assertEqual(
            summary,
            {
                "previous_effort_limit_sim": 10.0,
                "requested_effort_limit_sim": 0.1,
            },
        )
        self.assertAlmostEqual(actuator_cfg.effort_limit_sim, 0.1)
        self.assertFalse(env_cfg.dynamic_reset_gripper_effort_limit)

        live_limits = torch.full((1, 6), 10.0, dtype=torch.float32)
        live_limits[0, 5] = 0.1
        env = SimpleNamespace(
            cfg=SimpleNamespace(dynamic_reset_gripper_effort_limit=False),
            scene={
                "robot": SimpleNamespace(
                    data=SimpleNamespace(
                        joint_names=[
                            "shoulder_pan",
                            "shoulder_lift",
                            "elbow_flex",
                            "wrist_flex",
                            "wrist_roll",
                            "gripper",
                        ],
                        joint_effort_limits=live_limits,
                    ),
                )
            },
        )
        self.assertAlmostEqual(
            runner.validate_so101_gripper_effort_limit(env, args),
            0.1,
        )

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
            "multi-lego-tray-raw-a",
            "--target-object-key",
            "red_2x2_lego_brick",
            "--instruction",
            "Pick up the small red block and place it in the tray.",
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()
        self.assertEqual(args.scene_profile, "multi-lego-tray-raw-a")
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

    def test_task_owned_default_scene_rejects_unverifiable_real_red_override(self) -> None:
        args = SimpleNamespace(
            scene_profile=runner.scene_profiles.TASK_DEFAULT_SCENE_PROFILE,
            task="LeIsaac-SO101-SmartTask-Blue-v0",
            smart_scene_usd="auto",
            smart_target_asset="auto",
            smart_target_pos=None,
            smart_target_prim_path=runner.SMART_TARGET_MANAGED_PRIM_PATH,
            smart_target_cuboid_size=runner.SMART_TARGET_CUBOID_SIZE,
            red24_material_resolved=runner.scene_profiles.RED24_MATERIAL_REAL_RED,
        )
        with self.assertRaisesRegex(ValueError, "task-owned task-default scene"):
            runner.install_smart_task_asset_patch(args)

    def test_task_default_cuboid_uses_resolved_red24_material(self) -> None:
        class FakeCfg:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class FakeRigidObjectCfg(FakeCfg):
            class InitialStateCfg(FakeCfg):
                pass

        fake_isaaclab = ModuleType("isaaclab")
        fake_isaaclab.__path__ = []  # type: ignore[attr-defined]
        fake_sim = ModuleType("isaaclab.sim")
        for cfg_name in (
            "CuboidCfg",
            "RigidBodyPropertiesCfg",
            "MassPropertiesCfg",
            "CollisionPropertiesCfg",
            "RigidBodyMaterialCfg",
            "PreviewSurfaceCfg",
        ):
            setattr(fake_sim, cfg_name, FakeCfg)
        fake_assets = ModuleType("isaaclab.assets")
        fake_assets.RigidObjectCfg = FakeRigidObjectCfg
        fake_isaaclab.sim = fake_sim  # type: ignore[attr-defined]

        def build_env(material: str) -> SimpleNamespace:
            env_cfg = SimpleNamespace(scene=SimpleNamespace())
            with patch.dict(
                sys.modules,
                {
                    "isaaclab": fake_isaaclab,
                    "isaaclab.sim": fake_sim,
                    "isaaclab.assets": fake_assets,
                },
            ):
                runner.add_smart_target_cfg(
                    env_cfg,
                    "cuboid",
                    runner.SMART_TARGET_MANAGED_PRIM_PATH,
                    Path("/unused/scene.usd"),
                    (0.0, 0.25, 0.07),
                    runner.SMART_TARGET_CUBOID_SIZE,
                    material,
                )
            return env_cfg

        asset_env = build_env(runner.scene_profiles.RED24_MATERIAL_ASSET)
        real_env = build_env(runner.scene_profiles.RED24_MATERIAL_REAL_RED)
        asset_material = getattr(
            asset_env.scene,
            runner.SMART_TARGET_OBJECT_KEY,
        ).spawn.visual_material
        real_material = getattr(
            real_env.scene,
            runner.SMART_TARGET_OBJECT_KEY,
        ).spawn.visual_material

        self.assertEqual(asset_material.diffuse_color, runner.SMART_TARGET_CUBOID_ASSET_COLOR)
        self.assertEqual(
            real_material.diffuse_color,
            runner.scene_profiles.REAL_RED24_DIFFUSE_COLOR,
        )
        self.assertEqual(real_material.roughness, runner.SMART_TARGET_CUBOID_ROUGHNESS)

    def test_scene_owned_target_rejects_real_red_instead_of_silent_noop(self) -> None:
        with self.assertRaisesRegex(ValueError, "runner does not own"):
            runner.add_smart_target_cfg(
                SimpleNamespace(scene=SimpleNamespace()),
                "scene",
                runner.SMART_TARGET_MANAGED_PRIM_PATH,
                Path("/unused/scene.usd"),
                None,
                runner.SMART_TARGET_CUBOID_SIZE,
                runner.scene_profiles.RED24_MATERIAL_REAL_RED,
            )

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
        pose = runner.resolve_so101_initial_pose(args)
        pose = runner.apply_so101_initial_pose(env_cfg, pose)

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

        pose = runner.resolve_so101_initial_pose(args)
        self.assertIsNone(runner.apply_so101_initial_pose(env_cfg, pose))
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

    def test_injected_camera_mapping_accepts_explicit_slot_permutation(self) -> None:
        args = SimpleNamespace(
            isaac_front_camera_key="camera2",
            isaac_left_camera_key="camera1",
            isaac_wrist_camera_key="camera3",
        )
        physical_mapping = {
            "top": "camera3",
            "left": "camera2",
            "wrist": "camera1",
        }

        selected = runner.apply_injected_camera_mapping(args, physical_mapping)

        self.assertEqual(
            selected,
            {"top": "camera2", "left": "camera1", "wrist": "camera3"},
        )

    def test_injected_camera_mapping_rejects_non_injected_key(self) -> None:
        args = SimpleNamespace(
            isaac_front_camera_key="camera4",
            isaac_left_camera_key=None,
            isaac_wrist_camera_key=None,
        )
        physical_mapping = {
            "top": "camera3",
            "left": "camera2",
            "wrist": "camera1",
        }

        with self.assertRaisesRegex(ValueError, "only exposes"):
            runner.apply_injected_camera_mapping(args, physical_mapping)

    def test_injected_camera_mapping_rejects_duplicate_slots(self) -> None:
        args = SimpleNamespace(
            isaac_front_camera_key="camera3",
            isaac_left_camera_key="camera3",
            isaac_wrist_camera_key="camera1",
        )
        physical_mapping = {
            "top": "camera3",
            "left": "camera2",
            "wrist": "camera1",
        }

        with self.assertRaisesRegex(ValueError, "must use distinct"):
            runner.apply_injected_camera_mapping(args, physical_mapping)

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

    def test_degree_action_records_gripper_step_diagnostics(self) -> None:
        fallback_joint = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.279]],
            dtype=torch.float32,
        )
        diagnostics: dict[str, object] = {}

        command = runner.so101_new_embodiment_action_to_leisaac_tensor(
            {"gripper": np.asarray([[[0.0]]], dtype=np.float32)},
            "cpu",
            fallback_joint,
            max_arm_step_rad=None,
            max_gripper_step_rad=0.04,
            checkpoint_arm_units="degrees",
            diagnostics_out=diagnostics,
        ).numpy()[0]

        self.assertAlmostEqual(float(command[5]), 0.239, places=6)
        self.assertTrue(diagnostics["gripper_step_clipped"])
        self.assertAlmostEqual(float(diagnostics["current_gripper_rad"]), 0.279, places=6)
        self.assertAlmostEqual(
            float(diagnostics["applied_gripper_delta_rad"]),
            -0.04,
            places=6,
        )
        self.assertLess(float(diagnostics["requested_gripper_delta_rad"]), -0.4)

if __name__ == "__main__":
    unittest.main()
