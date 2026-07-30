from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1]
RUNNER_PATH = SCRIPT_DIR / "run_smart_task_closed_loop.py"


def _load_runner_without_isaac_app():
    fake_isaaclab = ModuleType("isaaclab")
    fake_isaaclab.__path__ = []  # type: ignore[attr-defined]
    fake_app = ModuleType("isaaclab.app")
    fake_app.AppLauncher = object  # type: ignore[attr-defined]

    spec = importlib.util.spec_from_file_location("action_trace_metrics_test_module", RUNNER_PATH)
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


class ActionTraceMetricsTest(unittest.TestCase):
    def test_smart_task_metrics_preserves_legacy_fields_and_xyz_is_additive(self) -> None:
        lego = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[1.0, 2.0, 0.5]], dtype=torch.float32)
            )
        )
        robot = SimpleNamespace(
            data=SimpleNamespace(
                body_names=["base"],
                body_pos_w=torch.tensor([[[0.0, 0.0, 0.1]]], dtype=torch.float32),
                joint_pos=torch.tensor(
                    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.25]],
                    dtype=torch.float32,
                ),
            )
        )
        ee_frame = SimpleNamespace(
            data=SimpleNamespace(
                target_pos_w=torch.tensor(
                    [[[0.0, 0.0, 0.0], [0.7, 2.4, 0.8]]],
                    dtype=torch.float32,
                )
            )
        )
        env = SimpleNamespace(
            scene={
                "target": lego,
                "robot": robot,
                "ee_frame": ee_frame,
            }
        )

        metrics = runner.smart_task_metrics(env, "so101", "target")
        spatial_metrics = runner.smart_task_spatial_metrics(env, "target")

        self.assertEqual(
            set(metrics),
            {"lego_z_minus_base", "jaw_to_lego", "gripper"},
        )
        self.assertAlmostEqual(metrics["lego_z_minus_base"], 0.4, places=6)
        self.assertAlmostEqual(metrics["jaw_to_lego"], 0.5830952, places=6)
        self.assertAlmostEqual(metrics["gripper"], 0.25, places=6)
        self.assertEqual(spatial_metrics["lego_pos_w_m"], [1.0, 2.0, 0.5])
        self.assertAlmostEqual(spatial_metrics["jaw_pos_w_m"][0], 0.7, places=6)
        self.assertAlmostEqual(spatial_metrics["jaw_pos_w_m"][1], 2.4, places=6)
        self.assertAlmostEqual(spatial_metrics["jaw_pos_w_m"][2], 0.8, places=6)
        self.assertAlmostEqual(spatial_metrics["jaw_minus_lego_w_m"][0], -0.3, places=6)
        self.assertAlmostEqual(spatial_metrics["jaw_minus_lego_w_m"][1], 0.4, places=6)
        self.assertAlmostEqual(spatial_metrics["jaw_minus_lego_w_m"][2], 0.3, places=6)
        self.assertAlmostEqual(spatial_metrics["jaw_to_lego_xy_m"], 0.5, places=6)

    def test_settle_environment_steps_physics_while_holding_reset_joint_pose(self) -> None:
        lego = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[0.0, 0.25, 0.025]], dtype=torch.float32)
            )
        )
        robot = SimpleNamespace(
            data=SimpleNamespace(
                body_names=["base"],
                body_pos_w=torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32),
                joint_pos=torch.tensor(
                    [[0.1, -0.2, 0.3, 0.4, -0.5, -0.1]],
                    dtype=torch.float32,
                ),
            )
        )
        ee_frame = SimpleNamespace(
            data=SimpleNamespace(
                target_pos_w=torch.tensor(
                    [[[0.0, 0.0, 0.0], [0.0, 0.25, 0.04]]],
                    dtype=torch.float32,
                )
            )
        )

        class FakeEnv:
            def __init__(self) -> None:
                self.scene = {
                    "target": lego,
                    "robot": robot,
                    "ee_frame": ee_frame,
                }
                self.commands: list[torch.Tensor] = []

            def step(self, command):
                self.commands.append(command.detach().clone())
                lego.data.root_pos_w[0, 2] -= 0.01
                obs = {"policy": {"joint_pos": robot.data.joint_pos.detach().clone()}}
                return (
                    obs,
                    None,
                    torch.tensor([False]),
                    torch.tensor([False]),
                    {},
                )

        env = FakeEnv()
        initial_policy_obs = {"joint_pos": robot.data.joint_pos.detach().clone()}
        final_policy_obs, records = runner.settle_environment_at_reset_pose(
            env,
            initial_policy_obs,
            env_steps=2,
            robot_kind="so101",
            target_object_name="target",
        )

        self.assertEqual(len(env.commands), 2)
        self.assertEqual(len(records), 2)
        self.assertTrue(torch.equal(env.commands[0], initial_policy_obs["joint_pos"]))
        self.assertTrue(torch.equal(env.commands[1], initial_policy_obs["joint_pos"]))
        self.assertAlmostEqual(
            records[-1]["spatial_metrics_after"]["lego_pos_w_m"][2],
            0.005,
            places=6,
        )
        self.assertEqual(
            final_policy_obs["joint_pos"].tolist(),
            initial_policy_obs["joint_pos"].tolist(),
        )

    def test_restore_robot_frame0_does_not_advance_settled_target(self) -> None:
        reset_joint_pos = torch.tensor(
            [[0.1, -0.2, 0.3, 0.4, -0.5, -0.1]],
            dtype=torch.float32,
        )
        target = SimpleNamespace(
            data=SimpleNamespace(
                root_state_w=torch.tensor(
                    [[0.01, 0.25, -0.00017, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                    dtype=torch.float32,
                )
            )
        )

        class FakeRobot:
            def __init__(self) -> None:
                self.data = SimpleNamespace(
                    joint_pos=reset_joint_pos + 0.01,
                    joint_vel=torch.ones_like(reset_joint_pos),
                )
                self.velocity_target = None

            def write_joint_state_to_sim(self, position, velocity) -> None:
                self.data.joint_pos = position.detach().clone()
                self.data.joint_vel = velocity.detach().clone()

            def set_joint_velocity_target(self, velocity) -> None:
                self.velocity_target = velocity.detach().clone()

        class FakeActionManager:
            def __init__(self) -> None:
                self.processed = None
                self.applied = False

            def process_action(self, action) -> None:
                self.processed = action.detach().clone()

            def apply_action(self) -> None:
                self.applied = True

        class FakeSensor:
            def __init__(self) -> None:
                self.reset_count = 0

            def reset(self) -> None:
                self.reset_count += 1

        class FakeScene:
            def __init__(self, robot, sensor) -> None:
                self._entities = {"robot": robot, "target": target}
                self.sensors = {"camera": sensor}
                self.write_count = 0

            def __getitem__(self, key):
                return self._entities[key]

            def write_data_to_sim(self) -> None:
                self.write_count += 1

        class FakeSim:
            def __init__(self) -> None:
                self.forward_count = 0
                self.render_count = 0

            def forward(self) -> None:
                self.forward_count += 1

            def has_rtx_sensors(self) -> bool:
                return True

            def render(self) -> None:
                self.render_count += 1

        robot = FakeRobot()
        sensor = FakeSensor()
        scene = FakeScene(robot, sensor)
        action_manager = FakeActionManager()
        sim = FakeSim()
        observation_manager = SimpleNamespace(
            compute=lambda update_history: {
                "policy": {"joint_pos": robot.data.joint_pos.detach().clone()}
            }
        )
        env = SimpleNamespace(
            scene=scene,
            action_manager=action_manager,
            sim=sim,
            observation_manager=observation_manager,
        )

        policy_obs, record = runner.restore_robot_frame0_after_settle(
            env,
            reset_joint_pos,
            target_object_name="target",
        )

        self.assertTrue(torch.equal(policy_obs["joint_pos"], reset_joint_pos))
        self.assertTrue(torch.equal(action_manager.processed, reset_joint_pos))
        self.assertTrue(action_manager.applied)
        self.assertTrue(torch.equal(robot.velocity_target, torch.zeros_like(reset_joint_pos)))
        self.assertEqual(scene.write_count, 1)
        self.assertEqual(sensor.reset_count, 1)
        self.assertEqual(sim.forward_count, 1)
        self.assertEqual(sim.render_count, 1)
        self.assertEqual(record["joint_max_abs_error_rad"], 0.0)
        self.assertEqual(record["target_root_state_max_abs_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
