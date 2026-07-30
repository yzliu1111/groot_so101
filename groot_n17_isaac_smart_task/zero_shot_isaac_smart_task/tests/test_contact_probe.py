from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import sys
import unittest

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import contact_probe  # noqa: E402


class _FakeContactSensorCfg:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeContactView:
    sensor_count = 1
    filter_count = 1

    def __init__(self, buffers):
        self.buffers = buffers
        self.last_dt = None

    def get_contact_data(self, dt):
        self.last_dt = dt
        return self.buffers


class ContactProbeConfigTest(unittest.TestCase):
    def test_disabled_probe_is_a_true_no_op_without_isaac_import(self) -> None:
        env_cfg = SimpleNamespace(
            scene=SimpleNamespace(
                robot=SimpleNamespace(
                    spawn=SimpleNamespace(activate_contact_sensors=False)
                )
            )
        )

        with patch.dict(sys.modules, {"isaaclab": None, "isaaclab.sensors": None}):
            installed = contact_probe.configure_so101_contact_probe(
                env_cfg,
                "{ENV_REGEX_NS}/Scene/target",
                enabled=False,
            )

        self.assertEqual(installed, {})
        self.assertFalse(env_cfg.scene.robot.spawn.activate_contact_sensors)
        self.assertFalse(
            hasattr(env_cfg.scene, contact_probe.GRIPPER_CONTACT_SENSOR_NAME)
        )

    def test_enabled_probe_installs_one_filtered_sensor_per_body(self) -> None:
        env_cfg = SimpleNamespace(
            scene=SimpleNamespace(
                robot=SimpleNamespace(
                    spawn=SimpleNamespace(activate_contact_sensors=False)
                )
            )
        )
        fake_isaaclab = ModuleType("isaaclab")
        fake_isaaclab.__path__ = []  # type: ignore[attr-defined]
        fake_sensors = ModuleType("isaaclab.sensors")
        fake_sensors.ContactSensorCfg = _FakeContactSensorCfg  # type: ignore[attr-defined]
        target_path = "{ENV_REGEX_NS}/Scene/red_2x4_lego_brick_managed"

        with patch.dict(
            sys.modules,
            {
                "isaaclab": fake_isaaclab,
                "isaaclab.sensors": fake_sensors,
            },
        ):
            installed = contact_probe.configure_so101_contact_probe(
                env_cfg,
                target_path,
                enabled=True,
                max_contact_data_count_per_prim=12,
            )

        self.assertTrue(env_cfg.scene.robot.spawn.activate_contact_sensors)
        self.assertEqual(
            installed,
            {
                "gripper": contact_probe.GRIPPER_CONTACT_SENSOR_NAME,
                "jaw": contact_probe.JAW_CONTACT_SENSOR_NAME,
            },
        )

        for body_name, sensor_name, body_path in contact_probe.SO101_CONTACT_SENSOR_SPECS:
            with self.subTest(body_name=body_name):
                cfg = getattr(env_cfg.scene, sensor_name)
                self.assertIsInstance(cfg, _FakeContactSensorCfg)
                self.assertEqual(cfg.kwargs["prim_path"], body_path)
                self.assertEqual(
                    cfg.kwargs["filter_prim_paths_expr"],
                    [target_path],
                )
                self.assertEqual(
                    cfg.kwargs["max_contact_data_count_per_prim"],
                    12,
                )
                self.assertTrue(cfg.kwargs["track_contact_points"])
                self.assertFalse(cfg.kwargs["track_pose"])
                self.assertEqual(cfg.kwargs["update_period"], 0.0)

    def test_runner_config_api_uses_target_cfg_prim_path_and_returns_names(self) -> None:
        target_path = "{ENV_REGEX_NS}/Scene/task_owned_target"
        env_cfg = SimpleNamespace(
            scene=SimpleNamespace(
                robot=SimpleNamespace(
                    spawn=SimpleNamespace(activate_contact_sensors=False)
                ),
                target=SimpleNamespace(prim_path=target_path),
            )
        )
        fake_isaaclab = ModuleType("isaaclab")
        fake_isaaclab.__path__ = []  # type: ignore[attr-defined]
        fake_sensors = ModuleType("isaaclab.sensors")
        fake_sensors.ContactSensorCfg = _FakeContactSensorCfg  # type: ignore[attr-defined]

        with patch.dict(
            sys.modules,
            {
                "isaaclab": fake_isaaclab,
                "isaaclab.sensors": fake_sensors,
            },
        ):
            sensor_names = contact_probe.configure_so101_target_contact_probe(
                env_cfg,
                "target",
            )

        self.assertEqual(
            sensor_names,
            (
                contact_probe.GRIPPER_CONTACT_SENSOR_NAME,
                contact_probe.JAW_CONTACT_SENSOR_NAME,
            ),
        )
        for sensor_name in sensor_names:
            self.assertEqual(
                getattr(env_cfg.scene, sensor_name).kwargs["filter_prim_paths_expr"],
                [target_path],
            )


class ContactProbeRuntimeTest(unittest.TestCase):
    def test_reads_only_active_contact_patches_and_derives_penetration(self) -> None:
        view = _FakeContactView(
            (
                torch.tensor([[99.0], [3.0], [5.0], [88.0]]),
                torch.tensor(
                    [
                        [9.0, 9.0, 9.0],
                        [0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6],
                        [8.0, 8.0, 8.0],
                    ]
                ),
                torch.tensor(
                    [
                        [9.0, 9.0, 9.0],
                        [1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [8.0, 8.0, 8.0],
                    ]
                ),
                torch.tensor([[9.0], [-0.002], [0.0005], [8.0]]),
                torch.tensor([[2]], dtype=torch.int32),
                torch.tensor([[1]], dtype=torch.int32),
            )
        )
        sensor = SimpleNamespace(contact_physx_view=view)

        result = contact_probe.read_contact_sensor(sensor, 1.0 / 60.0)

        self.assertAlmostEqual(view.last_dt, 1.0 / 60.0)
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["forces_n"], [3.0, 5.0])
        self.assertEqual(
            result["points_w_m"],
            [
                [0.10000000149011612, 0.20000000298023224, 0.30000001192092896],
                [0.4000000059604645, 0.5, 0.6000000238418579],
            ],
        )
        self.assertEqual(
            result["force_vectors_w_n"],
            [[3.0, 0.0, 0.0], [0.0, -5.0, 0.0]],
        )
        self.assertAlmostEqual(result["min_separation_m"], -0.002, places=7)
        self.assertAlmostEqual(result["max_penetration_m"], 0.002, places=7)
        self.assertEqual(result["pair_counts"], [[2]])
        json.dumps(result, allow_nan=False)

    def test_no_contact_is_empty_and_json_safe(self) -> None:
        view = _FakeContactView(
            (
                torch.zeros((2, 1)),
                torch.zeros((2, 3)),
                torch.zeros((2, 3)),
                torch.zeros((2, 1)),
                torch.tensor([[0]], dtype=torch.int32),
                torch.tensor([[0]], dtype=torch.int32),
            )
        )
        result = contact_probe.read_contact_sensor(
            SimpleNamespace(contact_physx_view=view),
            0.01,
        )

        self.assertEqual(result["count"], 0)
        self.assertEqual(result["forces_n"], [])
        self.assertEqual(result["points_w_m"], [])
        self.assertEqual(result["separations_m"], [])
        self.assertIsNone(result["min_separation_m"])
        self.assertEqual(result["max_penetration_m"], 0.0)
        json.dumps(result, allow_nan=False)

    def test_tip_transform_is_pure_testable(self) -> None:
        robot = SimpleNamespace(
            data=SimpleNamespace(
                body_names=["base", "gripper", "jaw"],
                body_pos_w=torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0],
                            [1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                        ]
                    ],
                    dtype=torch.float64,
                ),
                body_quat_w=torch.tensor(
                    [
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ],
                    dtype=torch.float64,
                ),
            )
        )
        result = contact_probe.read_so101_tip_points(robot)

        fixed_local = contact_probe.SO101_TIP_LOCAL_OFFSETS_M["gripper"]
        self.assertEqual(
            result["gripper"]["tip_pos_w_m"],
            [
                1.0 + fixed_local[0],
                2.0 + fixed_local[1],
                3.0 + fixed_local[2],
            ],
        )
        moving_local = contact_probe.SO101_TIP_LOCAL_OFFSETS_M["jaw"]
        # 180 degrees about world Z negates local X/Y and preserves Z.
        self.assertEqual(
            result["jaw"]["tip_pos_w_m"],
            [
                4.0 - moving_local[0],
                5.0 - moving_local[1],
                6.0 + moving_local[2],
            ],
        )
        json.dumps(result, allow_nan=False)

    def test_runner_read_api_aggregates_selected_sensors(self) -> None:
        first_buffers = (
            torch.tensor([[2.0]]),
            torch.tensor([[0.1, 0.2, 0.3]]),
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([[-0.001]]),
            torch.tensor([[1]], dtype=torch.int32),
            torch.tensor([[0]], dtype=torch.int32),
        )
        second_buffers = (
            torch.tensor([[4.0]]),
            torch.tensor([[0.4, 0.5, 0.6]]),
            torch.tensor([[0.0, 1.0, 0.0]]),
            torch.tensor([[-0.003]]),
            torch.tensor([[1]], dtype=torch.int32),
            torch.tensor([[0]], dtype=torch.int32),
        )
        sensor_names = ("sensor_a", "sensor_b")
        env = SimpleNamespace(
            physics_dt=1.0 / 60.0,
            scene={
                sensor_names[0]: SimpleNamespace(
                    contact_physx_view=_FakeContactView(first_buffers)
                ),
                sensor_names[1]: SimpleNamespace(
                    contact_physx_view=_FakeContactView(second_buffers)
                ),
            },
        )

        result = contact_probe.read_contact_probe(env, sensor_names)

        self.assertEqual(result["sensor_names"], list(sensor_names))
        self.assertEqual(result["total_contact_count"], 2)
        self.assertAlmostEqual(result["min_separation_m"], -0.003, places=7)
        self.assertAlmostEqual(result["max_penetration_m"], 0.003, places=7)
        self.assertEqual(set(result["sensors"]), set(sensor_names))
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
