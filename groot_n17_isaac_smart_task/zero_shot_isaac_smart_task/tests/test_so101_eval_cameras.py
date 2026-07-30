from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import so101_eval_cameras as eval_cameras  # noqa: E402


class _FakeOffsetCfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeTiledCameraCfg:
    OffsetCfg = _FakeOffsetCfg

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakePinholeCameraCfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeSceneEntityCfg:
    def __init__(self, name: str):
        self.name = name


class _FakeObservationTermCfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


_IMAGE_FUNC = object()
_FAKE_DEPS = eval_cameras._CameraDependencies(
    sim_utils=SimpleNamespace(PinholeCameraCfg=_FakePinholeCameraCfg),
    tiled_camera_cfg=_FakeTiledCameraCfg,
    observation_term_cfg=_FakeObservationTermCfg,
    scene_entity_cfg=_FakeSceneEntityCfg,
    image_func=_IMAGE_FUNC,
)


def _observation(sensor_name: str) -> _FakeObservationTermCfg:
    return _FakeObservationTermCfg(
        func=_IMAGE_FUNC,
        params={
            "sensor_cfg": _FakeSceneEntityCfg(sensor_name),
            "data_type": "rgb",
            "normalize": False,
        },
    )


def _base_env_cfg():
    wrist_sensor = SimpleNamespace(prim_path=eval_cameras.WRIST_CAMERA_PRIM_PATH)
    wrist_observation = _observation(eval_cameras.WRIST_CAMERA_KEY)
    return SimpleNamespace(
        scene=SimpleNamespace(camera1=wrist_sensor),
        observations=SimpleNamespace(policy=SimpleNamespace(camera1=wrist_observation)),
    )


class So101EvalCamerasTest(unittest.TestCase):
    def test_wrist_only_is_unchanged_and_does_not_load_isaac_dependencies(self) -> None:
        env_cfg = _base_env_cfg()
        scene_before = dict(vars(env_cfg.scene))
        policy_before = dict(vars(env_cfg.observations.policy))

        with patch.object(
            eval_cameras,
            "_load_camera_dependencies",
            side_effect=AssertionError("wrist-only must not load injection dependencies"),
        ):
            mapping = eval_cameras.configure_so101_eval_cameras(env_cfg, "wrist-only")

        self.assertEqual(mapping, {"wrist": "camera1"})
        self.assertEqual(vars(env_cfg.scene), scene_before)
        self.assertEqual(vars(env_cfg.observations.policy), policy_before)

    def test_dual_adds_front_top_camera3_with_authored_parameters(self) -> None:
        env_cfg = _base_env_cfg()

        with patch.object(eval_cameras, "_load_camera_dependencies", return_value=_FAKE_DEPS):
            mapping = eval_cameras.configure_so101_eval_cameras(env_cfg, "dual")

        self.assertEqual(mapping, {"top": "camera3", "wrist": "camera1"})
        self.assertFalse(hasattr(env_cfg.scene, "camera2"))
        camera3 = env_cfg.scene.camera3
        self.assertEqual(camera3.prim_path, eval_cameras.TOP_CAMERA_PRIM_PATH)
        self.assertEqual(camera3.offset.convention, "opengl")
        self.assertEqual(camera3.data_types, ["rgb"])
        self.assertEqual(camera3.spawn.focal_length, 30.0)
        self.assertEqual(camera3.spawn.focus_distance, 400.0)
        self.assertEqual(camera3.spawn.horizontal_aperture, 47)
        self.assertEqual(camera3.spawn.clipping_range, (0.01, 50.0))
        self.assertIs(camera3.spawn.lock_camera, True)
        self.assertEqual((camera3.width, camera3.height), (640, 480))
        self.assertAlmostEqual(camera3.update_period, 1 / 30.0)

        observation = env_cfg.observations.policy.camera3
        self.assertIs(observation.func, _IMAGE_FUNC)
        self.assertEqual(observation.params["sensor_cfg"].name, "camera3")
        self.assertEqual(observation.params["data_type"], "rgb")
        self.assertIs(observation.params["normalize"], False)

    def test_triple_adds_camera3_top_and_camera2_left(self) -> None:
        env_cfg = _base_env_cfg()

        with patch.object(eval_cameras, "_load_camera_dependencies", return_value=_FAKE_DEPS):
            mapping = eval_cameras.configure_so101_eval_cameras(env_cfg, "triple")

        self.assertEqual(
            mapping,
            {"top": "camera3", "left": "camera2", "wrist": "camera1"},
        )
        self.assertEqual(env_cfg.scene.camera3.prim_path, eval_cameras.TOP_CAMERA_PRIM_PATH)
        self.assertEqual(env_cfg.scene.camera2.prim_path, eval_cameras.LEFT_CAMERA_PRIM_PATH)
        self.assertEqual(
            env_cfg.observations.policy.camera3.params["sensor_cfg"].name,
            "camera3",
        )
        self.assertEqual(
            env_cfg.observations.policy.camera2.params["sensor_cfg"].name,
            "camera2",
        )

    def test_matching_existing_config_is_idempotent(self) -> None:
        env_cfg = _base_env_cfg()

        with patch.object(eval_cameras, "_load_camera_dependencies", return_value=_FAKE_DEPS):
            eval_cameras.configure_so101_eval_cameras(env_cfg, "triple")
        existing_camera2 = env_cfg.scene.camera2
        existing_camera3 = env_cfg.scene.camera3
        existing_observation2 = env_cfg.observations.policy.camera2
        existing_observation3 = env_cfg.observations.policy.camera3

        with patch.object(
            eval_cameras,
            "_load_camera_dependencies",
            side_effect=AssertionError("matching config must not be rebuilt"),
        ):
            eval_cameras.configure_so101_eval_cameras(env_cfg, "triple")

        self.assertIs(env_cfg.scene.camera2, existing_camera2)
        self.assertIs(env_cfg.scene.camera3, existing_camera3)
        self.assertIs(env_cfg.observations.policy.camera2, existing_observation2)
        self.assertIs(env_cfg.observations.policy.camera3, existing_observation3)

    def test_conflicting_sensor_fails_before_partial_injection(self) -> None:
        env_cfg = _base_env_cfg()
        env_cfg.scene.camera2 = SimpleNamespace(
            prim_path="{ENV_REGEX_NS}/Robot/gripper/old_wrist_camera"
        )

        with patch.object(eval_cameras, "_load_camera_dependencies", return_value=_FAKE_DEPS):
            with self.assertRaisesRegex(ValueError, "Refusing to replace existing scene sensor"):
                eval_cameras.configure_so101_eval_cameras(env_cfg, "triple")

        self.assertFalse(hasattr(env_cfg.scene, "camera3"))
        self.assertFalse(hasattr(env_cfg.observations.policy, "camera2"))
        self.assertFalse(hasattr(env_cfg.observations.policy, "camera3"))

    def test_conflicting_observation_fails_before_partial_injection(self) -> None:
        env_cfg = _base_env_cfg()
        env_cfg.scene.camera2 = SimpleNamespace(prim_path=eval_cameras.LEFT_CAMERA_PRIM_PATH)
        env_cfg.observations.policy.camera2 = _observation("different_sensor")

        with patch.object(eval_cameras, "_load_camera_dependencies", return_value=_FAKE_DEPS):
            with self.assertRaisesRegex(ValueError, "Refusing to replace existing policy observation"):
                eval_cameras.configure_so101_eval_cameras(env_cfg, "triple")

        self.assertFalse(hasattr(env_cfg.scene, "camera3"))
        self.assertFalse(hasattr(env_cfg.observations.policy, "camera3"))

    def test_missing_wrist_contract_fails_closed(self) -> None:
        env_cfg = _base_env_cfg()
        del env_cfg.scene.camera1

        with self.assertRaisesRegex(ValueError, "existing wrist sensor"):
            eval_cameras.configure_so101_eval_cameras(env_cfg, "dual")

        self.assertFalse(hasattr(env_cfg.scene, "camera3"))

    def test_unknown_layout_is_rejected_without_mutation(self) -> None:
        env_cfg = _base_env_cfg()
        scene_before = dict(vars(env_cfg.scene))
        policy_before = dict(vars(env_cfg.observations.policy))

        with self.assertRaisesRegex(ValueError, "Unsupported SO101 camera layout"):
            eval_cameras.configure_so101_eval_cameras(env_cfg, "quad")

        self.assertEqual(vars(env_cfg.scene), scene_before)
        self.assertEqual(vars(env_cfg.observations.policy), policy_before)


if __name__ == "__main__":
    unittest.main()
