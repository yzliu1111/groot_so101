"""Process-local SO101 camera configuration for isolated checkpoint evaluation.

The current LeIsaac SmartTask exposes only the robot-mounted wrist camera as
``camera1``.  This module adds the fixed left and front/top cameras to an
already-parsed environment config without modifying the LeIsaac checkout.

Call :func:`configure_so101_eval_cameras` after ``parse_env_cfg()`` and before
``gym.make()``.  IsaacLab discovers scene entities and observation terms by
iterating the config instances, so attributes added at this point are part of
the environment that Gym creates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


WRIST_CAMERA_KEY = "camera1"
LEFT_CAMERA_KEY = "camera2"
TOP_CAMERA_KEY = "camera3"

WRIST_CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/Robot/gripper/wrist_camera"
LEFT_CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/Scene/camera_left_xform/camera_left"
TOP_CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/Scene/camera_front_xform/camera_front"

CAMERA_LAYOUT_MAPPINGS = {
    "wrist-only": {"wrist": WRIST_CAMERA_KEY},
    "dual": {"top": TOP_CAMERA_KEY, "wrist": WRIST_CAMERA_KEY},
    "triple": {
        "top": TOP_CAMERA_KEY,
        "left": LEFT_CAMERA_KEY,
        "wrist": WRIST_CAMERA_KEY,
    },
}


@dataclass(frozen=True)
class _CameraSpec:
    key: str
    role: str
    prim_path: str


@dataclass(frozen=True)
class _CameraDependencies:
    sim_utils: Any
    tiled_camera_cfg: Any
    observation_term_cfg: Any
    scene_entity_cfg: Any
    image_func: Any


_WRIST_SPEC = _CameraSpec(
    key=WRIST_CAMERA_KEY,
    role="wrist",
    prim_path=WRIST_CAMERA_PRIM_PATH,
)
_LEFT_SPEC = _CameraSpec(
    key=LEFT_CAMERA_KEY,
    role="left",
    prim_path=LEFT_CAMERA_PRIM_PATH,
)
_TOP_SPEC = _CameraSpec(
    key=TOP_CAMERA_KEY,
    role="top",
    prim_path=TOP_CAMERA_PRIM_PATH,
)


def _load_camera_dependencies() -> _CameraDependencies:
    """Import Isaac/LeIsaac config types only when cameras must be added."""

    import isaaclab.sim as sim_utils
    from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
    from isaaclab.sensors import TiledCameraCfg
    from leisaac.tasks.smart_task import mdp

    return _CameraDependencies(
        sim_utils=sim_utils,
        tiled_camera_cfg=TiledCameraCfg,
        observation_term_cfg=ObservationTermCfg,
        scene_entity_cfg=SceneEntityCfg,
        image_func=mdp.image,
    )


def _layout_specs(camera_layout: str) -> tuple[_CameraSpec, ...]:
    if camera_layout == "wrist-only":
        return (_WRIST_SPEC,)
    if camera_layout == "dual":
        return (_WRIST_SPEC, _TOP_SPEC)
    if camera_layout == "triple":
        return (_WRIST_SPEC, _TOP_SPEC, _LEFT_SPEC)
    raise ValueError(
        f"Unsupported SO101 camera layout {camera_layout!r}; "
        f"expected one of {tuple(CAMERA_LAYOUT_MAPPINGS)}"
    )


def _sensor_prim_path(sensor_cfg: Any) -> str | None:
    prim_path = getattr(sensor_cfg, "prim_path", None)
    return prim_path if isinstance(prim_path, str) else None


def _observation_sensor_name(observation_cfg: Any) -> str | None:
    params = getattr(observation_cfg, "params", None)
    if not isinstance(params, dict):
        return None
    sensor_cfg = params.get("sensor_cfg")
    name = getattr(sensor_cfg, "name", None)
    return name if isinstance(name, str) else None


def _validate_existing_config(scene_cfg: Any, policy_cfg: Any, specs: tuple[_CameraSpec, ...]) -> None:
    """Check every requested key before mutating either config object."""

    for spec in specs:
        existing_sensor = getattr(scene_cfg, spec.key, None)
        if existing_sensor is None:
            if spec.key == WRIST_CAMERA_KEY:
                raise ValueError(
                    f"SO101 evaluation requires existing wrist sensor {spec.key!r}; "
                    "the selected environment does not expose it"
                )
        else:
            actual_prim_path = _sensor_prim_path(existing_sensor)
            if actual_prim_path != spec.prim_path:
                raise ValueError(
                    f"Refusing to replace existing scene sensor {spec.key!r}: "
                    f"expected role={spec.role!r} prim_path={spec.prim_path!r}, "
                    f"actual prim_path={actual_prim_path!r}"
                )

        existing_observation = getattr(policy_cfg, spec.key, None)
        if existing_observation is None:
            if spec.key == WRIST_CAMERA_KEY:
                raise ValueError(
                    f"SO101 evaluation requires existing wrist observation {spec.key!r}; "
                    "the selected policy observation config does not expose it"
                )
        else:
            actual_sensor_name = _observation_sensor_name(existing_observation)
            if actual_sensor_name != spec.key:
                raise ValueError(
                    f"Refusing to replace existing policy observation {spec.key!r}: "
                    f"expected sensor_cfg={spec.key!r}, actual sensor_cfg={actual_sensor_name!r}"
                )


def _make_fixed_scene_camera(spec: _CameraSpec, deps: _CameraDependencies) -> Any:
    """Build the LeIsaac-authored fixed left or front/top camera config."""

    return deps.tiled_camera_cfg(
        prim_path=spec.prim_path,
        offset=deps.tiled_camera_cfg.OffsetCfg(convention="opengl"),
        data_types=["rgb"],
        spawn=deps.sim_utils.PinholeCameraCfg(
            focal_length=30.0,
            focus_distance=400.0,
            horizontal_aperture=47,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )


def _make_camera_observation(spec: _CameraSpec, deps: _CameraDependencies) -> Any:
    return deps.observation_term_cfg(
        func=deps.image_func,
        params={
            "sensor_cfg": deps.scene_entity_cfg(spec.key),
            "data_type": "rgb",
            "normalize": False,
        },
    )


def configure_so101_eval_cameras(env_cfg: Any, camera_layout: str) -> dict[str, str]:
    """Expose the live cameras required by an SO101 checkpoint layout.

    The operation is idempotent when existing keys already have the expected
    semantics.  Any conflicting key fails before new attributes are added, so
    a historical three-camera layout cannot be silently relabeled.

    Returns:
        A fresh semantic-role to live-observation-key mapping suitable for the
        runner's explicit camera arguments.
    """

    specs = _layout_specs(camera_layout)
    try:
        scene_cfg = env_cfg.scene
        policy_cfg = env_cfg.observations.policy
    except AttributeError as exc:
        raise ValueError(
            "SO101 camera injection requires env_cfg.scene and env_cfg.observations.policy"
        ) from exc

    _validate_existing_config(scene_cfg, policy_cfg, specs)

    missing_specs = [
        spec
        for spec in specs
        if getattr(scene_cfg, spec.key, None) is None
        or getattr(policy_cfg, spec.key, None) is None
    ]
    if missing_specs:
        deps = _load_camera_dependencies()
        for spec in missing_specs:
            if getattr(scene_cfg, spec.key, None) is None:
                setattr(scene_cfg, spec.key, _make_fixed_scene_camera(spec, deps))
            if getattr(policy_cfg, spec.key, None) is None:
                setattr(policy_cfg, spec.key, _make_camera_observation(spec, deps))

    return dict(CAMERA_LAYOUT_MAPPINGS[camera_layout])
