"""Opt-in SO101 contact diagnostics for deployment evaluations.

This module deliberately avoids importing Isaac Lab at module-import time so
its buffer parsing and geometry helpers can be unit-tested without launching
Isaac Sim.  The runner calls :func:`configure_so101_target_contact_probe`
before ``gym.make`` and :func:`read_contact_probe` after an ``env.step``.

The SO101 asset calls the fixed gripper-side rigid body ``gripper`` and the
moving side ``jaw``.  Isaac Lab filtered contacts are one-to-many, so the two
bodies require separate contact sensors.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


GRIPPER_CONTACT_SENSOR_NAME = "contact_probe_gripper_to_target"
JAW_CONTACT_SENSOR_NAME = "contact_probe_jaw_to_target"

SO101_CONTACT_SENSOR_SPECS = (
    (
        "gripper",
        GRIPPER_CONTACT_SENSOR_NAME,
        "{ENV_REGEX_NS}/Robot/gripper",
    ),
    (
        "jaw",
        JAW_CONTACT_SENSOR_NAME,
        "{ENV_REGEX_NS}/Robot/jaw",
    ),
)

# Approximate distal-cap centers derived from the authored collision/visual
# meshes in leisaac/assets/robots/so101_follower.usd.  They are diagnostic
# keypoints, not signed-distance samples of the cooked PhysX colliders.
SO101_TIP_LOCAL_OFFSETS_M = {
    "gripper": (-0.009925, -0.000226, -0.103950),
    "jaw": (-0.010137, -0.081529, 0.018901),
}


def configure_so101_target_contact_probe(
    env_cfg: Any,
    target_key: str,
) -> tuple[str, ...]:
    """Install the runner-facing SO101 probe for one scene-entity key.

    The selected target's configured ``prim_path`` is preferred.  The managed
    scene-profile convention is used only as a fallback, which keeps this
    helper usable with both profile-injected and task-owned rigid objects.
    """

    if not isinstance(target_key, str) or not target_key.strip():
        raise ValueError("target_key must be a non-empty scene-entity key")

    target_cfg = getattr(env_cfg.scene, target_key, None)
    target_prim_path = getattr(target_cfg, "prim_path", None)
    if not isinstance(target_prim_path, str) or not target_prim_path.strip():
        target_prim_path = f"{{ENV_REGEX_NS}}/Scene/{target_key}_managed"

    installed = configure_so101_contact_probe(
        env_cfg,
        target_prim_path,
        enabled=True,
    )
    return tuple(installed[body_name] for body_name, _, _ in SO101_CONTACT_SENSOR_SPECS)


def configure_so101_contact_probe(
    env_cfg: Any,
    target_prim_path: str,
    *,
    enabled: bool = False,
    max_contact_data_count_per_prim: int = 16,
    debug_vis: bool = False,
) -> dict[str, str]:
    """Add two filtered SO101 contact sensors to an environment config.

    This function is intentionally opt-in.  With ``enabled=False`` it does not
    mutate ``env_cfg`` and does not import Isaac Lab.

    Args:
        env_cfg: Isaac Lab environment config containing ``scene.robot``.
        target_prim_path: Rigid-body prim expression for the managed target,
            for example
            ``{ENV_REGEX_NS}/Scene/red_2x4_lego_brick_managed``.
        enabled: Whether to install the probe.
        max_contact_data_count_per_prim: Maximum retained contact patches for
            each sensor body.
        debug_vis: Whether Isaac Lab should visualize sensor contact state.

    Returns:
        Mapping from SO101 rigid-body name to installed scene sensor name.

    Raises:
        ValueError: If an enabled configuration is invalid or would overwrite
            an existing scene field.
        AttributeError: If the environment does not expose the expected SO101
            robot spawn config.
    """

    if not enabled:
        return {}
    if not isinstance(target_prim_path, str) or not target_prim_path.strip():
        raise ValueError("target_prim_path must be a non-empty prim expression")
    if max_contact_data_count_per_prim < 1:
        raise ValueError("max_contact_data_count_per_prim must be at least 1")

    scene_cfg = env_cfg.scene
    robot_spawn_cfg = scene_cfg.robot.spawn

    # Delayed import: importing Isaac sensors requires an initialized Isaac
    # runtime in normal deployment, while the pure helpers below do not.
    from isaaclab.sensors import ContactSensorCfg

    for _, sensor_name, _ in SO101_CONTACT_SENSOR_SPECS:
        if hasattr(scene_cfg, sensor_name):
            raise ValueError(
                f"scene already defines {sensor_name!r}; refusing to overwrite it"
            )

    # ContactSensor checks PhysxContactReportAPI on the sensor body.  Enabling
    # it on the articulation spawner applies the reporter to gripper and jaw
    # without modifying the shared robot USD.
    robot_spawn_cfg.activate_contact_sensors = True

    installed: dict[str, str] = {}
    for body_name, sensor_name, body_prim_path in SO101_CONTACT_SENSOR_SPECS:
        setattr(
            scene_cfg,
            sensor_name,
            ContactSensorCfg(
                prim_path=body_prim_path,
                update_period=0.0,
                debug_vis=debug_vis,
                track_pose=False,
                track_contact_points=True,
                max_contact_data_count_per_prim=max_contact_data_count_per_prim,
                filter_prim_paths_expr=[target_prim_path],
            ),
        )
        installed[body_name] = sensor_name
    return installed


def _to_python(value: Any) -> Any:
    """Convert a tensor/array-like value to nested Python containers."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    elif hasattr(value, "numpy"):
        value = value.numpy().tolist()
    return value


def _flatten(value: Any) -> list[Any]:
    """Flatten nested tensor/list output while preserving scalar values."""

    value = _to_python(value)
    if isinstance(value, (list, tuple)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten(item))
        return flattened
    return [value]


def _rows(value: Any, width: int) -> list[list[Any]]:
    """Return a tensor-like buffer as fixed-width rows."""

    flattened = _flatten(value)
    if len(flattened) % width != 0:
        raise ValueError(
            f"buffer with {len(flattened)} values cannot be reshaped into width {width}"
        )
    return [
        flattened[index : index + width]
        for index in range(0, len(flattened), width)
    ]


def _json_float(value: Any) -> float | None:
    """Convert a numeric scalar to a finite JSON-safe float."""

    result = float(value)
    return result if math.isfinite(result) else None


def read_contact_sensor(sensor: Any, physics_dt: float) -> dict[str, Any]:
    """Read detailed patch data from one Isaac Lab ``ContactSensor``.

    The first value returned by ``RigidContactView.get_contact_data`` is the
    scalar normal force for each patch.  Multiplying it by the corresponding
    normal produces the world-frame force vector included in this result.
    All returned values are JSON-safe Python containers.
    """

    physics_dt = float(physics_dt)
    if not math.isfinite(physics_dt) or physics_dt <= 0.0:
        raise ValueError(f"physics_dt must be positive and finite, got {physics_dt!r}")

    view = sensor.contact_physx_view
    raw = view.get_contact_data(dt=physics_dt)
    if len(raw) != 6:
        raise ValueError(
            "RigidContactView.get_contact_data must return "
            "(forces, points, normals, separations, counts, starts)"
        )
    raw_forces, raw_points, raw_normals, raw_separations, raw_counts, raw_starts = raw

    forces = _flatten(raw_forces)
    points = _rows(raw_points, 3)
    normals = _rows(raw_normals, 3)
    separations = _flatten(raw_separations)
    counts = [int(value) for value in _flatten(raw_counts)]
    starts = [int(value) for value in _flatten(raw_starts)]

    sensor_count = int(getattr(view, "sensor_count", 1))
    filter_count = int(getattr(view, "filter_count", 1))
    expected_pair_slots = sensor_count * filter_count
    if len(counts) != expected_pair_slots or len(starts) != expected_pair_slots:
        raise ValueError(
            "contact pair buffers do not match view dimensions: "
            f"counts={len(counts)} starts={len(starts)} "
            f"sensor_count={sensor_count} filter_count={filter_count}"
        )

    active_indices: list[int] = []
    pairs: list[dict[str, int]] = []
    for pair_index, (count, start) in enumerate(zip(counts, starts)):
        if count < 0 or start < 0:
            raise ValueError(f"negative contact count/start at pair {pair_index}")
        stop = start + count
        if stop > min(len(forces), len(points), len(normals), len(separations)):
            raise ValueError(
                f"contact pair {pair_index} slice [{start}:{stop}] exceeds a data buffer"
            )
        active_indices.extend(range(start, stop))
        pairs.append(
            {
                "sensor_index": pair_index // filter_count,
                "filter_index": pair_index % filter_count,
                "count": count,
                "start_index": start,
            }
        )

    active_forces = [_json_float(forces[index]) for index in active_indices]
    active_points = [
        [_json_float(component) for component in points[index]]
        for index in active_indices
    ]
    active_normals = [
        [_json_float(component) for component in normals[index]]
        for index in active_indices
    ]
    active_separations = [
        _json_float(separations[index]) for index in active_indices
    ]

    force_vectors: list[list[float | None]] = []
    for force, normal in zip(active_forces, active_normals):
        if force is None or any(component is None for component in normal):
            force_vectors.append([None, None, None])
        else:
            force_vectors.append([force * float(component) for component in normal])

    finite_separations = [
        value for value in active_separations if value is not None
    ]
    min_separation = min(finite_separations) if finite_separations else None
    max_penetration = (
        max(0.0, -min_separation) if min_separation is not None else 0.0
    )

    return {
        "count": len(active_indices),
        "forces_n": active_forces,
        "force_vectors_w_n": force_vectors,
        "points_w_m": active_points,
        "normals_w": active_normals,
        "separations_m": active_separations,
        "min_separation_m": min_separation,
        "max_penetration_m": max_penetration,
        "pair_counts": [
            counts[index : index + filter_count]
            for index in range(0, len(counts), filter_count)
        ],
        "pairs": pairs,
    }


def transform_local_point_w(
    body_pos_w: Sequence[float],
    body_quat_wxyz: Sequence[float],
    local_point: Sequence[float],
) -> list[float]:
    """Transform one body-local point to world coordinates with pure Python."""

    if len(body_pos_w) != 3 or len(body_quat_wxyz) != 4 or len(local_point) != 3:
        raise ValueError("expected position(3), quaternion wxyz(4), and local point(3)")

    px, py, pz = (float(value) for value in body_pos_w)
    w, qx, qy, qz = (float(value) for value in body_quat_wxyz)
    vx, vy, vz = (float(value) for value in local_point)

    quat_norm = math.sqrt(w * w + qx * qx + qy * qy + qz * qz)
    if not math.isfinite(quat_norm) or quat_norm <= 1.0e-12:
        raise ValueError("body quaternion must have a finite non-zero norm")
    w, qx, qy, qz = (
        w / quat_norm,
        qx / quat_norm,
        qy / quat_norm,
        qz / quat_norm,
    )

    # Unit-quaternion vector rotation: v' = v + w*t + cross(q.xyz, t),
    # where t = 2*cross(q.xyz, v).
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    rotated_x = vx + w * tx + (qy * tz - qz * ty)
    rotated_y = vy + w * ty + (qz * tx - qx * tz)
    rotated_z = vz + w * tz + (qx * ty - qy * tx)
    return [px + rotated_x, py + rotated_y, pz + rotated_z]


def read_so101_tip_points(robot: Any) -> dict[str, dict[str, Any]]:
    """Read fixed/moving link poses and their mesh-derived tip keypoints."""

    data = robot.data
    body_names = list(data.body_names)
    body_positions = _to_python(data.body_pos_w)
    body_quaternions = _to_python(data.body_quat_w)

    result: dict[str, dict[str, Any]] = {}
    for body_name, local_tip in SO101_TIP_LOCAL_OFFSETS_M.items():
        if body_name not in body_names:
            raise ValueError(
                f"SO101 robot body {body_name!r} is missing from {body_names!r}"
            )
        body_index = body_names.index(body_name)
        body_pos = [float(value) for value in body_positions[0][body_index]]
        body_quat = [float(value) for value in body_quaternions[0][body_index]]
        result[body_name] = {
            "body_pos_w_m": body_pos,
            "body_quat_wxyz": body_quat,
            "tip_local_m": list(local_tip),
            "tip_pos_w_m": transform_local_point_w(
                body_pos,
                body_quat,
                local_tip,
            ),
        }
    return result


def read_contact_probe(
    env: Any,
    sensor_names: Sequence[str],
) -> dict[str, Any]:
    """Read runner-selected sensors and return JSON-safe aggregate metrics."""

    names = tuple(sensor_names)
    if len(set(names)) != len(names):
        raise ValueError(f"sensor_names contains duplicates: {names!r}")

    physics_dt = float(env.physics_dt)
    sensors: dict[str, dict[str, Any]] = {}
    for sensor_name in names:
        if not isinstance(sensor_name, str) or not sensor_name:
            raise ValueError(f"invalid contact sensor name: {sensor_name!r}")
        sensors[sensor_name] = read_contact_sensor(
            env.scene[sensor_name],
            physics_dt,
        )

    total_count = sum(int(sample["count"]) for sample in sensors.values())
    finite_minima = [
        float(sample["min_separation_m"])
        for sample in sensors.values()
        if sample["min_separation_m"] is not None
    ]
    min_separation = min(finite_minima) if finite_minima else None
    return {
        "physics_dt_s": physics_dt,
        "sensor_names": list(names),
        "total_contact_count": total_count,
        "min_separation_m": min_separation,
        "max_penetration_m": (
            max(0.0, -min_separation) if min_separation is not None else 0.0
        ),
        "sensors": sensors,
    }
