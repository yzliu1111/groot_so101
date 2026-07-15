"""Validation and slicing helpers for GR00T action chunks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


ACTION_TIME_KEYS = (
    "joint_position",
    "eef_9d",
    "gripper_position",
    "single_arm",
    "gripper",
)


def _action_array(action: Mapping[str, Any], key: str) -> np.ndarray:
    try:
        array = np.asarray(action[key], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"action[{key!r}] is not a numeric array") from exc
    if array.ndim != 3:
        raise ValueError(f"action[{key!r}] must have shape (B,T,D), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"action[{key!r}] contains NaN or infinity")
    return array


def action_chunk_length(action: Mapping[str, Any]) -> int:
    """Return T after validating that all known action arrays share one T."""

    lengths: dict[str, int] = {}
    for key in ACTION_TIME_KEYS:
        if key not in action:
            continue
        array = _action_array(action, key)
        if array.shape[1] < 1:
            raise ValueError(f"action[{key!r}] has an empty time dimension")
        lengths[key] = int(array.shape[1])

    if not lengths:
        raise ValueError(f"cannot infer action chunk length from keys: {sorted(action.keys())}")
    if len(set(lengths.values())) != 1:
        raise ValueError(f"action chunk time dimensions do not match: {lengths}")
    return next(iter(lengths.values()))


def slice_action_step(action: Mapping[str, Any], step_index: int, keys: Iterable[str]) -> dict[str, np.ndarray]:
    """Select one ``(B,1,D)`` timestep for the requested keys that exist."""

    if step_index < 0:
        raise ValueError(f"step_index must be non-negative, got {step_index}")

    selected: dict[str, np.ndarray] = {}
    for key in keys:
        if key not in action:
            continue
        array = _action_array(action, key)
        if step_index >= array.shape[1]:
            raise IndexError(
                f"step_index {step_index} is outside action[{key!r}] time dimension {array.shape[1]}"
            )
        selected[key] = array[:, step_index : step_index + 1, :]
    return selected
