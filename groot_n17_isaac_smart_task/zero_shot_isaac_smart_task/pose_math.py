"""Small NumPy-only pose conversions used by the Isaac runner.

Keeping this module independent from Torch and Isaac makes the EEF action math
testable without starting SimulationApp.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


_EPS = 1e-8


def _finite_vector(values: Any, size: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains NaN or infinity")
    return vector


def quat_wxyz_to_matrix(quat: Any) -> np.ndarray:
    """Convert a finite ``(w, x, y, z)`` quaternion to a rotation matrix."""

    w, x, y, z = _finite_vector(quat, 4, "quat")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < _EPS:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def quat_wxyz_to_rot6d(quat: Any) -> np.ndarray:
    """Convert a quaternion to GR00T's first-two-matrix-rows rot6d form."""

    return quat_wxyz_to_matrix(quat)[:2].reshape(6)


def matrix_to_quat_wxyz(rot: Any) -> np.ndarray:
    """Convert a finite 3x3 rotation matrix to a normalized wxyz quaternion."""

    matrix = np.asarray(rot, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rot must have shape (3, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rot contains NaN or infinity")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], _EPS)) * 2.0
        quat = (
            (matrix[2, 1] - matrix[1, 2]) / scale,
            0.25 * scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
        )
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], _EPS)) * 2.0
        quat = (
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            0.25 * scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
        )
    else:
        scale = math.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], _EPS)) * 2.0
        quat = (
            (matrix[1, 0] - matrix[0, 1]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            0.25 * scale,
        )

    result = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(result))
    if norm < _EPS:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return result / norm


def rot6d_to_matrix(rot6d: Any) -> np.ndarray:
    """Convert finite rot6d rows to an orthonormal right-handed matrix."""

    rows = _finite_vector(rot6d, 6, "rot6d").reshape(2, 3)
    row1 = rows[0]
    row1_norm = float(np.linalg.norm(row1))
    if row1_norm < _EPS:
        return np.eye(3, dtype=np.float32)
    row1 = row1 / row1_norm

    row2 = rows[1] - np.dot(rows[1], row1) * row1
    row2_norm = float(np.linalg.norm(row2))
    if row2_norm < _EPS:
        # Pick the axis least aligned with row1, then Gram-Schmidt it.
        basis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(row1)))]
        row2 = basis - np.dot(basis, row1) * row1
        row2_norm = float(np.linalg.norm(row2))
    row2 = row2 / row2_norm
    row3 = np.cross(row1, row2)
    return np.vstack([row1, row2, row3]).astype(np.float32)


def compose_pose_delta(base_pos: Any, base_quat: Any, delta_9d: Any) -> tuple[np.ndarray, np.ndarray]:
    """Compose a relative EEF-frame ``xyz + rot6d`` delta with a base pose."""

    position = _finite_vector(base_pos, 3, "base_pos")
    delta = _finite_vector(delta_9d, 9, "delta_9d")
    base_rot = quat_wxyz_to_matrix(base_quat)
    target_pos = position + base_rot @ delta[:3]
    target_rot = base_rot @ rot6d_to_matrix(delta[3:])
    return target_pos.astype(np.float32), matrix_to_quat_wxyz(target_rot)
