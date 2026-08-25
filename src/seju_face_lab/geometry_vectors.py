from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeometryVector:
    schema_version: str
    values: np.ndarray
    symmetry_residual: float


def normalize_landmarks_106(points: np.ndarray) -> np.ndarray:
    landmarks = np.asarray(points, dtype=np.float64)
    if landmarks.shape != (106, 2) or not np.isfinite(landmarks).all():
        raise ValueError("expected finite landmarks with shape (106, 2)")
    centered = landmarks - landmarks.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if scale <= 1e-12:
        raise ValueError("landmark scale is zero")
    normalized = centered / scale
    left_to_right = normalized[np.argmin(normalized[:, 0])]
    right_to_left = normalized[np.argmax(normalized[:, 0])]
    if left_to_right[0] > right_to_left[0]:
        normalized[:, 0] *= -1.0
    return normalized


def geometry_vector(points: np.ndarray) -> GeometryVector:
    normalized = normalize_landmarks_106(points)
    mirrored = normalized.copy()
    mirrored[:, 0] *= -1.0
    pairwise = np.linalg.norm(normalized[:, None, :] - mirrored[None, :, :], axis=2)
    symmetry_residual = float(np.mean(np.min(pairwise, axis=1)))
    values = np.concatenate((normalized.reshape(-1), np.asarray([symmetry_residual])))
    return GeometryVector(
        schema_version="insightface_2d_106_shape_v1",
        values=values,
        symmetry_residual=symmetry_residual,
    )
