from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubjectTemplate:
    subject_id: str
    neural_vector: np.ndarray
    geometry_vector: np.ndarray
    accepted_cluster_count: int
    angular_uncertainty: float


def unit_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or not np.isfinite(vector).all() or norm <= 1e-12:
        raise ValueError("expected a finite non-zero vector")
    return vector / norm


def spherical_geometric_median(
    vectors: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 500,
) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("expected a non-empty vector matrix")
    matrix = np.stack([unit_vector(row) for row in matrix])
    active_weights = (
        np.ones(matrix.shape[0], dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if active_weights.shape != (matrix.shape[0],) or np.any(active_weights < 0):
        raise ValueError("weights must be non-negative and match vector count")
    if float(active_weights.sum()) <= 0:
        raise ValueError("at least one weight must be positive")
    estimate = unit_vector(np.average(matrix, axis=0, weights=active_weights))
    for _ in range(max_iterations):
        angular = np.arccos(np.clip(matrix @ estimate, -1.0, 1.0))
        if float(angular.min()) <= tolerance:
            coincident = angular <= tolerance
            coincident_weight = float(active_weights[coincident].sum())
            if coincident_weight >= float(active_weights.sum()) / 2.0:
                return estimate
        inverse = active_weights / np.maximum(angular, tolerance)
        updated = unit_vector(np.sum(matrix * inverse[:, None], axis=0))
        if float(np.arccos(np.clip(updated @ estimate, -1.0, 1.0))) <= tolerance:
            return updated
        estimate = updated
    return estimate


def build_subject_template(
    subject_id: str,
    neural_vectors: np.ndarray,
    geometry_vectors: np.ndarray,
    quality_weights: np.ndarray,
    *,
    outlier_angle_radians: float = 0.65,
) -> SubjectTemplate:
    neural = np.asarray(neural_vectors, dtype=np.float64)
    geometry = np.asarray(geometry_vectors, dtype=np.float64)
    weights = np.asarray(quality_weights, dtype=np.float64)
    if neural.shape[0] < 3:
        raise ValueError("subject requires at least three independent portraits")
    if geometry.shape[0] != neural.shape[0] or weights.shape != (neural.shape[0],):
        raise ValueError("subject evidence arrays must have matching rows")
    first = spherical_geometric_median(neural, weights)
    angles = np.arccos(np.clip(np.stack([unit_vector(row) for row in neural]) @ first, -1.0, 1.0))
    keep = angles <= outlier_angle_radians
    if int(keep.sum()) < 3:
        keep = np.ones(neural.shape[0], dtype=bool)
    center = spherical_geometric_median(neural[keep], weights[keep])
    geometry_center = np.median(geometry[keep], axis=0)
    retained_angles = np.arccos(
        np.clip(np.stack([unit_vector(row) for row in neural[keep]]) @ center, -1.0, 1.0)
    )
    return SubjectTemplate(
        subject_id=subject_id,
        neural_vector=center,
        geometry_vector=geometry_center,
        accepted_cluster_count=int(keep.sum()),
        angular_uncertainty=float(np.median(retained_angles)),
    )


def build_global_center(templates: Iterable[SubjectTemplate]) -> np.ndarray:
    rows = list(templates)
    if not rows:
        raise ValueError("at least one subject template is required")
    return spherical_geometric_median(np.stack([row.neural_vector for row in rows]))
