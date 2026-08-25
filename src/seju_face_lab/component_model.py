from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .robust_templates import spherical_geometric_median, unit_vector


@dataclass(frozen=True)
class ComponentModel:
    center: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    support_low: np.ndarray
    support_high: np.ndarray


def tangent_log(center: np.ndarray, vector: np.ndarray) -> np.ndarray:
    mu = unit_vector(center)
    x = unit_vector(vector)
    cosine = float(np.clip(mu @ x, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle <= 1e-12:
        return np.zeros_like(mu)
    direction = x - cosine * mu
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("antipodal vector cannot be mapped deterministically")
    return direction * (angle / norm)


def fit_component_model(
    subject_vectors: np.ndarray,
    *,
    max_components: int = 8,
    variance_threshold: float = 0.90,
) -> ComponentModel:
    matrix = np.asarray(subject_vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("component model requires at least two subject vectors")
    if max_components < 1 or not 0.0 < variance_threshold <= 1.0:
        raise ValueError("invalid component selection settings")
    normalized = np.stack([unit_vector(row) for row in matrix])
    center = spherical_geometric_median(normalized)
    tangent = np.stack([tangent_log(center, row) for row in normalized])
    _, singular, right = np.linalg.svd(tangent, full_matrices=False)
    variance = singular**2
    ratios = variance / variance.sum() if float(variance.sum()) > 0 else np.zeros_like(variance)
    cumulative = np.cumsum(ratios)
    selected = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
    selected = min(selected, max_components, matrix.shape[0] - 1, matrix.shape[1])
    components = right[:selected]
    coordinates = tangent @ components.T
    return ComponentModel(
        center=center,
        components=components,
        explained_variance_ratio=ratios[:selected],
        support_low=np.quantile(coordinates, 0.05, axis=0),
        support_high=np.quantile(coordinates, 0.95, axis=0),
    )


def project_component(model: ComponentModel, vector: np.ndarray) -> np.ndarray:
    return tangent_log(model.center, vector) @ model.components.T


def reconstruction_residual(model: ComponentModel, vector: np.ndarray) -> float:
    tangent = tangent_log(model.center, vector)
    coordinates = tangent @ model.components.T
    reconstructed = coordinates @ model.components
    return float(np.linalg.norm(tangent - reconstructed))


def component_support(model: ComponentModel, vector: np.ndarray) -> bool:
    coordinates = project_component(model, vector)
    return bool(np.all(coordinates >= model.support_low) and np.all(coordinates <= model.support_high))
