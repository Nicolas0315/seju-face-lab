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
    return fit_component_model_at_center(
        normalized,
        center,
        max_components=max_components,
        variance_threshold=variance_threshold,
    )


def fit_component_model_at_center(
    subject_vectors: np.ndarray,
    center: np.ndarray,
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
    center = unit_vector(center)
    tangent = np.stack([tangent_log(center, row) for row in normalized])
    _, singular, right = np.linalg.svd(tangent, full_matrices=False)
    variance = singular**2
    ratios = variance / variance.sum() if float(variance.sum()) > 0 else np.zeros_like(variance)
    cumulative = np.cumsum(ratios)
    selected = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
    selected = min(selected, max_components, normalized.shape[0] - 1, normalized.shape[1])
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


def bootstrap_component_stability(
    subject_vectors: np.ndarray,
    *,
    seed: int,
    samples: int = 200,
    max_components: int = 8,
    variance_threshold: float = 0.90,
) -> dict[str, float | int | list[float]]:
    matrix = np.asarray(subject_vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 3:
        raise ValueError("component bootstrap requires at least three subject vectors")
    if samples < 1:
        raise ValueError("component bootstrap samples must be positive")
    reference = fit_component_model(
        matrix,
        max_components=max_components,
        variance_threshold=variance_threshold,
    )
    rng = np.random.default_rng(seed)
    center_cosines = []
    subspace_similarities = []
    for _ in range(samples):
        indices = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        fitted = fit_component_model(
            matrix[indices],
            max_components=max_components,
            variance_threshold=variance_threshold,
        )
        center_cosines.append(float(reference.center @ fitted.center))
        common = min(reference.components.shape[0], fitted.components.shape[0])
        singular = np.linalg.svd(
            reference.components[:common] @ fitted.components[:common].T,
            compute_uv=False,
        )
        subspace_similarities.append(float(np.mean(singular**2)))
    center_interval = [
        float(value) for value in np.quantile(center_cosines, [0.025, 0.975])
    ]
    subspace_interval = [
        float(value) for value in np.quantile(subspace_similarities, [0.025, 0.975])
    ]
    return {
        "bootstrap_samples": samples,
        "center_cosine_median": float(np.median(center_cosines)),
        "center_cosine_95_interval": center_interval,
        "subspace_similarity_median": float(np.median(subspace_similarities)),
        "subspace_similarity_95_interval": subspace_interval,
    }
