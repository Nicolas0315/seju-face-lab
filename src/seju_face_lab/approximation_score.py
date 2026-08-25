from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .component_model import ComponentModel, component_support, reconstruction_residual
from .robust_templates import unit_vector


@dataclass(frozen=True)
class ScoreCalibration:
    center_agreements: np.ndarray
    reconstruction_residuals: np.ndarray
    definition_version: str = "loso_hmean_v1"


@dataclass(frozen=True)
class ApproximationScore:
    center_percentile: float
    manifold_percentile: float
    seju_approximation: float
    out_of_support: bool
    definition_version: str
    boundary: str = "cleaned local Seju snapshot only; not identity or membership probability"


def combine_percentiles(*, center_percentile: float, manifold_percentile: float) -> float:
    left = float(np.clip(center_percentile, 0.0, 100.0))
    right = float(np.clip(manifold_percentile, 0.0, 100.0))
    if left <= 0.0 or right <= 0.0:
        return 0.0
    return float(2.0 * left * right / (left + right))


def empirical_percentile(value: float, reference: np.ndarray) -> float:
    values = np.asarray(reference, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("calibration reference must be a non-empty finite vector")
    return float(np.mean(values <= value) * 100.0)


def score_vector(
    vector: np.ndarray,
    model: ComponentModel,
    calibration: ScoreCalibration,
) -> ApproximationScore:
    x = unit_vector(vector)
    center_agreement = float(x @ model.center)
    residual = reconstruction_residual(model, x)
    center_percentile = empirical_percentile(center_agreement, calibration.center_agreements)
    manifold_percentile = float(
        (1.0 - np.mean(calibration.reconstruction_residuals < residual)) * 100.0
    )
    return ApproximationScore(
        center_percentile=center_percentile,
        manifold_percentile=manifold_percentile,
        seju_approximation=combine_percentiles(
            center_percentile=center_percentile,
            manifold_percentile=manifold_percentile,
        ),
        out_of_support=not component_support(model, x),
        definition_version=calibration.definition_version,
    )
