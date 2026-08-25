from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .approximation_score import ScoreCalibration, score_vector
from .component_model import ComponentModel
from .face_observations import FaceObservationExtractor, extract_face_observation
from .model_contract import ModelContract
from .vector_pipeline import load_component_model


@dataclass(frozen=True)
class ScoreBundle:
    evaluation: dict[str, Any]
    component_model: ComponentModel
    calibration: ScoreCalibration
    contract: ModelContract


def load_score_bundle(evaluation_dir: Path) -> ScoreBundle:
    evaluation = json.loads(
        (evaluation_dir / "evaluation.json").read_text(encoding="utf-8")
    )
    promotion = evaluation.get("promotion")
    if not isinstance(promotion, dict) or promotion.get("status") != "promoted":
        raise ValueError("promoted evaluation required before scoring")
    component_model = load_component_model(evaluation_dir / "promoted_component_model.npz")
    calibration_data = np.load(evaluation_dir / "score_calibration.npz", allow_pickle=False)
    calibration = ScoreCalibration(
        center_agreements=calibration_data["center_agreements"],
        reconstruction_residuals=calibration_data["reconstruction_residuals"],
        definition_version=str(calibration_data["definition_version"][0]),
    )
    contract = ModelContract.from_mapping(
        json.loads((evaluation_dir / "model_contract.json").read_text(encoding="utf-8"))
    )
    return ScoreBundle(evaluation, component_model, calibration, contract)


def score_face_image(
    image_path: Path,
    evaluation_dir: Path,
    extractor: FaceObservationExtractor,
) -> dict[str, Any]:
    bundle = load_score_bundle(evaluation_dir)
    observation = extract_face_observation(
        image_path,
        "synthetic_candidate",
        extractor,
        bundle.contract,
    )
    if observation.quality < 0.05:
        raise ValueError("face quality gate failed")
    score = score_vector(
        observation.neural_vector,
        bundle.component_model,
        bundle.calibration,
    )
    return {
        "schema_version": "seju_face_score_v1",
        "image_id": image_path.stem,
        "center_percentile": score.center_percentile,
        "manifold_percentile": score.manifold_percentile,
        "seju_approximation": score.seju_approximation,
        "out_of_support": score.out_of_support,
        "quality": observation.quality,
        "score_definition_version": score.definition_version,
        "algorithm": bundle.evaluation["score_algorithm"],
        "boundary": score.boundary,
    }
