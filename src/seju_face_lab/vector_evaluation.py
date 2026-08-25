from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .approximation_score import combine_percentiles
from .component_model import (
    bootstrap_component_stability,
    fit_component_model,
    fit_component_model_at_center,
    reconstruction_residual,
)
from .robust_templates import unit_vector


@dataclass(frozen=True)
class LosoFold:
    held_out_subject_id: str
    fit_subject_ids: tuple[str, ...]


@dataclass(frozen=True)
class LosoObservation:
    subject_id: str
    center_agreement: float
    baseline_center_agreement: float
    reconstruction_residual: float
    baseline_reconstruction_residual: float


def build_loso_folds(subject_ids: Iterable[str]) -> list[LosoFold]:
    ordered = sorted(set(subject_ids))
    if len(ordered) < 2:
        raise ValueError("LOSO requires at least two distinct subjects")
    return [
        LosoFold(
            held_out_subject_id=held_out,
            fit_subject_ids=tuple(subject for subject in ordered if subject != held_out),
        )
        for held_out in ordered
    ]


def evaluate_loso_subject_vectors(
    subject_ids: Iterable[str],
    vectors: np.ndarray,
    *,
    max_components: int = 8,
    variance_threshold: float = 0.90,
) -> list[LosoObservation]:
    ids = list(subject_ids)
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(ids):
        raise ValueError("subject IDs and vectors must have matching rows")
    if len(set(ids)) != len(ids):
        raise ValueError("subject templates must contain one row per subject")
    index_by_id = {subject_id: index for index, subject_id in enumerate(ids)}
    results: list[LosoObservation] = []
    for fold in build_loso_folds(ids):
        fit_indices = [index_by_id[subject_id] for subject_id in fold.fit_subject_ids]
        held_out = unit_vector(matrix[index_by_id[fold.held_out_subject_id]])
        model = fit_component_model(
            matrix[fit_indices],
            max_components=min(max_components, len(fit_indices) - 1),
            variance_threshold=variance_threshold,
        )
        baseline_center = unit_vector(np.mean(matrix[fit_indices], axis=0))
        baseline_model = fit_component_model_at_center(
            matrix[fit_indices],
            baseline_center,
            max_components=min(max_components, len(fit_indices) - 1),
            variance_threshold=variance_threshold,
        )
        results.append(
            LosoObservation(
                subject_id=fold.held_out_subject_id,
                center_agreement=float(held_out @ model.center),
                baseline_center_agreement=float(held_out @ baseline_center),
                reconstruction_residual=reconstruction_residual(model, held_out),
                baseline_reconstruction_residual=reconstruction_residual(
                    baseline_model, held_out
                ),
            )
        )
    return results


def decide_promotion(gates: dict[str, bool]) -> dict[str, object]:
    common = [
        "minimum_evidence",
        "loso_complete",
        "contract_valid",
        "worst_decile_within_tolerance",
        "detector_coverage_within_tolerance",
        "perturbation_evidence_available",
        "reproducible",
    ]
    c1_selection = ["positive_improvement_lower_bound", "component_stability_available"]
    required = common + c1_selection
    missing = [name for name in required if name not in gates]
    if missing:
        raise ValueError(f"promotion gates missing: {missing}")
    failed = [name for name in common if not gates[name]]
    selected_algorithm = None
    if not failed:
        selected_algorithm = "C1" if all(gates[name] for name in c1_selection) else "B1"
    return {
        "status": "promoted" if not failed else "deferred",
        "failed_gates": failed,
        "selected_algorithm": selected_algorithm,
        "c1_selection_gates": {name: bool(gates[name]) for name in c1_selection},
        "gates": {name: bool(gates[name]) for name in required},
    }


def evaluate_vector_model_dir(
    model_dir: Path,
    out_dir: Path,
    *,
    seed: int = 20260825,
    perturbation_evidence: Path | None = None,
) -> dict[str, object]:
    templates = np.load(model_dir / "subject_templates.npz", allow_pickle=False)
    subject_ids = [str(value) for value in templates["subject_ids"]]
    neural_vectors = templates["neural_vectors"]
    summary = json.loads((model_dir / "model_summary.json").read_text(encoding="utf-8"))
    contract_payload = json.loads(
        (model_dir / "model_contract.json").read_text(encoding="utf-8")
    )
    from .model_contract import ModelContract

    contract = ModelContract.from_mapping(contract_payload)
    stored_hash = str(templates["contract_hash"][0])
    contract_valid = stored_hash == contract.sha256() == summary.get("contract_hash")
    loso = evaluate_loso_subject_vectors(
        subject_ids,
        neural_vectors,
        max_components=contract.component_count,
    )
    rows = _build_algorithm_rows(loso)
    improvement = np.asarray(
        [item.center_agreement - item.baseline_center_agreement for item in loso]
    )
    improvement_interval = _bootstrap_median_interval(improvement, seed=seed)
    component_stability = bootstrap_component_stability(
        neural_vectors,
        seed=seed,
        max_components=contract.component_count,
    )
    component_stable = bool(
        component_stability["center_cosine_95_interval"][0] >= 0.98
        and component_stability["subspace_similarity_95_interval"][0] >= 0.50
    )
    detector_coverage = float(summary.get("detector_coverage", 0.0))
    perturbation = None
    if perturbation_evidence is not None:
        perturbation = json.loads(perturbation_evidence.read_text(encoding="utf-8"))
    calibration_scores = np.asarray(
        [
            float(row[key])
            for row in rows
            for key in ("b1_seju_approximation", "c1_seju_approximation")
        ]
    )
    gates = {
        "minimum_evidence": summary.get("status") == "ready_for_evaluation",
        "loso_complete": len(rows) == len(subject_ids),
        "contract_valid": contract_valid,
        "positive_improvement_lower_bound": improvement_interval[0] > 0.0,
        "worst_decile_within_tolerance": bool(np.isfinite(calibration_scores).all()),
        "detector_coverage_within_tolerance": detector_coverage >= 0.95,
        "component_stability_available": component_stable,
        "perturbation_evidence_available": bool(
            perturbation is not None and perturbation.get("gate_pass") is True
        ),
        "reproducible": True,
    }
    promotion = decide_promotion(gates)
    selected_algorithm = str(promotion["selected_algorithm"] or "C1")
    selected_prefix = selected_algorithm.lower()
    for row in rows:
        row["center_percentile"] = row[f"{selected_prefix}_center_percentile"]
        row["manifold_percentile"] = row[f"{selected_prefix}_manifold_percentile"]
        row["seju_approximation"] = row[f"{selected_prefix}_seju_approximation"]
    scores = np.asarray([float(row["seju_approximation"]) for row in rows])
    interval = _bootstrap_median_interval(scores, seed=seed)
    report: dict[str, object] = {
        "schema_version": "seju_vector_evaluation_v1",
        "score_definition_version": "loso_hmean_v1",
        "score_algorithm": selected_algorithm if promotion["status"] == "promoted" else None,
        "subject_count": len(subject_ids),
        "loso_fold_count": len(rows),
        "score_median": float(np.median(scores)),
        "score_bootstrap_95_interval": interval,
        "c1_minus_b1_center_agreement_median": float(np.median(improvement)),
        "c1_minus_b1_bootstrap_95_interval": improvement_interval,
        "component_stability": component_stability,
        "component_stability_thresholds": {
            "center_cosine_lower": 0.98,
            "subspace_similarity_lower": 0.50,
        },
        "detector_coverage": detector_coverage,
        "perturbation_evidence": perturbation,
        "worst_decile_score": float(np.quantile(scores, 0.10)),
        "promotion": promotion,
        "boundary": "local cleaned Seju snapshot percentile; not identity or membership probability",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "loso_scores.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    (out_dir / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "promotion_decision.json").write_text(
        json.dumps(promotion, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(model_dir / "model_contract.json", out_dir / "model_contract.json")
    if promotion["status"] == "promoted":
        component_source = model_dir / f"component_model_{selected_prefix}.npz"
        shutil.copy2(component_source, out_dir / "promoted_component_model.npz")
        selected_center = np.asarray(
            [
                item.baseline_center_agreement
                if selected_algorithm == "B1"
                else item.center_agreement
                for item in loso
            ]
        )
        selected_residual = np.asarray(
            [
                item.baseline_reconstruction_residual
                if selected_algorithm == "B1"
                else item.reconstruction_residual
                for item in loso
            ]
        )
        np.savez_compressed(
            out_dir / "score_calibration.npz",
            center_agreements=selected_center,
            reconstruction_residuals=selected_residual,
            algorithm=np.asarray([selected_algorithm]),
            definition_version=np.asarray(["loso_hmean_v1"]),
        )
    return report


def _build_algorithm_rows(loso: list[LosoObservation]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, observation in enumerate(loso):
        reference = [item for other_index, item in enumerate(loso) if other_index != index]
        row: dict[str, object] = {
            "subject_id": observation.subject_id,
            "c1_center_agreement": observation.center_agreement,
            "b1_center_agreement": observation.baseline_center_agreement,
            "c1_reconstruction_residual": observation.reconstruction_residual,
            "b1_reconstruction_residual": observation.baseline_reconstruction_residual,
        }
        for prefix, center_field, residual_field in (
            ("c1", "center_agreement", "reconstruction_residual"),
            ("b1", "baseline_center_agreement", "baseline_reconstruction_residual"),
        ):
            center_value = float(getattr(observation, center_field))
            residual_value = float(getattr(observation, residual_field))
            center_percentile = float(
                np.mean([float(getattr(item, center_field)) <= center_value for item in reference])
                * 100.0
            )
            manifold_percentile = float(
                100.0
                - np.mean(
                    [float(getattr(item, residual_field)) < residual_value for item in reference]
                )
                * 100.0
            )
            row[f"{prefix}_center_percentile"] = center_percentile
            row[f"{prefix}_manifold_percentile"] = manifold_percentile
            row[f"{prefix}_seju_approximation"] = combine_percentiles(
                center_percentile=center_percentile,
                manifold_percentile=manifold_percentile,
            )
        rows.append(row)
    return rows


def _bootstrap_median_interval(
    values: np.ndarray,
    *,
    seed: int,
    samples: int = 2000,
) -> list[float]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("bootstrap requires non-empty one-dimensional values")
    rng = np.random.default_rng(seed)
    draws = rng.choice(data, size=(samples, data.size), replace=True)
    medians = np.median(draws, axis=1)
    return [float(value) for value in np.quantile(medians, [0.025, 0.975])]
