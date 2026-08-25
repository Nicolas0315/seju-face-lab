from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .component_model import (
    ComponentModel,
    fit_component_model,
    fit_component_model_at_center,
)
from .data_gate import cluster_near_duplicates
from .model_contract import ModelContract
from .robust_templates import SubjectTemplate, build_subject_template, unit_vector


def build_vector_model_records(
    records: list[Mapping[str, Any]],
    neural_vectors: np.ndarray,
    geometry_vectors: np.ndarray,
    contract: ModelContract,
    out_dir: Path,
    *,
    min_subjects: int = 20,
    min_portraits: int = 3,
) -> dict[str, Any]:
    neural = np.asarray(neural_vectors, dtype=np.float64)
    geometry = np.asarray(geometry_vectors, dtype=np.float64)
    if neural.shape[0] != len(records) or geometry.shape[0] != len(records):
        raise ValueError("observation records and vector arrays must have matching rows")
    if neural.ndim != 2 or neural.shape[1] != contract.embedding_dimension:
        raise ValueError("neural vectors do not match model contract")
    if geometry.ndim != 2:
        raise ValueError("geometry vectors must be a matrix")

    templates: list[SubjectTemplate] = []
    excluded: list[dict[str, Any]] = []
    subject_ids = sorted({str(record["subject_id"]) for record in records})
    for subject_id in subject_ids:
        indices = [
            index for index, record in enumerate(records) if str(record["subject_id"]) == subject_id
        ]
        if len(indices) < min_portraits:
            excluded.append(
                {
                    "subject_id": subject_id,
                    "reason": "insufficient_independent_portraits",
                    "portrait_count": len(indices),
                }
            )
            continue
        weights = np.asarray(
            [max(float(records[index].get("quality", 0.0)), 0.05) for index in indices]
        )
        templates.append(
            build_subject_template(
                subject_id,
                neural[indices],
                geometry[indices],
                weights,
            )
        )
    if len(templates) < 2:
        raise ValueError("at least two valid subject templates are required")

    template_neural = np.stack([item.neural_vector for item in templates])
    component_model = fit_component_model(
        template_neural,
        max_components=min(contract.component_count, len(templates) - 1),
    )
    b1_center = unit_vector(np.mean(template_neural, axis=0))
    b1_component_model = fit_component_model_at_center(
        template_neural,
        b1_center,
        max_components=min(contract.component_count, len(templates) - 1),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "subject_templates.npz",
        subject_ids=np.asarray([item.subject_id for item in templates]),
        neural_vectors=template_neural,
        geometry_vectors=np.stack([item.geometry_vector for item in templates]),
        accepted_cluster_counts=np.asarray([item.accepted_cluster_count for item in templates]),
        angular_uncertainty=np.asarray([item.angular_uncertainty for item in templates]),
        contract_hash=np.asarray([contract.sha256()]),
    )
    _save_component_model(component_model, out_dir / "component_model.npz")
    _save_component_model(component_model, out_dir / "component_model_c1.npz")
    _save_component_model(b1_component_model, out_dir / "component_model_b1.npz")
    (out_dir / "model_contract.json").write_text(
        json.dumps(contract.to_mapping(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": "seju_vector_model_v1",
        "status": "ready_for_evaluation" if len(templates) >= min_subjects else "diagnostic",
        "valid_subject_count": len(templates),
        "excluded_subjects": excluded,
        "minimum_subjects": min_subjects,
        "minimum_portraits_per_subject": min_portraits,
        "component_count": int(component_model.components.shape[0]),
        "b1_component_count": int(b1_component_model.components.shape[0]),
        "contract_hash": contract.sha256(),
        "boundary": "cleaned local Seju snapshot only; not identity or membership probability",
    }
    (out_dir / "model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_vector_model(observations_dir: Path, out_dir: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in (observations_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    neural_data = np.load(observations_dir / "neural_vectors.npz", allow_pickle=False)
    geometry_data = np.load(observations_dir / "geometry_vectors.npz", allow_pickle=False)
    image_ids = [str(value) for value in neural_data["image_ids"]]
    if image_ids != [str(record["image_id"]) for record in records]:
        raise ValueError("observation manifest and neural vector order mismatch")
    if image_ids != [str(value) for value in geometry_data["image_ids"]]:
        raise ValueError("neural and geometry vector order mismatch")
    contract = ModelContract.from_mapping(
        json.loads((observations_dir / "model_contract.json").read_text(encoding="utf-8"))
    )
    stored_hash = str(neural_data["contract_hash"][0])
    if stored_hash != contract.sha256():
        raise ValueError("observation vectors and model contract mismatch")

    duplicate_rows = [
        {
            "subject_id": record["subject_id"],
            "image_id": record["image_id"],
            "path": record["path"],
            "quality": record["quality"],
        }
        for record in records
    ]
    clusters = cluster_near_duplicates(duplicate_rows)
    representatives = {str(cluster["representative_id"]) for cluster in clusters}
    keep = [index for index, image_id in enumerate(image_ids) if image_id in representatives]
    selected_records = [records[index] for index in keep]
    summary = build_vector_model_records(
        selected_records,
        neural_data["vectors"][keep],
        geometry_data["vectors"][keep],
        contract,
        out_dir,
    )
    observation_summary = json.loads(
        (observations_dir / "observations_summary.json").read_text(encoding="utf-8")
    )
    total_observations = int(observation_summary["accepted_count"]) + int(
        observation_summary["rejected_count"]
    )
    suppressed_count = sum(len(cluster["suppressed_ids"]) for cluster in clusters)
    summary.update(
        {
            "observation_accepted_count": int(observation_summary["accepted_count"]),
            "observation_rejected_count": int(observation_summary["rejected_count"]),
            "detector_coverage": (
                int(observation_summary["accepted_count"]) / total_observations
                if total_observations
                else 0.0
            ),
            "independent_image_count": len(keep),
            "near_duplicate_suppressed_count": suppressed_count,
        }
    )
    (out_dir / "model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "duplicate_clusters.json").write_text(
        json.dumps(clusters, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def load_component_model(path: Path) -> ComponentModel:
    data = np.load(path, allow_pickle=False)
    return ComponentModel(
        center=data["center"],
        components=data["components"],
        explained_variance_ratio=data["explained_variance_ratio"],
        support_low=data["support_low"],
        support_high=data["support_high"],
    )


def _save_component_model(model: ComponentModel, path: Path) -> None:
    np.savez_compressed(
        path,
        center=model.center,
        components=model.components,
        explained_variance_ratio=model.explained_variance_ratio,
        support_low=model.support_low,
        support_high=model.support_high,
    )
