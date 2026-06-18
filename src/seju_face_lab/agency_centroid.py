from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .model import CentroidModel, build_centroid_model, save_model

AGENCY_CENTROID_BOUNDARY = (
    "Agency centroid approximates only the provided member vector folders; it does not represent "
    "the agency's real population, identity, attractiveness, demographic attributes, or personal value."
)


def build_agency_centroid(subject_vectors_dir: Path, agency: str) -> CentroidModel:
    vector_paths = sorted(subject_vectors_dir.glob("*/vector.npz"), key=lambda path: path.parent.name)
    if not vector_paths:
        raise SystemExit(f"No member vector.npz found under {subject_vectors_dir} for agency {agency}")

    members: list[str] = []
    embeddings: list[np.ndarray] = []
    appearances: list[np.ndarray] = []
    for vector_path in vector_paths:
        with np.load(vector_path) as data:
            members.append(vector_path.parent.name)
            embeddings.append(np.asarray(data["mean_embedding"], dtype=np.float32))
            appearances.append(np.asarray(data["mean_appearance"], dtype=np.float32))

    model = build_centroid_model(
        image_ids=members,
        source_paths=[str(path) for path in vector_paths],
        embeddings=np.stack(embeddings),
        appearances=np.stack(appearances),
        centroid_mode="image_weighted",
    )
    return replace(model, subject_count=len(members), subject_counts={member: 1 for member in members})


def write_agency_centroid(model: CentroidModel, out_dir: Path, agency: str) -> None:
    save_model(
        model,
        out_dir,
        profile_metadata={
            "model_type": "agency_centroid",
            "agency": agency,
            "boundary": AGENCY_CENTROID_BOUNDARY,
            "notes": [
                AGENCY_CENTROID_BOUNDARY,
                "Agency member-vector centroid summarizes only the provided local member vector folders.",
                "Generated-image scores are approximate vector similarity, not ground truth.",
            ],
        },
        report_title="agency member-vector centroid report",
        evaluation_model_path=str(out_dir),
        manifest_label=f"{agency} agency member-vector centroid",
    )
