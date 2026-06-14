from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .backends import VectorBackend, get_vector_backend
from .embeddings import ImageVector, iter_image_paths
from .model import CentroidModel


@dataclass(frozen=True)
class Score:
    image_id: str
    path: str
    cosine_to_mean: float
    cosine_to_median: float
    euclidean_to_mean: float
    euclidean_to_median: float
    centroid_score: float


def score_generated_images(
    model: CentroidModel,
    images_dir: Path,
    crop: str = "center",
    backend: VectorBackend | None = None,
) -> list[Score]:
    active_backend = backend or get_vector_backend("deterministic")
    vectors = [active_backend.vectorize(path, crop=crop) for path in iter_image_paths(images_dir)]
    scores = [_score_vector(model, vector) for vector in vectors]
    return sorted(scores, key=lambda item: item.centroid_score, reverse=True)


def write_scores(scores: list[Score], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_lines = [
        "image_id,path,cosine_to_mean,cosine_to_median,euclidean_to_mean,euclidean_to_median,centroid_score"
    ]
    for score in scores:
        csv_lines.append(
            ",".join(
                [
                    _csv(score.image_id),
                    _csv(score.path),
                    f"{score.cosine_to_mean:.6f}",
                    f"{score.cosine_to_median:.6f}",
                    f"{score.euclidean_to_mean:.6f}",
                    f"{score.euclidean_to_median:.6f}",
                    f"{score.centroid_score:.6f}",
                ]
            )
        )
    (out_dir / "scores.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8-sig")
    (out_dir / "evaluation.md").write_text(_render_scores(scores), encoding="utf-8")


def _score_vector(model: CentroidModel, vector: ImageVector) -> Score:
    cosine_mean = _cosine(vector.embedding, model.mean_embedding)
    cosine_median = _cosine(vector.embedding, model.median_embedding)
    euclidean_mean = float(np.linalg.norm(vector.embedding - model.mean_embedding))
    euclidean_median = float(np.linalg.norm(vector.embedding - model.median_embedding))
    centroid_score = (cosine_mean + cosine_median) / 2.0
    return Score(
        image_id=vector.image_id,
        path=str(vector.path),
        cosine_to_mean=cosine_mean,
        cosine_to_median=cosine_median,
        euclidean_to_mean=euclidean_mean,
        euclidean_to_median=euclidean_median,
        centroid_score=centroid_score,
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _render_scores(scores: list[Score]) -> str:
    lines = ["# generated-image centroid evaluation", ""]
    if not scores:
        lines.extend(["No generated images found.", ""])
        return "\n".join(lines)
    lines.append("| rank | image_id | centroid_score | cosine_mean | cosine_median |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for rank, score in enumerate(scores, start=1):
        lines.append(
            f"| {rank} | {score.image_id} | {score.centroid_score:.4f} | "
            f"{score.cosine_to_mean:.4f} | {score.cosine_to_median:.4f} |"
        )
    lines.extend(
        [
            "",
            "Scores are approximate vector similarity against this local centroid model.",
            "",
        ]
    )
    return "\n".join(lines)


def _csv(value: str) -> str:
    escaped = value.replace('"', '""')
    return f'"{escaped}"'
