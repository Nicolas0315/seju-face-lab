from __future__ import annotations

from dataclasses import dataclass
import json
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


@dataclass(frozen=True)
class SubjectReview:
    subject: str
    image_count: int
    failed_count: int
    best_image_id: str | None
    best_image_path: str | None
    best_centroid_score: float | None
    mean_centroid_score: float | None
    median_centroid_score: float | None
    mean_cosine_to_mean: float | None
    mean_cosine_to_median: float | None


def score_generated_images(
    model: CentroidModel,
    images_dir: Path,
    crop: str = "center",
    backend: VectorBackend | None = None,
    failed_paths: list[str] | None = None,
) -> list[Score]:
    active_backend = backend or get_vector_backend("deterministic")
    vectors: list[ImageVector] = []
    for path in iter_image_paths(images_dir):
        try:
            vectors.append(active_backend.vectorize(path, crop=crop))
        except Exception:  # noqa: BLE001 - keep batch evaluation running and report failures.
            if failed_paths is None:
                raise
            failed_paths.append(str(path))
    scores = [_score_vector(model, vector) for vector in vectors]
    return sorted(scores, key=lambda item: item.centroid_score, reverse=True)


def review_subject_directories(
    model: CentroidModel,
    subjects_dir: Path,
    crop: str = "center",
    backend: VectorBackend | None = None,
) -> list[SubjectReview]:
    active_backend = backend or get_vector_backend("deterministic")
    subject_dirs = sorted(path for path in subjects_dir.iterdir() if path.is_dir())
    reviews = [
        _review_subject(model, subject_dir, crop=crop, backend=active_backend)
        for subject_dir in subject_dirs
    ]
    return sorted(
        reviews,
        key=lambda item: item.mean_centroid_score if item.mean_centroid_score is not None else -1.0,
        reverse=True,
    )


def write_subject_reviews(reviews: list[SubjectReview], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_lines = [
        "subject,image_count,failed_count,best_image_id,best_image_path,best_centroid_score,"
        "mean_centroid_score,median_centroid_score,mean_cosine_to_mean,mean_cosine_to_median"
    ]
    for review in reviews:
        csv_lines.append(
            ",".join(
                [
                    _csv(review.subject),
                    str(review.image_count),
                    str(review.failed_count),
                    _csv(review.best_image_id or ""),
                    _csv(review.best_image_path or ""),
                    _optional_float(review.best_centroid_score),
                    _optional_float(review.mean_centroid_score),
                    _optional_float(review.median_centroid_score),
                    _optional_float(review.mean_cosine_to_mean),
                    _optional_float(review.mean_cosine_to_median),
                ]
            )
        )
    (out_dir / "subject_reviews.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8-sig")
    (out_dir / "subject_reviews.md").write_text(_render_subject_reviews(reviews), encoding="utf-8")
    (out_dir / "subject_reviews.json").write_text(
        json.dumps(_subject_review_summary(reviews), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_scores(scores: list[Score], out_dir: Path, failed_paths: list[str] | None = None) -> None:
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
    (out_dir / "evaluation.md").write_text(
        _render_scores(scores, failed_paths or []),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(_score_summary(scores, failed_paths or []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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


def _review_subject(
    model: CentroidModel,
    subject_dir: Path,
    crop: str,
    backend: VectorBackend,
) -> SubjectReview:
    vectors: list[ImageVector] = []
    failed_count = 0
    for path in iter_image_paths(subject_dir):
        try:
            vectors.append(backend.vectorize(path, crop=crop))
        except Exception:  # noqa: BLE001 - keep a subject review running and report failures.
            failed_count += 1
    scores = sorted(
        (_score_vector(model, vector) for vector in vectors),
        key=lambda item: item.centroid_score,
        reverse=True,
    )
    if not scores:
        return SubjectReview(
            subject=subject_dir.name,
            image_count=0,
            failed_count=failed_count,
            best_image_id=None,
            best_image_path=None,
            best_centroid_score=None,
            mean_centroid_score=None,
            median_centroid_score=None,
            mean_cosine_to_mean=None,
            mean_cosine_to_median=None,
        )
    centroid_scores = np.asarray([score.centroid_score for score in scores], dtype=np.float32)
    cosine_mean_scores = np.asarray([score.cosine_to_mean for score in scores], dtype=np.float32)
    cosine_median_scores = np.asarray([score.cosine_to_median for score in scores], dtype=np.float32)
    best = scores[0]
    return SubjectReview(
        subject=subject_dir.name,
        image_count=len(scores),
        failed_count=failed_count,
        best_image_id=best.image_id,
        best_image_path=best.path,
        best_centroid_score=float(best.centroid_score),
        mean_centroid_score=float(np.mean(centroid_scores)),
        median_centroid_score=float(np.median(centroid_scores)),
        mean_cosine_to_mean=float(np.mean(cosine_mean_scores)),
        mean_cosine_to_median=float(np.mean(cosine_median_scores)),
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _render_scores(scores: list[Score], failed_paths: list[str]) -> str:
    lines = ["# generated-image centroid evaluation", ""]
    if not scores:
        lines.extend(["No generated images found.", ""])
    else:
        lines.append("| rank | image_id | centroid_score | cosine_mean | cosine_median |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for rank, score in enumerate(scores, start=1):
            lines.append(
                f"| {rank} | {score.image_id} | {score.centroid_score:.4f} | "
                f"{score.cosine_to_mean:.4f} | {score.cosine_to_median:.4f} |"
            )
        lines.append("")
    if failed_paths:
        lines.append("## Failed Images")
        lines.append("")
        lines.extend(f"- {path}" for path in failed_paths)
        lines.append("")
    lines.extend(
        [
            "Scores are approximate vector similarity against this local centroid model.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_subject_reviews(reviews: list[SubjectReview]) -> str:
    lines = ["# subject seju-face similarity review", ""]
    if not reviews:
        lines.extend(["No subject directories found.", ""])
        return "\n".join(lines)
    lines.append(
        "| rank | subject | images | failed | mean_score | median_score | best_score | best_image |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for rank, review in enumerate(reviews, start=1):
        lines.append(
            f"| {rank} | {review.subject} | {review.image_count} | {review.failed_count} | "
            f"{_optional_float(review.mean_centroid_score)} | "
            f"{_optional_float(review.median_centroid_score)} | "
            f"{_optional_float(review.best_centroid_score)} | "
            f"{review.best_image_id or ''} |"
        )
    lines.extend(
        [
            "",
            "Scores are approximate vector similarity against this local seju centroid model.",
            "They are not identity, attractiveness, ethnicity, or objective face-type labels.",
            "",
        ]
    )
    return "\n".join(lines)


def _score_summary(scores: list[Score], failed_paths: list[str]) -> dict:
    if not scores:
        return {
            "image_count": 0,
            "failed_count": len(failed_paths),
            "failed_paths": failed_paths[:20],
            "best_image_id": None,
            "best_centroid_score": None,
            "mean_centroid_score": None,
            "median_centroid_score": None,
            "top_images": [],
        }
    centroid_scores = np.asarray([score.centroid_score for score in scores], dtype=np.float32)
    best = scores[0]
    return {
        "image_count": len(scores),
        "failed_count": len(failed_paths),
        "failed_paths": failed_paths[:20],
        "best_image_id": best.image_id,
        "best_centroid_score": round(float(best.centroid_score), 6),
        "mean_centroid_score": round(float(np.mean(centroid_scores)), 6),
        "median_centroid_score": round(float(np.median(centroid_scores)), 6),
        "top_images": [
            {
                "image_id": score.image_id,
                "path": score.path,
                "centroid_score": round(float(score.centroid_score), 6),
                "cosine_to_mean": round(float(score.cosine_to_mean), 6),
                "cosine_to_median": round(float(score.cosine_to_median), 6),
                "euclidean_to_mean": round(float(score.euclidean_to_mean), 6),
                "euclidean_to_median": round(float(score.euclidean_to_median), 6),
            }
            for score in scores[:5]
        ],
        "boundary": "Approximate vector similarity for this local centroid model only.",
    }


def _subject_review_summary(reviews: list[SubjectReview]) -> dict:
    return {
        "subject_count": len(reviews),
        "subjects": [
            {
                "subject": review.subject,
                "image_count": review.image_count,
                "failed_count": review.failed_count,
                "best_image_id": review.best_image_id,
                "best_image_path": review.best_image_path,
                "best_centroid_score": _round_optional(review.best_centroid_score),
                "mean_centroid_score": _round_optional(review.mean_centroid_score),
                "median_centroid_score": _round_optional(review.median_centroid_score),
                "mean_cosine_to_mean": _round_optional(review.mean_cosine_to_mean),
                "mean_cosine_to_median": _round_optional(review.mean_cosine_to_median),
            }
            for review in reviews
        ],
        "boundary": "Approximate vector similarity for this local centroid model only.",
    }


def _optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _csv(value: str) -> str:
    escaped = value.replace('"', '""')
    return f'"{escaped}"'
