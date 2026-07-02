from __future__ import annotations

from dataclasses import dataclass
from html import escape
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
    top_images: tuple[dict[str, object], ...]


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
    (out_dir / "subject_reviews.html").write_text(_render_subject_reviews_html(reviews), encoding="utf-8")
    (out_dir / "subject_reviews.json").write_text(
        json.dumps(_subject_review_summary(reviews), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_scores(
    scores: list[Score],
    out_dir: Path,
    failed_paths: list[str] | None = None,
    model: CentroidModel | None = None,
) -> None:
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
        _render_scores(scores, failed_paths or [], model=model),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(_score_summary(scores, failed_paths or [], model=model), ensure_ascii=False, indent=2),
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
            top_images=(),
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
        top_images=tuple(_score_dict(score) for score in scores[:5]),
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _render_scores(
    scores: list[Score],
    failed_paths: list[str],
    model: CentroidModel | None = None,
) -> str:
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
    calibration = _score_null_calibration(scores, model)
    if calibration.get("available"):
        null_distribution = calibration["null_distribution"]
        observed = calibration["observed_percentiles"]
        lines.extend(
            [
                "## Null Distribution Calibration",
                "",
                f"- sample_count: {null_distribution['sample_count']}",
                f"- seed: {null_distribution['seed']}",
                f"- p95: {_optional_float(null_distribution['p95'])}",
                f"- p99: {_optional_float(null_distribution['p99'])}",
                f"- best_score_percentile: {_optional_float(observed.get('best_centroid_score'))}",
                f"- mean_score_percentile: {_optional_float(observed.get('mean_centroid_score'))}",
                "",
            ]
        )
    lines.extend(
        [
            "Scores are approximate vector similarity against this local centroid model.",
            "Null calibration uses random unit vectors as a local sanity baseline, not a population claim.",
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
    lines.extend(_render_subject_analysis(_subject_review_analysis(reviews)))
    lines.extend(
        [
            "",
            "Scores are approximate vector similarity against this local seju centroid model.",
            "They are not identity, attractiveness, ethnicity, or objective face-type labels.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_subject_analysis(analysis: dict[str, object]) -> list[str]:
    if not analysis:
        return []
    stats = analysis.get("score_stats")
    lines = ["", "## Vector Analysis", ""]
    if isinstance(stats, dict):
        lines.extend(
            [
                f"- reviewed_images: {stats.get('reviewed_image_count', 0)}",
                f"- failed_images: {stats.get('failed_image_count', 0)}",
                f"- mean_of_subject_means: {_optional_float(stats.get('mean_of_subject_means'))}",
                f"- median_of_subject_means: {_optional_float(stats.get('median_of_subject_means'))}",
                "",
            ]
        )
    lines.extend(_render_subject_analysis_table("Stable Mean Leaders", analysis.get("top_mean_subjects")))
    lines.extend(_render_subject_analysis_table("Peak Best Leaders", analysis.get("top_best_subjects")))
    lines.extend(_render_subject_analysis_table("Single Image Lift", analysis.get("single_image_lift")))
    lines.extend(_render_subject_analysis_table("Mean Vector Affinity", analysis.get("mean_vector_leaders")))
    lines.extend(_render_subject_analysis_table("Median Vector Affinity", analysis.get("median_vector_leaders")))
    return lines


def _render_subject_analysis_table(title: str, rows: object) -> list[str]:
    if not isinstance(rows, list) or not rows:
        return []
    lines = [f"### {title}", "", "| rank | subject | metric | mean | best | median |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for rank, row in enumerate(rows, start=1):
        if isinstance(row, dict):
            lines.append(
                f"| {rank} | {row.get('subject', '')} | "
                f"{_optional_float(row.get('metric'))} | "
                f"{_optional_float(row.get('mean_centroid_score'))} | "
                f"{_optional_float(row.get('best_centroid_score'))} | "
                f"{_optional_float(row.get('median_centroid_score'))} |"
            )
    lines.append("")
    return lines


def _render_subject_reviews_html(reviews: list[SubjectReview]) -> str:
    cards = "\n".join(_render_subject_card(rank, review) for rank, review in enumerate(reviews, start=1))
    analysis = _render_subject_analysis_html(_subject_review_analysis(reviews))
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>subject seju-face similarity review</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f4;color:#1f2933}",
            "h1{font-size:24px;margin:0 0 16px}",
            ".subjects{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}",
            ".subject{border:1px solid #d7d7d0;border-radius:8px;background:#fff;padding:10px}",
            ".subject img{width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:6px;background:#e7e5dc}",
            ".meta{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0;color:#4b5563;font-size:12px}",
            ".pill{background:#eef2f7;border-radius:999px;padding:3px 8px}",
            ".scores{font-size:12px;line-height:1.45;margin-top:6px}",
            ".path{overflow-wrap:anywhere;color:#59636e}",
            ".analysis{margin-top:24px}",
            ".analysis h2{font-size:18px;margin:18px 0 8px}",
            "table{border-collapse:collapse;width:100%;font-size:12px;background:#fff;margin-bottom:12px}",
            "th,td{border:1px solid #d7d7d0;padding:6px;text-align:left}",
            "td.metric,td.score{text-align:right;font-variant-numeric:tabular-nums}",
            ".boundary{font-size:12px;color:#667085;margin-top:18px}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>subject seju-face similarity review</h1>",
            f'<div class="subjects">{cards}</div>' if cards else "<p>No subject directories found.</p>",
            analysis,
            '<p class="boundary">Scores are approximate local triage against this centroid model. '
            "They are not identity, attractiveness, ethnicity, or objective face-type labels.</p>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _render_subject_analysis_html(analysis: dict[str, object]) -> str:
    if not analysis:
        return ""
    sections = [
        _render_subject_analysis_html_table("Stable Mean Leaders", analysis.get("top_mean_subjects")),
        _render_subject_analysis_html_table("Peak Best Leaders", analysis.get("top_best_subjects")),
        _render_subject_analysis_html_table("Single Image Lift", analysis.get("single_image_lift")),
        _render_subject_analysis_html_table("Mean Vector Affinity", analysis.get("mean_vector_leaders")),
        _render_subject_analysis_html_table("Median Vector Affinity", analysis.get("median_vector_leaders")),
    ]
    body = "\n".join(section for section in sections if section)
    return f'<section class="analysis"><h2>Vector Analysis</h2>{body}</section>' if body else ""


def _render_subject_analysis_html_table(title: str, rows: object) -> str:
    if not isinstance(rows, list) or not rows:
        return ""
    lines = [
        f"<h2>{escape(title)}</h2>",
        "<table>",
        "<thead><tr><th>rank</th><th>subject</th><th>metric</th><th>mean</th><th>best</th><th>median</th></tr></thead>",
        "<tbody>",
    ]
    for rank, row in enumerate(rows, start=1):
        if isinstance(row, dict):
            lines.append(
                "<tr>"
                f"<td>{rank}</td>"
                f"<td>{escape(str(row.get('subject', '')))}</td>"
                f"<td class=\"metric\">{_optional_float(row.get('metric'))}</td>"
                f"<td class=\"score\">{_optional_float(row.get('mean_centroid_score'))}</td>"
                f"<td class=\"score\">{_optional_float(row.get('best_centroid_score'))}</td>"
                f"<td class=\"score\">{_optional_float(row.get('median_centroid_score'))}</td>"
                "</tr>"
            )
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def _render_subject_card(rank: int, review: SubjectReview) -> str:
    best = review.top_images[0] if review.top_images else {}
    image_path = str(best.get("path") or "")
    image = f'<img src="{escape(_image_src(image_path))}" alt="{escape(review.subject)}">' if image_path else ""
    return "\n".join(
        [
            '<article class="subject">',
            f"<h2>#{rank} {escape(review.subject)}</h2>",
            image,
            '<div class="meta">',
            f'<span class="pill">images {review.image_count}</span>',
            f'<span class="pill">failed {review.failed_count}</span>',
            f'<span class="pill">mean {_optional_float(review.mean_centroid_score)}</span>',
            f'<span class="pill">best {_optional_float(review.best_centroid_score)}</span>',
            "</div>",
            '<div class="scores">',
            f"<div>median_score: {_optional_float(review.median_centroid_score)}</div>",
            f"<div>mean_cosine_to_mean: {_optional_float(review.mean_cosine_to_mean)}</div>",
            f"<div>mean_cosine_to_median: {_optional_float(review.mean_cosine_to_median)}</div>",
            f'<div class="path">{escape(image_path)}</div>' if image_path else "",
            "</div>",
            "</article>",
        ]
    )


def _image_src(value: str) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve(strict=False)
    try:
        return path.as_uri()
    except ValueError:
        return value


def _score_summary(
    scores: list[Score],
    failed_paths: list[str],
    model: CentroidModel | None = None,
) -> dict:
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
            "null_calibration": _score_null_calibration(scores, model),
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
        "null_calibration": _score_null_calibration(scores, model),
        "boundary": "Approximate vector similarity for this local centroid model only.",
    }


def _score_null_calibration(
    scores: list[Score],
    model: CentroidModel | None,
    sample_count: int = 4096,
    seed: int = 260623,
) -> dict[str, object]:
    if model is None:
        return {"available": False, "reason": "model_not_supplied"}
    if model.mean_embedding.size == 0 or model.median_embedding.size == 0:
        return {"available": False, "reason": "empty_centroid"}
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(sample_count, model.mean_embedding.shape[0])).astype(np.float32)
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    samples = samples / np.maximum(norms, 1e-12)
    mean = _unit(model.mean_embedding)
    median = _unit(model.median_embedding)
    null_scores = (samples @ mean + samples @ median) / 2.0
    observed_scores = np.asarray([score.centroid_score for score in scores], dtype=np.float32)
    best_score = float(np.max(observed_scores)) if observed_scores.size else None
    mean_score = float(np.mean(observed_scores)) if observed_scores.size else None
    median_score = float(np.median(observed_scores)) if observed_scores.size else None
    return {
        "available": True,
        "method": "random_unit_vector_centroid_score",
        "sample_count": sample_count,
        "seed": seed,
        "embedding_dim": int(model.mean_embedding.shape[0]),
        "null_distribution": {
            "sample_count": sample_count,
            "seed": seed,
            "mean": round(float(np.mean(null_scores)), 6),
            "std": round(float(np.std(null_scores)), 6),
            "p50": round(float(np.percentile(null_scores, 50)), 6),
            "p90": round(float(np.percentile(null_scores, 90)), 6),
            "p95": round(float(np.percentile(null_scores, 95)), 6),
            "p99": round(float(np.percentile(null_scores, 99)), 6),
        },
        "observed_percentiles": {
            "best_centroid_score": _null_percentile(null_scores, best_score),
            "mean_centroid_score": _null_percentile(null_scores, mean_score),
            "median_centroid_score": _null_percentile(null_scores, median_score),
        },
        "boundary": (
            "Random-unit-vector baseline for this local embedding dimension only; "
            "not demographic, identity, attractiveness, or population calibration."
        ),
    }


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _null_percentile(null_scores: np.ndarray, observed: float | None) -> float | None:
    if observed is None:
        return None
    return round(float(np.mean(null_scores <= observed)), 6)


def _subject_review_summary(reviews: list[SubjectReview]) -> dict:
    return {
        "subject_count": len(reviews),
        "analysis": _subject_review_analysis(reviews),
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
                "top_images": list(review.top_images),
            }
            for review in reviews
        ],
        "boundary": "Approximate vector similarity for this local centroid model only.",
    }


def _subject_review_analysis(reviews: list[SubjectReview]) -> dict[str, object]:
    scored = [review for review in reviews if review.mean_centroid_score is not None]
    if not scored:
        return {
            "score_stats": {
                "reviewed_image_count": sum(review.image_count for review in reviews),
                "failed_image_count": sum(review.failed_count for review in reviews),
                "mean_of_subject_means": None,
                "median_of_subject_means": None,
            },
            "top_mean_subjects": [],
            "top_best_subjects": [],
            "single_image_lift": [],
            "mean_vector_leaders": [],
            "median_vector_leaders": [],
        }
    mean_scores = np.asarray([review.mean_centroid_score for review in scored], dtype=np.float32)
    return {
        "score_stats": {
            "reviewed_image_count": sum(review.image_count for review in reviews),
            "failed_image_count": sum(review.failed_count for review in reviews),
            "mean_of_subject_means": round(float(np.mean(mean_scores)), 6),
            "median_of_subject_means": round(float(np.median(mean_scores)), 6),
        },
        "top_mean_subjects": _rank_subjects(scored, "mean_centroid_score"),
        "top_best_subjects": _rank_subjects(scored, "best_centroid_score"),
        "single_image_lift": _rank_subjects(scored, "single_image_lift"),
        "mean_vector_leaders": _rank_subjects(scored, "mean_cosine_to_mean"),
        "median_vector_leaders": _rank_subjects(scored, "mean_cosine_to_median"),
    }


def _rank_subjects(reviews: list[SubjectReview], metric: str, limit: int = 5) -> list[dict[str, object]]:
    rows = [_subject_metric_row(review, metric) for review in reviews]
    rows = [row for row in rows if row["metric"] is not None]
    return sorted(rows, key=lambda row: row["metric"], reverse=True)[:limit]


def _subject_metric_row(review: SubjectReview, metric: str) -> dict[str, object]:
    if metric == "single_image_lift":
        metric_value = _score_delta(review.best_centroid_score, review.mean_centroid_score)
    else:
        metric_value = getattr(review, metric)
    return {
        "subject": review.subject,
        "metric": _round_optional(metric_value),
        "mean_centroid_score": _round_optional(review.mean_centroid_score),
        "best_centroid_score": _round_optional(review.best_centroid_score),
        "median_centroid_score": _round_optional(review.median_centroid_score),
        "best_image_id": review.best_image_id,
    }


def _score_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def _score_dict(score: Score) -> dict[str, object]:
    return {
        "image_id": score.image_id,
        "path": score.path,
        "centroid_score": round(float(score.centroid_score), 6),
        "cosine_to_mean": round(float(score.cosine_to_mean), 6),
        "cosine_to_median": round(float(score.cosine_to_median), 6),
        "euclidean_to_mean": round(float(score.euclidean_to_mean), 6),
        "euclidean_to_median": round(float(score.euclidean_to_median), 6),
    }


def _optional_float(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _csv(value: str) -> str:
    escaped = value.replace('"', '""')
    return f'"{escaped}"'
