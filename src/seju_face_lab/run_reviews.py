from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class GenerationRunReview:
    run_dir: str
    provider: str
    model_id: str
    status: str
    image_count: int
    failed_count: int
    best_image_id: str | None
    best_centroid_score: float | None
    mean_centroid_score: float | None
    median_centroid_score: float | None
    best_style_score: float | None
    mean_style_score: float | None
    median_style_score: float | None
    best_combined_image_id: str | None
    best_combined_path: str | None
    best_combined_score: float | None
    prompt_words: int | None


def review_generation_runs(run_dirs: list[Path]) -> list[GenerationRunReview]:
    reviews = [_review_generation_run(run_dir) for run_dir in run_dirs]
    return sorted(
        reviews,
        key=_review_sort_key,
        reverse=True,
    )


def write_generation_run_reviews(reviews: list[GenerationRunReview], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "generation_run_reviews.csv").write_text(
        _render_generation_run_reviews_csv(reviews),
        encoding="utf-8-sig",
    )
    (out_dir / "generation_run_reviews.md").write_text(
        _render_generation_run_reviews_md(reviews),
        encoding="utf-8",
    )
    (out_dir / "generation_run_reviews.json").write_text(
        json.dumps(_generation_run_reviews_summary(reviews), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _review_generation_run(run_dir: Path) -> GenerationRunReview:
    summary_path = _summary_path(run_dir)
    if not summary_path.exists():
        raise ValueError(
            f"Missing evaluation summary for {run_dir}. Run evaluate with "
            f"--out {run_dir / 'evaluation'} first, or pass an evaluation output directory."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    generation_run_path = _generation_run_path(run_dir)
    generation_run = (
        json.loads(generation_run_path.read_text(encoding="utf-8"))
        if generation_run_path.exists()
        else {}
    )
    config = generation_run.get("config", {})
    result = generation_run.get("result", {})
    prompt = config.get("prompt")
    style_summary_path = _style_summary_path(run_dir)
    style_summary = (
        json.loads(style_summary_path.read_text(encoding="utf-8"))
        if style_summary_path.exists()
        else {}
    )
    best_centroid = _optional_float(summary.get("best_centroid_score"))
    best_style = _optional_float(style_summary.get("best_style_score"))
    best_combined_image_id, best_combined_path, best_combined = _best_combined_score(run_dir)
    return GenerationRunReview(
        run_dir=str(run_dir),
        provider=str(config.get("provider", "")),
        model_id=str(config.get("model_id", "")),
        status=str(result.get("status", "")),
        image_count=int(summary.get("image_count", 0)),
        failed_count=int(summary.get("failed_count", 0)),
        best_image_id=summary.get("best_image_id"),
        best_centroid_score=best_centroid,
        mean_centroid_score=_optional_float(summary.get("mean_centroid_score")),
        median_centroid_score=_optional_float(summary.get("median_centroid_score")),
        best_style_score=best_style,
        mean_style_score=_optional_float(style_summary.get("mean_style_score")),
        median_style_score=_optional_float(style_summary.get("median_style_score")),
        best_combined_image_id=best_combined_image_id,
        best_combined_path=best_combined_path,
        best_combined_score=best_combined,
        prompt_words=len(prompt.split()) if isinstance(prompt, str) else None,
    )


def _summary_path(run_dir: Path) -> Path:
    nested = run_dir / "evaluation" / "summary.json"
    if nested.exists():
        return nested
    return run_dir / "summary.json"


def _generation_run_path(run_dir: Path) -> Path:
    direct = run_dir / "generation_run.json"
    if direct.exists():
        return direct
    return run_dir.parent / "generation_run.json"


def _style_summary_path(run_dir: Path) -> Path:
    nested = run_dir / "style_evaluation" / "style_summary.json"
    if nested.exists():
        return nested
    if _is_evaluation_output_dir(run_dir):
        sibling = run_dir.parent / "style_evaluation" / "style_summary.json"
        if sibling.exists():
            return sibling
    return run_dir / "style_summary.json"


def _scores_csv_path(run_dir: Path) -> Path:
    nested = run_dir / "evaluation" / "scores.csv"
    if nested.exists():
        return nested
    return run_dir / "scores.csv"


def _style_scores_csv_path(run_dir: Path) -> Path:
    nested = run_dir / "style_evaluation" / "style_scores.csv"
    if nested.exists():
        return nested
    if _is_evaluation_output_dir(run_dir):
        sibling = run_dir.parent / "style_evaluation" / "style_scores.csv"
        if sibling.exists():
            return sibling
    return run_dir / "style_scores.csv"


def _is_evaluation_output_dir(run_dir: Path) -> bool:
    return (run_dir / "summary.json").exists() and (run_dir / "scores.csv").exists()


def _render_generation_run_reviews_csv(reviews: list[GenerationRunReview]) -> str:
    lines = [
        "rank,run_dir,provider,model_id,status,image_count,best_image_id,"
        "failed_count,best_centroid_score,mean_centroid_score,median_centroid_score,"
        "best_style_score,mean_style_score,median_style_score,best_combined_image_id,"
        "best_combined_path,best_combined_score,prompt_words"
    ]
    for rank, review in enumerate(reviews, start=1):
        lines.append(
            ",".join(
                [
                    str(rank),
                    _csv(review.run_dir),
                    _csv(review.provider),
                    _csv(review.model_id),
                    _csv(review.status),
                    str(review.image_count),
                    _csv(review.best_image_id or ""),
                    str(review.failed_count),
                    _format_optional(review.best_centroid_score),
                    _format_optional(review.mean_centroid_score),
                    _format_optional(review.median_centroid_score),
                    _format_optional(review.best_style_score),
                    _format_optional(review.mean_style_score),
                    _format_optional(review.median_style_score),
                    _csv(review.best_combined_image_id or ""),
                    _csv(review.best_combined_path or ""),
                    _format_optional(review.best_combined_score),
                    "" if review.prompt_words is None else str(review.prompt_words),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _render_generation_run_reviews_md(reviews: list[GenerationRunReview]) -> str:
    lines = ["# generation run comparison", ""]
    if not reviews:
        lines.extend(["No generation runs provided.", ""])
        return "\n".join(lines)
    lines.append(
        "| rank | run | images | failed | best_face | best_style | combined | best_image | combined_image | combined_path |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |")
    for rank, review in enumerate(reviews, start=1):
        lines.append(
            f"| {rank} | {review.run_dir} | {review.image_count} | {review.failed_count} | "
            f"{_format_optional(review.best_centroid_score)} | "
            f"{_format_optional(review.best_style_score)} | "
            f"{_format_optional(review.best_combined_score)} | "
            f"{review.best_image_id or ''} | "
            f"{review.best_combined_image_id or ''} | "
            f"{review.best_combined_path or ''} |"
        )
    lines.extend(
        [
            "",
            "Face scores are approximate vector similarity against the local centroid model.",
            "Style scores are approximate OpenCLIP image-style similarity. Combined score is the best per-image average when both are present.",
            "",
        ]
    )
    return "\n".join(lines)


def _generation_run_reviews_summary(reviews: list[GenerationRunReview]) -> dict:
    best = next((review for review in reviews if _has_review_score(review)), None)
    return {
        "run_count": len(reviews),
        "best_run_dir": best.run_dir if best else None,
        "best_centroid_score": _round_optional(best.best_centroid_score) if best else None,
        "best_style_score": _round_optional(best.best_style_score) if best else None,
        "best_combined_image_id": best.best_combined_image_id if best else None,
        "best_combined_path": best.best_combined_path if best else None,
        "best_combined_score": _round_optional(best.best_combined_score) if best else None,
        "runs": [
            {
                **asdict(review),
                "best_centroid_score": _round_optional(review.best_centroid_score),
                "mean_centroid_score": _round_optional(review.mean_centroid_score),
                "median_centroid_score": _round_optional(review.median_centroid_score),
                "best_style_score": _round_optional(review.best_style_score),
                "mean_style_score": _round_optional(review.mean_style_score),
                "median_style_score": _round_optional(review.median_style_score),
                "best_combined_image_id": review.best_combined_image_id,
                "best_combined_path": review.best_combined_path,
                "best_combined_score": _round_optional(review.best_combined_score),
            }
            for review in reviews
        ],
        "boundary": (
            "Approximate local generation-run triage only. Combined score mixes face geometry "
            "and style axes when both are present."
        ),
    }


def _best_combined_score(run_dir: Path) -> tuple[str | None, str | None, float | None]:
    face_scores = _read_score_column(_scores_csv_path(run_dir), "centroid_score")
    style_scores = _read_score_column(_style_scores_csv_path(run_dir), "style_score")
    best_image_id = None
    best_path = None
    best_score = None
    for image_path in sorted(face_scores.keys() & style_scores.keys()):
        image_id, raw_path, face_score = face_scores[image_path]
        _, _, style_score = style_scores[image_path]
        combined = (face_score + style_score) / 2.0
        if best_score is None or combined > best_score:
            best_image_id = image_id
            best_path = raw_path
            best_score = combined
    return best_image_id, best_path, best_score


def _read_score_column(path: Path, column: str) -> dict[str, tuple[str, str, float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if (
            not rows.fieldnames
            or "image_id" not in rows.fieldnames
            or "path" not in rows.fieldnames
            or column not in rows.fieldnames
        ):
            return {}
        scores: dict[str, tuple[str, str, float]] = {}
        base_dirs = _score_path_base_dirs(path)
        for row in rows:
            if not row.get("image_id") or not row.get("path") or not row.get(column):
                continue
            value = (str(row["image_id"]), str(row["path"]), float(row[column]))
            for key in _path_keys(str(row["path"]), base_dirs):
                scores[key] = value
        return scores


def _score_path_base_dirs(csv_path: Path) -> list[Path]:
    bases = [Path.cwd(), csv_path.parent]
    if csv_path.parent.name in {"evaluation", "style_evaluation"}:
        bases.append(csv_path.parent.parent)
    return bases


def _path_keys(value: str, base_dirs: list[Path]) -> set[str]:
    path = Path(value).expanduser()
    if path.is_absolute():
        return {_normal_path(path)}
    return {_normal_path(base / path) for base in base_dirs}


def _normal_path(path: Path) -> str:
    return str(path.resolve(strict=False)).casefold()


def _has_review_score(review: GenerationRunReview) -> bool:
    return review.best_combined_score is not None or review.best_centroid_score is not None


def _review_sort_key(review: GenerationRunReview) -> tuple[int, float]:
    if review.best_combined_score is not None:
        return (1, review.best_combined_score)
    if review.best_centroid_score is not None:
        return (1, review.best_centroid_score)
    return (0, 0.0)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _format_optional(value: float | None) -> str:
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
