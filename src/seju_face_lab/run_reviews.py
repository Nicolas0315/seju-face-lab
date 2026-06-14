from __future__ import annotations

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
    best_image_id: str | None
    best_centroid_score: float | None
    mean_centroid_score: float | None
    median_centroid_score: float | None
    prompt_words: int | None


def review_generation_runs(run_dirs: list[Path]) -> list[GenerationRunReview]:
    reviews = [_review_generation_run(run_dir) for run_dir in run_dirs]
    return sorted(
        reviews,
        key=lambda review: review.best_centroid_score
        if review.best_centroid_score is not None
        else -1.0,
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
    summary_path = run_dir / "evaluation" / "summary.json"
    if not summary_path.exists():
        raise ValueError(
            f"Missing evaluation summary for {run_dir}. Run evaluate with "
            f"--out {run_dir / 'evaluation'} first."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    generation_run_path = run_dir / "generation_run.json"
    generation_run = (
        json.loads(generation_run_path.read_text(encoding="utf-8"))
        if generation_run_path.exists()
        else {}
    )
    config = generation_run.get("config", {})
    result = generation_run.get("result", {})
    prompt = config.get("prompt")
    return GenerationRunReview(
        run_dir=str(run_dir),
        provider=str(config.get("provider", "")),
        model_id=str(config.get("model_id", "")),
        status=str(result.get("status", "")),
        image_count=int(summary.get("image_count", 0)),
        best_image_id=summary.get("best_image_id"),
        best_centroid_score=_optional_float(summary.get("best_centroid_score")),
        mean_centroid_score=_optional_float(summary.get("mean_centroid_score")),
        median_centroid_score=_optional_float(summary.get("median_centroid_score")),
        prompt_words=len(prompt.split()) if isinstance(prompt, str) else None,
    )


def _render_generation_run_reviews_csv(reviews: list[GenerationRunReview]) -> str:
    lines = [
        "rank,run_dir,provider,model_id,status,image_count,best_image_id,"
        "best_centroid_score,mean_centroid_score,median_centroid_score,prompt_words"
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
                    _format_optional(review.best_centroid_score),
                    _format_optional(review.mean_centroid_score),
                    _format_optional(review.median_centroid_score),
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
    lines.append("| rank | run | images | best_score | mean_score | median_score | best_image |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for rank, review in enumerate(reviews, start=1):
        lines.append(
            f"| {rank} | {review.run_dir} | {review.image_count} | "
            f"{_format_optional(review.best_centroid_score)} | "
            f"{_format_optional(review.mean_centroid_score)} | "
            f"{_format_optional(review.median_centroid_score)} | "
            f"{review.best_image_id or ''} |"
        )
    lines.extend(
        [
            "",
            "Scores are approximate vector similarity for local generation-run triage only.",
            "",
        ]
    )
    return "\n".join(lines)


def _generation_run_reviews_summary(reviews: list[GenerationRunReview]) -> dict:
    best = next((review for review in reviews if review.best_centroid_score is not None), None)
    return {
        "run_count": len(reviews),
        "best_run_dir": best.run_dir if best else None,
        "best_centroid_score": _round_optional(best.best_centroid_score) if best else None,
        "runs": [
            {
                **asdict(review),
                "best_centroid_score": _round_optional(review.best_centroid_score),
                "mean_centroid_score": _round_optional(review.mean_centroid_score),
                "median_centroid_score": _round_optional(review.median_centroid_score),
            }
            for review in reviews
        ],
        "boundary": "Approximate vector similarity for local generation-run triage only.",
    }


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
