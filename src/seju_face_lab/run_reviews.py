from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from html import escape
import json
from pathlib import Path


@dataclass(frozen=True)
class GenerationRunReview:
    run_dir: str
    provider: str
    model_id: str
    status: str
    centroid_kind: str
    prompt_profile: str
    seed: int | None
    planned_count: int | None
    steps: int | None
    guidance_scale: float | None
    size: str
    device: str
    dtype: str
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
    qa_pass_count: int | None
    qa_fail_count: int | None
    qa_pass_rate: float | None
    best_qa_image_id: str | None
    best_qa_path: str | None
    best_qa_centroid_score: float | None
    prompt_words: int | None


@dataclass(frozen=True)
class CandidateReview:
    image_id: str
    path: str
    centroid_score: float | None
    style_score: float | None
    combined_score: float | None
    qa_pass: bool | None
    qa_reason: str | None


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
    (out_dir / "generation_run_reviews.html").write_text(
        _render_generation_run_reviews_html(reviews),
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
    quality_summary = _quality_summary(run_dir)
    best_qa_image_id, best_qa_path, best_qa_centroid = _best_quality_centroid_score(run_dir)
    return GenerationRunReview(
        run_dir=str(run_dir),
        provider=str(config.get("provider", "")),
        model_id=str(config.get("model_id", "")),
        status=str(result.get("status", "")),
        centroid_kind=str(config.get("centroid_kind", "")),
        prompt_profile=str(config.get("prompt_profile", "")),
        seed=_optional_int(config.get("seed")),
        planned_count=_optional_int(config.get("count")),
        steps=_optional_int(config.get("steps")),
        guidance_scale=_optional_float(config.get("guidance_scale")),
        size=_generation_size(config),
        device=str(config.get("device", "")),
        dtype=str(config.get("dtype", "")),
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
        qa_pass_count=_optional_int(quality_summary.get("pass_count")),
        qa_fail_count=_optional_int(quality_summary.get("fail_count")),
        qa_pass_rate=_optional_float(quality_summary.get("pass_rate")),
        best_qa_image_id=best_qa_image_id,
        best_qa_path=best_qa_path,
        best_qa_centroid_score=best_qa_centroid,
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


def _quality_summary_path(run_dir: Path) -> Path:
    nested = run_dir / "quality" / "image_quality.json"
    if nested.exists():
        return nested
    if _is_evaluation_output_dir(run_dir):
        sibling = run_dir.parent / "quality" / "image_quality.json"
        if sibling.exists():
            return sibling
    return run_dir / "image_quality.json"


def _quality_csv_path(run_dir: Path) -> Path:
    nested = run_dir / "quality" / "image_quality.csv"
    if nested.exists():
        return nested
    if _is_evaluation_output_dir(run_dir):
        sibling = run_dir.parent / "quality" / "image_quality.csv"
        if sibling.exists():
            return sibling
    return run_dir / "image_quality.csv"


def _is_evaluation_output_dir(run_dir: Path) -> bool:
    return (run_dir / "summary.json").exists() and (run_dir / "scores.csv").exists()


def _render_generation_run_reviews_csv(reviews: list[GenerationRunReview]) -> str:
    lines = [
        "rank,run_dir,provider,model_id,status,centroid_kind,image_count,best_image_id,"
        "failed_count,best_centroid_score,mean_centroid_score,median_centroid_score,"
        "best_style_score,mean_style_score,median_style_score,best_combined_image_id,"
        "best_combined_path,best_combined_score,qa_pass_count,qa_fail_count,qa_pass_rate,"
        "best_qa_image_id,best_qa_path,best_qa_centroid_score,prompt_profile,seed,"
        "planned_count,steps,guidance_scale,size,device,dtype,prompt_words"
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
                    _csv(review.centroid_kind),
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
                    "" if review.qa_pass_count is None else str(review.qa_pass_count),
                    "" if review.qa_fail_count is None else str(review.qa_fail_count),
                    _format_optional(review.qa_pass_rate),
                    _csv(review.best_qa_image_id or ""),
                    _csv(review.best_qa_path or ""),
                    _format_optional(review.best_qa_centroid_score),
                    _csv(review.prompt_profile),
                    "" if review.seed is None else str(review.seed),
                    "" if review.planned_count is None else str(review.planned_count),
                    "" if review.steps is None else str(review.steps),
                    _format_optional(review.guidance_scale),
                    _csv(review.size),
                    _csv(review.device),
                    _csv(review.dtype),
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
        "| rank | run | provider | centroid | profile | seed | images | failed | qa_pass | best_face | qa_face | best_style | combined | best_image | qa_image |"
    )
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for rank, review in enumerate(reviews, start=1):
        lines.append(
            f"| {rank} | {review.run_dir} | {review.provider} | {review.centroid_kind} | {review.prompt_profile} | "
            f"{'' if review.seed is None else review.seed} | {review.image_count} | {review.failed_count} | "
            f"{'' if review.qa_pass_count is None else review.qa_pass_count} | "
            f"{_format_optional(review.best_centroid_score)} | "
            f"{_format_optional(review.best_qa_centroid_score)} | "
            f"{_format_optional(review.best_style_score)} | "
            f"{_format_optional(review.best_combined_score)} | "
            f"{review.best_image_id or ''} | "
            f"{review.best_qa_image_id or ''} |"
        )
    lines.extend(
        [
            "",
            "Face scores are approximate vector similarity against the local centroid model.",
            "QA face scores use only generated images that pass the OpenCV single-centered-face quality gate when available.",
            "Style scores are approximate OpenCLIP image-style similarity. Combined score is the best per-image average when both are present.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_generation_run_reviews_html(reviews: list[GenerationRunReview]) -> str:
    cards = "\n".join(_render_run_card(rank, review) for rank, review in enumerate(reviews, start=1))
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>generation run comparison</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f4;color:#1f2933}",
            "h1{font-size:24px;margin:0 0 16px}",
            ".run{background:#fff;border:1px solid #d7d7d0;border-radius:8px;margin:0 0 18px;padding:14px}",
            ".meta{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0 12px;color:#4b5563;font-size:13px}",
            ".pill{background:#eef2f7;border-radius:999px;padding:3px 8px}",
            ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}",
            ".candidate{border:1px solid #e2e2dc;border-radius:8px;padding:8px;background:#fbfbf9}",
            ".candidate img{width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:6px;background:#e7e5dc}",
            ".scores{font-size:12px;line-height:1.45;margin-top:6px}",
            ".path{overflow-wrap:anywhere;color:#59636e}",
            ".boundary{font-size:12px;color:#667085;margin-top:18px}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>generation run comparison</h1>",
            cards or "<p>No generation runs provided.</p>",
            '<p class="boundary">Scores are approximate local triage against this centroid model. '
            "They are not identity, attractiveness, ethnicity, or objective face-type labels.</p>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _render_run_card(rank: int, review: GenerationRunReview) -> str:
    run_dir = Path(review.run_dir)
    candidates = _candidate_reviews(run_dir)[:12]
    image_base_dir = _image_base_dir(run_dir)
    candidate_html = "\n".join(
        _render_candidate_card(candidate, image_base_dir) for candidate in candidates
    )
    if not candidate_html:
        candidate_html = "<p>No scored candidate rows found.</p>"
    title = f"#{rank} {escape(review.run_dir)}"
    return "\n".join(
        [
            '<section class="run">',
            f"<h2>{title}</h2>",
            '<div class="meta">',
            f'<span class="pill">images {review.image_count}</span>',
            f'<span class="pill">failed {review.failed_count}</span>',
            f'<span class="pill">provider {escape(review.provider)}</span>',
            f'<span class="pill">centroid {escape(review.centroid_kind)}</span>',
            f'<span class="pill">profile {escape(review.prompt_profile)}</span>',
            f'<span class="pill">seed {"" if review.seed is None else review.seed}</span>',
            f'<span class="pill">best face {_format_optional(review.best_centroid_score)}</span>',
            f'<span class="pill">QA face {_format_optional(review.best_qa_centroid_score)}</span>',
            f'<span class="pill">style {_format_optional(review.best_style_score)}</span>',
            f'<span class="pill">combined {_format_optional(review.best_combined_score)}</span>',
            "</div>",
            f'<div class="grid">{candidate_html}</div>',
            "</section>",
        ]
    )


def _render_candidate_card(candidate: CandidateReview, run_dir: Path) -> str:
    image_src = _image_src(candidate.path, run_dir)
    qa = "unknown" if candidate.qa_pass is None else ("pass" if candidate.qa_pass else "fail")
    reason = f"<div>qa_reason: {escape(candidate.qa_reason)}</div>" if candidate.qa_reason else ""
    return "\n".join(
        [
            '<article class="candidate">',
            f'<img src="{escape(image_src)}" alt="{escape(candidate.image_id)}">',
            '<div class="scores">',
            f"<strong>{escape(candidate.image_id)}</strong>",
            f"<div>face: {_format_optional(candidate.centroid_score)}</div>",
            f"<div>style: {_format_optional(candidate.style_score)}</div>",
            f"<div>combined: {_format_optional(candidate.combined_score)}</div>",
            f"<div>qa: {qa}</div>",
            reason,
            f'<div class="path">{escape(candidate.path)}</div>',
            "</div>",
            "</article>",
        ]
    )


def _candidate_reviews(run_dir: Path) -> list[CandidateReview]:
    face_rows = _read_score_rows(_scores_csv_path(run_dir), "centroid_score")
    style_rows = _read_score_rows(_style_scores_csv_path(run_dir), "style_score")
    style_scores = _read_score_column(_style_scores_csv_path(run_dir), "style_score")
    quality = _read_quality_rows(_quality_csv_path(run_dir))
    candidates: list[CandidateReview] = []
    consumed_style_keys: set[str] = set()
    for image_id, path, face_score, keys in face_rows:
        style = _first_lookup(style_scores, keys)
        qa_pass, qa_reason = _first_lookup(quality, keys) or (None, None)
        style_score = style[2] if style else None
        if style:
            consumed_style_keys.update(keys)
        candidates.append(_candidate_review(image_id, path, face_score, style_score, qa_pass, qa_reason))
    for image_id, path, style_score, keys in style_rows:
        if consumed_style_keys & keys:
            continue
        qa_pass, qa_reason = _first_lookup(quality, keys) or (None, None)
        candidates.append(_candidate_review(image_id, path, None, style_score, qa_pass, qa_reason))
    return sorted(candidates, key=_candidate_sort_key, reverse=True)


def _candidate_review(
    image_id: str,
    path: str,
    face_score: float | None,
    style_score: float | None,
    qa_pass: bool | None,
    qa_reason: str | None,
) -> CandidateReview:
    combined = (face_score + style_score) / 2.0 if face_score is not None and style_score is not None else None
    return CandidateReview(
        image_id=image_id,
        path=path,
        centroid_score=face_score,
        style_score=style_score,
        combined_score=combined,
        qa_pass=qa_pass,
        qa_reason=qa_reason,
    )


def _first_lookup(mapping: dict[str, object], keys: set[str]) -> object | None:
    for key in sorted(keys):
        if key in mapping:
            return mapping[key]
    return None


def _candidate_sort_key(candidate: CandidateReview) -> tuple[int, float]:
    if candidate.qa_pass and candidate.centroid_score is not None:
        return (3, candidate.centroid_score)
    if candidate.combined_score is not None:
        return (2, candidate.combined_score)
    if candidate.centroid_score is not None:
        return (1, candidate.centroid_score)
    if candidate.style_score is not None:
        return (1, candidate.style_score)
    return (0, 0.0)


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
        "best_qa_image_id": best.best_qa_image_id if best else None,
        "best_qa_path": best.best_qa_path if best else None,
        "best_qa_centroid_score": _round_optional(best.best_qa_centroid_score) if best else None,
        "best_generation": _generation_config_summary(best),
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
                "qa_pass_rate": _round_optional(review.qa_pass_rate),
                "best_qa_image_id": review.best_qa_image_id,
                "best_qa_path": review.best_qa_path,
                "best_qa_centroid_score": _round_optional(review.best_qa_centroid_score),
                "guidance_scale": _round_optional(review.guidance_scale),
            }
            for review in reviews
        ],
        "boundary": (
            "Approximate local generation-run triage only. Combined score mixes face geometry "
            "and style axes when both are present. QA scores are OpenCV detector-gated triage only."
        ),
    }


def _quality_summary(run_dir: Path) -> dict:
    path = _quality_summary_path(run_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _generation_config_summary(review: GenerationRunReview | None) -> dict[str, object] | None:
    if review is None:
        return None
    return {
        "provider": review.provider,
        "model_id": review.model_id,
        "status": review.status,
        "centroid_kind": review.centroid_kind,
        "prompt_profile": review.prompt_profile,
        "seed": review.seed,
        "planned_count": review.planned_count,
        "steps": review.steps,
        "guidance_scale": _round_optional(review.guidance_scale),
        "size": review.size,
        "device": review.device,
        "dtype": review.dtype,
        "prompt_words": review.prompt_words,
    }


def _generation_size(config: dict) -> str:
    width = config.get("width")
    height = config.get("height")
    if width is None or height is None:
        return ""
    return f"{width}x{height}"


def _best_quality_centroid_score(run_dir: Path) -> tuple[str | None, str | None, float | None]:
    face_scores = _read_score_column(_scores_csv_path(run_dir), "centroid_score")
    qa_passes = _read_quality_passes(_quality_csv_path(run_dir))
    best_image_id = None
    best_path = None
    best_score = None
    for image_path in sorted(face_scores.keys() & qa_passes):
        image_id, raw_path, face_score = face_scores[image_path]
        if best_score is None or face_score > best_score:
            best_image_id = image_id
            best_path = raw_path
            best_score = face_score
    return best_image_id, best_path, best_score


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


def _read_score_rows(path: Path, column: str) -> list[tuple[str, str, float, set[str]]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if (
            not rows.fieldnames
            or "image_id" not in rows.fieldnames
            or "path" not in rows.fieldnames
            or column not in rows.fieldnames
        ):
            return []
        base_dirs = _score_path_base_dirs(path)
        score_rows = []
        for row in rows:
            if not row.get("image_id") or not row.get("path") or not row.get(column):
                continue
            raw_path = str(row["path"])
            score_rows.append(
                (
                    str(row["image_id"]),
                    raw_path,
                    float(row[column]),
                    _path_keys(raw_path, base_dirs),
                )
            )
        return score_rows


def _read_quality_passes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or "path" not in rows.fieldnames or "qa_pass" not in rows.fieldnames:
            return set()
        passes: set[str] = set()
        base_dirs = _score_path_base_dirs(path)
        for row in rows:
            if str(row.get("qa_pass", "")).lower() != "true" or not row.get("path"):
                continue
            passes.update(_path_keys(str(row["path"]), base_dirs))
        return passes


def _read_quality_rows(path: Path) -> dict[str, tuple[bool, str | None]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or "path" not in rows.fieldnames or "qa_pass" not in rows.fieldnames:
            return {}
        quality: dict[str, tuple[bool, str | None]] = {}
        base_dirs = _score_path_base_dirs(path)
        for row in rows:
            if not row.get("path"):
                continue
            qa_pass = str(row.get("qa_pass", "")).lower() == "true"
            reason = row.get("reason")
            for key in _path_keys(str(row["path"]), base_dirs):
                quality[key] = (qa_pass, reason)
        return quality


def _score_path_base_dirs(csv_path: Path) -> list[Path]:
    bases = [Path.cwd(), csv_path.parent]
    if csv_path.parent.name in {"evaluation", "style_evaluation", "quality"}:
        bases.append(csv_path.parent.parent)
    return bases


def _path_keys(value: str, base_dirs: list[Path]) -> set[str]:
    path = Path(value).expanduser()
    if path.is_absolute():
        return {_normal_path(path)}
    return {_normal_path(base / path) for base in base_dirs}


def _normal_path(path: Path) -> str:
    return str(path.resolve(strict=False)).casefold()


def _image_src(value: str, run_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        cwd_path = (Path.cwd() / path).resolve(strict=False)
        run_path = (run_dir / path).resolve(strict=False)
        path = cwd_path if cwd_path.exists() else run_path
    try:
        return path.as_uri()
    except ValueError:
        return value


def _image_base_dir(run_dir: Path) -> Path:
    if _is_evaluation_output_dir(run_dir):
        return run_dir.parent
    return run_dir


def _has_review_score(review: GenerationRunReview) -> bool:
    return (
        review.best_qa_centroid_score is not None
        or review.best_combined_score is not None
        or review.best_centroid_score is not None
    )


def _review_sort_key(review: GenerationRunReview) -> tuple[int, float]:
    if review.best_qa_centroid_score is not None:
        return (2, review.best_qa_centroid_score)
    if review.best_combined_score is not None:
        return (1, review.best_combined_score)
    if review.best_centroid_score is not None:
        return (1, review.best_centroid_score)
    return (0, 0.0)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


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
