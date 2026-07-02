from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STRICT_RUBRIC_BOUNDARY = (
    "Strict rubric review is local generated-image triage only. It is not identity, "
    "attractiveness, ethnicity, personality, or person-matching analysis."
)

SEVERE_PRESENTATION_FLAGS = {
    "dark_or_underlit_image",
    "off_center_or_asymmetric_visibility",
    "high_texture_or_messy_edges",
}


def write_pairwise_rubric_review(
    evaluation: Path,
    out_dir: Path,
    quality: Path | None = None,
    face_axes: Path | None = None,
    min_centroid_score: float = 0.35,
    min_null_percentile: float = 0.95,
    min_pairwise_gap: float = 0.05,
) -> dict[str, Any]:
    evaluation_summary = _load_json_report(evaluation, "summary.json")
    quality_summary = _load_optional_json_report(quality, "image_quality.json")
    face_axis_report = _load_optional_json_report(face_axes, "face_axis_report.json")

    quality_by_image = _index_by_image(quality_summary.get("images", []) if quality_summary else [])
    axes_by_image = _index_by_image(face_axis_report.get("images_detail", []) if face_axis_report else [])
    top_images = _top_images(evaluation_summary)

    batch_null = _batch_null_review(evaluation_summary, min_null_percentile)
    candidates = [
        _candidate_review(
            image,
            quality_by_image.get(str(image.get("image_id", ""))),
            axes_by_image.get(str(image.get("image_id", ""))),
            min_centroid_score,
        )
        for image in top_images
    ]
    comparisons = _pairwise_comparisons(candidates, min_pairwise_gap)
    recommendation = _batch_recommendation(candidates, batch_null)
    report = {
        "evaluation": str(_resolve_report_path(evaluation, "summary.json")),
        "quality": str(_resolve_report_path(quality, "image_quality.json")) if quality else None,
        "face_axes": str(_resolve_report_path(face_axes, "face_axis_report.json")) if face_axes else None,
        "thresholds": {
            "min_centroid_score": min_centroid_score,
            "min_null_percentile": min_null_percentile,
            "min_pairwise_gap": min_pairwise_gap,
        },
        "batch_null_review": batch_null,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "pairwise_comparisons": comparisons,
        "recommendation": recommendation,
        "boundary": STRICT_RUBRIC_BOUNDARY,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pairwise_rubric.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "pairwise_rubric.csv").write_text(_render_candidate_csv(candidates), encoding="utf-8-sig")
    (out_dir / "pairwise_rubric.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


def _load_json_report(path: Path, filename: str) -> dict[str, Any]:
    report_path = _resolve_report_path(path, filename)
    if not report_path.exists():
        raise SystemExit(f"Missing report file: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def _load_optional_json_report(path: Path | None, filename: str) -> dict[str, Any] | None:
    if path is None:
        return None
    return _load_json_report(path, filename)


def _resolve_report_path(path: Path | None, filename: str) -> Path:
    if path is None:
        raise ValueError("path is required")
    if path.is_dir():
        return path / filename
    return path


def _top_images(summary: dict[str, Any]) -> list[dict[str, Any]]:
    images = summary.get("top_images", [])
    if not isinstance(images, list):
        return []
    return sorted(
        [image for image in images if isinstance(image, dict)],
        key=lambda item: _float(item.get("centroid_score")),
        reverse=True,
    )


def _index_by_image(rows: list[Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        image_id = row.get("image_id")
        if image_id:
            indexed[str(image_id)] = row
    return indexed


def _batch_null_review(summary: dict[str, Any], min_null_percentile: float) -> dict[str, Any]:
    calibration = summary.get("null_calibration", {})
    if not isinstance(calibration, dict) or not calibration.get("available"):
        return {
            "available": False,
            "pass": None,
            "observed_best_percentile": None,
            "reason": "null calibration unavailable",
        }
    observed = calibration.get("observed_percentiles", {})
    percentile = _float(observed.get("best_centroid_score") if isinstance(observed, dict) else None)
    return {
        "available": True,
        "pass": percentile >= min_null_percentile,
        "observed_best_percentile": round(percentile, 6),
        "reason": (
            "best centroid score clears null baseline"
            if percentile >= min_null_percentile
            else "best centroid score is not sufficiently above null baseline"
        ),
    }


def _candidate_review(
    image: dict[str, Any],
    quality: dict[str, Any] | None,
    axes: dict[str, Any] | None,
    min_centroid_score: float,
) -> dict[str, Any]:
    image_id = str(image.get("image_id", ""))
    centroid_score = _float(image.get("centroid_score"))
    quality_pass = quality.get("qa_pass") if quality else None
    quality_reason = quality.get("reason") if quality else "quality report unavailable"
    presentation_flags = _presentation_flags(axes)
    severe_flags = sorted(flag for flag in presentation_flags if flag in SEVERE_PRESENTATION_FLAGS)
    checks = {
        "centroid_score": {
            "pass": centroid_score >= min_centroid_score,
            "value": round(centroid_score, 6),
            "threshold": min_centroid_score,
        },
        "quality": {
            "pass": quality_pass,
            "reason": quality_reason,
        },
        "presentation": {
            "pass": not severe_flags if axes else None,
            "flags": presentation_flags,
            "severe_flags": severe_flags,
        },
    }
    failed = [
        name
        for name, check in checks.items()
        if isinstance(check, dict) and check.get("pass") is False
    ]
    if failed:
        recommendation = "reject"
    elif any(isinstance(check, dict) and check.get("pass") is None for check in checks.values()):
        recommendation = "review"
    else:
        recommendation = "promote"
    return {
        "image_id": image_id,
        "path": str(image.get("path", "")),
        "centroid_score": round(centroid_score, 6),
        "checks": checks,
        "failed_checks": failed,
        "recommendation": recommendation,
    }


def _presentation_flags(axes: dict[str, Any] | None) -> list[str]:
    if not axes:
        return []
    flags = axes.get("presentation_flags", [])
    if isinstance(flags, list):
        return [str(flag) for flag in flags]
    return []


def _pairwise_comparisons(candidates: list[dict[str, Any]], min_pairwise_gap: float) -> list[dict[str, Any]]:
    comparisons = []
    for index, winner in enumerate(candidates):
        for loser in candidates[index + 1 :]:
            gap = _float(winner.get("centroid_score")) - _float(loser.get("centroid_score"))
            comparisons.append(
                {
                    "winner": winner["image_id"],
                    "loser": loser["image_id"],
                    "centroid_gap": round(gap, 6),
                    "decision": "clear_win" if gap >= min_pairwise_gap else "too_close",
                }
            )
    return comparisons


def _batch_recommendation(candidates: list[dict[str, Any]], batch_null: dict[str, Any]) -> dict[str, Any]:
    if not candidates:
        return {"decision": "reject", "image_id": None, "reason": "no candidates"}
    best = candidates[0]
    if batch_null.get("pass") is False:
        return {
            "decision": "review",
            "image_id": best["image_id"],
            "reason": "batch null calibration did not clear strict threshold",
        }
    return {
        "decision": best["recommendation"],
        "image_id": best["image_id"],
        "reason": "best candidate strict rubric result",
    }


def _render_candidate_csv(candidates: list[dict[str, Any]]) -> str:
    lines = [
        "image_id,path,centroid_score,recommendation,failed_checks,qa_pass,quality_reason,presentation_flags,severe_flags"
    ]
    for candidate in candidates:
        checks = candidate["checks"]
        quality = checks["quality"]
        presentation = checks["presentation"]
        lines.append(
            ",".join(
                [
                    _csv(candidate["image_id"]),
                    _csv(candidate["path"]),
                    f"{candidate['centroid_score']:.6f}",
                    _csv(candidate["recommendation"]),
                    _csv(";".join(candidate["failed_checks"])),
                    _csv("" if quality["pass"] is None else str(quality["pass"]).lower()),
                    _csv(str(quality["reason"])),
                    _csv(";".join(presentation["flags"])),
                    _csv(";".join(presentation["severe_flags"])),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pairwise Strict Rubric Review",
        "",
        f"- recommendation: {report['recommendation']['decision']} ({report['recommendation']['image_id']})",
        f"- candidates: {report['candidate_count']}",
        f"- null calibration: {report['batch_null_review']['reason']}",
        "",
        "## Candidates",
        "",
        "| image_id | centroid_score | recommendation | failed_checks |",
        "| --- | ---: | --- | --- |",
    ]
    for candidate in report["candidates"]:
        failed = ", ".join(candidate["failed_checks"]) or "-"
        lines.append(
            f"| {candidate['image_id']} | {candidate['centroid_score']:.6f} | "
            f"{candidate['recommendation']} | {failed} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise Comparisons",
            "",
            "| winner | loser | centroid_gap | decision |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for comparison in report["pairwise_comparisons"]:
        lines.append(
            f"| {comparison['winner']} | {comparison['loser']} | "
            f"{comparison['centroid_gap']:.6f} | {comparison['decision']} |"
        )
    lines.extend(["", f"Boundary: {report['boundary']}", ""])
    return "\n".join(lines)


def _csv(value: object) -> str:
    text = str(value)
    if any(char in text for char in [",", '"', "\n"]):
        return '"' + text.replace('"', '""') + '"'
    return text


def _float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
