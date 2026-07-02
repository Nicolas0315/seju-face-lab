from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
from typing import Any

from .agency import build_agency_average_params
from .backends import get_vector_backend
from .face_axes import AXES, write_face_axis_report
from .metrics import score_generated_images, write_scores
from .model import load_model


DEFAULT_WEIGHTS = {
    "descriptor_similarity": 0.35,
    "image_centroid_score": 0.45,
    "axis_alignment": 0.20,
}


def write_agency_enhancement_report(
    model_dir: Path,
    agencies_config: Path,
    images: Path,
    out_dir: Path,
    crop: str = "center",
    backend_name: str = "deterministic",
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    report = build_agency_enhancement_report(
        model_dir=model_dir,
        agencies_config=agencies_config,
        images=images,
        crop=crop,
        backend_name=backend_name,
        weights=weights,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agency_enhancement_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(report["agencies"], out_dir / "agency_enhancement_scores.csv")
    (out_dir / "agency_enhancement_report.md").write_text(_render_report(report), encoding="utf-8")
    return report


def build_agency_enhancement_report(
    model_dir: Path,
    agencies_config: Path,
    images: Path,
    crop: str = "center",
    backend_name: str = "deterministic",
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    active_weights = _normalize_weights(weights or DEFAULT_WEIGHTS)
    agency_report = build_agency_average_params(model_dir, agencies_config)
    model = load_model(model_dir)
    backend = get_vector_backend(backend_name)
    failed_paths: list[str] = []
    image_scores = score_generated_images(
        model,
        images,
        crop=crop,
        backend=backend,
        failed_paths=failed_paths,
    )
    with tempfile.TemporaryDirectory() as tmp:
        axis_report = write_face_axis_report(
            images=images,
            out_dir=Path(tmp),
            crop=crop,
            backend_name=backend_name,
        )
    score_map = {score.image_id: score for score in image_scores}
    axis_map = {row["image_id"]: row for row in axis_report.get("images_detail", [])}
    agencies = [
        _enhanced_agency(agency, score_map, axis_map, active_weights)
        for agency in agency_report["agencies"]
    ]
    agencies = sorted(agencies, key=lambda item: item["enhancement_score"], reverse=True)
    for rank, agency in enumerate(agencies, start=1):
        agency["rank"] = rank
    return {
        "model_dir": str(model_dir),
        "agencies_config": str(agencies_config),
        "images": str(images),
        "backend": backend_name,
        "crop": crop,
        "weights": active_weights,
        "image_count": len(image_scores),
        "failed_count": len(failed_paths) + int(axis_report.get("failed_count", 0)),
        "failed_paths": failed_paths[:20],
        "agencies": agencies,
        "summary": _summary(agencies),
        "boundary": (
            "Enhancement scores combine hypothesis descriptors, local image-vector scores, "
            "and neutral visual-axis alignment. They are not identity, attractiveness, "
            "popularity, ethnicity, or personal-value scores."
        ),
    }


def write_agency_enhancement_bundle(
    model_dir: Path,
    agencies_config: Path,
    images: Path,
    out_dir: Path,
    crop: str = "center",
    backend_name: str = "deterministic",
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(model_dir)
    backend = get_vector_backend(backend_name)
    failed_paths: list[str] = []
    scores = score_generated_images(model, images, crop=crop, backend=backend, failed_paths=failed_paths)
    write_scores(scores, out_dir / "image_scores", failed_paths=failed_paths, model=model)
    axis_report = write_face_axis_report(
        images=images,
        out_dir=out_dir / "face_axes",
        crop=crop,
        backend_name=backend_name,
    )
    active_weights = _normalize_weights(weights or DEFAULT_WEIGHTS)
    agency_report = build_agency_average_params(model_dir, agencies_config)
    score_map = {score.image_id: score for score in scores}
    axis_map = {row["image_id"]: row for row in axis_report.get("images_detail", [])}
    agencies = [
        _enhanced_agency(agency, score_map, axis_map, active_weights)
        for agency in agency_report["agencies"]
    ]
    agencies = sorted(agencies, key=lambda item: item["enhancement_score"], reverse=True)
    for rank, agency in enumerate(agencies, start=1):
        agency["rank"] = rank
    report = {
        "model_dir": str(model_dir),
        "agencies_config": str(agencies_config),
        "images": str(images),
        "backend": backend_name,
        "crop": crop,
        "weights": active_weights,
        "image_count": len(scores),
        "failed_count": len(failed_paths) + int(axis_report.get("failed_count", 0)),
        "failed_paths": failed_paths[:20],
        "agencies": agencies,
        "summary": _summary(agencies),
        "outputs": {
            "image_scores": str(out_dir / "image_scores"),
            "face_axes": str(out_dir / "face_axes"),
        },
        "boundary": (
            "Enhancement scores combine hypothesis descriptors, local image-vector scores, "
            "and neutral visual-axis alignment. They are not identity, attractiveness, "
            "popularity, ethnicity, or personal-value scores."
        ),
    }
    (out_dir / "agency_enhancement_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(report["agencies"], out_dir / "agency_enhancement_scores.csv")
    (out_dir / "agency_enhancement_report.md").write_text(_render_report(report), encoding="utf-8")
    return report


def _enhanced_agency(
    agency: dict[str, Any],
    score_map: dict[str, Any],
    axis_map: dict[str, dict[str, Any]],
    weights: dict[str, float],
) -> dict[str, Any]:
    slug = agency["slug"]
    image_score = score_map.get(slug)
    image_axes = axis_map.get(slug)
    descriptor_similarity = float(agency["descriptor_similarity"])
    centroid_score = float(image_score.centroid_score) if image_score else None
    axis_alignment = _axis_alignment(agency.get("axis_vector", {}), image_axes.get("axes", {}) if image_axes else {})
    normalized_centroid = _normalize_centroid_score(centroid_score)
    components = {
        "descriptor_similarity": descriptor_similarity,
        "image_centroid_score": centroid_score,
        "image_centroid_score_normalized": normalized_centroid,
        "axis_alignment": axis_alignment,
    }
    enhancement_score = (
        descriptor_similarity * weights["descriptor_similarity"]
        + normalized_centroid * weights["image_centroid_score"]
        + axis_alignment * weights["axis_alignment"]
    )
    return {
        "slug": slug,
        "name": agency["name"],
        "rank": None,
        "enhancement_score": round(enhancement_score, 6),
        "confidence": _confidence(image_score is not None, image_axes is not None),
        "components": {key: _round_optional(value) for key, value in components.items()},
        "hypothesis_axis_vector": agency.get("axis_vector", {}),
        "observed_axis_vector": image_axes.get("axes", {}) if image_axes else {},
        "observed_distribution": _observed_distribution(image_axes),
        "presentation_flags": image_axes.get("presentation_flags", []) if image_axes else ["missing_image"],
        "improvement_actions": _improvement_actions(centroid_score, axis_alignment, image_axes),
        "prompt_path_hint": f"prompts/{slug}.txt",
        "boundary": "Agency row is aggregate research evidence, not a label for any person.",
    }


def _axis_alignment(expected: dict[str, Any], observed: dict[str, Any]) -> float:
    if not expected or not observed:
        return 0.0
    deltas = []
    for axis in AXES:
        deltas.append(abs(float(expected.get(axis, 0.0)) - float(observed.get(axis, 0.0))) / 2.0)
    score = 1.0 - (sum(deltas) / len(deltas))
    return round(max(0.0, min(1.0, score)), 6)


def _normalize_centroid_score(value: float | None) -> float:
    if value is None:
        return 0.0
    return round(max(0.0, min(1.0, (value + 1.0) / 2.0)), 6)


def _confidence(has_score: bool, has_axes: bool) -> str:
    if has_score and has_axes:
        return "measured"
    if has_score or has_axes:
        return "partial"
    return "hypothesis_only"


def _observed_distribution(image_axes: dict[str, Any] | None) -> dict[str, Any]:
    if not image_axes:
        return {}
    return {
        "quadrant": image_axes.get("quadrant"),
        "corner": image_axes.get("corner"),
        "cross_label": image_axes.get("cross_label"),
        "outlier_score": image_axes.get("outlier_score"),
    }


def _improvement_actions(
    centroid_score: float | None,
    axis_alignment: float,
    image_axes: dict[str, Any] | None,
) -> list[str]:
    actions = []
    flags = image_axes.get("presentation_flags", []) if image_axes else ["missing_image"]
    if centroid_score is None:
        actions.append("add_or_regenerate_named_sample")
    elif centroid_score < 0.1:
        actions.append("regenerate_with_detector_friendly_prompt")
    elif centroid_score < 0.25:
        actions.append("run_second_seed_and_compare")
    if axis_alignment < 0.55:
        actions.append("adjust_prompt_to_match_hypothesis_axes")
    if "dark_or_underlit_image" in flags:
        actions.append("increase_even_front_lighting")
    if "off_center_or_asymmetric_visibility" in flags:
        actions.append("enforce_centered_frontal_crop")
    if "dark_upper_band_or_hair_shadow" in flags:
        actions.append("reduce_hair_shadow_over_face")
    if "high_texture_or_messy_edges" in flags:
        actions.append("reduce_edge_noise_and_hair_flyaways")
    if not actions:
        actions.append("keep_as_baseline_candidate")
    return actions


def _summary(agencies: list[dict[str, Any]]) -> dict[str, Any]:
    if not agencies:
        return {"agency_count": 0}
    measured = [agency for agency in agencies if agency["confidence"] == "measured"]
    top = agencies[0]
    return {
        "agency_count": len(agencies),
        "measured_count": len(measured),
        "top_slug": top["slug"],
        "top_score": top["enhancement_score"],
        "next_actions": _top_actions(agencies),
    }


def _top_actions(agencies: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for agency in agencies:
        for action in agency["improvement_actions"]:
            counts[action] = counts.get(action, 0) + 1
    return [
        action
        for action, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]
    ]


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    required = DEFAULT_WEIGHTS.keys()
    raw = {key: max(0.0, float(weights.get(key, 0.0))) for key in required}
    total = sum(raw.values()) or 1.0
    return {key: round(value / total, 6) for key, value in raw.items()}


def _write_csv(agencies: list[dict[str, Any]], path: Path) -> None:
    headers = [
        "rank",
        "slug",
        "name",
        "enhancement_score",
        "confidence",
        "descriptor_similarity",
        "image_centroid_score",
        "axis_alignment",
        "quadrant",
        "corner",
        "presentation_flags",
        "improvement_actions",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for agency in agencies:
            components = agency["components"]
            distribution = agency["observed_distribution"]
            writer.writerow(
                {
                    "rank": agency["rank"],
                    "slug": agency["slug"],
                    "name": agency["name"],
                    "enhancement_score": agency["enhancement_score"],
                    "confidence": agency["confidence"],
                    "descriptor_similarity": components["descriptor_similarity"],
                    "image_centroid_score": components["image_centroid_score"],
                    "axis_alignment": components["axis_alignment"],
                    "quadrant": distribution.get("quadrant", ""),
                    "corner": distribution.get("corner", ""),
                    "presentation_flags": ";".join(agency["presentation_flags"]),
                    "improvement_actions": ";".join(agency["improvement_actions"]),
                }
            )


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        "# agency enhancement report",
        "",
        f"- model_dir: {report['model_dir']}",
        f"- agencies_config: {report['agencies_config']}",
        f"- images: {report['images']}",
        f"- backend: {report['backend']}",
        f"- image_count: {report['image_count']}",
        f"- failed_count: {report['failed_count']}",
        f"- top_slug: {report['summary'].get('top_slug')}",
        f"- top_score: {report['summary'].get('top_score')}",
        "",
        "## Ranking",
        "",
        "| rank | agency | score | confidence | descriptor | image | axis | actions |",
        "| ---: | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for agency in report["agencies"]:
        components = agency["components"]
        lines.append(
            f"| {agency['rank']} | {agency['name']} | {agency['enhancement_score']} | "
            f"{agency['confidence']} | {components['descriptor_similarity']} | "
            f"{components['image_centroid_score']} | {components['axis_alignment']} | "
            f"{', '.join(agency['improvement_actions'])} |"
        )
    lines.extend(["", "## Boundary", "", report["boundary"], ""])
    return "\n".join(lines)


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)
