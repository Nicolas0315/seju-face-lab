from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .backends import get_vector_backend
from .embeddings import iter_image_paths


AXES = [
    "soft_defined",
    "cool_warm",
    "deep_bright",
    "natural_styled",
    "muted_vivid",
    "soft_crisp",
    "light_dark_hair",
    "dynamic_symmetric",
]


def axis_vector_from_descriptors(descriptors: dict[str, Any]) -> dict[str, float]:
    d = {str(key): _float(value) for key, value in descriptors.items()}
    contrast = _normalize(d.get("contrast", 0.0), 0.04, 0.24)
    edge = _normalize(d.get("edge_density", 0.0), 0.01, 0.12)
    luminance = _normalize(d.get("luminance", 0.5), 0.35, 0.82)
    saturation = _normalize(d.get("saturation", 0.0), 0.02, 0.22)
    warmth = _normalize(d.get("warmth", 0.0), -0.08, 0.08)
    symmetry = _normalize(d.get("symmetry", 0.85), 0.75, 0.99)
    hair_darkness = _normalize(d.get("upper_band_darkness", 0.2), 0.05, 0.48)
    middle_luminance = _normalize(d.get("middle_luminance", 0.5), 0.35, 0.82)
    lower_luminance = _normalize(d.get("lower_luminance", 0.5), 0.35, 0.82)
    makeup_proxy = _average([contrast, saturation])
    return {
        "soft_defined": _round(_average([contrast, edge])),
        "cool_warm": _round(warmth),
        "deep_bright": _round(_average([luminance, middle_luminance, lower_luminance])),
        "natural_styled": _round(makeup_proxy),
        "muted_vivid": _round(saturation),
        "soft_crisp": _round(edge),
        "light_dark_hair": _round(hair_darkness),
        "dynamic_symmetric": _round(symmetry),
    }


def axis_distribution(axis_vector: dict[str, float]) -> dict[str, Any]:
    x = _round(_average([axis_vector["soft_defined"], axis_vector["natural_styled"]]))
    y = _round(_average([axis_vector["deep_bright"], axis_vector["cool_warm"]]))
    cross_x = _round(axis_vector["dynamic_symmetric"])
    cross_y = _round(axis_vector["light_dark_hair"])
    return {
        "quadrant_x": x,
        "quadrant_y": y,
        "quadrant": _quadrant(x, y),
        "corner": _corner(x, y),
        "cross_x": cross_x,
        "cross_y": cross_y,
        "cross_label": _cross_label(cross_x, cross_y),
        "presentation_flags": presentation_flags(axis_vector),
        "outlier_score": outlier_score(axis_vector),
        "axes": axis_vector,
    }


def presentation_flags(axis_vector: dict[str, float]) -> list[str]:
    flags = []
    if axis_vector["deep_bright"] <= -0.55:
        flags.append("dark_or_underlit_image")
    if axis_vector["soft_defined"] >= 0.70:
        flags.append("high_contrast_or_heavy_definition")
    if axis_vector["soft_crisp"] >= 0.70:
        flags.append("high_texture_or_messy_edges")
    if axis_vector["dynamic_symmetric"] <= -0.45:
        flags.append("off_center_or_asymmetric_visibility")
    if axis_vector["light_dark_hair"] >= 0.75 and axis_vector["deep_bright"] <= -0.20:
        flags.append("dark_upper_band_or_hair_shadow")
    if axis_vector["natural_styled"] >= 0.70:
        flags.append("strong_styling_or_makeup_signal")
    if not flags:
        flags.append("no_major_presentation_flags")
    return flags


def outlier_score(axis_vector: dict[str, float]) -> float:
    distance = sum(abs(axis_vector[axis]) for axis in AXES) / len(AXES)
    return _round(distance)


def write_face_axis_report(
    images: Path,
    out_dir: Path,
    crop: str = "center",
    backend_name: str = "deterministic",
) -> dict[str, Any]:
    backend = get_vector_backend(backend_name)
    rows = []
    failures = []
    for path in iter_image_paths(images):
        try:
            vector = backend.vectorize(path, crop=crop)
        except Exception as exc:  # noqa: BLE001 - report bad images without stopping the batch.
            failures.append({"path": str(path), "error": str(exc)})
            continue
        axes = axis_vector_from_descriptors(vector.descriptors)
        distribution = axis_distribution(axes)
        rows.append(
            {
                "image_id": vector.image_id,
                "path": str(path),
                "descriptors": vector.descriptors,
                **distribution,
            }
        )
    summary = _summary(rows)
    report = {
        "images": str(images),
        "backend": backend_name,
        "crop": crop,
        "axis_definitions": _axis_definitions(),
        "image_count": len(rows),
        "failed_count": len(failures),
        "summary": summary,
        "images_detail": rows,
        "failures": failures,
        "boundary": (
            "Face axes are visual descriptor coordinates only. They are not identity, "
            "attractiveness, ethnicity, personality, or popularity labels."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "face_axis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_axis_csv(rows, out_dir / "face_axis_scores.csv")
    (out_dir / "face_axis_report.md").write_text(_render_axis_report(report), encoding="utf-8")
    return report


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"available": False}
    axis_means = {
        axis: _round(sum(row["axes"][axis] for row in rows) / len(rows))
        for axis in AXES
    }
    quadrant_counts: dict[str, int] = {}
    cross_counts: dict[str, int] = {}
    for row in rows:
        quadrant_counts[row["quadrant"]] = quadrant_counts.get(row["quadrant"], 0) + 1
        cross_counts[row["cross_label"]] = cross_counts.get(row["cross_label"], 0) + 1
    return {
        "available": True,
        "axis_means": axis_means,
        "distribution": axis_distribution(axis_means),
        "quadrant_counts": quadrant_counts,
        "cross_counts": cross_counts,
    }


def _write_axis_csv(rows: list[dict[str, Any]], path: Path) -> None:
    headers = [
        "image_id",
        "path",
        "quadrant",
        "corner",
        "quadrant_x",
        "quadrant_y",
        "cross_label",
        "outlier_score",
        "presentation_flags",
        "cross_x",
        "cross_y",
        *AXES,
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            flat = {
                "image_id": row["image_id"],
                "path": row["path"],
                "quadrant": row["quadrant"],
                "corner": row["corner"],
                "quadrant_x": row["quadrant_x"],
                "quadrant_y": row["quadrant_y"],
                "cross_label": row["cross_label"],
                "outlier_score": row["outlier_score"],
                "presentation_flags": ";".join(row["presentation_flags"]),
                "cross_x": row["cross_x"],
                "cross_y": row["cross_y"],
            }
            flat.update(row["axes"])
            writer.writerow(flat)


def _render_axis_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# face axis report",
        "",
        f"- images: {report['images']}",
        f"- backend: {report['backend']}",
        f"- image_count: {report['image_count']}",
        f"- failed_count: {report['failed_count']}",
        "",
        "## Axis Definitions",
        "",
    ]
    for definition in report["axis_definitions"]:
        lines.append(f"- {definition['axis']}: {definition['low']} <-> {definition['high']}")
    if summary.get("available"):
        dist = summary["distribution"]
        lines.extend(
            [
                "",
                "## Distribution",
                "",
                f"- quadrant: {dist['quadrant']}",
                f"- corner: {dist['corner']}",
                f"- quadrant_x: {dist['quadrant_x']}",
                f"- quadrant_y: {dist['quadrant_y']}",
                f"- cross_label: {dist['cross_label']}",
                f"- outlier_score: {dist['outlier_score']}",
                f"- presentation_flags: {', '.join(dist['presentation_flags'])}",
                "",
                "## Axis Means",
                "",
                "| axis | value |",
                "| --- | ---: |",
            ]
        )
        for axis, value in summary["axis_means"].items():
            lines.append(f"| {axis} | {value} |")
    lines.extend(["", "## Boundary", "", report["boundary"], ""])
    return "\n".join(lines)


def _axis_definitions() -> list[dict[str, str]]:
    return [
        {"axis": "soft_defined", "low": "soft/low contrast", "high": "defined feature separation"},
        {"axis": "cool_warm", "low": "cool/neutral", "high": "warm"},
        {"axis": "deep_bright", "low": "deep/shaded", "high": "bright/airy"},
        {"axis": "natural_styled", "low": "natural/no-makeup-like", "high": "styled/makeup-visible"},
        {"axis": "muted_vivid", "low": "muted color", "high": "vivid color"},
        {"axis": "soft_crisp", "low": "soft silhouette", "high": "crisp hairline/detail"},
        {"axis": "light_dark_hair", "low": "lighter upper-band hair signal", "high": "dark hair signal"},
        {"axis": "dynamic_symmetric", "low": "dynamic/asymmetric", "high": "centered/symmetric"},
    ]


def _quadrant(x: float, y: float) -> str:
    if x >= 0 and y >= 0:
        return "defined_bright"
    if x < 0 and y >= 0:
        return "soft_bright"
    if x < 0 and y < 0:
        return "soft_deep"
    return "defined_deep"


def _corner(x: float, y: float) -> str:
    if x >= 0.33 and y >= 0.33:
        return "defined_warm_bright_corner"
    if x <= -0.33 and y >= 0.33:
        return "soft_clear_bright_corner"
    if x <= -0.33 and y <= -0.33:
        return "soft_muted_deep_corner"
    if x >= 0.33 and y <= -0.33:
        return "defined_cool_deep_corner"
    return "center_cross"


def _cross_label(x: float, y: float) -> str:
    horizontal = "symmetric" if x >= 0 else "dynamic"
    vertical = "dark_hair" if y >= 0 else "light_hair"
    return f"{horizontal}_{vertical}"


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    normalized = ((value - low) / (high - low)) * 2.0 - 1.0
    return max(-1.0, min(1.0, normalized))


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _float(value: Any) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def _round(value: float) -> float:
    return round(float(value), 6)
