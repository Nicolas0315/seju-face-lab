from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_ingredients_report(model_dir: Path, out_dir: Path) -> dict[str, Any]:
    profile = json.loads((model_dir / "profile.json").read_text(encoding="utf-8"))
    descriptors = profile.get("descriptors", {})
    mean = _descriptor(descriptors.get("mean"))
    median = _descriptor(descriptors.get("median"))
    report = {
        "model_dir": str(model_dir),
        "image_count": profile.get("image_count"),
        "embedding_dim": profile.get("embedding_dim"),
        "appearance_shape": profile.get("appearance_shape"),
        "descriptor_values": {
            "mean": mean,
            "median": median,
            "delta_median_minus_mean": _descriptor_delta(mean, median),
        },
        "ingredients": _ingredients(mean, median),
        "prompt_guidance": _prompt_guidance(mean, median),
        "boundary": (
            "Aggregate visual ingredient analysis only. This is not identity recognition, "
            "attractiveness scoring, ethnicity classification, medical inference, or an "
            "objective face-type label."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "face_ingredients.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "face_ingredients.md").write_text(_render(report), encoding="utf-8")
    return report


def _descriptor(value: Any) -> dict[str, float | None]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): float(item) if item is not None else None
        for key, item in value.items()
        if isinstance(item, int | float) or item is None
    }


def _descriptor_delta(
    mean: dict[str, float | None],
    median: dict[str, float | None],
) -> dict[str, float | None]:
    delta: dict[str, float | None] = {}
    for key in sorted(set(mean) | set(median)):
        left = mean.get(key)
        right = median.get(key)
        delta[key] = None if left is None or right is None else round(right - left, 6)
    return delta


def _ingredients(
    mean: dict[str, float | None],
    median: dict[str, float | None],
) -> dict[str, dict[str, Any]]:
    representative = median or mean
    return {
        "overall": {
            "summary": _overall_summary(representative),
            "evidence": _pick(representative, ["luminance", "contrast", "symmetry"]),
        },
        "face_parts": {
            "summary": _face_parts_summary(representative),
            "evidence": _pick(representative, ["symmetry", "edge_density", "middle_luminance", "lower_luminance"]),
        },
        "color_tone": {
            "summary": _color_summary(representative),
            "evidence": _pick(representative, ["luminance", "warmth", "saturation"]),
        },
        "makeup_texture": {
            "summary": _makeup_summary(representative),
            "evidence": _pick(representative, ["contrast", "saturation", "edge_density"]),
        },
        "hair": {
            "summary": _hair_summary(representative),
            "evidence": _pick(representative, ["upper_band_darkness", "edge_density", "warmth"]),
        },
    }


def _prompt_guidance(
    mean: dict[str, float | None],
    median: dict[str, float | None],
) -> list[str]:
    representative = median or mean
    luminance = _number(representative.get("luminance"))
    contrast = _number(representative.get("contrast"))
    saturation = _number(representative.get("saturation"))
    edge = _number(representative.get("edge_density"))
    symmetry = _number(representative.get("symmetry"))
    upper_darkness = _number(representative.get("upper_band_darkness"))
    guidance = [
        "bright portrait lighting" if luminance >= 0.7 else "moderate portrait lighting",
        "low-contrast face rendering" if contrast < 0.1 else "clearer facial contrast",
        "natural clean makeup texture"
        if contrast < 0.1 and saturation < 0.12
        else "more defined makeup texture",
        "dark natural hair kept away from both eyes"
        if upper_darkness >= 0.2
        else "lighter hair signal kept away from both eyes",
        "soft hairline and facial contours" if edge < 0.03 else "defined hairline and facial contours",
        "centered face with high left-right balance"
        if symmetry >= 0.95
        else "centered face with moderate left-right balance",
    ]
    if _number(representative.get("saturation")) > 0.08:
        guidance.append("subtle but visible color in lips and cheeks")
    if _number(representative.get("warmth")) < 0:
        guidance.append("neutral to slightly cool color grade")
    return guidance


def _overall_summary(d: dict[str, float | None]) -> str:
    luminance = _number(d.get("luminance"))
    contrast = _number(d.get("contrast"))
    symmetry = _number(d.get("symmetry"))
    brightness = "bright" if luminance >= 0.7 else "moderately bright"
    contrast_label = "soft low-contrast" if contrast < 0.1 else "clearer contrast"
    symmetry_label = "highly centered/symmetric" if symmetry >= 0.95 else "moderately balanced"
    return f"{brightness}, {contrast_label}, {symmetry_label} aggregate face impression"


def _face_parts_summary(d: dict[str, float | None]) -> str:
    edge = _number(d.get("edge_density"))
    symmetry = _number(d.get("symmetry"))
    contour = "soft facial and hairline edges" if edge < 0.03 else "crisper facial edges"
    balance = "strong left-right balance" if symmetry >= 0.95 else "less stable left-right balance"
    return f"{balance} with {contour}; best treated as frontal and unobscured for generation"


def _color_summary(d: dict[str, float | None]) -> str:
    warmth = _number(d.get("warmth"))
    saturation = _number(d.get("saturation"))
    tone = "slightly cool/neutral" if warmth < 0 else "warm"
    sat = "muted color" if saturation < 0.1 else "moderate color"
    return f"{tone} palette with {sat} and bright skin/background values"


def _makeup_summary(d: dict[str, float | None]) -> str:
    contrast = _number(d.get("contrast"))
    saturation = _number(d.get("saturation"))
    edge = _number(d.get("edge_density"))
    base = "natural, sheer-looking makeup" if contrast < 0.1 else "more defined makeup"
    color = "low-color" if saturation < 0.08 else "subtle visible color"
    texture = "smooth texture" if edge < 0.03 else "more crisp detail texture"
    return f"{base}; {color}; {texture}; avoid heavy contour or high-gloss extremes"


def _hair_summary(d: dict[str, float | None]) -> str:
    upper_darkness = _number(d.get("upper_band_darkness"))
    edge = _number(d.get("edge_density"))
    darkness = "dark natural hair signal" if upper_darkness >= 0.2 else "lighter upper-band signal"
    edge_label = "soft boundary" if edge < 0.03 else "defined boundary"
    return f"{darkness} with {edge_label}; keep hair from covering eyes in generated candidates"


def _pick(d: dict[str, float | None], keys: list[str]) -> dict[str, float | None]:
    return {key: _round_optional(d.get(key)) for key in keys}


def _number(value: float | None) -> float:
    return 0.0 if value is None else float(value)


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _render(report: dict[str, Any]) -> str:
    lines = [
        "# seju-face ingredient report",
        "",
        f"- model_dir: {report['model_dir']}",
        f"- image_count: {_value(report.get('image_count'))}",
        f"- embedding_dim: {_value(report.get('embedding_dim'))}",
        "",
        "## Ingredients",
        "",
    ]
    ingredients = report.get("ingredients")
    if isinstance(ingredients, dict):
        for name, block in ingredients.items():
            if isinstance(block, dict):
                lines.extend(
                    [
                        f"### {name.replace('_', ' ').title()}",
                        "",
                        str(block.get("summary") or ""),
                        "",
                        "| descriptor | value |",
                        "| --- | ---: |",
                    ]
                )
                evidence = block.get("evidence")
                if isinstance(evidence, dict):
                    for key, value in evidence.items():
                        lines.append(f"| {key} | {_value(value)} |")
                lines.append("")
    guidance = report.get("prompt_guidance")
    if isinstance(guidance, list) and guidance:
        lines.extend(["## Prompt Guidance", ""])
        lines.extend(f"- {item}" for item in guidance)
        lines.append("")
    lines.extend(["## Boundary", "", str(report["boundary"]), ""])
    return "\n".join(lines)


def _value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
