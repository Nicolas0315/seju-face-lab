from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_TARGETS = {
    "image_centroid_score": 0.35,
    "axis_alignment": 0.62,
    "enhancement_score": 0.76,
}

AXIS_PROMPT_RULES = [
    ("soft_defined", "softer lower-contrast facial separation", "more defined facial separation"),
    ("cool_warm", "cooler neutral skin balance", "warmer skin balance"),
    ("deep_bright", "deeper shaded tone", "brighter airy tone"),
    ("natural_styled", "more natural styling", "more visibly styled makeup"),
    ("muted_vivid", "more muted color", "more vivid color"),
    ("soft_crisp", "softer hairline and silhouette", "crisper hairline detail"),
    ("light_dark_hair", "lighter upper hair band", "darker upper hair band"),
    ("dynamic_symmetric", "more dynamic asymmetry", "more centered symmetry"),
]

AXIS_DESCRIPTOR_SOURCES = {
    "soft_defined": ("contrast", "edge_density"),
    "cool_warm": ("warmth",),
    "deep_bright": ("luminance", "middle_luminance", "lower_luminance"),
    "natural_styled": ("contrast", "saturation"),
    "muted_vivid": ("saturation",),
    "soft_crisp": ("edge_density",),
    "light_dark_hair": ("upper_band_darkness",),
    "dynamic_symmetric": ("symmetry",),
}


def write_generation_calibration(
    enhancement_report: Path,
    agency_params: Path,
    out_dir: Path,
    target_image_score: float = DEFAULT_TARGETS["image_centroid_score"],
    target_axis_alignment: float = DEFAULT_TARGETS["axis_alignment"],
    target_enhancement_score: float = DEFAULT_TARGETS["enhancement_score"],
    seed_start: int = 260623,
    variants_per_agency: int = 3,
) -> dict[str, Any]:
    report = build_generation_calibration(
        enhancement_report=enhancement_report,
        agency_params=agency_params,
        target_image_score=target_image_score,
        target_axis_alignment=target_axis_alignment,
        target_enhancement_score=target_enhancement_score,
        seed_start=seed_start,
        variants_per_agency=variants_per_agency,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = out_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    for agency in report["agencies"]:
        slug = agency["slug"]
        (prompts_dir / f"{slug}_calibrated.txt").write_text(
            agency["calibrated_prompt"] + "\n",
            encoding="utf-8",
        )
    (out_dir / "generation_calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(report["agencies"], out_dir / "generation_calibration.csv")
    (out_dir / "generation_calibration.md").write_text(_render_report(report), encoding="utf-8")
    return report


def build_generation_calibration(
    enhancement_report: Path,
    agency_params: Path,
    target_image_score: float,
    target_axis_alignment: float,
    target_enhancement_score: float,
    seed_start: int,
    variants_per_agency: int,
) -> dict[str, Any]:
    enhancement = _read_json(enhancement_report)
    params = _read_json(agency_params)
    params_by_slug = {agency["slug"]: agency for agency in params.get("agencies", [])}
    agencies = []
    for index, agency in enumerate(enhancement.get("agencies", [])):
        param = params_by_slug.get(agency["slug"], {})
        agencies.append(
            _agency_calibration(
                agency=agency,
                param=param,
                target_image_score=target_image_score,
                target_axis_alignment=target_axis_alignment,
                target_enhancement_score=target_enhancement_score,
                seed=seed_start + index * variants_per_agency,
                variants_per_agency=variants_per_agency,
            )
        )
    return {
        "enhancement_report": str(enhancement_report),
        "agency_params": str(agency_params),
        "targets": {
            "image_centroid_score": target_image_score,
            "axis_alignment": target_axis_alignment,
            "enhancement_score": target_enhancement_score,
        },
        "seed_start": seed_start,
        "variants_per_agency": variants_per_agency,
        "agencies": agencies,
        "summary": _summary(agencies),
        "boundary": (
            "Calibration improves fictional aggregate prompt quality only. It must not be used "
            "to imitate a real person or to score personal worth, attractiveness, identity, "
            "ethnicity, or popularity."
        ),
    }


def _agency_calibration(
    agency: dict[str, Any],
    param: dict[str, Any],
    target_image_score: float,
    target_axis_alignment: float,
    target_enhancement_score: float,
    seed: int,
    variants_per_agency: int,
) -> dict[str, Any]:
    components = agency.get("components", {})
    image_score = _float_or_none(components.get("image_centroid_score"))
    axis_alignment = _float_or_none(components.get("axis_alignment"))
    enhancement_score = _float_or_none(agency.get("enhancement_score"))
    gaps = {
        "image_centroid_score": _gap(target_image_score, image_score),
        "axis_alignment": _gap(target_axis_alignment, axis_alignment),
        "enhancement_score": _gap(target_enhancement_score, enhancement_score),
    }
    priority = _priority(gaps)
    prompt = _calibrated_prompt(agency, param, gaps)
    axis_delta = _axis_delta(
        agency.get("hypothesis_axis_vector", {}),
        agency.get("observed_axis_vector", {}),
    )
    return {
        "slug": agency["slug"],
        "name": agency["name"],
        "rank": agency.get("rank"),
        "current": {
            "image_centroid_score": _round_optional(image_score),
            "axis_alignment": _round_optional(axis_alignment),
            "enhancement_score": _round_optional(enhancement_score),
        },
        "gaps_to_target": {key: round(value, 6) for key, value in gaps.items()},
        "priority": priority,
        "presentation_flags": agency.get("presentation_flags", []),
        "improvement_actions": agency.get("improvement_actions", []),
        "axis_delta": axis_delta,
        "prompt_attribution": _prompt_attribution(agency, param, axis_delta),
        "calibrated_prompt": prompt,
        "negative_prompt": _negative_prompt(agency),
        "generation_plan": {
            "provider": "dry-run",
            "prompt_profile": "detector-friendly",
            "count": variants_per_agency,
            "seed": seed,
            "review_after_generation": True,
            "recommended_output_dir": f"outputs/agency_generation_refined/{agency['slug']}",
        },
    }


def _calibrated_prompt(
    agency: dict[str, Any],
    param: dict[str, Any],
    gaps: dict[str, float],
) -> str:
    base_prompt = str(param.get("imagegen_prompt", "")).strip()
    axis_notes = _axis_notes(
        agency.get("hypothesis_axis_vector", {}),
        agency.get("observed_axis_vector", {}),
    )
    action_notes = _action_notes(agency.get("improvement_actions", []))
    intensity = _intensity(gaps)
    return "\n".join(
        part
        for part in [
            base_prompt,
            "",
            f"Calibration strength: {intensity}.",
            "Precision target: improve seju-centroid similarity while matching the agency hypothesis axes.",
            "Axis corrections: " + "; ".join(axis_notes),
            "Presentation corrections: " + "; ".join(action_notes),
            (
                "Keep the face centered, evenly lit, frontal, natural, fictional, and detector-friendly. "
                "Preserve realistic skin texture and avoid over-sharpened hair edges."
            ),
        ]
        if part
    )


def _negative_prompt(agency: dict[str, Any]) -> str:
    terms = [
        "specific celebrity likeness",
        "copied identity",
        "named person",
        "minor",
        "text",
        "watermark",
        "logo",
        "collage",
        "multiple people",
        "side profile",
        "hair covering eyes",
        "hands on face",
        "extreme crop",
    ]
    flags = set(agency.get("presentation_flags", []))
    if "dark_or_underlit_image" in flags:
        terms.append("underexposed face")
    if "dark_upper_band_or_hair_shadow" in flags:
        terms.append("heavy hair shadow over face")
    if "off_center_or_asymmetric_visibility" in flags:
        terms.append("off-center face")
    if "high_texture_or_messy_edges" in flags:
        terms.append("messy hair flyaways")
    return ", ".join(_dedupe(terms))


def _axis_notes(expected: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    if not expected or not observed:
        return ["collect a measured image sample for this agency"]
    notes = []
    for axis, low_label, high_label in AXIS_PROMPT_RULES:
        delta = float(expected.get(axis, 0.0)) - float(observed.get(axis, 0.0))
        if abs(delta) < 0.25:
            continue
        notes.append(high_label if delta > 0 else low_label)
    return notes or ["keep current visual axis balance"]


def _prompt_attribution(
    agency: dict[str, Any],
    param: dict[str, Any],
    axis_delta: dict[str, float],
) -> list[dict[str, Any]]:
    expected = agency.get("hypothesis_axis_vector", {})
    observed = agency.get("observed_axis_vector", {})
    descriptor_offsets = param.get("descriptor_offsets", {})
    if not isinstance(descriptor_offsets, dict):
        descriptor_offsets = {}
    rows = []
    for axis, low_label, high_label in AXIS_PROMPT_RULES:
        delta = float(axis_delta.get(axis, 0.0))
        descriptor_delta = {
            key: round(float(descriptor_offsets.get(key, 0.0)), 6)
            for key in AXIS_DESCRIPTOR_SOURCES.get(axis, ())
            if descriptor_offsets.get(key) not in (None, 0, 0.0)
        }
        if abs(delta) < 0.25 and not descriptor_delta:
            continue
        prompt_clause = high_label if delta > 0 else low_label
        rows.append(
            {
                "axis": axis,
                "descriptor_delta": descriptor_delta,
                "axis_delta": round(delta, 6),
                "prompt_clause": prompt_clause,
                "measured_effect": {
                    "hypothesis_axis": _round_optional(_float_or_none(expected.get(axis))),
                    "observed_axis": _round_optional(_float_or_none(observed.get(axis))),
                    "gap": round(delta, 6),
                },
            }
        )
    return rows


def _action_notes(actions: list[str]) -> list[str]:
    mapping = {
        "regenerate_with_detector_friendly_prompt": "use a cleaner detector-friendly frontal portrait composition",
        "run_second_seed_and_compare": "run multiple seeds and retain only the best centroid/axis candidate",
        "adjust_prompt_to_match_hypothesis_axes": "prioritize the hypothesis axis corrections over decorative styling",
        "increase_even_front_lighting": "increase even front lighting and reduce underexposure",
        "enforce_centered_frontal_crop": "center the face with both eyes clearly visible",
        "reduce_hair_shadow_over_face": "keep hair away from the face and reduce upper-face shadows",
        "reduce_edge_noise_and_hair_flyaways": "reduce flyaway hair and over-sharpened edges",
        "keep_as_baseline_candidate": "keep this prompt as a baseline control",
        "add_or_regenerate_named_sample": "create a named agency sample before scoring",
    }
    return [mapping.get(action, action) for action in actions] or ["keep as baseline control"]


def _axis_delta(expected: dict[str, Any], observed: dict[str, Any]) -> dict[str, float]:
    axes = sorted(set(expected) | set(observed))
    return {
        axis: round(float(expected.get(axis, 0.0)) - float(observed.get(axis, 0.0)), 6)
        for axis in axes
    }


def _priority(gaps: dict[str, float]) -> str:
    if gaps["image_centroid_score"] >= 0.25:
        return "regenerate"
    if gaps["axis_alignment"] >= 0.20:
        return "axis_calibration"
    if gaps["enhancement_score"] >= 0.08:
        return "seed_sweep"
    return "baseline_control"


def _intensity(gaps: dict[str, float]) -> str:
    largest = max(gaps.values())
    if largest >= 0.30:
        return "high"
    if largest >= 0.15:
        return "medium"
    return "low"


def _summary(agencies: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for agency in agencies:
        counts[agency["priority"]] = counts.get(agency["priority"], 0) + 1
    return {
        "agency_count": len(agencies),
        "priority_counts": counts,
        "regenerate_first": [
            agency["slug"]
            for agency in agencies
            if agency["priority"] in {"regenerate", "axis_calibration"}
        ],
    }


def _write_csv(agencies: list[dict[str, Any]], path: Path) -> None:
    headers = [
        "slug",
        "name",
        "priority",
        "current_image_centroid_score",
        "current_axis_alignment",
        "current_enhancement_score",
        "gap_image_centroid_score",
        "gap_axis_alignment",
        "gap_enhancement_score",
        "seed",
        "output_dir",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for agency in agencies:
            writer.writerow(
                {
                    "slug": agency["slug"],
                    "name": agency["name"],
                    "priority": agency["priority"],
                    "current_image_centroid_score": agency["current"]["image_centroid_score"],
                    "current_axis_alignment": agency["current"]["axis_alignment"],
                    "current_enhancement_score": agency["current"]["enhancement_score"],
                    "gap_image_centroid_score": agency["gaps_to_target"]["image_centroid_score"],
                    "gap_axis_alignment": agency["gaps_to_target"]["axis_alignment"],
                    "gap_enhancement_score": agency["gaps_to_target"]["enhancement_score"],
                    "seed": agency["generation_plan"]["seed"],
                    "output_dir": agency["generation_plan"]["recommended_output_dir"],
                }
            )


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        "# agency generation calibration",
        "",
        f"- enhancement_report: {report['enhancement_report']}",
        f"- agency_params: {report['agency_params']}",
        f"- seed_start: {report['seed_start']}",
        f"- variants_per_agency: {report['variants_per_agency']}",
        "",
        "## Targets",
        "",
    ]
    for key, value in report["targets"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Calibration Plan",
            "",
            "| agency | priority | image | axis | enhancement | next output |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for agency in report["agencies"]:
        current = agency["current"]
        lines.append(
            f"| {agency['name']} | {agency['priority']} | "
            f"{_display(current['image_centroid_score'])} | "
            f"{_display(current['axis_alignment'])} | "
            f"{_display(current['enhancement_score'])} | "
            f"{agency['generation_plan']['recommended_output_dir']} |"
        )
    lines.extend(["", "## Prompt Attribution", ""])
    for agency in report["agencies"]:
        lines.extend(
            [
                f"### {agency['name']}",
                "",
                "| axis | descriptor_delta | axis_delta | prompt_clause | measured_effect |",
                "| --- | --- | ---: | --- | --- |",
            ]
        )
        attribution = agency.get("prompt_attribution", [])
        if not attribution:
            lines.append("|  |  |  | keep current visual axis balance | no measured axis gap |")
            continue
        for item in attribution:
            measured = item["measured_effect"]
            lines.append(
                f"| {item['axis']} | {_display_descriptor_delta(item['descriptor_delta'])} | "
                f"{_display(item['axis_delta'])} | {item['prompt_clause']} | "
                f"hypothesis={_display(measured.get('hypothesis_axis'))}, "
                f"observed={_display(measured.get('observed_axis'))}, "
                f"gap={_display(measured.get('gap'))} |"
            )
    lines.extend(["", "## Boundary", "", report["boundary"], ""])
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gap(target: float, value: float | None) -> float:
    if value is None:
        return target
    return max(0.0, float(target) - float(value))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _display(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _display_descriptor_delta(values: dict[str, float]) -> str:
    if not values:
        return ""
    return "; ".join(f"{key}={_display(value)}" for key, value in values.items())


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output
