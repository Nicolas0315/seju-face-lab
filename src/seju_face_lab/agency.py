from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .face_axes import axis_distribution, axis_vector_from_descriptors
from .prompting import prompt_from_descriptors


DESCRIPTOR_KEYS = [
    "luminance",
    "contrast",
    "saturation",
    "warmth",
    "symmetry",
    "edge_density",
    "upper_band_darkness",
    "middle_luminance",
    "lower_luminance",
]


def write_agency_average_params(
    model_dir: Path,
    agencies_config: Path,
    out_dir: Path,
) -> dict[str, Any]:
    report = build_agency_average_params(model_dir, agencies_config)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agency_average_params.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "agency_average_params.md").write_text(_render_report(report), encoding="utf-8")
    prompts_dir = out_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    for agency in report["agencies"]:
        slug = agency["slug"]
        (prompts_dir / f"{slug}.txt").write_text(agency["imagegen_prompt"] + "\n", encoding="utf-8")
    return report


def build_agency_average_params(model_dir: Path, agencies_config: Path) -> dict[str, Any]:
    model_profile = json.loads((model_dir / "profile.json").read_text(encoding="utf-8"))
    config = json.loads(agencies_config.read_text(encoding="utf-8"))
    base_descriptors = _descriptor_values(model_profile.get("descriptors", {}).get("median"))
    agencies = [
        _agency_summary(agency, base_descriptors)
        for agency in config.get("agencies", [])
        if isinstance(agency, dict)
    ]
    return {
        "model_dir": str(model_dir),
        "agencies_config": str(agencies_config),
        "retrieved_at": config.get("retrieved_at"),
        "base_agency": config.get("base_agency", "seju"),
        "descriptor_keys": DESCRIPTOR_KEYS,
        "analysis_logic": _analysis_logic(),
        "agencies": agencies,
        "rankings": {
            "by_descriptor_similarity": sorted(
                [
                    {
                        "rank": index + 1,
                        "slug": agency["slug"],
                        "name": agency["name"],
                        "descriptor_similarity": agency["descriptor_similarity"],
                    }
                    for index, agency in enumerate(
                        sorted(agencies, key=lambda item: item["descriptor_similarity"], reverse=True)
                    )
                ],
                key=lambda item: item["rank"],
            )
        },
        "boundary": (
            "Agency profiles are aggregate research hypotheses from official public agency sources "
            "and local seju centroid descriptors. They are not popularity rankings, identity claims, "
            "or attractiveness scores. Use image-level vector scoring only for generated or explicitly "
            "curated local images."
        ),
    }


def _agency_summary(agency: dict[str, Any], base_descriptors: dict[str, float]) -> dict[str, Any]:
    offsets = _descriptor_values(agency.get("descriptor_offsets"))
    average = {
        key: round(_clamp_descriptor(key, base_descriptors.get(key, 0.0) + offsets.get(key, 0.0)), 6)
        for key in DESCRIPTOR_KEYS
    }
    distance = _descriptor_distance(base_descriptors, average)
    similarity = round(1.0 / (1.0 + distance), 6)
    axis_vector = axis_vector_from_descriptors(average)
    return {
        "slug": str(agency["slug"]),
        "name": str(agency["name"]),
        "official_sources": agency.get("official_sources", []),
        "public_examples": agency.get("public_examples", []),
        "positioning": agency.get("positioning", []),
        "parameter_basis": agency.get("parameter_basis", []),
        "average_descriptors": average,
        "axis_vector": axis_vector,
        "axis_distribution": axis_distribution(axis_vector),
        "descriptor_offsets": {key: offsets.get(key, 0.0) for key in DESCRIPTOR_KEYS},
        "descriptor_distance_to_seju": round(distance, 6),
        "descriptor_similarity": similarity,
        "imagegen_prompt": _imagegen_prompt(str(agency["name"]), agency, average),
    }


def _descriptor_values(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): float(item)
        for key, item in value.items()
        if key in DESCRIPTOR_KEYS and isinstance(item, int | float)
    }


def _descriptor_distance(left: dict[str, float], right: dict[str, float]) -> float:
    values = []
    for key in DESCRIPTOR_KEYS:
        values.append((left.get(key, 0.0) - right.get(key, 0.0)) ** 2)
    return math.sqrt(sum(values) / len(values))


def _clamp_descriptor(key: str, value: float) -> float:
    if key == "warmth":
        return _clamp(value, -1.0, 1.0)
    return _clamp(value, 0.0, 1.0)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _imagegen_prompt(name: str, agency: dict[str, Any], descriptors: dict[str, float]) -> str:
    base_prompt = prompt_from_descriptors(descriptors, profile="detector-friendly")
    positioning = ", ".join(str(item) for item in agency.get("positioning", [])[:4])
    return (
        "Use case: photorealistic-natural\n"
        f"Asset type: agency aggregate face research sample for {name}\n"
        "Primary request: create one fictional young adult Japanese woman portrait that represents "
        "aggregate visual tendencies, not any real person.\n"
        f"Research positioning: {positioning}\n"
        f"Aggregate descriptor prompt: {base_prompt}\n"
        "Composition/framing: centered frontal portrait, head and shoulders, both eyes visible, "
        "neutral background, detector-friendly crop.\n"
        "Lighting/mood: clean commercial portrait lighting, natural skin texture, calm expression.\n"
        "Constraints: fictional person only, no celebrity likeness, no text, no watermark, no logo, "
        "no minors, no heavy retouching, no cosplay, no duplicated faces."
    )


def _analysis_logic() -> list[dict[str, str]]:
    return [
        {
            "step": "official_source",
            "logic": "Record official agency list URLs and public examples; do not scrape private data.",
        },
        {
            "step": "parameter_hypothesis",
            "logic": "Map agency positioning into descriptor offsets over the seju median centroid.",
        },
        {
            "step": "average_profile",
            "logic": "Create agency average descriptors by applying bounded offsets to seju descriptors.",
        },
        {
            "step": "image_generation",
            "logic": "Generate fictional aggregate portraits from descriptors; avoid real-person likeness.",
        },
        {
            "step": "precision_measurement",
            "logic": "Score generated images against the seju centroid and keep face/style/QA axes separate.",
        },
    ]


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        "# agency average face parameter report",
        "",
        f"- model_dir: {report['model_dir']}",
        f"- agencies_config: {report['agencies_config']}",
        f"- retrieved_at: {report.get('retrieved_at')}",
        f"- base_agency: {report.get('base_agency')}",
        "",
        "## Analysis Logic",
        "",
    ]
    for item in report["analysis_logic"]:
        lines.append(f"- {item['step']}: {item['logic']}")
    lines.extend(["", "## Similarity Ranking", "", "| rank | agency | descriptor_similarity |", "| ---: | --- | ---: |"])
    for item in report["rankings"]["by_descriptor_similarity"]:
        lines.append(f"| {item['rank']} | {item['name']} | {item['descriptor_similarity']} |")
    lines.extend(["", "## Agencies", ""])
    for agency in report["agencies"]:
        lines.extend(
            [
                f"### {agency['name']}",
                "",
                f"- slug: {agency['slug']}",
                f"- descriptor_similarity: {agency['descriptor_similarity']}",
                f"- descriptor_distance_to_seju: {agency['descriptor_distance_to_seju']}",
                f"- quadrant: {agency['axis_distribution']['quadrant']}",
                f"- corner: {agency['axis_distribution']['corner']}",
                f"- cross_label: {agency['axis_distribution']['cross_label']}",
                f"- positioning: {', '.join(agency['positioning'])}",
                f"- public_examples: {', '.join(agency['public_examples'])}",
                f"- sources: {', '.join(source.get('url', '') for source in agency['official_sources'])}",
                "",
                "#### Average Descriptors",
                "",
                "| descriptor | value | offset |",
                "| --- | ---: | ---: |",
            ]
        )
        for key in DESCRIPTOR_KEYS:
            lines.append(
                f"| {key} | {agency['average_descriptors'][key]} | "
                f"{agency['descriptor_offsets'][key]} |"
            )
        lines.extend(["", "#### 8-Axis Vector", "", "| axis | value |", "| --- | ---: |"])
        for key, value in agency["axis_vector"].items():
            lines.append(f"| {key} | {value} |")
        lines.extend(["", "#### ImageGen Prompt", "", agency["imagegen_prompt"], ""])
    lines.extend(["## Boundary", "", report["boundary"], ""])
    return "\n".join(lines)
