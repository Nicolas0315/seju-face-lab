from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RETRIEVED_AT = "2026-06-15"


def build_benchmark_research() -> dict[str, Any]:
    sources = [
        {
            "name": "NIST FRTE/FATE",
            "url": "https://www.nist.gov/programs-projects/face-technology-evaluations-frtefate",
            "source_type": "official benchmark program",
            "scope": "face recognition, face analysis, quality, presentation attack, age estimation",
            "use_for_seju_face_lab": "Treat as the benchmark taxonomy: separate identity recognition from face analysis and quality gates.",
            "adoption": "Use FRTE/FATE categories to keep backend comparison, QA, and analysis axes separate.",
        },
        {
            "name": "NEC FRTE 1:N result",
            "url": "https://www.nec.com/en/press/202603/global_20260309_02.html",
            "source_type": "vendor press release tied to NIST benchmark",
            "scope": "large-scale face identification benchmark signal",
            "use_for_seju_face_lab": "Evidence that industrial face recognition quality is measured with NIST-style 1:N/1:1 tests, not local aesthetic labels.",
            "adoption": "Use only as benchmark context; NEC SDKs are not an OSS dependency here.",
        },
        {
            "name": "InsightFace",
            "url": "https://github.com/deepinsight/insightface",
            "source_type": "OSS face analysis toolkit",
            "scope": "face detection, alignment, recognition, model evaluation, ArcFace-family embeddings",
            "use_for_seju_face_lab": "Primary neural face-vector candidate on RTX machines through the existing insightface backend.",
            "adoption": "Keep as preferred 512D neural embedding cross-check; compare ranks against deterministic and OpenCV-face.",
        },
        {
            "name": "DeepFace",
            "url": "https://github.com/serengil/deepface",
            "source_type": "OSS face recognition wrapper",
            "scope": "multiple detector/model choices through a common represent API",
            "use_for_seju_face_lab": "Detector/model sensitivity audit, especially ArcFace with RetinaFace after OpenCV detector divergence.",
            "adoption": "Keep deepface-retinaface as the default DeepFace cross-check path; use detector sweeps before trusting rank changes.",
        },
        {
            "name": "OpenCLIP",
            "url": "https://github.com/mlfoundations/open_clip",
            "source_type": "OSS image-text/image embedding toolkit",
            "scope": "general image style and photographic embedding axis",
            "use_for_seju_face_lab": "Style/photographic similarity axis, separate from face-geometry embeddings.",
            "adoption": "Keep style-evaluate separate; never merge style score into identity-like face-vector claims without an explicit report.",
        },
        {
            "name": "Worldcoin Open IRIS",
            "url": "https://github.com/worldcoin/open-iris",
            "source_type": "OSS iris recognition pipeline",
            "scope": "iris segmentation, feature extraction, iris code matching, scalable biometric verification",
            "use_for_seju_face_lab": "Research reference for segmentation-template-matching architecture, not a face-vector backend.",
            "adoption": "Do not mix iris codes with face vectors; track as a separate biometric modality and privacy boundary.",
        },
    ]
    recommendations = [
        {
            "priority": "P1",
            "title": "Make InsightFace/ArcFace the primary neural cross-check",
            "why": "It is already wired, produces normalized 512D embeddings, and maps closest to benchmark-style face recognition.",
            "implementation": "Run compare-backends with deterministic, opencv-face, insightface, and deepface-retinaface on the same reviewed image sets.",
        },
        {
            "priority": "P1",
            "title": "Keep detector acceptance as a first-class metric",
            "why": "Face-vector quality changes when detectors reject images or crop different regions.",
            "implementation": "Continue recording reference/generated acceptance counts and rank agreement per backend.",
        },
        {
            "priority": "P1",
            "title": "Separate face geometry, style, and aggregate ingredients",
            "why": "NIST-style face recognition, FATE-style analysis, and CLIP-style image similarity measure different things.",
            "implementation": "Report face scores, style scores, QA, and ingredients as separate axes in precision reviews.",
        },
        {
            "priority": "P2",
            "title": "Add benchmark-inspired local probe sets",
            "why": "Local seju data is not an FRTE benchmark; robustness must be checked with paired crops, lighting changes, and detector-visible variants.",
            "implementation": "Create ignored probe folders for same-person/multiple-crop, generated variants, and distractor folders, then compare backend rank stability.",
        },
        {
            "priority": "P2",
            "title": "Treat World IRIS as architecture inspiration only",
            "why": "Iris templates solve proof-of-personhood uniqueness, while this repo analyzes aggregate face impressions.",
            "implementation": "Borrow the segmentation-template-match audit shape; do not ingest iris data or combine iris and face embeddings.",
        },
    ]
    return {
        "retrieved_at": RETRIEVED_AT,
        "sources": sources,
        "vectorization_strategy": {
            "primary_face_embedding": "insightface ArcFace-family 512D embeddings when optional dependencies and CUDA providers are available",
            "face_crop_baseline": "opencv-face deterministic vectors for detector-normalized local continuity",
            "neural_cross_check": "deepface-retinaface after detector acceptance has been audited",
            "style_axis": "OpenCLIP via style-evaluate, kept separate from face-vector similarity",
            "ingredient_axis": "ingredients-report from centroid descriptors, kept separate from recognition embeddings",
            "iris_axis": "out of scope for face-vector scoring; separate modality only",
        },
        "recommendations": recommendations,
        "boundary": (
            "This catalog guides local aggregate analysis and backend selection. It is not an identity "
            "recognition claim, biometric enrollment system, NIST-equivalent benchmark, or approval to "
            "collect biometric identifiers."
        ),
    }


def write_benchmark_research(out_dir: Path) -> dict[str, Any]:
    report = build_benchmark_research()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark_research.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "benchmark_research.md").write_text(_render(report), encoding="utf-8")
    return report


def _render(report: dict[str, Any]) -> str:
    lines = [
        "# benchmark and OSS research catalog",
        "",
        f"- retrieved_at: {report['retrieved_at']}",
        "",
        "## Sources",
        "",
        "| name | type | scope | adoption |",
        "| --- | --- | --- | --- |",
    ]
    for source in report["sources"]:
        lines.append(
            "| {name} | {source_type} | {scope} | {adoption} |".format(
                name=source["name"],
                source_type=source["source_type"],
                scope=source["scope"],
                adoption=source["adoption"],
            )
        )
    lines.extend(["", "## Vectorization Strategy", ""])
    strategy = report["vectorization_strategy"]
    lines.extend(f"- {key}: {value}" for key, value in strategy.items())
    lines.extend(["", "## Recommendations", ""])
    for item in report["recommendations"]:
        lines.append(f"- {item['priority']} {item['title']}: {item['implementation']}")
    lines.extend(["", "## Boundary", "", report["boundary"], ""])
    return "\n".join(lines)
