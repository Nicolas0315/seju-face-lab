from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from html import escape
import json
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit research data quality and evidence types.")
    parser.add_argument("--seju-subjects", type=Path, default=Path("data/raw/seju_by_talent"))
    parser.add_argument("--agencies", type=Path, default=Path("configs/agencies/seju_like_agencies.json"))
    parser.add_argument("--agency-real-root", type=Path, default=Path("data/raw/agencies"))
    parser.add_argument(
        "--agency-enhancement",
        type=Path,
        default=Path("outputs/agency_generation_refined/v2_contrast/enhancement/agency_enhancement_report.json"),
    )
    parser.add_argument(
        "--subject-review",
        type=Path,
        default=Path("outputs/seju_subject_reviews/subject_reviews.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/data_quality_audit"))
    args = parser.parse_args()

    report = build_report(
        seju_subjects=args.seju_subjects,
        agencies=args.agencies,
        agency_real_root=args.agency_real_root,
        agency_enhancement=args.agency_enhancement,
        subject_review=args.subject_review,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "data_quality_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out / "data_quality_audit.md").write_text(render_markdown(report), encoding="utf-8")
    (args.out / "data_quality_audit.html").write_text(render_html(report), encoding="utf-8")
    print(f"data quality audit: {args.out / 'data_quality_audit.md'}")
    print(f"risk level: {report['summary']['risk_level']}")
    print(f"engine items: {len(report['engine_items'])}")
    return 0


def build_report(
    seju_subjects: Path,
    agencies: Path,
    agency_real_root: Path,
    agency_enhancement: Path,
    subject_review: Path,
) -> dict[str, Any]:
    seju = audit_subject_images(seju_subjects)
    agency = audit_agency_evidence(agencies, agency_real_root, agency_enhancement, seju["image_count"])
    subject_stats = load_subject_review(subject_review)
    issues = collect_issues(seju, agency, subject_stats)
    return {
        "inputs": {
            "seju_subjects": str(seju_subjects),
            "agencies": str(agencies),
            "agency_real_root": str(agency_real_root),
            "agency_enhancement": str(agency_enhancement),
            "subject_review": str(subject_review),
        },
        "summary": {
            "risk_level": risk_level(issues),
            "issue_count": len(issues),
            "blocking_issue_count": sum(1 for item in issues if item["severity"] == "blocking"),
            "warning_issue_count": sum(1 for item in issues if item["severity"] == "warning"),
        },
        "seju_subject_images": seju,
        "subject_review": subject_stats,
        "agency_evidence": agency,
        "issues": issues,
        "engine_items": engine_items(issues),
        "boundary": (
            "This audit checks evidence quality and data provenance. It must not be used for "
            "identity, attractiveness, popularity, ethnicity, or personal-value labels."
        ),
    }


def audit_subject_images(root: Path) -> dict[str, Any]:
    subject_dirs = sorted(path for path in root.iterdir() if path.is_dir()) if root.exists() else []
    subject_counts: dict[str, int] = {}
    image_rows: list[dict[str, Any]] = []
    corrupt_paths: list[str] = []
    hash_to_paths: dict[str, list[str]] = defaultdict(list)
    small_images: list[str] = []
    wide_or_tall_images: list[str] = []
    for subject_dir in subject_dirs:
        paths = [path for path in sorted(subject_dir.rglob("*")) if is_image(path)]
        subject_counts[subject_dir.name] = len(paths)
        for path in paths:
            digest = sha256_file(path)
            hash_to_paths[digest].append(str(path))
            try:
                with Image.open(path) as image:
                    width, height = image.size
            except (OSError, UnidentifiedImageError):
                corrupt_paths.append(str(path))
                continue
            min_side = min(width, height)
            ratio = width / height if height else 0.0
            if min_side < 384:
                small_images.append(str(path))
            if ratio < 0.55 or ratio > 1.9:
                wide_or_tall_images.append(str(path))
            image_rows.append(
                {
                    "subject": subject_dir.name,
                    "path": str(path),
                    "width": width,
                    "height": height,
                    "min_side": min_side,
                    "aspect_ratio": round(ratio, 4),
                    "sha256": digest,
                }
            )
    duplicate_groups = [
        {"sha256": digest, "paths": paths}
        for digest, paths in sorted(hash_to_paths.items())
        if len(paths) > 1
    ]
    counts = list(subject_counts.values())
    return {
        "subject_count": len(subject_dirs),
        "image_count": sum(counts),
        "min_images_per_subject": min(counts) if counts else 0,
        "max_images_per_subject": max(counts) if counts else 0,
        "mean_images_per_subject": round(sum(counts) / len(counts), 3) if counts else 0.0,
        "imbalance_ratio": round((max(counts) / max(1, min(counts))), 3) if counts else 0.0,
        "empty_subjects": [subject for subject, count in subject_counts.items() if count == 0],
        "corrupt_paths": corrupt_paths,
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_image_count": sum(len(group["paths"]) for group in duplicate_groups),
        "duplicate_groups": duplicate_groups[:25],
        "small_image_count": len(small_images),
        "small_images_sample": small_images[:25],
        "wide_or_tall_image_count": len(wide_or_tall_images),
        "wide_or_tall_images_sample": wide_or_tall_images[:25],
        "subject_counts": subject_counts,
        "image_rows_sample": image_rows[:25],
    }


def audit_agency_evidence(
    agencies_path: Path,
    agency_real_root: Path,
    agency_enhancement: Path,
    seju_real_image_count: int,
) -> dict[str, Any]:
    config = read_json(agencies_path)
    enhancement = read_json(agency_enhancement) if agency_enhancement.exists() else {}
    enhanced_by_slug = {row.get("slug"): row for row in enhancement.get("agencies", [])}
    rows = []
    evidence_counts: Counter[str] = Counter()
    quadrant_counts: Counter[str] = Counter()
    for agency in config.get("agencies", []):
        slug = str(agency.get("slug", ""))
        real_dir = agency_real_root / slug
        real_images = count_images(real_dir)
        if slug == "seju":
            real_images = seju_real_image_count
        generated = enhanced_by_slug.get(slug, {})
        evidence_type = "real_and_generated" if real_images else "hypothesis_and_generated"
        if slug == "seju":
            evidence_type = "real_centroid_baseline"
        evidence_counts[evidence_type] += 1
        distribution = generated.get("observed_distribution", {})
        quadrant = str(distribution.get("quadrant", "unmeasured"))
        quadrant_counts[quadrant] += 1
        rows.append(
            {
                "slug": slug,
                "name": agency.get("name"),
                "evidence_type": evidence_type,
                "real_image_count": real_images,
                "has_generated_score": bool(generated),
                "enhancement_score": generated.get("enhancement_score"),
                "image_centroid_score": generated.get("components", {}).get("image_centroid_score"),
                "axis_alignment": generated.get("components", {}).get("axis_alignment"),
                "quadrant": quadrant,
                "presentation_flags": generated.get("presentation_flags", []),
            }
        )
    return {
        "agency_count": len(rows),
        "evidence_counts": dict(evidence_counts),
        "quadrant_counts": dict(quadrant_counts),
        "unique_quadrants": len([key for key in quadrant_counts if key != "unmeasured"]),
        "agencies": rows,
    }


def load_subject_review(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    payload = read_json(path)
    stats = payload.get("analysis", {}).get("score_stats", {})
    return {
        "available": True,
        "subject_count": payload.get("subject_count"),
        "reviewed_image_count": stats.get("reviewed_image_count"),
        "failed_image_count": stats.get("failed_image_count"),
        "mean_of_subject_means": stats.get("mean_of_subject_means"),
        "median_of_subject_means": stats.get("median_of_subject_means"),
    }


def collect_issues(
    seju: dict[str, Any],
    agency: dict[str, Any],
    subject_stats: dict[str, Any],
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if seju["subject_count"] < 50:
        issues.append(
            issue(
                "warning",
                "seju_subject_count_low",
                f"Only {seju['subject_count']} seju subjects are available; stronger centroid stability needs more subjects.",
            )
        )
    if seju["imbalance_ratio"] > 2.0:
        issues.append(
            issue(
                "warning",
                "subject_image_imbalance",
                f"Subject image imbalance ratio is {seju['imbalance_ratio']}; unbalanced centroids can overweight frequent subjects.",
            )
        )
    if seju["duplicate_group_count"]:
        issues.append(
            issue(
                "warning",
                "duplicate_images_present",
                f"{seju['duplicate_group_count']} exact duplicate image groups found.",
            )
        )
    if seju["small_image_count"]:
        issues.append(
            issue(
                "info",
                "small_images_present",
                f"{seju['small_image_count']} images have min side below 384 px.",
            )
        )
    if seju["wide_or_tall_image_count"]:
        issues.append(
            issue(
                "info",
                "aspect_outliers_present",
                f"{seju['wide_or_tall_image_count']} images have extreme aspect ratios.",
            )
        )
    if subject_stats.get("failed_image_count"):
        issues.append(
            issue(
                "warning",
                "subject_review_failures",
                f"{subject_stats['failed_image_count']} images failed subject review scoring.",
            )
        )
    hypothesis_count = agency["evidence_counts"].get("hypothesis_and_generated", 0)
    if hypothesis_count:
        issues.append(
            issue(
                "blocking",
                "non_seju_real_data_missing",
                f"{hypothesis_count} agencies are hypothesis/generated only; do not treat them as real agency averages.",
            )
        )
    if agency["unique_quadrants"] < 2 and agency["agency_count"] > 1:
        issues.append(
            issue(
                "warning",
                "quadrant_separability_low",
                "Agency samples mostly occupy one quadrant; rely on 8-axis bars or adaptive projection.",
            )
        )
    unmeasured = [
        row["slug"]
        for row in agency["agencies"]
        if not row["has_generated_score"]
    ]
    if unmeasured:
        issues.append(
            issue(
                "warning",
                "agency_generated_scores_missing",
                f"Missing generated scores for agencies: {', '.join(unmeasured)}.",
            )
        )
    return issues


def engine_items(issues: list[dict[str, str]]) -> list[dict[str, str]]:
    mapping = {
        "seju_subject_count_low": "expand-seju-real-subject-set",
        "subject_image_imbalance": "subject-balanced-centroid",
        "duplicate_images_present": "deduplicate-before-centroid",
        "small_images_present": "image-resolution-quality-gate",
        "aspect_outliers_present": "aspect-and-crop-quality-gate",
        "subject_review_failures": "failed-image-quarantine",
        "non_seju_real_data_missing": "real-agency-data-collection",
        "quadrant_separability_low": "adaptive-8-axis-projection",
        "agency_generated_scores_missing": "agency-score-coverage-gate",
    }
    return [
        {
            "item": mapping.get(issue_row["code"], issue_row["code"]),
            "source_issue": issue_row["code"],
            "severity": issue_row["severity"],
        }
        for issue_row in issues
    ]


def issue(severity: str, code: str, message: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "message": message}


def risk_level(issues: list[dict[str, str]]) -> str:
    if any(item["severity"] == "blocking" for item in issues):
        return "needs_real_data_before_strong_claims"
    if any(item["severity"] == "warning" for item in issues):
        return "usable_with_caveats"
    return "clean"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# data quality audit",
        "",
        f"- risk_level: {report['summary']['risk_level']}",
        f"- issue_count: {report['summary']['issue_count']}",
        f"- seju_subjects: {report['seju_subject_images']['subject_count']}",
        f"- seju_images: {report['seju_subject_images']['image_count']}",
        f"- agency_count: {report['agency_evidence']['agency_count']}",
        "",
        "## Issues",
        "",
    ]
    for item in report["issues"]:
        lines.append(f"- {item['severity']} / {item['code']}: {item['message']}")
    lines.extend(["", "## Engine Items", ""])
    for item in report["engine_items"]:
        lines.append(f"- {item['severity']} / {item['item']} from `{item['source_issue']}`")
    lines.extend(["", "## Agency Evidence", ""])
    lines.append("| agency | evidence | real_images | generated | score | quadrant |")
    lines.append("| --- | --- | ---: | --- | ---: | --- |")
    for row in report["agency_evidence"]["agencies"]:
        lines.append(
            f"| {row['slug']} | {row['evidence_type']} | {row['real_image_count']} | "
            f"{row['has_generated_score']} | {fmt(row['enhancement_score'])} | {row['quadrant']} |"
        )
    lines.extend(["", "## Boundary", "", report["boundary"], ""])
    return "\n".join(lines)


def render_html(report: dict[str, Any]) -> str:
    issues = "".join(
        f"<li><strong>{escape(item['severity'])}</strong> {escape(item['code'])}: {escape(item['message'])}</li>"
        for item in report["issues"]
    )
    rows = "".join(
        "<tr>"
        f"<td>{escape(row['slug'])}</td>"
        f"<td>{escape(row['evidence_type'])}</td>"
        f"<td>{row['real_image_count']}</td>"
        f"<td>{escape(str(row['has_generated_score']))}</td>"
        f"<td>{escape(fmt(row['enhancement_score']))}</td>"
        f"<td>{escape(row['quadrant'])}</td>"
        "</tr>"
        for row in report["agency_evidence"]["agencies"]
    )
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>data quality audit</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;line-height:1.55;background:#fbfaf7;color:#202124}}
table{{border-collapse:collapse;width:100%;background:white}}
th,td{{border:1px solid #d9dde3;padding:8px;text-align:left}}
th{{background:#f5f6f7}}
.risk{{font-weight:700;color:#8a3d3d}}
</style>
</head>
<body>
<h1>data quality audit</h1>
<p class="risk">risk: {escape(report['summary']['risk_level'])}</p>
<h2>Issues</h2>
<ul>{issues}</ul>
<h2>Agency Evidence</h2>
<table>
<thead><tr><th>agency</th><th>evidence</th><th>real images</th><th>generated</th><th>score</th><th>quadrant</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<p>{escape(report['boundary'])}</p>
</body>
</html>
"""


def count_images(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if is_image(path))


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
