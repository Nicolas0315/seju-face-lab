from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .sns_metrics import SnsEngagement, read_engagement_manifest


@dataclass
class TalentCorrelationRow:
    talent_slug: str
    name: str | None
    # face metrics
    face_mean_centroid_score: float | None
    face_best_centroid_score: float | None
    face_image_count: int | None
    # per-platform engagement (raw values)
    ig_followers: int | None
    ig_posts: int | None
    ig_engagement_rate: float | None
    ig_total_engagement: int | None
    tw_followers: int | None
    tw_posts: int | None
    tk_followers: int | None
    tk_posts: int | None
    tk_total_engagement: int | None
    tk_engagement_rate: float | None
    # aggregated
    total_followers: int | None
    max_followers: int | None


@dataclass
class CorrelationResult:
    variable_a: str
    variable_b: str
    n: int
    pearson_r: float | None
    pearson_p: float | None
    spearman_r: float | None
    spearman_p: float | None
    interpretation: str


def build_correlation_dataset(
    subject_reviews_json: Path,
    engagement_manifest: Path,
) -> list[TalentCorrelationRow]:
    """Join face scores (from review-subjects summary.json) with SNS engagement."""
    face_scores = _load_face_scores(subject_reviews_json)
    engagements = _load_engagement(engagement_manifest)

    all_slugs = sorted(set(face_scores) | set(engagements))
    rows: list[TalentCorrelationRow] = []
    for slug in all_slugs:
        face = face_scores.get(slug, {})
        eng_list: list[SnsEngagement] = engagements.get(slug, {}).get("engagements", [])
        eng_by_platform = {e.platform: e for e in eng_list}
        name = engagements.get(slug, {}).get("name") or face.get("subject")

        ig = eng_by_platform.get("instagram")
        tw = eng_by_platform.get("twitter")
        tk = eng_by_platform.get("tiktok")

        followers_list = [
            v for v in [
                ig.followers if ig else None,
                tw.followers if tw else None,
                tk.followers if tk else None,
            ] if v is not None
        ]

        rows.append(TalentCorrelationRow(
            talent_slug=slug,
            name=name,
            face_mean_centroid_score=face.get("mean_centroid_score"),
            face_best_centroid_score=face.get("best_centroid_score"),
            face_image_count=face.get("image_count"),
            ig_followers=ig.followers if ig else None,
            ig_posts=ig.posts if ig else None,
            ig_engagement_rate=ig.engagement_rate if ig else None,
            ig_total_engagement=ig.total_engagement if ig else None,
            tw_followers=tw.followers if tw else None,
            tw_posts=tw.posts if tw else None,
            tk_followers=tk.followers if tk else None,
            tk_posts=tk.posts if tk else None,
            tk_total_engagement=tk.total_engagement if tk else None,
            tk_engagement_rate=tk.engagement_rate if tk else None,
            total_followers=sum(followers_list) if followers_list else None,
            max_followers=max(followers_list) if followers_list else None,
        ))
    return rows


def compute_correlations(rows: list[TalentCorrelationRow]) -> list[CorrelationResult]:
    """Compute Pearson + Spearman correlations for face score vs engagement metrics."""
    try:
        from scipy import stats as scipy_stats
        has_scipy = True
    except ImportError:
        has_scipy = False

    face_fields = ["face_mean_centroid_score", "face_best_centroid_score"]
    engagement_fields = [
        "ig_followers", "ig_total_engagement", "ig_engagement_rate",
        "tw_followers",
        "tk_followers", "tk_total_engagement", "tk_engagement_rate",
        "total_followers", "max_followers",
    ]

    results: list[CorrelationResult] = []
    for fa in face_fields:
        for eb in engagement_fields:
            pairs = [
                (getattr(r, fa), getattr(r, eb))
                for r in rows
                if getattr(r, fa) is not None and getattr(r, eb) is not None
            ]
            if len(pairs) < 3:
                results.append(CorrelationResult(
                    variable_a=fa, variable_b=eb, n=len(pairs),
                    pearson_r=None, pearson_p=None,
                    spearman_r=None, spearman_p=None,
                    interpretation="insufficient_data",
                ))
                continue

            a_vals = np.array([p[0] for p in pairs])
            b_vals = np.array([p[1] for p in pairs])
            n = len(pairs)

            if has_scipy:
                pr, pp = scipy_stats.pearsonr(a_vals, b_vals)
                sr, sp = scipy_stats.spearmanr(a_vals, b_vals)
                pearson_r = float(pr)
                pearson_p = float(pp)
                spearman_r = float(sr)
                spearman_p = float(sp)
            else:
                # Fallback numpy Pearson (no p-value)
                pearson_r = float(np.corrcoef(a_vals, b_vals)[0, 1])
                pearson_p = None
                spearman_r = float(_numpy_spearman(a_vals, b_vals))
                spearman_p = None

            results.append(CorrelationResult(
                variable_a=fa, variable_b=eb, n=n,
                pearson_r=round(pearson_r, 4),
                pearson_p=round(pearson_p, 4) if pearson_p is not None else None,
                spearman_r=round(spearman_r, 4),
                spearman_p=round(spearman_p, 4) if spearman_p is not None else None,
                interpretation=_interpret_correlation(spearman_r, spearman_p),
            ))
    return results


def write_correlation_report(
    rows: list[TalentCorrelationRow],
    correlations: list[CorrelationResult],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset CSV
    csv_header = (
        "talent_slug,name,face_mean_score,face_best_score,face_images,"
        "ig_followers,ig_posts,ig_eng_rate,ig_total_eng,"
        "tw_followers,tw_posts,"
        "tk_followers,tk_posts,tk_total_eng,tk_eng_rate,"
        "total_followers,max_followers"
    )
    csv_lines = [csv_header]
    for r in rows:
        csv_lines.append(",".join([
            _csv(r.talent_slug), _csv(r.name or ""),
            _f(r.face_mean_centroid_score), _f(r.face_best_centroid_score),
            _i(r.face_image_count),
            _i(r.ig_followers), _i(r.ig_posts), _f(r.ig_engagement_rate), _i(r.ig_total_engagement),
            _i(r.tw_followers), _i(r.tw_posts),
            _i(r.tk_followers), _i(r.tk_posts), _i(r.tk_total_engagement), _f(r.tk_engagement_rate),
            _i(r.total_followers), _i(r.max_followers),
        ]))
    (out_dir / "correlation_dataset.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8-sig"
    )

    # Correlation results CSV
    corr_header = "variable_a,variable_b,n,pearson_r,pearson_p,spearman_r,spearman_p,interpretation"
    corr_lines = [corr_header]
    for c in correlations:
        corr_lines.append(",".join([
            _csv(c.variable_a), _csv(c.variable_b),
            str(c.n), _f(c.pearson_r), _f(c.pearson_p),
            _f(c.spearman_r), _f(c.spearman_p),
            _csv(c.interpretation),
        ]))
    (out_dir / "correlations.csv").write_text(
        "\n".join(corr_lines) + "\n", encoding="utf-8-sig"
    )

    # Markdown report
    md = _render_report_md(rows, correlations)
    (out_dir / "correlation_report.md").write_text(md, encoding="utf-8")

    # JSON summary
    summary = {
        "talent_count": len(rows),
        "with_face_score": sum(1 for r in rows if r.face_mean_centroid_score is not None),
        "with_ig": sum(1 for r in rows if r.ig_followers is not None),
        "with_twitter": sum(1 for r in rows if r.tw_followers is not None),
        "with_tiktok": sum(1 for r in rows if r.tk_followers is not None),
        "correlations": [
            {
                "a": c.variable_a, "b": c.variable_b, "n": c.n,
                "spearman_r": c.spearman_r, "spearman_p": c.spearman_p,
                "pearson_r": c.pearson_r, "pearson_p": c.pearson_p,
                "interpretation": c.interpretation,
            }
            for c in correlations
        ],
    }
    (out_dir / "correlation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _render_report_md(rows: list[TalentCorrelationRow], correlations: list[CorrelationResult]) -> str:
    lines = [
        "# seju-face-lab: face score × SNS engagement correlation analysis",
        "",
        f"- talents analyzed: {len(rows)}",
        f"- with face score: {sum(1 for r in rows if r.face_mean_centroid_score is not None)}",
        f"- with Instagram: {sum(1 for r in rows if r.ig_followers is not None)}",
        f"- with Twitter: {sum(1 for r in rows if r.tw_followers is not None)}",
        f"- with TikTok: {sum(1 for r in rows if r.tk_followers is not None)}",
        "",
        "## Correlation Results",
        "",
        "| face_var | engagement_var | n | Spearman r | p | Pearson r | interpretation |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for c in sorted(correlations, key=lambda x: -(abs(x.spearman_r) if x.spearman_r is not None else 0)):
        lines.append(
            f"| {c.variable_a} | {c.variable_b} | {c.n} "
            f"| {_f4(c.spearman_r)} | {_f4(c.spearman_p)} "
            f"| {_f4(c.pearson_r)} | {c.interpretation} |"
        )

    lines += [
        "",
        "## Talent Rankings (face score)",
        "",
        "| rank | talent | face_mean | ig_followers | tw_followers | tk_followers | total_followers |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    ranked = sorted(rows, key=lambda r: r.face_mean_centroid_score or 0.0, reverse=True)
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {r.name or r.talent_slug} "
            f"| {_f4(r.face_mean_centroid_score)} "
            f"| {_i(r.ig_followers)} | {_i(r.tw_followers)} "
            f"| {_i(r.tk_followers)} | {_i(r.total_followers)} |"
        )

    lines += [
        "",
        "---",
        "Spearman ρ is robust to outliers and monotonic relationships.",
        "Correlation ≠ causation. Small n reduces statistical power.",
        "SNS metrics are best-effort scraped public data and may be incomplete.",
        "",
    ]
    return "\n".join(lines)


def _load_face_scores(path: Path) -> dict[str, dict]:
    """Load face scores from review-subjects summary.json or subject_reviews.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict] = {}

    # subject_reviews.json format (from write_subject_reviews)
    if "subjects" in data:
        for s in data["subjects"]:
            slug = s.get("subject", "")
            result[slug] = {
                "subject": slug,
                "mean_centroid_score": s.get("mean_centroid_score"),
                "best_centroid_score": s.get("best_centroid_score"),
                "image_count": s.get("image_count"),
            }
        return result

    # Alternative: direct slug → score mapping
    if isinstance(data, dict):
        for slug, v in data.items():
            if isinstance(v, dict):
                result[slug] = v
            else:
                result[slug] = {"mean_centroid_score": float(v)}
    return result


def _load_engagement(path: Path) -> dict[str, dict]:
    records = read_engagement_manifest(path)
    return {
        r.talent_slug: {"name": r.name, "engagements": r.engagements}
        for r in records
    }


def _numpy_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a_rank = np.argsort(np.argsort(a)).astype(float)
    b_rank = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(a_rank, b_rank)[0, 1])


def _interpret_correlation(r: float | None, p: float | None) -> str:
    if r is None:
        return "no_data"
    sig = "" if p is None else ("*" if p < 0.05 else ("+" if p < 0.1 else ""))
    abs_r = abs(r)
    if abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.4:
        strength = "moderate"
    elif abs_r >= 0.2:
        strength = "weak"
    else:
        strength = "negligible"
    direction = "positive" if r >= 0 else "negative"
    return f"{strength}_{direction}{sig}"


def _csv(v: str) -> str:
    return f'"{v.replace(chr(34), chr(34)+chr(34))}"'


def _f(v: float | None) -> str:
    return "" if v is None else f"{v:.6f}"


def _f4(v: float | None) -> str:
    return "" if v is None else f"{v:.4f}"


def _i(v: int | None) -> str:
    return "" if v is None else str(v)
