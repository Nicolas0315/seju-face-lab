from __future__ import annotations

import argparse
from base64 import b64encode
from datetime import date
from html import escape
import json
from pathlib import Path
import shutil
from typing import Any


AXIS_LABELS = {
    "soft_defined": ("soft", "defined"),
    "cool_warm": ("cool", "warm"),
    "deep_bright": ("deep", "bright"),
    "natural_styled": ("natural", "styled"),
    "muted_vivid": ("muted", "vivid"),
    "soft_crisp": ("soft detail", "crisp detail"),
    "light_dark_hair": ("lighter hair", "dark hair"),
    "dynamic_symmetric": ("dynamic", "symmetric"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the agency average-face static site.")
    parser.add_argument("--agencies", type=Path, default=Path("configs/agencies/seju_like_agencies.json"))
    parser.add_argument("--average-params", type=Path, default=Path("outputs/agency_reviews/seju_like/agency_average_params.json"))
    parser.add_argument("--enhancement", type=Path, default=Path("outputs/agency_enhancement/agency_enhancement_report.json"))
    parser.add_argument("--calibration", type=Path, default=Path("outputs/agency_generation_calibration/generation_calibration.json"))
    parser.add_argument("--data-quality", type=Path, default=Path("outputs/data_quality_audit_v1/data_quality_audit.json"))
    parser.add_argument("--images", type=Path, default=Path("outputs/agency_imagegen_samples"))
    parser.add_argument("--out", type=Path, default=Path("outputs/agency_site"))
    args = parser.parse_args()

    config = _read_json(args.agencies)
    params = _read_json(args.average_params)
    enhancement = _read_json(args.enhancement)
    calibration = _read_json(args.calibration) if args.calibration.exists() else {}
    data_quality = _read_json(args.data_quality) if args.data_quality.exists() else {}
    build_site(config, params, enhancement, calibration, data_quality, args.images, args.out)
    print(f"site: {args.out / 'index.html'}")
    return 0


def build_site(
    config: dict[str, Any],
    params: dict[str, Any],
    enhancement: dict[str, Any],
    calibration: dict[str, Any],
    data_quality: dict[str, Any],
    images_dir: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    _copy_images(images_dir, assets_dir)
    agencies = _merge_agencies(config, params, enhancement, calibration, data_quality)
    (out_dir / "index.html").write_text(
        _render_html(config, enhancement, calibration, data_quality, agencies),
        encoding="utf-8",
    )
    (out_dir / "data.json").write_text(
        json.dumps(
            {
                "agencies": agencies,
                "summary": enhancement.get("summary", {}),
                "calibration_summary": calibration.get("summary", {}),
                "data_quality_summary": data_quality.get("summary", {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "_headers").write_text(_headers(), encoding="utf-8")


def _merge_agencies(
    config: dict[str, Any],
    params: dict[str, Any],
    enhancement: dict[str, Any],
    calibration: dict[str, Any],
    data_quality: dict[str, Any],
) -> list[dict[str, Any]]:
    config_by_slug = {agency["slug"]: agency for agency in config.get("agencies", [])}
    params_by_slug = {agency["slug"]: agency for agency in params.get("agencies", [])}
    calibration_by_slug = {agency["slug"]: agency for agency in calibration.get("agencies", [])}
    evidence_by_slug = {
        agency.get("slug"): agency
        for agency in data_quality.get("agency_evidence", {}).get("agencies", [])
        if agency.get("slug")
    }
    enhanced = enhancement.get("agencies", [])
    rows = []
    for agency in enhanced:
        slug = agency["slug"]
        cfg = config_by_slug.get(slug, {})
        param = params_by_slug.get(slug, {})
        calibrated = calibration_by_slug.get(slug, {})
        evidence = evidence_by_slug.get(slug, {})
        rows.append(
            {
                "slug": slug,
                "name": agency["name"],
                "rank": agency["rank"],
                "enhancement_score": agency["enhancement_score"],
                "confidence": agency["confidence"],
                "components": agency["components"],
                "distribution": agency.get("observed_distribution", {}),
                "observed_axis_vector": agency.get("observed_axis_vector", {}),
                "hypothesis_axis_vector": agency.get("hypothesis_axis_vector", {}),
                "presentation_flags": agency.get("presentation_flags", []),
                "improvement_actions": agency.get("improvement_actions", []),
                "members": cfg.get("public_examples", []),
                "positioning": cfg.get("positioning", []),
                "official_sources": cfg.get("official_sources", []),
                "average_descriptors": param.get("average_descriptors", {}),
                "axis_vector": param.get("axis_vector", {}),
                "calibration": calibrated,
                "evidence": {
                    "type": evidence.get("evidence_type", "unverified"),
                    "real_image_count": evidence.get("real_image_count", 0),
                    "generated_image_count": evidence.get("generated_image_count", 0),
                    "quality_risk": data_quality.get("summary", {}).get("risk_level", "unknown"),
                },
                "image": f"assets/{slug}.png",
            }
        )
    return rows


def _render_html(
    config: dict[str, Any],
    enhancement: dict[str, Any],
    calibration: dict[str, Any],
    data_quality: dict[str, Any],
    agencies: list[dict[str, Any]],
) -> str:
    cards = "\n".join(_agency_card(agency) for agency in agencies)
    rows = "\n".join(_ranking_row(agency) for agency in agencies)
    calibration_rows = "\n".join(_calibration_row(agency) for agency in agencies)
    axis_map = _axis_map_section(agencies)
    nav = "\n".join(f'<a href="#{escape(agency["slug"])}">{escape(agency["name"])}</a>' for agency in agencies)
    generated_at = date.today().isoformat()
    retrieved_at = config.get("retrieved_at", "unknown")
    summary = enhancement.get("summary", {})
    calibration_summary = calibration.get("summary", {})
    data_quality_summary = data_quality.get("summary", {})
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agency Average Face Research</title>
  <meta name="description" content="Agency member examples, official roster links, generated aggregate face images, and seju-face enhancement scores.">
  <link rel="icon" href="data:,">
  <style>{_css()}</style>
</head>
<body>
  <header class="hero">
    <div class="hero-inner">
      <p class="eyebrow">seju-face-lab / agency research</p>
      <h1>事務所別 平均顔リサーチ</h1>
      <p class="lead">
        公式所属一覧へのリンク、公開メンバー例、架空の平均顔生成画像、seju重心への近似スコアを
        1ページで確認するための研究ビューです。
      </p>
      <div class="meta">
        <span>generated: {escape(generated_at)}</span>
        <span>source retrieved: {escape(str(retrieved_at))}</span>
        <span>top: {escape(str(summary.get("top_slug", "")))} {escape(str(summary.get("top_score", "")))}</span>
        <span>regen first: {escape(", ".join(calibration_summary.get("regenerate_first", [])))}</span>
        <span>data quality: {escape(str(data_quality_summary.get("risk_level", "unknown")))}</span>
      </div>
    </div>
  </header>

  <nav class="agency-nav" aria-label="Agency sections">
    {nav}
  </nav>

  <main>
    <section class="summary" aria-labelledby="summary-title">
      <div>
        <h2 id="summary-title">Ranking</h2>
        <p>
          enhancement score は、事務所別の仮説パラメータ、生成画像のseju重心スコア、
          8軸観測の一致度を融合した研究用スコアです。
        </p>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>rank</th>
              <th>agency</th>
              <th>score</th>
              <th>image</th>
              <th>axis</th>
              <th>quadrant</th>
              <th>evidence</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
      </div>
    </section>

    {axis_map}

    <section class="cards" aria-label="Agency cards">
      {cards}
    </section>

    <section class="calibration" aria-labelledby="calibration-title">
      <div>
        <h2 id="calibration-title">生成精度改善プラン</h2>
        <p>
          現在の生成画像を測定し、目標 image score 0.35、axis alignment 0.62、
          enhancement score 0.76 に届いていない箇所を補正した次回生成プランです。
        </p>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>agency</th>
              <th>priority</th>
              <th>image gap</th>
              <th>axis gap</th>
              <th>seed</th>
              <th>next</th>
            </tr>
          </thead>
          <tbody>
            {calibration_rows}
          </tbody>
        </table>
      </div>
    </section>

    <section class="boundary" aria-labelledby="boundary-title">
      <h2 id="boundary-title">Research Boundary</h2>
      <p>
        このページの画像は架空の集約サンプルで、実在人物の似顔ではありません。
        スコアはローカル画像セット由来の重心への近似であり、本人識別、魅力度判定、
        人気、人格、民族性、個人価値を表しません。
      </p>
    </section>
  </main>
</body>
</html>
"""


def _axis_map_section(agencies: list[dict[str, Any]]) -> str:
    if not any(agency.get("observed_axis_vector") for agency in agencies):
        return ""
    axis_rows = "\n".join(_axis_vector_row(agency) for agency in agencies)
    return f"""
    <section class="axis-map-section" aria-labelledby="axis-map-title">
      <div class="axis-map-copy">
        <h2 id="axis-map-title">8軸方向性マップ</h2>
        <p>
          生成された事務所別平均顔画像を、8つの観測軸に分解した位置図です。
          大きな象限は defined/natural と bright/warm の合成、十字軸は symmetric と dark hair の傾向を示します。
        </p>
        <p class="axis-caveat">
          これは画像特徴の研究用マップであり、実在人物や所属者への評価ラベルではありません。
        </p>
      </div>
      <div class="axis-map-panel">
        {_axis_map_svg(agencies)}
      </div>
      <div class="axis-bars" aria-label="8-axis vectors by agency">
        {axis_rows}
      </div>
    </section>
"""


def _axis_map_svg(agencies: list[dict[str, Any]]) -> str:
    points = "\n".join(_axis_map_point(agency, index) for index, agency in enumerate(agencies))
    return f"""
<svg class="axis-map" viewBox="0 0 640 420" role="img" aria-labelledby="axis-map-svg-title axis-map-svg-desc">
  <title id="axis-map-svg-title">Agency 8-axis quadrant map</title>
  <desc id="axis-map-svg-desc">Two-dimensional projection of observed 8-axis vectors for each agency aggregate image.</desc>
  <rect class="axis-map-bg" x="54" y="34" width="520" height="300" rx="6"></rect>
  <line class="axis-map-mid" x1="314" y1="34" x2="314" y2="334"></line>
  <line class="axis-map-mid" x1="54" y1="184" x2="574" y2="184"></line>
  <text class="axis-map-label center" x="314" y="24">bright / warm</text>
  <text class="axis-map-label center" x="314" y="356">deep / cool</text>
  <text class="axis-map-label side" x="24" y="190" transform="rotate(-90 24 190)">natural / soft</text>
  <text class="axis-map-label side" x="612" y="190" transform="rotate(90 612 190)">defined / styled</text>
  <text class="axis-map-corner" x="74" y="72">soft bright</text>
  <text class="axis-map-corner end" x="554" y="72">defined bright</text>
  <text class="axis-map-corner" x="74" y="314">soft deep</text>
  <text class="axis-map-corner end" x="554" y="314">defined deep</text>
  {points}
</svg>
"""


def _axis_map_point(agency: dict[str, Any], index: int) -> str:
    axes = agency.get("observed_axis_vector", {})
    x_axis = _axis_average(axes, ["soft_defined", "natural_styled"])
    y_axis = _axis_average(axes, ["deep_bright", "cool_warm"])
    cross_x = _axis_value(axes, "dynamic_symmetric")
    cross_y = _axis_value(axes, "light_dark_hair")
    x = _scale_axis(x_axis, 54, 574)
    y = _scale_axis(y_axis, 334, 34)
    r = 8 + max(0.0, _axis_value(agency.get("components", {}), "axis_alignment")) * 6
    label_y = y - 14 if index % 2 == 0 else y + 25
    slug = str(agency.get("slug", ""))
    name = str(agency.get("name", slug))
    return f"""
  <g class="axis-point axis-point-{index + 1}" tabindex="0" aria-label="{escape(name)} x {_fmt(x_axis)} y {_fmt(y_axis)} cross symmetric {_fmt(cross_x)} dark hair {_fmt(cross_y)}">
    <circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}"></circle>
    <text x="{x:.1f}" y="{label_y:.1f}">{escape(slug)}</text>
  </g>
"""


def _axis_vector_row(agency: dict[str, Any]) -> str:
    axes = agency.get("observed_axis_vector", {})
    bars = "\n".join(_axis_bar(axis, axes.get(axis)) for axis in AXIS_LABELS)
    distribution = agency.get("distribution", {})
    return f"""
<article class="axis-vector-card">
  <div class="axis-vector-head">
    <h3>{escape(str(agency.get("name", "")))}</h3>
    <span>{escape(str(distribution.get("quadrant", "")))}</span>
  </div>
  <div class="axis-vector-grid">
    {bars}
  </div>
</article>
"""


def _axis_bar(axis: str, value: Any) -> str:
    low, high = AXIS_LABELS[axis]
    numeric = _axis_value({axis: value}, axis)
    left = 50 if numeric >= 0 else 50 + numeric * 50
    width = abs(numeric) * 50
    side = "pos" if numeric >= 0 else "neg"
    return f"""
<div class="axis-bar {side}">
  <div class="axis-bar-label">
    <span>{escape(low)}</span>
    <strong>{escape(axis.replace("_", " "))}</strong>
    <span>{escape(high)}</span>
  </div>
  <div class="axis-track" aria-label="{escape(axis)} {_fmt(numeric)}">
    <span class="axis-zero"></span>
    <span class="axis-fill" style="left: {left:.1f}%; width: {width:.1f}%"></span>
  </div>
  <span class="axis-value">{_fmt(numeric)}</span>
</div>
"""


def _agency_card(agency: dict[str, Any]) -> str:
    members = "".join(f"<li>{escape(str(member))}</li>" for member in agency["members"])
    sources = "".join(
        f'<a href="{escape(source.get("url", ""))}" target="_blank" rel="noreferrer">{escape(source.get("name", "official"))}</a>'
        for source in agency["official_sources"]
    )
    flags = "".join(f"<span>{escape(str(flag))}</span>" for flag in agency["presentation_flags"])
    actions = "".join(f"<li>{escape(str(action))}</li>" for action in agency["improvement_actions"])
    calibration = agency.get("calibration", {})
    gaps = calibration.get("gaps_to_target", {}) if isinstance(calibration, dict) else {}
    plan = calibration.get("generation_plan", {}) if isinstance(calibration, dict) else {}
    positioning = " / ".join(escape(str(item)) for item in agency["positioning"])
    image_alt = f"{agency['name']} fictional aggregate average face sample"
    components = agency["components"]
    distribution = agency["distribution"]
    evidence = agency.get("evidence", {})
    evidence_type = str(evidence.get("type", "unverified"))
    evidence_label = evidence_type.replace("_", " ")
    evidence_class = evidence_type.replace("_", "-")
    real_images = evidence.get("real_image_count", 0)
    return f"""
<article class="card" id="{escape(agency["slug"])}">
  <div class="portrait">
    <img src="{escape(agency["image"])}" alt="{escape(image_alt)}" loading="lazy">
  </div>
  <div class="card-body">
    <div class="card-head">
      <span class="rank">#{escape(str(agency["rank"]))}</span>
      <h2>{escape(agency["name"])}</h2>
      <span class="evidence evidence-{escape(evidence_class)}">{escape(evidence_label)}</span>
    </div>
    <p class="positioning">{positioning}</p>
    <p class="evidence-note">real images: {escape(str(real_images))} / risk: {escape(str(evidence.get("quality_risk", "unknown")))}</p>
    <dl class="metrics">
      <div><dt>enhancement</dt><dd>{_fmt(agency["enhancement_score"])}</dd></div>
      <div><dt>descriptor</dt><dd>{_fmt(components.get("descriptor_similarity"))}</dd></div>
      <div><dt>image</dt><dd>{_fmt(components.get("image_centroid_score"))}</dd></div>
      <div><dt>axis</dt><dd>{_fmt(components.get("axis_alignment"))}</dd></div>
    </dl>
    <div class="split">
      <section>
        <h3>所属メンバー例</h3>
        <ul class="members">{members}</ul>
      </section>
      <section>
        <h3>公式一覧</h3>
        <div class="sources">{sources}</div>
      </section>
    </div>
    <div class="axis-line">
      <span>{escape(str(distribution.get("quadrant", "")))}</span>
      <span>{escape(str(distribution.get("corner", "")))}</span>
      <span>{escape(str(distribution.get("cross_label", "")))}</span>
    </div>
    <div class="flags">{flags}</div>
    <section class="actions">
      <h3>次の改善</h3>
      <ul>{actions}</ul>
    </section>
    <section class="calibration-note">
      <h3>精度改善プラン</h3>
      <p>
        priority: <strong>{escape(str(calibration.get("priority", "n/a")))}</strong> /
        image gap: {_fmt(gaps.get("image_centroid_score"))} /
        axis gap: {_fmt(gaps.get("axis_alignment"))} /
        seed: {escape(str(plan.get("seed", "n/a")))}
      </p>
    </section>
  </div>
</article>
"""


def _ranking_row(agency: dict[str, Any]) -> str:
    components = agency["components"]
    distribution = agency["distribution"]
    evidence = agency.get("evidence", {})
    return f"""
<tr>
  <td>{escape(str(agency["rank"]))}</td>
  <td><a href="#{escape(agency["slug"])}">{escape(agency["name"])}</a></td>
  <td>{_fmt(agency["enhancement_score"])}</td>
  <td>{_fmt(components.get("image_centroid_score"))}</td>
  <td>{_fmt(components.get("axis_alignment"))}</td>
  <td>{escape(str(distribution.get("quadrant", "")))}</td>
  <td>{escape(str(evidence.get("type", "unverified")).replace("_", " "))}</td>
</tr>
"""


def _calibration_row(agency: dict[str, Any]) -> str:
    calibration = agency.get("calibration", {})
    gaps = calibration.get("gaps_to_target", {}) if isinstance(calibration, dict) else {}
    plan = calibration.get("generation_plan", {}) if isinstance(calibration, dict) else {}
    return f"""
<tr>
  <td><a href="#{escape(agency["slug"])}">{escape(agency["name"])}</a></td>
  <td>{escape(str(calibration.get("priority", "")))}</td>
  <td>{_fmt(gaps.get("image_centroid_score"))}</td>
  <td>{_fmt(gaps.get("axis_alignment"))}</td>
  <td>{escape(str(plan.get("seed", "")))}</td>
  <td>{escape(str(plan.get("recommended_output_dir", "")))}</td>
</tr>
"""


def _copy_images(images_dir: Path, assets_dir: Path) -> None:
    for path in sorted(images_dir.glob("*.png")):
        shutil.copy2(path, assets_dir / path.name)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _axis_value(values: dict[str, Any], key: str) -> float:
    try:
        return max(-1.0, min(1.0, float(values.get(key, 0.0))))
    except (TypeError, ValueError):
        return 0.0


def _axis_average(values: dict[str, Any], keys: list[str]) -> float:
    if not keys:
        return 0.0
    return sum(_axis_value(values, key) for key in keys) / len(keys)


def _scale_axis(value: float, start: float, end: float) -> float:
    normalized = (_axis_value({"value": value}, "value") + 1.0) / 2.0
    return start + normalized * (end - start)


def _headers() -> str:
    nonce = b64encode(b"seju-face-lab").decode("ascii")
    return f"""/*
  X-Content-Type-Options: nosniff
  Referrer-Policy: strict-origin-when-cross-origin
  Permissions-Policy: interest-cohort=()
  Content-Security-Policy: default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'nonce-{nonce}';
"""


def _css() -> str:
    return """
:root {
  color-scheme: light;
  --ink: #202124;
  --muted: #5f6368;
  --line: #d9dde3;
  --paper: #fbfaf7;
  --surface: #ffffff;
  --accent: #0f766e;
  --accent-soft: #d9f0ec;
  --rose: #b4535f;
  --gold: #8a6b16;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Arial, "Hiragino Kaku Gothic ProN", "Yu Gothic", Meiryo, sans-serif;
  color: var(--ink);
  background: var(--paper);
  line-height: 1.55;
}
a { color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 3px; }
.hero {
  background: #132522;
  color: white;
  padding: 48px 20px 28px;
}
.hero-inner { max-width: 1120px; margin: 0 auto; }
.eyebrow { color: #9fd5cc; text-transform: uppercase; letter-spacing: .08em; font-size: 12px; }
h1 { font-size: clamp(32px, 6vw, 68px); line-height: 1.05; margin: 10px 0 16px; letter-spacing: 0; }
.lead { max-width: 780px; font-size: 18px; color: #eef7f5; }
.meta { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 22px; }
.meta span {
  border: 1px solid rgba(255,255,255,.28);
  border-radius: 999px;
  padding: 6px 10px;
  font-size: 13px;
}
.agency-nav {
  position: sticky;
  top: 0;
  z-index: 2;
  display: flex;
  gap: 8px;
  overflow-x: auto;
  padding: 10px 20px;
  border-bottom: 1px solid var(--line);
  background: rgba(251,250,247,.94);
  backdrop-filter: blur(10px);
}
.agency-nav a {
  white-space: nowrap;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 7px 11px;
  background: white;
  font-size: 13px;
  text-decoration: none;
}
main { max-width: 1120px; margin: 0 auto; padding: 24px 20px 56px; }
.summary {
  display: grid;
  grid-template-columns: minmax(0, 320px) minmax(0, 1fr);
  gap: 24px;
  align-items: start;
  padding: 18px 0 26px;
}
h2 { font-size: 24px; margin: 0 0 10px; }
h3 { font-size: 13px; margin: 0 0 8px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); background: white; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { padding: 10px; border-bottom: 1px solid var(--line); text-align: left; }
th { color: var(--muted); font-weight: 600; background: #f5f6f7; }
td:nth-child(1), td:nth-child(3), td:nth-child(4), td:nth-child(5) {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.cards { display: grid; gap: 18px; }
.axis-map-section {
  display: grid;
  grid-template-columns: minmax(0, 320px) minmax(0, 1fr);
  gap: 24px;
  align-items: start;
  padding: 20px 0 26px;
  border-top: 1px solid var(--line);
}
.axis-map-copy p { color: var(--muted); margin: 0 0 12px; }
.axis-caveat { font-size: 13px; }
.axis-map-panel {
  border: 1px solid var(--line);
  background: white;
  padding: 12px;
}
.axis-map { display: block; width: 100%; height: auto; }
.axis-map-bg { fill: #fbfcfb; stroke: var(--line); }
.axis-map-mid { stroke: #b8c0c8; stroke-width: 1; stroke-dasharray: 5 5; }
.axis-map-label, .axis-map-corner {
  fill: var(--muted);
  font-size: 13px;
}
.axis-map-corner { font-size: 12px; fill: #7a8188; }
.axis-map-label.end, .axis-map-corner.end { text-anchor: end; }
.axis-map-label.center, .axis-map-label.side { text-anchor: middle; }
.axis-point circle {
  fill: var(--accent);
  fill-opacity: .82;
  stroke: white;
  stroke-width: 2;
}
.axis-point text {
  fill: var(--ink);
  font-size: 12px;
  font-weight: 700;
  text-anchor: middle;
  paint-order: stroke;
  stroke: white;
  stroke-width: 4px;
}
.axis-point-2 circle { fill: #b4535f; }
.axis-point-3 circle { fill: #8a6b16; }
.axis-point-4 circle { fill: #4f6f9f; }
.axis-point-5 circle { fill: #6f5b8c; }
.axis-point-6 circle { fill: #62764f; }
.axis-point-7 circle { fill: #9a5f3f; }
.axis-point-8 circle { fill: #3f7688; }
.axis-bars {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
.axis-vector-card {
  border: 1px solid var(--line);
  background: white;
  padding: 12px;
}
.axis-vector-head {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  align-items: baseline;
  margin-bottom: 10px;
}
.axis-vector-head h3 {
  margin: 0;
  color: var(--ink);
  text-transform: none;
  letter-spacing: 0;
  font-size: 15px;
}
.axis-vector-head span {
  color: var(--muted);
  font-size: 12px;
  white-space: nowrap;
}
.axis-vector-grid { display: grid; gap: 7px; }
.axis-bar {
  display: grid;
  grid-template-columns: minmax(150px, 1fr) minmax(120px, 1.1fr) 46px;
  gap: 8px;
  align-items: center;
  font-size: 12px;
}
.axis-bar-label {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 6px;
  color: var(--muted);
}
.axis-bar-label strong {
  color: var(--ink);
  font-weight: 600;
}
.axis-bar-label span:last-child { text-align: right; }
.axis-track {
  position: relative;
  height: 10px;
  border-radius: 999px;
  background: #edf1f0;
  overflow: hidden;
}
.axis-zero {
  position: absolute;
  left: 50%;
  top: 0;
  bottom: 0;
  width: 1px;
  background: #8f979f;
}
.axis-fill {
  position: absolute;
  top: 0;
  bottom: 0;
  border-radius: 999px;
  background: var(--accent);
}
.axis-bar.neg .axis-fill { background: var(--rose); }
.axis-value {
  text-align: right;
  font-variant-numeric: tabular-nums;
  color: var(--muted);
}
.card {
  display: grid;
  grid-template-columns: minmax(240px, 360px) minmax(0, 1fr);
  border: 1px solid var(--line);
  background: var(--surface);
}
.portrait { background: #e9eceb; min-height: 320px; }
.portrait img { display: block; width: 100%; height: 100%; object-fit: cover; aspect-ratio: 4 / 5; }
.card-body { padding: 18px; }
.card-head { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.card-head h2 { margin: 0; }
.rank { color: var(--rose); font-weight: 700; font-variant-numeric: tabular-nums; }
.positioning { color: var(--muted); margin: 8px 0 14px; }
.evidence {
  display: inline-flex;
  align-items: center;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 4px 8px;
  color: #164640;
  background: var(--accent-soft);
  font-size: 12px;
  font-weight: 700;
}
.evidence-hypothesis-and-generated { color: var(--gold); background: #f6ead1; }
.evidence-real-and-generated { color: #164640; background: #d9f0ec; }
.evidence-real-centroid-baseline { color: #123b6d; background: #dfeafb; }
.evidence-unverified { color: var(--muted); background: #eef0f2; }
.evidence-note { margin: -8px 0 14px; color: var(--muted); font-size: 12px; }
.metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  margin: 0 0 16px;
}
.metrics div { border: 1px solid var(--line); padding: 10px; background: #fbfcfb; }
.metrics dt { color: var(--muted); font-size: 12px; }
.metrics dd { margin: 2px 0 0; font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
.split { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.members { margin: 0; padding-left: 18px; }
.sources { display: flex; flex-direction: column; gap: 6px; }
.axis-line { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 10px; }
.axis-line span, .flags span {
  display: inline-flex;
  border-radius: 999px;
  padding: 5px 8px;
  font-size: 12px;
  background: var(--accent-soft);
  color: #164640;
}
.flags { display: flex; flex-wrap: wrap; gap: 6px; }
.flags span { background: #f6ead1; color: var(--gold); }
.actions { margin-top: 16px; }
.actions ul { margin: 0; padding-left: 18px; }
.calibration {
  display: grid;
  grid-template-columns: minmax(0, 320px) minmax(0, 1fr);
  gap: 24px;
  align-items: start;
  margin-top: 24px;
  padding: 20px 0;
}
.calibration-note {
  margin-top: 14px;
  border-top: 1px solid var(--line);
  padding-top: 12px;
}
.calibration-note p { margin: 0; color: var(--muted); }
.boundary {
  margin-top: 28px;
  border-top: 1px solid var(--line);
  padding-top: 18px;
  color: var(--muted);
}
@media (max-width: 820px) {
  .summary, .card, .split, .calibration, .axis-map-section, .axis-bars { grid-template-columns: 1fr; }
  .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .axis-bar { grid-template-columns: 1fr; gap: 4px; }
  .axis-bar-label { grid-template-columns: 1fr auto 1fr; }
  .axis-value { text-align: left; }
  .portrait { min-height: 0; }
}
"""


if __name__ == "__main__":
    raise SystemExit(main())
