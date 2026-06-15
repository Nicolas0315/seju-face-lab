from __future__ import annotations

import argparse
from base64 import b64encode
from datetime import date
from html import escape
import json
from pathlib import Path
import shutil
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the agency average-face static site.")
    parser.add_argument("--agencies", type=Path, default=Path("configs/agencies/seju_like_agencies.json"))
    parser.add_argument("--average-params", type=Path, default=Path("outputs/agency_reviews/seju_like/agency_average_params.json"))
    parser.add_argument("--enhancement", type=Path, default=Path("outputs/agency_enhancement/agency_enhancement_report.json"))
    parser.add_argument("--images", type=Path, default=Path("outputs/agency_imagegen_samples"))
    parser.add_argument("--out", type=Path, default=Path("outputs/agency_site"))
    args = parser.parse_args()

    config = _read_json(args.agencies)
    params = _read_json(args.average_params)
    enhancement = _read_json(args.enhancement)
    build_site(config, params, enhancement, args.images, args.out)
    print(f"site: {args.out / 'index.html'}")
    return 0


def build_site(
    config: dict[str, Any],
    params: dict[str, Any],
    enhancement: dict[str, Any],
    images_dir: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    _copy_images(images_dir, assets_dir)
    agencies = _merge_agencies(config, params, enhancement)
    (out_dir / "index.html").write_text(_render_html(config, enhancement, agencies), encoding="utf-8")
    (out_dir / "data.json").write_text(
        json.dumps({"agencies": agencies, "summary": enhancement.get("summary", {})}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "_headers").write_text(_headers(), encoding="utf-8")


def _merge_agencies(
    config: dict[str, Any],
    params: dict[str, Any],
    enhancement: dict[str, Any],
) -> list[dict[str, Any]]:
    config_by_slug = {agency["slug"]: agency for agency in config.get("agencies", [])}
    params_by_slug = {agency["slug"]: agency for agency in params.get("agencies", [])}
    enhanced = enhancement.get("agencies", [])
    rows = []
    for agency in enhanced:
        slug = agency["slug"]
        cfg = config_by_slug.get(slug, {})
        param = params_by_slug.get(slug, {})
        rows.append(
            {
                "slug": slug,
                "name": agency["name"],
                "rank": agency["rank"],
                "enhancement_score": agency["enhancement_score"],
                "confidence": agency["confidence"],
                "components": agency["components"],
                "distribution": agency.get("observed_distribution", {}),
                "presentation_flags": agency.get("presentation_flags", []),
                "improvement_actions": agency.get("improvement_actions", []),
                "members": cfg.get("public_examples", []),
                "positioning": cfg.get("positioning", []),
                "official_sources": cfg.get("official_sources", []),
                "average_descriptors": param.get("average_descriptors", {}),
                "axis_vector": param.get("axis_vector", {}),
                "image": f"assets/{slug}.png",
            }
        )
    return rows


def _render_html(config: dict[str, Any], enhancement: dict[str, Any], agencies: list[dict[str, Any]]) -> str:
    cards = "\n".join(_agency_card(agency) for agency in agencies)
    rows = "\n".join(_ranking_row(agency) for agency in agencies)
    nav = "\n".join(f'<a href="#{escape(agency["slug"])}">{escape(agency["name"])}</a>' for agency in agencies)
    generated_at = date.today().isoformat()
    retrieved_at = config.get("retrieved_at", "unknown")
    summary = enhancement.get("summary", {})
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
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
      </div>
    </section>

    <section class="cards" aria-label="Agency cards">
      {cards}
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


def _agency_card(agency: dict[str, Any]) -> str:
    members = "".join(f"<li>{escape(str(member))}</li>" for member in agency["members"])
    sources = "".join(
        f'<a href="{escape(source.get("url", ""))}" target="_blank" rel="noreferrer">{escape(source.get("name", "official"))}</a>'
        for source in agency["official_sources"]
    )
    flags = "".join(f"<span>{escape(str(flag))}</span>" for flag in agency["presentation_flags"])
    actions = "".join(f"<li>{escape(str(action))}</li>" for action in agency["improvement_actions"])
    positioning = " / ".join(escape(str(item)) for item in agency["positioning"])
    image_alt = f"{agency['name']} fictional aggregate average face sample"
    components = agency["components"]
    distribution = agency["distribution"]
    return f"""
<article class="card" id="{escape(agency["slug"])}">
  <div class="portrait">
    <img src="{escape(agency["image"])}" alt="{escape(image_alt)}" loading="lazy">
  </div>
  <div class="card-body">
    <div class="card-head">
      <span class="rank">#{escape(str(agency["rank"]))}</span>
      <h2>{escape(agency["name"])}</h2>
    </div>
    <p class="positioning">{positioning}</p>
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
  </div>
</article>
"""


def _ranking_row(agency: dict[str, Any]) -> str:
    components = agency["components"]
    distribution = agency["distribution"]
    return f"""
<tr>
  <td>{escape(str(agency["rank"]))}</td>
  <td><a href="#{escape(agency["slug"])}">{escape(agency["name"])}</a></td>
  <td>{_fmt(agency["enhancement_score"])}</td>
  <td>{_fmt(components.get("image_centroid_score"))}</td>
  <td>{_fmt(components.get("axis_alignment"))}</td>
  <td>{escape(str(distribution.get("quadrant", "")))}</td>
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
.card {
  display: grid;
  grid-template-columns: minmax(240px, 360px) minmax(0, 1fr);
  border: 1px solid var(--line);
  background: var(--surface);
}
.portrait { background: #e9eceb; min-height: 320px; }
.portrait img { display: block; width: 100%; height: 100%; object-fit: cover; aspect-ratio: 4 / 5; }
.card-body { padding: 18px; }
.card-head { display: flex; align-items: baseline; gap: 10px; }
.card-head h2 { margin: 0; }
.rank { color: var(--rose); font-weight: 700; font-variant-numeric: tabular-nums; }
.positioning { color: var(--muted); margin: 8px 0 14px; }
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
.boundary {
  margin-top: 28px;
  border-top: 1px solid var(--line);
  padding-top: 18px;
  color: var(--muted);
}
@media (max-width: 820px) {
  .summary, .card, .split { grid-template-columns: 1fr; }
  .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .portrait { min-height: 0; }
}
"""


if __name__ == "__main__":
    raise SystemExit(main())
