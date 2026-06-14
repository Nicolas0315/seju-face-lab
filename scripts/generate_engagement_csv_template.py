"""Generate a blank CSV template for manual SNS engagement entry.

Reads the SNS handles manifest and writes a CSV with one row per (talent, platform).
Fill in followers/posts/total_engagement columns, then import with:

    python -m seju_face_lab sources import-engagement \
        --csv data/processed/sns_engagement_manual.csv \
        --out data/processed/sns_engagement.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


COLUMNS = [
    "talent_slug", "name", "platform", "handle",
    "followers", "following", "posts", "total_engagement", "engagement_rate",
    "display_name", "bio",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manual engagement CSV template.")
    parser.add_argument("--handles", type=Path, default=Path("data/processed/sns_handles.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/sns_engagement_manual.csv"))
    args = parser.parse_args()

    handles_path: Path = args.handles
    if not handles_path.exists():
        raise SystemExit(f"Handles manifest not found: {handles_path}")

    rows = []
    for line in handles_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        slug = data["talent_slug"]
        name = data.get("name") or ""
        for platform, handle in data.get("sns_handles", {}).items():
            rows.append({
                "talent_slug": slug,
                "name": name,
                "platform": platform,
                "handle": handle,
                "followers": "",
                "following": "",
                "posts": "",
                "total_engagement": "",
                "engagement_rate": "",
                "display_name": "",
                "bio": "",
            })

    rows.sort(key=lambda r: (r["talent_slug"], r["platform"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"template written: {args.out}")
    print(f"rows: {len(rows)} (talents x platforms)")
    print()
    print("Fill in at minimum: followers")
    print("Then run:")
    print(
        f"  python -m seju_face_lab sources import-engagement "
        f"--csv {args.out} --out data/processed/sns_engagement.jsonl"
    )


if __name__ == "__main__":
    main()
