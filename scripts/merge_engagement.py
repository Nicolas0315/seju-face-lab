"""Merge platform-specific engagement data into a single JSONL manifest.

Usage:
    python scripts/merge_engagement.py \
        --twitter data/processed/sns_engagement_twitter.jsonl \
        --instagram data/processed/ig_results_m5.json \
        --tiktok data/processed/tk_results_m5.json \
        # optional
        --handles data/processed/sns_handles.jsonl \
        --out data/processed/sns_engagement.jsonl
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--twitter", type=Path)
    parser.add_argument("--instagram", type=Path)
    parser.add_argument("--tiktok", type=Path, default=None)
    parser.add_argument("--handles", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    retrieved_at = datetime.now(timezone.utc).isoformat()

    # Load handle map: slug -> {platform: handle}
    slug_to_handles: dict[str, dict[str, str]] = {}
    slug_to_name: dict[str, str | None] = {}
    for line in args.handles.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        slug_to_handles[d["talent_slug"]] = d.get("sns_handles", {})
        slug_to_name[d["talent_slug"]] = d.get("name")

    all_slugs = sorted(slug_to_handles)

    # Load Twitter data (JSONL from fetch-engagement)
    twitter_by_slug: dict[str, dict] = {}
    if args.twitter and args.twitter.exists():
        for line in args.twitter.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            tw_engs = [e for e in rec.get("engagements", []) if e["platform"] == "twitter"]
            if tw_engs:
                twitter_by_slug[rec["talent_slug"]] = tw_engs[0]

    # Load Instagram data (JSON from M5 batch script)
    ig_by_slug: dict[str, dict] = {}
    if args.instagram and args.instagram.exists():
        ig_by_slug = json.loads(args.instagram.read_text(encoding="utf-8"))

    # Load TikTok data (JSON from M5 batch script, optional)
    tk_by_slug: dict[str, dict] = {}
    if args.tiktok and args.tiktok.exists():
        tk_by_slug = json.loads(args.tiktok.read_text(encoding="utf-8"))

    def _engagement_rate(total_engagement: int | None, followers: int | None, posts: int | None) -> float | None:
        if total_engagement is None or followers is None or followers == 0:
            return None
        if posts and posts > 0:
            return round((total_engagement / posts) / followers, 6)
        return round(total_engagement / followers, 6)

    def _ig_engagement(slug: str) -> dict | None:
        ig = ig_by_slug.get(slug)
        if not ig:
            return None
        handle = ig.get("handle", slug_to_handles.get(slug, {}).get("instagram", ""))
        fol = ig.get("followers")
        return {
            "platform": "instagram", "handle": handle,
            "profile_url": "https://www.instagram.com/" + handle + "/",
            "followers": fol, "following": None, "posts": None,
            "total_engagement": None, "engagement_rate": None,
            "bio": None, "display_name": None,
            "fetch_status": "ok" if fol is not None else ig.get("status", "partial"),
            "fetch_error": None, "retrieved_at": retrieved_at,
        }

    def _tw_engagement(slug: str) -> dict | None:
        tw = twitter_by_slug.get(slug)
        return tw

    def _tk_engagement(slug: str) -> dict | None:
        tk = tk_by_slug.get(slug)
        if not tk:
            return None
        handle = tk.get("handle", slug_to_handles.get(slug, {}).get("tiktok", ""))
        fol = tk.get("followers")
        posts = tk.get("posts")
        hearts = tk.get("hearts")
        eng_rate = _engagement_rate(hearts, fol, posts)
        return {
            "platform": "tiktok", "handle": handle,
            "profile_url": "https://www.tiktok.com/@" + handle,
            "followers": fol, "following": None, "posts": posts,
            "total_engagement": hearts, "engagement_rate": eng_rate,
            "bio": None, "display_name": None,
            "fetch_status": "ok" if fol is not None else tk.get("status", "partial"),
            "fetch_error": None, "retrieved_at": retrieved_at,
        }

    records = []
    for slug in all_slugs:
        engs = []
        ig = _ig_engagement(slug)
        if ig:
            engs.append(ig)
        tw = _tw_engagement(slug)
        if tw:
            engs.append(tw)
        tk = _tk_engagement(slug)
        if tk:
            engs.append(tk)
        if engs:
            records.append({"talent_slug": slug, "name": slug_to_name.get(slug), "engagements": engs})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ig_ok = sum(1 for r in records for e in r["engagements"] if e["platform"] == "instagram" and e.get("followers"))
    tw_ok = sum(1 for r in records for e in r["engagements"] if e["platform"] == "twitter" and e.get("followers"))
    tk_ok = sum(1 for r in records for e in r["engagements"] if e["platform"] == "tiktok" and e.get("followers"))
    print(f"Merged engagement manifest: {args.out}")
    print(f"  talents: {len(records)}")
    print(f"  instagram with followers: {ig_ok}")
    print(f"  twitter with followers: {tw_ok}")
    print(f"  tiktok with followers: {tk_ok}")


if __name__ == "__main__":
    main()
