from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from threading import Lock
from typing import Iterable

# SNS URL patterns for handle extraction
_SNS_PATTERNS: dict[str, re.Pattern[str]] = {
    "instagram": re.compile(r"(?:www\.)?instagram\.com/([A-Za-z0-9_.](?:[A-Za-z0-9_.]{0,29}))/?(?:\?|$|#)"),
    "twitter": re.compile(r"(?:www\.)?(?:twitter|x)\.com/([A-Za-z0-9_]{1,15})/?(?:\?|$|#)"),
    "tiktok": re.compile(r"(?:www\.)?tiktok\.com/@([A-Za-z0-9_.]{2,24})/?(?:\?|$|#)"),
}
# Handles that are navigation paths, not user accounts
_IGNORED_HANDLES: dict[str, frozenset[str]] = {
    "instagram": frozenset({"p", "reel", "reels", "explore", "stories", "tv", "accounts", "web",
                             "direct", "login", "signup", "challenge", "legal"}),
    "twitter": frozenset({"intent", "share", "home", "i", "settings", "privacy", "tos",
                          "en", "login", "signup", "search", "explore", "notifications"}),
    "tiktok": frozenset({"login", "signup", "trending", "live", "music", "foryou", "explore",
                         "discover", "tag", "sound", "effect", "embed"}),
}

_IG_API_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_DEFAULT_UA = "seju-face-lab/0.1 (+local research; contact: local)"
_SIGI_STATE_RE = re.compile(r'<script[^>]+id="SIGI_STATE"[^>]*>(.*?)</script>', re.DOTALL)
_UNIVERSAL_DATA_RE = re.compile(
    r'<script[^>]*>\s*window\.__UNIVERSAL_DATA_FOR_REHYDRATION__\s*=\s*(\{.*?\})\s*;?\s*</script>',
    re.DOTALL,
)
_NITTER_STATS_RE = re.compile(r'<li[^>]*class="[^"]*followers[^"]*"[^>]*>.*?<span[^>]*>([\d,]+)</span>', re.DOTALL)
_FOLLOWER_TEXT_RE = re.compile(r"([\d,]+(?:\.\d+)?[KMBkmb]?)\s*(?:フォロワー|followers|Followers)")


@dataclass
class SnsHandleRecord:
    talent_slug: str
    name: str | None
    profile_url: str
    sns_handles: dict[str, str]
    retrieved_at: str


@dataclass
class SnsEngagement:
    platform: str
    handle: str
    profile_url: str
    followers: int | None
    following: int | None
    posts: int | None
    total_engagement: int | None  # sum of all available engagement signals
    engagement_rate: float | None  # total_engagement / followers if calculable
    bio: str | None
    display_name: str | None
    fetch_status: str  # "ok" | "partial" | "blocked" | "not_found" | "error"
    fetch_error: str | None
    retrieved_at: str


@dataclass
class TalentEngagementRecord:
    talent_slug: str
    name: str | None
    engagements: list[SnsEngagement]


# ─── handle extraction from profile page HTML ────────────────────────────────

def extract_sns_handles_from_links(links: list[tuple[str, str | None]]) -> dict[str, str]:
    """Return {platform: handle} from a list of (href, text) link tuples."""
    found: dict[str, str] = {}
    for href, _text in links:
        for platform, pattern in _SNS_PATTERNS.items():
            if platform in found:
                continue
            m = pattern.search(href)
            if not m:
                continue
            handle = m.group(1).rstrip("/")
            if handle.lower() in _IGNORED_HANDLES[platform]:
                continue
            found[platform] = handle
    return found


# ─── scraping talent profile pages for SNS handles ───────────────────────────

def scrape_talent_sns_handles(
    manifest_path: Path,
    out_path: Path,
    delay_seconds: float = 0.8,
    max_profiles: int | None = None,
    user_agent: str = _DEFAULT_UA,
) -> list[SnsHandleRecord]:
    """Re-fetch seju.tokyo talent profiles and extract SNS handles."""
    from .sources import read_source_manifest

    candidates = read_source_manifest(manifest_path)
    seen_slugs: dict[str, tuple[str | None, str]] = {}
    for c in candidates:
        if c.talent_slug not in seen_slugs:
            seen_slugs[c.talent_slug] = (c.name, c.profile_url)

    slugs = list(seen_slugs.items())
    if max_profiles is not None:
        slugs = slugs[:max_profiles]

    fetcher = _Fetcher(user_agent=user_agent, delay_seconds=delay_seconds)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    records: list[SnsHandleRecord] = []

    for slug, (name, profile_url) in slugs:
        try:
            html = fetcher.fetch_text(profile_url)
            parser = _LinkParser()
            parser.feed(html)
            handles = extract_sns_handles_from_links(parser.links)
        except Exception as exc:  # noqa: BLE001
            handles = {}
            print(f"  warn: handle scrape failed for {slug}: {exc}")

        records.append(SnsHandleRecord(
            talent_slug=slug,
            name=name,
            profile_url=profile_url,
            sns_handles=handles,
            retrieved_at=retrieved_at,
        ))

    write_handles_manifest(records, out_path)
    return records


def write_handles_manifest(records: Iterable[SnsHandleRecord], out_path: Path) -> None:
    items = list(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(r), ensure_ascii=False) for r in items]
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_handles_manifest(path: Path) -> list[SnsHandleRecord]:
    records: list[SnsHandleRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        records.append(SnsHandleRecord(**data))
    return records


# ─── per-platform engagement fetching ────────────────────────────────────────

def fetch_instagram_engagement(handle: str) -> SnsEngagement:
    """Fetch public Instagram profile data via unofficial JSON endpoint + page fallback."""
    retrieved_at = datetime.now(timezone.utc).isoformat()
    profile_url = f"https://www.instagram.com/{handle}/"
    fetcher = _Fetcher(user_agent=_IG_API_UA, delay_seconds=0.5)

    # Strategy 1: undocumented profile API (often works for public profiles)
    api_url = f"https://www.instagram.com/api/v1/users/web_profile_info/?username={handle}"
    try:
        text = fetcher.fetch_text(api_url, extra_headers={
            "X-IG-App-ID": "936619743392459",
            "Accept": "application/json",
            "Referer": "https://www.instagram.com/",
        })
        data = json.loads(text)
        user = data.get("data", {}).get("user") or data.get("user") or {}
        if user:
            followers = _int_or_none(user.get("edge_followed_by", {}).get("count"))
            following = _int_or_none(user.get("edge_follow", {}).get("count"))
            posts = _int_or_none(user.get("edge_owner_to_timeline_media", {}).get("count"))
            bio = user.get("biography") or None
            display_name = user.get("full_name") or None
            total_eng = _estimate_ig_engagement(user)
            return SnsEngagement(
                platform="instagram", handle=handle, profile_url=profile_url,
                followers=followers, following=following, posts=posts,
                total_engagement=total_eng,
                engagement_rate=_engagement_rate(total_eng, followers, posts),
                bio=bio, display_name=display_name,
                fetch_status="ok", fetch_error=None, retrieved_at=retrieved_at,
            )
    except Exception:  # noqa: BLE001
        pass

    # Strategy 2: parse public page meta tags
    try:
        html = fetcher.fetch_text(profile_url)
        meta = _parse_meta_tags(html)
        desc = meta.get("og:description") or meta.get("description") or ""
        followers = _parse_follower_count_from_text(desc)
        display_name = meta.get("og:title") or None
        return SnsEngagement(
            platform="instagram", handle=handle, profile_url=profile_url,
            followers=followers, following=None, posts=None,
            total_engagement=None, engagement_rate=None,
            bio=None, display_name=display_name,
            fetch_status="partial" if followers else "blocked",
            fetch_error=None, retrieved_at=retrieved_at,
        )
    except urllib.error.HTTPError as exc:
        status = "not_found" if exc.code == 404 else "blocked"
        return _error_engagement("instagram", handle, profile_url, status, str(exc), retrieved_at)
    except Exception as exc:  # noqa: BLE001
        return _error_engagement("instagram", handle, profile_url, "error", str(exc), retrieved_at)


def fetch_twitter_engagement(handle: str) -> SnsEngagement:
    """Fetch public Twitter/X profile data."""
    retrieved_at = datetime.now(timezone.utc).isoformat()
    profile_url = f"https://x.com/{handle}"
    fetcher = _Fetcher(user_agent=_IG_API_UA, delay_seconds=0.5)

    # Try Nitter public instance (lighter on anti-bot)
    nitter_hosts = ["nitter.net", "nitter.privacydev.net", "nitter.poast.org"]
    for host in nitter_hosts:
        nitter_url = f"https://{host}/{handle}"
        try:
            html = fetcher.fetch_text(nitter_url)
            if "User not found" in html or "page not found" in html.lower():
                return _error_engagement("twitter", handle, profile_url, "not_found", "nitter: user not found", retrieved_at)
            followers = _parse_nitter_followers(html)
            stats = _parse_nitter_stats(html)
            meta = _parse_meta_tags(html)
            display_name = meta.get("og:title") or None
            return SnsEngagement(
                platform="twitter", handle=handle, profile_url=profile_url,
                followers=followers, following=stats.get("following"),
                posts=stats.get("tweets"),
                total_engagement=None, engagement_rate=None,
                bio=meta.get("og:description") or None, display_name=display_name,
                fetch_status="ok" if followers else "partial",
                fetch_error=None, retrieved_at=retrieved_at,
            )
        except Exception:  # noqa: BLE001
            continue

    # Fall back: parse x.com directly
    try:
        html = fetcher.fetch_text(profile_url)
        meta = _parse_meta_tags(html)
        desc = meta.get("og:description") or ""
        followers = _parse_follower_count_from_text(desc)
        display_name = meta.get("og:title") or None
        return SnsEngagement(
            platform="twitter", handle=handle, profile_url=profile_url,
            followers=followers, following=None, posts=None,
            total_engagement=None, engagement_rate=None,
            bio=None, display_name=display_name,
            fetch_status="partial" if followers else "blocked",
            fetch_error=None, retrieved_at=retrieved_at,
        )
    except Exception as exc:  # noqa: BLE001
        return _error_engagement("twitter", handle, profile_url, "error", str(exc), retrieved_at)


def fetch_tiktok_engagement(handle: str) -> SnsEngagement:
    """Fetch public TikTok profile data via multiple strategies."""
    retrieved_at = datetime.now(timezone.utc).isoformat()
    profile_url = f"https://www.tiktok.com/@{handle}"
    fetcher = _Fetcher(user_agent=_IG_API_UA, delay_seconds=0.5)

    # Strategy 1: undocumented user detail API (no auth, sometimes works)
    api_url = (
        f"https://www.tiktok.com/api/user/detail/?uniqueId={handle}"
        f"&count=0&cursor=0&from_page=user"
    )
    try:
        text = fetcher.fetch_text(api_url, extra_headers={
            "Referer": "https://www.tiktok.com/",
            "Accept": "application/json, text/plain, */*",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
        })
        data = json.loads(text)
        user_info = data.get("userInfo") or {}
        user = user_info.get("user") or {}
        stats = user_info.get("stats") or {}
        if stats and stats.get("followerCount") is not None:
            followers = _int_or_none(stats.get("followerCount"))
            hearts = _int_or_none(stats.get("heartCount") or stats.get("diggCount"))
            return SnsEngagement(
                platform="tiktok", handle=handle, profile_url=profile_url,
                followers=followers, following=_int_or_none(stats.get("followingCount")),
                posts=_int_or_none(stats.get("videoCount")),
                total_engagement=hearts,
                engagement_rate=_engagement_rate(hearts, followers, _int_or_none(stats.get("videoCount"))),
                bio=user.get("signature") or None,
                display_name=user.get("nickname") or None,
                fetch_status="ok", fetch_error=None, retrieved_at=retrieved_at,
            )
    except Exception:  # noqa: BLE001
        pass

    # Strategy 2: HTML page with SIGI_STATE / UNIVERSAL_DATA
    try:
        html = fetcher.fetch_text(profile_url)

        m = _SIGI_STATE_RE.search(html)
        if m:
            data = json.loads(m.group(1))
            user_info = (
                data.get("UserPage", {}).get("userInfo", {})
                or _deep_find(data, "userInfo")
                or {}
            )
            user = user_info.get("user") or {}
            stats = user_info.get("stats") or {}
            if stats:
                followers = _int_or_none(stats.get("followerCount"))
                hearts = _int_or_none(stats.get("heartCount") or stats.get("diggCount"))
                return SnsEngagement(
                    platform="tiktok", handle=handle, profile_url=profile_url,
                    followers=followers, following=None,
                    posts=_int_or_none(stats.get("videoCount")),
                    total_engagement=hearts,
                    engagement_rate=_engagement_rate(hearts, followers, None),
                    bio=user.get("signature") or None,
                    display_name=user.get("nickname") or None,
                    fetch_status="ok" if followers else "partial",
                    fetch_error=None, retrieved_at=retrieved_at,
                )

        m2 = _UNIVERSAL_DATA_RE.search(html)
        if m2:
            data = json.loads(m2.group(1))
            user_info = _deep_find(data, "userInfo") or {}
            stats = user_info.get("stats") or {}
            user = user_info.get("user") or {}
            if stats and stats.get("followerCount") is not None:
                followers = _int_or_none(stats.get("followerCount"))
                hearts = _int_or_none(stats.get("heartCount") or stats.get("diggCount"))
                return SnsEngagement(
                    platform="tiktok", handle=handle, profile_url=profile_url,
                    followers=followers, following=None,
                    posts=_int_or_none(stats.get("videoCount")),
                    total_engagement=hearts,
                    engagement_rate=_engagement_rate(hearts, followers, None),
                    bio=user.get("signature") or None,
                    display_name=user.get("nickname") or None,
                    fetch_status="ok" if followers else "partial",
                    fetch_error=None, retrieved_at=retrieved_at,
                )

        meta = _parse_meta_tags(html)
        desc = meta.get("og:description") or ""
        followers = _parse_follower_count_from_text(desc)
        return SnsEngagement(
            platform="tiktok", handle=handle, profile_url=profile_url,
            followers=followers, following=None, posts=None,
            total_engagement=None, engagement_rate=None,
            bio=None, display_name=meta.get("og:title") or None,
            fetch_status="partial" if followers else "blocked",
            fetch_error=None, retrieved_at=retrieved_at,
        )
    except urllib.error.HTTPError as exc:
        status = "not_found" if exc.code == 404 else "blocked"
        return _error_engagement("tiktok", handle, profile_url, status, str(exc), retrieved_at)
    except Exception as exc:  # noqa: BLE001
        return _error_engagement("tiktok", handle, profile_url, "error", str(exc), retrieved_at)


def fetch_sns_engagement(platform: str, handle: str) -> SnsEngagement:
    """Dispatch to per-platform fetcher."""
    if platform == "instagram":
        return fetch_instagram_engagement(handle)
    if platform == "twitter":
        return fetch_twitter_engagement(handle)
    if platform == "tiktok":
        return fetch_tiktok_engagement(handle)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    return _error_engagement(platform, handle, f"https://{platform}.com/{handle}",
                              "error", f"unsupported platform: {platform}", retrieved_at)


def fetch_all_talent_engagement(
    handles_path: Path,
    out_path: Path,
    delay_between_talents: float = 2.0,
    platforms: list[str] | None = None,
) -> list[TalentEngagementRecord]:
    """Fetch SNS engagement for all talents in a handles manifest."""
    active_platforms = set(platforms or ["instagram", "twitter", "tiktok"])
    records = read_handles_manifest(handles_path)
    results: list[TalentEngagementRecord] = []

    for record in records:
        engagements: list[SnsEngagement] = []
        for platform, handle in record.sns_handles.items():
            if platform not in active_platforms:
                continue
            print(f"  {record.talent_slug} [{platform}] @{handle}")
            eng = fetch_sns_engagement(platform, handle)
            engagements.append(eng)
            time.sleep(delay_between_talents)

        results.append(TalentEngagementRecord(
            talent_slug=record.talent_slug,
            name=record.name,
            engagements=engagements,
        ))

    write_engagement_manifest(results, out_path)
    return results


def write_engagement_manifest(records: Iterable[TalentEngagementRecord], out_path: Path) -> None:
    items = list(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for item in items:
        lines.append(json.dumps({
            "talent_slug": item.talent_slug,
            "name": item.name,
            "engagements": [asdict(e) for e in item.engagements],
        }, ensure_ascii=False))
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_engagement_manifest(path: Path) -> list[TalentEngagementRecord]:
    records: list[TalentEngagementRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        records.append(TalentEngagementRecord(
            talent_slug=data["talent_slug"],
            name=data.get("name"),
            engagements=[SnsEngagement(**e) for e in data.get("engagements", [])],
        ))
    return records


def import_engagement_csv(
    csv_path: Path,
    out_path: Path,
    existing_path: Path | None = None,
    overwrite_platforms: bool = True,
) -> list[TalentEngagementRecord]:
    """Import SNS engagement data from a user-supplied CSV and write to a JSONL manifest.

    CSV format (UTF-8 with or without BOM):
        talent_slug, platform, handle, followers, following, posts,
        total_engagement, engagement_rate, display_name, bio

    Only talent_slug, platform, and handle are required.
    Missing numeric columns are stored as null.

    If existing_path is given, its records are loaded first; rows from the CSV
    then replace any matching (talent_slug, platform) entry when overwrite_platforms=True,
    or are skipped when False.
    """
    import csv

    retrieved_at = datetime.now(timezone.utc).isoformat()

    # Load existing data keyed by (slug, platform)
    existing: dict[str, TalentEngagementRecord] = {}
    if existing_path and existing_path.exists():
        for rec in read_engagement_manifest(existing_path):
            existing[rec.talent_slug] = rec

    def _csv_int(val: str) -> int | None:
        v = val.strip()
        if not v or v.lower() in ("", "null", "none", "n/a", "-"):
            return None
        # strip commas and K/M suffixes
        v = v.replace(",", "")
        if v[-1].upper() == "K":
            return int(float(v[:-1]) * 1_000)
        if v[-1].upper() == "M":
            return int(float(v[:-1]) * 1_000_000)
        return int(float(v))

    def _csv_float(val: str) -> float | None:
        v = val.strip()
        if not v or v.lower() in ("", "null", "none", "n/a", "-"):
            return None
        v = v.rstrip("%")
        try:
            f = float(v)
            return f / 100 if "%" in val else f
        except ValueError:
            return None

    raw = csv_path.read_bytes()
    text = raw.decode("utf-8-sig")  # strips BOM if present

    reader = csv.DictReader(text.splitlines())
    rows_by_slug: dict[str, list[SnsEngagement]] = {}

    for row in reader:
        slug = row.get("talent_slug", "").strip()
        platform = row.get("platform", "").strip().lower()
        handle = row.get("handle", "").strip().lstrip("@")
        if not slug or not platform or not handle:
            continue

        followers = _csv_int(row.get("followers", ""))
        following = _csv_int(row.get("following", ""))
        posts = _csv_int(row.get("posts", ""))
        total_eng = _csv_int(row.get("total_engagement", ""))
        eng_rate = _csv_float(row.get("engagement_rate", ""))
        display_name = row.get("display_name", "").strip() or None
        bio = row.get("bio", "").strip() or None

        if eng_rate is None and total_eng is not None and followers:
            eng_rate = _engagement_rate(total_eng, followers, posts)

        profile_url = _platform_profile_url(platform, handle)
        eng = SnsEngagement(
            platform=platform, handle=handle, profile_url=profile_url,
            followers=followers, following=following, posts=posts,
            total_engagement=total_eng, engagement_rate=eng_rate,
            bio=bio, display_name=display_name,
            fetch_status="ok" if followers is not None else "partial",
            fetch_error=None, retrieved_at=retrieved_at,
        )
        rows_by_slug.setdefault(slug, []).append(eng)

    # Merge into existing records
    for slug, new_engs in rows_by_slug.items():
        if slug not in existing:
            existing[slug] = TalentEngagementRecord(
                talent_slug=slug, name=None, engagements=[]
            )
        rec = existing[slug]
        for new_eng in new_engs:
            if overwrite_platforms:
                rec.engagements = [
                    e for e in rec.engagements if e.platform != new_eng.platform
                ]
            else:
                if any(e.platform == new_eng.platform for e in rec.engagements):
                    continue
            rec.engagements.append(new_eng)

    result = sorted(existing.values(), key=lambda r: r.talent_slug)
    write_engagement_manifest(result, out_path)
    return result


def _platform_profile_url(platform: str, handle: str) -> str:
    if platform == "instagram":
        return f"https://www.instagram.com/{handle}/"
    if platform in ("twitter", "x"):
        return f"https://x.com/{handle}"
    if platform == "tiktok":
        return f"https://www.tiktok.com/@{handle}"
    return f"https://{platform}.com/{handle}"


# ─── internal helpers ─────────────────────────────────────────────────────────

def _error_engagement(
    platform: str, handle: str, profile_url: str,
    status: str, error: str, retrieved_at: str,
) -> SnsEngagement:
    return SnsEngagement(
        platform=platform, handle=handle, profile_url=profile_url,
        followers=None, following=None, posts=None,
        total_engagement=None, engagement_rate=None,
        bio=None, display_name=None,
        fetch_status=status, fetch_error=error, retrieved_at=retrieved_at,
    )


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _engagement_rate(total_eng: int | None, followers: int | None, posts: int | None) -> float | None:
    if total_eng is None or followers is None or followers == 0:
        return None
    if posts and posts > 0:
        per_post = total_eng / posts
        return round(per_post / followers, 6)
    return round(total_eng / followers, 6)


def _estimate_ig_engagement(user: dict) -> int | None:
    """Estimate total engagement from recent media edge counts."""
    media = user.get("edge_owner_to_timeline_media", {})
    edges = media.get("edges") or []
    if not edges:
        return None
    likes = sum(
        e.get("node", {}).get("edge_liked_by", {}).get("count", 0) for e in edges
    )
    comments = sum(
        e.get("node", {}).get("edge_media_to_comment", {}).get("count", 0) for e in edges
    )
    return likes + comments


def _parse_follower_count_from_text(text: str) -> int | None:
    m = _FOLLOWER_TEXT_RE.search(text)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    multiplier = 1
    if raw[-1].lower() == "k":
        multiplier = 1_000
        raw = raw[:-1]
    elif raw[-1].lower() == "m":
        multiplier = 1_000_000
        raw = raw[:-1]
    elif raw[-1].lower() == "b":
        multiplier = 1_000_000_000
        raw = raw[:-1]
    try:
        return int(float(raw) * multiplier)
    except ValueError:
        return None


def _parse_nitter_followers(html: str) -> int | None:
    m = _NITTER_STATS_RE.search(html)
    if m:
        return _int_or_none(m.group(1).replace(",", ""))
    # fallback: look for Followers in stat items
    pattern = re.compile(r'<span[^>]*class="[^"]*followers[^"]*"[^>]*>([\d,]+)</span>', re.DOTALL)
    m2 = pattern.search(html)
    if m2:
        return _int_or_none(m2.group(1).replace(",", ""))
    return _parse_follower_count_from_text(html)


def _parse_nitter_stats(html: str) -> dict[str, int | None]:
    stats: dict[str, int | None] = {}
    patterns = {
        "tweets": re.compile(r'<span[^>]*class="[^"]*tweets[^"]*"[^>]*>([\d,]+)</span>', re.DOTALL),
        "following": re.compile(r'<span[^>]*class="[^"]*following[^"]*"[^>]*>([\d,]+)</span>', re.DOTALL),
    }
    for key, pat in patterns.items():
        m = pat.search(html)
        if m:
            stats[key] = _int_or_none(m.group(1).replace(",", ""))
    return stats


def _parse_meta_tags(html: str) -> dict[str, str]:
    parser = _MetaParser()
    parser.feed(html)
    return parser.meta


def _deep_find(data: dict | list, key: str) -> object:
    """Recursively search for a key in nested dicts/lists."""
    if isinstance(data, dict):
        if key in data:
            return data[key]
        for v in data.values():
            result = _deep_find(v, key)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = _deep_find(item, key)
            if result is not None:
                return result
    return None


class _Fetcher:
    def __init__(self, user_agent: str, delay_seconds: float) -> None:
        self.user_agent = user_agent
        self.delay_seconds = max(0.0, delay_seconds)
        self._lock = Lock()
        self._next_at = 0.0

    def fetch_text(self, url: str, extra_headers: dict[str, str] | None = None) -> str:
        with self._lock:
            now = time.monotonic()
            if now < self._next_at:
                time.sleep(self._next_at - now)
            self._next_at = time.monotonic() + self.delay_seconds

        headers: dict[str, str] = {"User-Agent": self.user_agent}
        if extra_headers:
            headers.update(extra_headers)
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=20) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            attrs_dict = {k.lower(): v for k, v in attrs if v is not None}
            if "href" in attrs_dict:
                self.links.append((attrs_dict["href"], None))


class _MetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        attrs_dict = {k.lower(): v for k, v in attrs if v is not None}
        key = attrs_dict.get("property") or attrs_dict.get("name")
        content = attrs_dict.get("content")
        if key and content:
            self.meta[key] = content
