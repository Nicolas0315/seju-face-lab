"""Enhanced SNS exploration engine for seju-face-lab.

Architecture:
  SnsRouter — tries multiple backends per platform, falls back gracefully
  RemoteInstagramFetcher — optionally runs Instagram fetch on a trusted SSH host
  PostEngagementFetcher — fetches recent posts to compute real engagement rates
  SnsStore — SQLite cache to avoid redundant fetches (TTL-aware)
  SnsDiscovery — keyword/name search to discover new handles

Supported data sources:
  Instagram: optional SSH remote → requests.Session → manual
  Twitter: FxTwitter API → Nitter → manual
  TikTok: kalodata creator search → oEmbed → manual CSV

Usage:
    from seju_face_lab.sns_explorer import SnsRouter, SnsStore
    router = SnsRouter(store=SnsStore("data/processed/sns_cache.db"))
    profile = router.fetch("instagram", "kasumi_mori")
    print(profile.followers, profile.engagement_rate)
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# ─── Data models ─────────────────────────────────────────────────────────────

@dataclass
class SnsProfile:
    platform: str
    handle: str
    profile_url: str
    followers: int | None = None
    following: int | None = None
    posts: int | None = None
    avg_likes: float | None = None          # average likes on recent posts
    avg_comments: float | None = None       # average comments on recent posts
    total_engagement: int | None = None     # sum of likes+comments on recent posts
    engagement_rate: float | None = None    # (avg_likes+avg_comments)/followers
    bio: str | None = None
    display_name: str | None = None
    source: str = "unknown"                 # which backend returned this data
    fetch_status: str = "ok"               # ok|partial|blocked|not_found|error
    fetch_error: str | None = None
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_engagement_dict(self) -> dict:
        """Convert to SnsEngagement-compatible dict for legacy manifests."""
        return {
            "platform": self.platform,
            "handle": self.handle,
            "profile_url": self.profile_url,
            "followers": self.followers,
            "following": self.following,
            "posts": self.posts,
            "total_engagement": self.total_engagement,
            "engagement_rate": self.engagement_rate,
            "bio": self.bio,
            "display_name": self.display_name,
            "fetch_status": self.fetch_status,
            "fetch_error": self.fetch_error,
            "retrieved_at": self.retrieved_at,
        }


# ─── SQLite cache (TTL-aware) ─────────────────────────────────────────────────

class SnsStore:
    """Persistent SQLite cache for SNS profiles. TTL prevents stale data."""

    def __init__(self, db_path: Path | str, ttl_hours: float = 168.0) -> None:
        self.db_path = Path(db_path)
        self.ttl_seconds = ttl_hours * 3600
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sns_profiles (
                platform TEXT NOT NULL,
                handle TEXT NOT NULL,
                data_json TEXT NOT NULL,
                fetched_at REAL NOT NULL,
                PRIMARY KEY (platform, handle)
            )
        """)
        self._conn.commit()

    def get(self, platform: str, handle: str) -> SnsProfile | None:
        cur = self._conn.execute(
            "SELECT data_json, fetched_at FROM sns_profiles WHERE platform=? AND handle=?",
            (platform, handle.lower()),
        )
        row = cur.fetchone()
        if not row:
            return None
        data_json, fetched_at = row
        if (time.time() - fetched_at) > self.ttl_seconds:
            return None
        data = json.loads(data_json)
        return SnsProfile(**data)

    def put(self, profile: SnsProfile) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO sns_profiles (platform, handle, data_json, fetched_at) VALUES (?,?,?,?)",
            (profile.platform, profile.handle.lower(), json.dumps(asdict(profile)), time.time()),
        )
        self._conn.commit()

    def invalidate(self, platform: str, handle: str) -> None:
        self._conn.execute(
            "DELETE FROM sns_profiles WHERE platform=? AND handle=?",
            (platform, handle.lower()),
        )
        self._conn.commit()

    def all_profiles(self) -> list[SnsProfile]:
        cur = self._conn.execute("SELECT data_json FROM sns_profiles")
        return [SnsProfile(**json.loads(r[0])) for r in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()


# ─── Remote (SSH) Instagram fetcher ──────────────────────────────────────────

_REMOTE_IG_SCRIPT = """
import json, sys, time, traceback
import requests
handles = json.loads(sys.argv[1])
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "X-IG-App-ID": "936619743392459",
    "Accept": "application/json",
    "Referer": "https://www.instagram.com/",
    "Accept-Language": "ja-JP,ja;q=0.9",
})
try: session.get("https://www.instagram.com/", timeout=10)
except: pass
results = {}
for handle in handles:
    try:
        r = session.get(f"https://www.instagram.com/api/v1/users/web_profile_info/?username={handle}", timeout=15)
        if r.status_code == 200:
            u = r.json().get("data", {}).get("user") or {}
            fol = (u.get("edge_followed_by") or {}).get("count")
            following = (u.get("edge_follow") or {}).get("count")
            posts_count = (u.get("edge_owner_to_timeline_media") or {}).get("count")
            media_edges = (u.get("edge_owner_to_timeline_media") or {}).get("edges") or []
            likes = sum(e.get("node", {}).get("edge_liked_by", {}).get("count", 0) for e in media_edges)
            comments = sum(e.get("node", {}).get("edge_media_to_comment", {}).get("count", 0) for e in media_edges)
            n_posts = len(media_edges)
            avg_likes = round(likes / n_posts, 1) if n_posts else None
            avg_comments = round(comments / n_posts, 1) if n_posts else None
            eng = (likes + comments) if n_posts else None
            er = round((likes + comments) / n_posts / fol, 6) if (n_posts and fol) else None
            results[handle] = {"followers": fol, "following": following, "posts": posts_count,
                "avg_likes": avg_likes, "avg_comments": avg_comments,
                "total_engagement": eng, "engagement_rate": er,
                "display_name": u.get("full_name"), "bio": u.get("biography"),
                "status": "ok" if fol is not None else "partial"}
        else:
            results[handle] = {"followers": None, "status": f"http_{r.status_code}"}
    except Exception as e:
        results[handle] = {"followers": None, "status": "error", "error": str(e)[:120]}
    time.sleep(1.2)
print(json.dumps(results))
"""


class RemoteInstagramFetcher:
    """Fetch Instagram profiles via SSH to an explicitly configured trusted host."""

    def __init__(
        self,
        ssh_host: str,
        python_bin: str = "python3",
        timeout: float = 120.0,
    ) -> None:
        self.ssh_host = ssh_host
        self.python_bin = python_bin
        self.timeout = timeout

    def available(self) -> bool:
        try:
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", self.ssh_host, "echo ok"],
                capture_output=True, text=True, timeout=8,
            )
            return result.returncode == 0 and "ok" in result.stdout
        except Exception:
            return False

    def fetch_batch(self, handles: list[str]) -> dict[str, dict]:
        """Fetch multiple IG handles in one SSH round-trip. Returns {handle: {...}}."""
        handles_json = json.dumps(handles)
        script = _REMOTE_IG_SCRIPT.strip()
        cmd = ["ssh", "-o", "ConnectTimeout=10", self.ssh_host,
               f"{self.python_bin} - {handles_json!r}"]
        try:
            result = subprocess.run(
                cmd,
                input=script,
                capture_output=True, text=True, timeout=self.timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr[:200])
            # find last JSON line in stdout
            for line in reversed(result.stdout.splitlines()):
                line = line.strip()
                if line.startswith("{"):
                    return json.loads(line)
            raise ValueError("no JSON in output")
        except Exception as exc:
            return {h: {"followers": None, "status": f"ssh_error: {exc}"} for h in handles}

    def fetch(self, handle: str) -> SnsProfile:
        results = self.fetch_batch([handle])
        data = results.get(handle, {})
        fol = data.get("followers")
        return SnsProfile(
            platform="instagram", handle=handle,
            profile_url=f"https://www.instagram.com/{handle}/",
            followers=fol,
            following=data.get("following"),
            posts=data.get("posts"),
            avg_likes=data.get("avg_likes"),
            avg_comments=data.get("avg_comments"),
            total_engagement=data.get("total_engagement"),
            engagement_rate=data.get("engagement_rate"),
            bio=data.get("bio"),
            display_name=data.get("display_name"),
            source=f"remote_ig:{self.ssh_host}",
            fetch_status="ok" if fol is not None else data.get("status", "error"),
            fetch_error=data.get("error"),
        )


# ─── Platform-specific fetchers ──────────────────────────────────────────────

_IG_API_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _fetch_instagram_local(handle: str) -> SnsProfile:
    """Fetch Instagram via local requests.Session (works on trusted IPs)."""
    try:
        import requests as _requests
    except ImportError:
        return SnsProfile(
            platform="instagram", handle=handle,
            profile_url=f"https://www.instagram.com/{handle}/",
            fetch_status="error", fetch_error="requests not installed",
            source="local_ig",
        )
    s = _requests.Session()
    s.headers.update({
        "User-Agent": _IG_API_UA,
        "X-IG-App-ID": "936619743392459",
        "Accept": "application/json",
        "Referer": "https://www.instagram.com/",
        "Accept-Language": "ja-JP,ja;q=0.9",
    })
    try:
        s.get("https://www.instagram.com/", timeout=10)
    except Exception:
        pass
    try:
        r = s.get(
            f"https://www.instagram.com/api/v1/users/web_profile_info/?username={handle}",
            timeout=15,
        )
        if r.status_code == 404:
            return SnsProfile(platform="instagram", handle=handle,
                              profile_url=f"https://www.instagram.com/{handle}/",
                              fetch_status="not_found", source="local_ig")
        if r.status_code != 200:
            return SnsProfile(platform="instagram", handle=handle,
                              profile_url=f"https://www.instagram.com/{handle}/",
                              fetch_status="blocked", fetch_error=f"HTTP {r.status_code}", source="local_ig")
        u = r.json().get("data", {}).get("user") or {}
        fol = (u.get("edge_followed_by") or {}).get("count")
        media_edges = (u.get("edge_owner_to_timeline_media") or {}).get("edges") or []
        n = len(media_edges)
        likes = sum(e.get("node", {}).get("edge_liked_by", {}).get("count", 0) for e in media_edges)
        comments = sum(e.get("node", {}).get("edge_media_to_comment", {}).get("count", 0) for e in media_edges)
        return SnsProfile(
            platform="instagram", handle=handle,
            profile_url=f"https://www.instagram.com/{handle}/",
            followers=fol,
            following=(u.get("edge_follow") or {}).get("count"),
            posts=(u.get("edge_owner_to_timeline_media") or {}).get("count"),
            avg_likes=round(likes / n, 1) if n else None,
            avg_comments=round(comments / n, 1) if n else None,
            total_engagement=likes + comments if n else None,
            engagement_rate=round((likes + comments) / n / fol, 6) if (n and fol) else None,
            bio=u.get("biography"),
            display_name=u.get("full_name"),
            source="local_ig",
            fetch_status="ok" if fol is not None else "partial",
        )
    except Exception as exc:
        return SnsProfile(platform="instagram", handle=handle,
                          profile_url=f"https://www.instagram.com/{handle}/",
                          fetch_status="blocked", fetch_error=str(exc)[:200], source="local_ig")


def _fetch_twitter_fxtwitter(handle: str) -> SnsProfile:
    """Fetch Twitter/X profile via FxTwitter public API (no auth)."""
    try:
        req = urllib.request.Request(
            f"https://api.fxtwitter.com/{handle}",
            headers={"User-Agent": "seju-face-lab/0.2"},
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode())
        u = data.get("user") or {}
        if not u:
            return SnsProfile(platform="twitter", handle=handle,
                              profile_url=f"https://x.com/{handle}",
                              fetch_status="not_found", source="fxtwitter")
        fol = u.get("followers")
        return SnsProfile(
            platform="twitter", handle=handle,
            profile_url=f"https://x.com/{handle}",
            followers=fol,
            following=u.get("following"),
            posts=u.get("tweets"),
            bio=u.get("description"),
            display_name=u.get("name"),
            source="fxtwitter",
            fetch_status="ok" if fol is not None else "partial",
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return SnsProfile(platform="twitter", handle=handle,
                              profile_url=f"https://x.com/{handle}",
                              fetch_status="not_found", source="fxtwitter")
        return SnsProfile(platform="twitter", handle=handle,
                          profile_url=f"https://x.com/{handle}",
                          fetch_status="blocked", fetch_error=f"HTTP {exc.code}", source="fxtwitter")
    except Exception as exc:
        return SnsProfile(platform="twitter", handle=handle,
                          profile_url=f"https://x.com/{handle}",
                          fetch_status="error", fetch_error=str(exc)[:200], source="fxtwitter")


def _fetch_tiktok_oembed(handle: str) -> SnsProfile:
    """Attempt TikTok profile via oEmbed (limited — no followers count, but display name/avatar)."""
    try:
        oembed_url = f"https://www.tiktok.com/oembed?url=https://www.tiktok.com/@{handle}"
        req = urllib.request.Request(oembed_url, headers={"User-Agent": "seju-face-lab/0.2"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode())
        name = data.get("author_name") or data.get("title")
        return SnsProfile(
            platform="tiktok", handle=handle,
            profile_url=f"https://www.tiktok.com/@{handle}",
            followers=None,
            display_name=name,
            source="tiktok_oembed",
            fetch_status="partial",  # oEmbed doesn't include follower count
            fetch_error="oEmbed: no follower count available",
        )
    except Exception as exc:
        return SnsProfile(platform="tiktok", handle=handle,
                          profile_url=f"https://www.tiktok.com/@{handle}",
                          fetch_status="blocked", fetch_error=str(exc)[:200], source="tiktok_oembed")


def _fetch_tiktok_kalodata_search(handle: str) -> SnsProfile | None:
    """Search kalodata.com creator list for a matching handle.

    kalodata serves TikTok Shop JP market analytics. Works only if the creator
    is registered on TikTok Shop JP. Requires a running authenticated kalodata
    session (patchright) on the local machine — falls back to None if unavailable.
    """
    # Kalodata requires a patchright browser session — only available when
    # kalodata-scraper is running locally. Check for the IPC socket or skip.
    kalodata_ipc = Path.home() / ".kalodata" / "session.json"
    if not kalodata_ipc.exists():
        return None
    try:
        session_data = json.loads(kalodata_ipc.read_text(encoding="utf-8"))
        # session_data should contain {"cookies": [...], "base_url": "..."}
        cookies = {c["name"]: c["value"] for c in session_data.get("cookies", [])}
        if not cookies:
            return None
        import requests as _requests
        s = _requests.Session()
        s.headers.update({"User-Agent": _IG_API_UA, "Referer": "https://kalodata.com/"})
        s.cookies.update(cookies)
        payload = {"handle": handle, "market": "JP", "page": 1, "pageSize": 5}
        r = s.post("https://www.kalodata.com/api/creator/search", json=payload, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        items = data.get("data", {}).get("items") or []
        for item in items:
            if (item.get("handle") or "").lower() == handle.lower():
                fol = item.get("followers_raw") or item.get("followers")
                return SnsProfile(
                    platform="tiktok", handle=handle,
                    profile_url=f"https://www.tiktok.com/@{handle}",
                    followers=int(fol) if fol else None,
                    display_name=item.get("nickname"),
                    engagement_rate=float(str(item.get("video_engagement_rate", "0")).rstrip("%")) / 100
                    if item.get("video_engagement_rate") else None,
                    source="kalodata",
                    fetch_status="ok" if fol else "partial",
                )
    except Exception:  # noqa: BLE001
        pass
    return None


# ─── Main router ──────────────────────────────────────────────────────────────

class SnsRouter:
    """Multi-source SNS profile router with fallback chain and cache.

    Strategy per platform:
      instagram: optional_remote > local_requests > partial
      twitter: fxtwitter > nitter > partial
      tiktok: kalodata_search > oembed (partial, no followers) > partial

    Args:
        store: optional SnsStore for caching (TTL 7 days default)
        remote_host: optional SSH host for trusted-IP Instagram fetches
        delay: seconds between requests per platform (polite crawl)
    """

    def __init__(
        self,
        store: SnsStore | None = None,
        remote_host: str | None = None,
        delay: float = 1.0,
    ) -> None:
        self.store = store
        self._remote_ig = RemoteInstagramFetcher(ssh_host=remote_host) if remote_host else None
        self._remote_available: bool | None = None  # lazy probe
        self.delay = delay

    def _remote_ok(self) -> bool:
        if self._remote_ig is None:
            return False
        if self._remote_available is None:
            self._remote_available = self._remote_ig.available()
        return self._remote_available

    def fetch(self, platform: str, handle: str, force: bool = False) -> SnsProfile:
        """Fetch a single profile. Uses cache unless force=True."""
        if self.store and not force:
            cached = self.store.get(platform, handle)
            if cached and cached.fetch_status in ("ok", "partial"):
                return cached

        profile = self._fetch_live(platform, handle)

        if self.store:
            self.store.put(profile)
        return profile

    def _fetch_live(self, platform: str, handle: str) -> SnsProfile:
        if platform == "instagram":
            if self._remote_ok():
                assert self._remote_ig is not None
                p = self._remote_ig.fetch(handle)
                if p.fetch_status in ("ok", "partial") and p.followers is not None:
                    return p
            time.sleep(self.delay)
            p = _fetch_instagram_local(handle)
            return p

        if platform == "twitter":
            p = _fetch_twitter_fxtwitter(handle)
            if p.fetch_status in ("ok",) and p.followers is not None:
                return p
            time.sleep(self.delay)
            # Nitter fallback (reuse existing sns_metrics)
            try:
                from .sns_metrics import fetch_twitter_engagement
                legacy = fetch_twitter_engagement(handle)
                if legacy.followers is not None:
                    return SnsProfile(
                        platform="twitter", handle=handle,
                        profile_url=legacy.profile_url,
                        followers=legacy.followers,
                        following=legacy.following,
                        posts=legacy.posts,
                        bio=legacy.bio, display_name=legacy.display_name,
                        source="nitter_fallback",
                        fetch_status=legacy.fetch_status,
                    )
            except Exception:  # noqa: BLE001
                pass
            return p

        if platform == "tiktok":
            kalo = _fetch_tiktok_kalodata_search(handle)
            if kalo is not None:
                return kalo
            return _fetch_tiktok_oembed(handle)

        return SnsProfile(
            platform=platform, handle=handle,
            profile_url=f"https://{platform}.com/{handle}",
            fetch_status="error", fetch_error=f"unsupported platform: {platform}",
            source="none",
        )

    def fetch_batch(
        self,
        items: Iterable[tuple[str, str]],  # (platform, handle)
        delay_between: float = 1.5,
        force: bool = False,
    ) -> list[SnsProfile]:
        """Fetch a list of (platform, handle) pairs."""
        items_list = list(items)

        # Batch only uncached Instagram handles via an explicitly configured SSH host.
        ig_handles = [
            handle
            for platform, handle in items_list
            if platform == "instagram" and (force or not self._has_cached_profile(platform, handle))
        ]
        ig_results: dict[str, dict] = {}
        if ig_handles and self._remote_ok():
            assert self._remote_ig is not None
            print(f"  [remote_ig] fetching {len(ig_handles)} handles via {self._remote_ig.ssh_host}...")
            ig_results = self._remote_ig.fetch_batch(ig_handles)

        profiles: list[SnsProfile] = []
        for platform, handle in items_list:
            if self.store and not force:
                cached = self.store.get(platform, handle)
                if cached and cached.fetch_status in ("ok", "partial") and cached.followers is not None:
                    profiles.append(cached)
                    continue

            if platform == "instagram" and handle in ig_results:
                data = ig_results[handle]
                fol = data.get("followers")
                if fol is not None:
                    p = SnsProfile(
                        platform="instagram", handle=handle,
                        profile_url=f"https://www.instagram.com/{handle}/",
                        followers=fol, following=data.get("following"),
                        posts=data.get("posts"),
                        avg_likes=data.get("avg_likes"),
                        avg_comments=data.get("avg_comments"),
                        total_engagement=data.get("total_engagement"),
                        engagement_rate=data.get("engagement_rate"),
                        bio=data.get("bio"), display_name=data.get("display_name"),
                        source=f"remote_ig:{self._remote_ig.ssh_host}" if self._remote_ig else "remote_ig",
                        fetch_status="ok",
                        fetch_error=data.get("error"),
                    )
                else:
                    p = _fetch_instagram_local(handle)
            else:
                p = self._fetch_live(platform, handle)
                time.sleep(delay_between)

            if self.store:
                self.store.put(p)
            profiles.append(p)

        return profiles

    def _has_cached_profile(self, platform: str, handle: str) -> bool:
        if self.store is None:
            return False
        cached = self.store.get(platform, handle)
        return bool(cached and cached.fetch_status in ("ok", "partial") and cached.followers is not None)


# ─── Discovery: find new handles by name/keyword ─────────────────────────────

@dataclass
class DiscoveredHandle:
    platform: str
    handle: str
    display_name: str | None
    followers: int | None
    relevance_score: float  # higher = more likely to be the right person
    source: str


def discover_instagram_by_name(name: str, max_results: int = 5) -> list[DiscoveredHandle]:
    """Search Instagram for profiles matching a name (uses undocumented search endpoint)."""
    try:
        import requests as _requests
        s = _requests.Session()
        s.headers.update({
            "User-Agent": _IG_API_UA,
            "X-IG-App-ID": "936619743392459",
            "Accept": "application/json",
            "Referer": "https://www.instagram.com/",
        })
        s.get("https://www.instagram.com/", timeout=10)
        r = s.get(
            f"https://www.instagram.com/api/v1/users/search/?query={urllib.request.quote(name)}&count={max_results}",
            timeout=15,
        )
        if r.status_code != 200:
            return []
        users = r.json().get("users") or []
        results = []
        for u in users[:max_results]:
            user_data = u.get("user") or u
            handle = user_data.get("username", "")
            display = user_data.get("full_name", "")
            fol = user_data.get("follower_count")
            name_lower = name.lower().replace(" ", "")
            display_lower = display.lower().replace(" ", "")
            handle_lower = handle.lower()
            # simple relevance: name overlap
            score = 0.0
            if name_lower in display_lower or display_lower in name_lower:
                score += 0.8
            elif name_lower[:3] in handle_lower:
                score += 0.3
            results.append(DiscoveredHandle(
                platform="instagram", handle=handle,
                display_name=display, followers=fol,
                relevance_score=score, source="ig_search",
            ))
        return sorted(results, key=lambda x: -x.relevance_score)
    except Exception:  # noqa: BLE001
        return []


def discover_twitter_by_name(name: str, max_results: int = 5) -> list[DiscoveredHandle]:
    """Search FxTwitter for users matching a name via nitter search."""
    # FxTwitter doesn't support user search; use Nitter search
    try:
        url = f"https://nitter.net/search?f=users&q={urllib.request.quote(name)}"
        req = urllib.request.Request(url, headers={"User-Agent": "seju-face-lab/0.2"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        import re
        pattern = re.compile(r'href="/([A-Za-z0-9_]{1,15})"[^>]*>@\1</a>.*?class="fullname">([^<]+)</a>', re.DOTALL)
        found = []
        for m in pattern.finditer(html):
            handle, display = m.group(1), m.group(2).strip()
            if handle.lower() in ("search", "login", "about"):
                continue
            found.append(DiscoveredHandle(
                platform="twitter", handle=handle,
                display_name=display, followers=None,
                relevance_score=0.5, source="nitter_search",
            ))
            if len(found) >= max_results:
                break
        return found
    except Exception:  # noqa: BLE001
        return []


def discover_handles_for_talent(
    talent_name: str,
    platforms: list[str] | None = None,
) -> dict[str, list[DiscoveredHandle]]:
    """Discover SNS handles for a talent name across platforms."""
    platforms = platforms or ["instagram", "twitter"]
    results: dict[str, list[DiscoveredHandle]] = {}
    if "instagram" in platforms:
        results["instagram"] = discover_instagram_by_name(talent_name)
    if "twitter" in platforms:
        results["twitter"] = discover_twitter_by_name(talent_name)
    return results


# ─── CLI integration helpers ──────────────────────────────────────────────────

def build_router(
    cache_path: Path | str | None = None,
    remote_host: str | None = None,
    ttl_hours: float = 168.0,
) -> SnsRouter:
    """Build a SnsRouter with optional cache."""
    store = SnsStore(Path(cache_path), ttl_hours=ttl_hours) if cache_path else None
    return SnsRouter(store=store, remote_host=remote_host)
