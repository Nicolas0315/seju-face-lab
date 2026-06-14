from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from threading import Lock
from typing import Iterable

IMAGE_RE = re.compile(r"\.(?:jpg|jpeg|png|webp)(?:\?|$)", re.IGNORECASE)
BIRTHDATE_RE = re.compile(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日")


@dataclass(frozen=True)
class SourceCandidate:
    profile_url: str
    talent_slug: str
    name: str | None
    birthdate: str | None
    age_as_of: int | None
    image_url: str
    image_kind: str
    alt: str | None
    eligible_for_analysis: bool
    exclusion_reason: str | None
    retrieved_at: str
    source_policy: str


def discover_sources(
    index_url: str,
    out_path: Path,
    as_of: str | None,
    min_age: int,
    include_under_min_age: bool,
    max_profiles: int | None,
    workers: int,
    delay_seconds: float,
    user_agent: str,
) -> list[SourceCandidate]:
    as_of_date = date.fromisoformat(as_of) if as_of else date.today()
    fetcher = _ThrottledFetcher(user_agent=user_agent, delay_seconds=delay_seconds)
    _assert_robots_allowed(index_url, user_agent)
    index_html = fetcher.fetch_text(index_url)
    profile_urls = parse_talent_links(index_html, index_url)
    if max_profiles is not None:
        profile_urls = profile_urls[:max_profiles]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    results: list[SourceCandidate] = []
    if workers <= 1:
        for profile_url in profile_urls:
            results.extend(
                _discover_profile(
                    profile_url,
                    fetcher,
                    as_of_date,
                    min_age,
                    include_under_min_age,
                    retrieved_at,
                    user_agent,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(
                    _discover_profile,
                    profile_url,
                    fetcher,
                    as_of_date,
                    min_age,
                    include_under_min_age,
                    retrieved_at,
                    user_agent,
                ): profile_url
                for profile_url in profile_urls
            }
            for future in as_completed(futures):
                results.extend(future.result())

    return sorted(results, key=lambda item: (item.talent_slug, item.image_url))


def parse_talent_links(html: str, base_url: str) -> list[str]:
    parser = _SourceHTMLParser()
    parser.feed(html)
    seen: set[str] = set()
    urls: list[str] = []
    for href, _text in parser.links:
        url = urllib.parse.urljoin(base_url, href)
        parsed = urllib.parse.urlparse(url)
        if parsed.netloc != urllib.parse.urlparse(base_url).netloc:
            continue
        path = parsed.path.rstrip("/") + "/"
        if not path.startswith("/talents/") or path == "/talents/":
            continue
        normalized = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
        if normalized not in seen:
            seen.add(normalized)
            urls.append(normalized)
    return urls


def parse_profile(html: str, profile_url: str) -> tuple[dict[str, str | None], list[tuple[str, str, str | None]]]:
    parser = _SourceHTMLParser()
    parser.feed(html)

    title = parser.meta.get("og:title") or parser.title
    name = _clean_name(title)
    description = parser.meta.get("description") or parser.meta.get("og:description") or " ".join(parser.text)
    birthdate = _extract_birthdate(description)

    images: list[tuple[str, str, str | None]] = []
    for key in ("og:image", "twitter:image"):
        if key in parser.meta:
            images.append((parser.meta[key], key, None))
    for src, alt in parser.images:
        images.append((src, "img", alt))
    for href, text in parser.links:
        if IMAGE_RE.search(href):
            images.append((href, "link", text))

    deduped: list[tuple[str, str, str | None]] = []
    seen: set[str] = set()
    for raw_url, kind, alt in images:
        image_url = urllib.parse.urljoin(profile_url, _best_srcset_url(raw_url))
        if not _is_profile_image_candidate(image_url):
            continue
        if image_url not in seen:
            seen.add(image_url)
            deduped.append((image_url, kind, alt))

    return {"name": name, "birthdate": birthdate}, deduped


def write_source_manifest(candidates: Iterable[SourceCandidate], out_path: Path) -> None:
    items = list(candidates)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(candidate), ensure_ascii=False) for candidate in items]
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    out_path.with_suffix(".audit.md").write_text(_render_source_audit(items), encoding="utf-8")


def _discover_profile(
    profile_url: str,
    fetcher: _ThrottledFetcher,
    as_of_date: date,
    min_age: int,
    include_under_min_age: bool,
    retrieved_at: str,
    user_agent: str,
) -> list[SourceCandidate]:
    _assert_robots_allowed(profile_url, user_agent)
    html = fetcher.fetch_text(profile_url)
    profile, images = parse_profile(html, profile_url)
    birthdate = profile["birthdate"]
    age = _age_on_date(date.fromisoformat(birthdate), as_of_date) if birthdate else None
    eligible, reason = _eligibility(age, min_age, include_under_min_age)
    slug = urllib.parse.urlparse(profile_url).path.rstrip("/").split("/")[-1]
    return [
        SourceCandidate(
            profile_url=profile_url,
            talent_slug=slug,
            name=profile["name"],
            birthdate=birthdate,
            age_as_of=age,
            image_url=image_url,
            image_kind=kind,
            alt=alt,
            eligible_for_analysis=eligible,
            exclusion_reason=reason,
            retrieved_at=retrieved_at,
            source_policy="manifest_only_review_before_download",
        )
        for image_url, kind, alt in images
    ]


def _eligibility(age: int | None, min_age: int, include_under_min_age: bool) -> tuple[bool, str | None]:
    if age is None:
        return (include_under_min_age, None if include_under_min_age else "age_unknown")
    if age < min_age and not include_under_min_age:
        return (False, f"under_min_age_{min_age}")
    return (True, None)


def _assert_robots_allowed(url: str, user_agent: str) -> None:
    parsed = urllib.parse.urlparse(url)
    robots_url = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "/robots.txt", "", "", ""))
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    parser.read()
    if not parser.can_fetch(user_agent, url):
        raise PermissionError(f"robots.txt disallows fetch: {url}")


def _age_on_date(birthdate: date, as_of_date: date) -> int:
    years = as_of_date.year - birthdate.year
    if (as_of_date.month, as_of_date.day) < (birthdate.month, birthdate.day):
        years -= 1
    return years


def _extract_birthdate(text: str) -> str | None:
    match = BIRTHDATE_RE.search(text)
    if not match:
        return None
    year, month, day = (int(group) for group in match.groups())
    return date(year, month, day).isoformat()


def _clean_name(title: str | None) -> str | None:
    if not title:
        return None
    name = title.split("|", 1)[0].strip()
    return name or None


def _best_srcset_url(value: str) -> str:
    first = value.split(",", 1)[0].strip()
    return first.split(" ", 1)[0].strip()


def _is_profile_image_candidate(url: str) -> bool:
    lowered = url.lower()
    if not IMAGE_RE.search(lowered):
        return False
    if "/wp-content/uploads/" not in lowered:
        return False
    blocked = ("favicon", "logo", "cropped-", "seju_logo")
    return not any(token in lowered for token in blocked)


def _render_source_audit(candidates: list[SourceCandidate]) -> str:
    profiles = sorted({candidate.profile_url for candidate in candidates})
    eligible = sum(1 for candidate in candidates if candidate.eligible_for_analysis)
    excluded = len(candidates) - eligible
    lines = [
        "# seju source discovery audit",
        "",
        f"- candidates: {len(candidates)}",
        f"- eligible_for_analysis: {eligible}",
        f"- excluded_or_review_required: {excluded}",
        f"- profiles: {len(profiles)}",
        "",
        "## Policy",
        "",
        "- Manifest-only discovery; image download is a separate reviewed step.",
        "- robots.txt is checked before fetching the index and profile pages.",
        "- Under-min-age and age-unknown profiles are excluded by default.",
        "- Social platforms are intentionally not scraped.",
        "",
    ]
    return "\n".join(lines)


class _ThrottledFetcher:
    def __init__(self, user_agent: str, delay_seconds: float) -> None:
        self.user_agent = user_agent
        self.delay_seconds = max(0.0, delay_seconds)
        self._lock = Lock()
        self._next_at = 0.0

    def fetch_text(self, url: str) -> str:
        with self._lock:
            now = time.monotonic()
            if now < self._next_at:
                time.sleep(self._next_at - now)
            self._next_at = time.monotonic() + self.delay_seconds

        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(request, timeout=20) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")


class _SourceHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str | None]] = []
        self.images: list[tuple[str, str | None]] = []
        self.meta: dict[str, str] = {}
        self.text: list[str] = []
        self.title: str | None = None
        self._in_title = False
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key.lower(): value for key, value in attrs if value is not None}
        if tag == "title":
            self._in_title = True
        if tag == "a" and "href" in attrs_dict:
            self.links.append((attrs_dict["href"], None))
        if tag == "img":
            alt = attrs_dict.get("alt")
            for key in ("src", "data-src", "data-lazy-src", "srcset", "data-srcset"):
                if key in attrs_dict:
                    self.images.append((attrs_dict[key], alt))
        if tag == "meta":
            key = attrs_dict.get("property") or attrs_dict.get("name")
            content = attrs_dict.get("content")
            if key and content:
                self.meta[key] = content

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            self.title = "".join(self._title_parts).strip()

    def handle_data(self, data: str) -> None:
        value = data.strip()
        if value:
            self.text.append(value)
        if self._in_title:
            self._title_parts.append(data)
