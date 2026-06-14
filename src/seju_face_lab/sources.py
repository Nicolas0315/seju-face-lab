from __future__ import annotations

import json
import hashlib
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
from typing import Callable, Iterable

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


@dataclass(frozen=True)
class DownloadResult:
    profile_url: str
    talent_slug: str
    image_url: str
    status: str
    path: str | None
    sha256: str | None
    bytes: int
    reason: str | None


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
    text_candidates = [
        parser.meta.get("description"),
        parser.meta.get("og:description"),
        " ".join(parser.text),
    ]
    birthdate = _extract_first_birthdate(value for value in text_candidates if value)

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


def read_source_manifest(path: Path) -> list[SourceCandidate]:
    candidates: list[SourceCandidate] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        candidates.append(SourceCandidate(**data))
    return candidates


def download_source_images(
    candidates: Iterable[SourceCandidate],
    out_dir: Path,
    max_count: int | None = None,
    dry_run: bool = False,
    include_ineligible: bool = False,
    delay_seconds: float = 0.5,
    max_bytes: int = 20_000_000,
    user_agent: str = "seju-face-lab/0.1 (+local research; contact: local)",
    fetch_bytes: Callable[[str], tuple[bytes, str | None]] | None = None,
    check_robots: Callable[[str, str], None] | None = None,
) -> list[DownloadResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = _select_download_candidates(candidates, include_ineligible=include_ineligible)
    if max_count is not None:
        selected = selected[:max_count]

    fetcher = _ThrottledFetcher(user_agent=user_agent, delay_seconds=delay_seconds)
    robots_cache = _RobotsPolicyCache()
    robots_check = check_robots or robots_cache.assert_allowed
    results: list[DownloadResult] = []
    for candidate in selected:
        target = out_dir / _download_filename(candidate)
        if dry_run:
            results.append(
                DownloadResult(
                    profile_url=candidate.profile_url,
                    talent_slug=candidate.talent_slug,
                    image_url=candidate.image_url,
                    status="planned",
                    path=str(target),
                    sha256=None,
                    bytes=0,
                    reason=None,
                )
            )
            continue
        existing_target = _existing_download_path(out_dir, candidate, target)
        if existing_target is not None:
            results.append(
                DownloadResult(
                    profile_url=candidate.profile_url,
                    talent_slug=candidate.talent_slug,
                    image_url=candidate.image_url,
                    status="skipped",
                    path=str(existing_target),
                    sha256=_sha256_file(existing_target),
                    bytes=existing_target.stat().st_size,
                    reason="exists",
                )
            )
            continue
        try:
            robots_check(candidate.image_url, user_agent)
            if fetch_bytes is None:
                payload, content_type = fetcher.fetch_bytes(candidate.image_url, max_bytes=max_bytes)
            else:
                payload, content_type = fetch_bytes(candidate.image_url)
            if len(payload) > max_bytes:
                raise ValueError(f"image exceeds max bytes: {len(payload)} > {max_bytes}")
            if not _is_supported_content_type(content_type, candidate.image_url):
                raise ValueError(f"unsupported content type: {content_type}")
            target.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            results.append(
                DownloadResult(
                    profile_url=candidate.profile_url,
                    talent_slug=candidate.talent_slug,
                    image_url=candidate.image_url,
                    status="downloaded",
                    path=str(target),
                    sha256=digest,
                    bytes=len(payload),
                    reason=None,
                )
            )
        except Exception as exc:  # noqa: BLE001 - report per-image failures and keep batch going.
            results.append(
                DownloadResult(
                    profile_url=candidate.profile_url,
                    talent_slug=candidate.talent_slug,
                    image_url=candidate.image_url,
                    status="failed",
                    path=str(target),
                    sha256=None,
                    bytes=0,
                    reason=str(exc),
                )
            )

    if not dry_run:
        _write_download_manifest(results, out_dir)
    return results


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
    _RobotsPolicyCache().assert_allowed(url, user_agent)


class _RobotsPolicyCache:
    def __init__(self) -> None:
        self._parsers: dict[tuple[str, str], urllib.robotparser.RobotFileParser] = {}

    def assert_allowed(self, url: str, user_agent: str) -> None:
        parsed = urllib.parse.urlparse(url)
        key = (parsed.scheme, parsed.netloc)
        parser = self._parsers.get(key)
        if parser is None:
            parser = self._read_parser(parsed)
            self._parsers[key] = parser
        if not parser.can_fetch(user_agent, url):
            raise PermissionError(f"robots.txt disallows fetch: {url}")

    @staticmethod
    def _read_parser(parsed: urllib.parse.ParseResult) -> urllib.robotparser.RobotFileParser:
        robots_url = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "/robots.txt", "", "", ""))
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        parser.read()
        return parser

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


def _extract_first_birthdate(texts: Iterable[str]) -> str | None:
    for text in texts:
        birthdate = _extract_birthdate(text)
        if birthdate:
            return birthdate
    return None


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

        request = urllib.request.Request(_quote_url(url), headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(request, timeout=20) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")

    def fetch_bytes(self, url: str, max_bytes: int) -> tuple[bytes, str | None]:
        with self._lock:
            now = time.monotonic()
            if now < self._next_at:
                time.sleep(self._next_at - now)
            self._next_at = time.monotonic() + self.delay_seconds

        request = urllib.request.Request(_quote_url(url), headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(request, timeout=30) as response:
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"image exceeds max bytes: {total} > {max_bytes}")
                chunks.append(chunk)
            return b"".join(chunks), response.headers.get("Content-Type")


def _select_download_candidates(
    candidates: Iterable[SourceCandidate],
    include_ineligible: bool,
) -> list[SourceCandidate]:
    selected: list[SourceCandidate] = []
    seen_urls: set[str] = set()
    for candidate in candidates:
        if not candidate.eligible_for_analysis and not include_ineligible:
            continue
        if candidate.image_url in seen_urls:
            continue
        seen_urls.add(candidate.image_url)
        selected.append(candidate)
    return selected


def _download_filename(candidate: SourceCandidate) -> str:
    extension = _extension_from_url(candidate.image_url)
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", candidate.talent_slug).strip("-") or "talent"
    digest = hashlib.sha1(candidate.image_url.encode("utf-8")).hexdigest()[:10]
    return f"{slug}_{digest}{extension}"


def _existing_download_path(out_dir: Path, candidate: SourceCandidate, target: Path) -> Path | None:
    if target.exists():
        return target
    extension = _extension_from_url(candidate.image_url)
    digest = hashlib.sha1(candidate.image_url.encode("utf-8")).hexdigest()[:10]
    legacy_matches = sorted(out_dir.glob(f"*_{digest}{extension}"))
    return legacy_matches[0] if legacy_matches else None


def _extension_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path.lower()
    for extension in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        if path.endswith(extension):
            return extension
    return ".img"


def _is_supported_content_type(content_type: str | None, url: str) -> bool:
    if content_type is None or not content_type.strip():
        return IMAGE_RE.search(url) is not None
    return content_type.split(";", 1)[0].strip().lower().startswith("image/")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_download_manifest(results: Iterable[DownloadResult], out_dir: Path) -> None:
    items = list(results)
    lines = [json.dumps(asdict(result), ensure_ascii=False) for result in items]
    (out_dir / "download_manifest.jsonl").write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )
    summary = {
        "downloaded": sum(1 for result in items if result.status == "downloaded"),
        "planned": sum(1 for result in items if result.status == "planned"),
        "skipped": sum(1 for result in items if result.status == "skipped"),
        "failed": sum(1 for result in items if result.status == "failed"),
        "boundary": "Local reviewed image staging only; do not commit raw images.",
    }
    (out_dir / "download_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _quote_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    path = urllib.parse.quote(parsed.path, safe="/%")
    query = urllib.parse.quote_plus(parsed.query, safe="=&%")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, query, parsed.fragment))


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
