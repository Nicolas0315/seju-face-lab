# Web Source Strategy

Retrieval date: 2026-06-14.

## Sources

- Primary source: `https://seju.tokyo/talents/`
- Profile pages: `https://seju.tokyo/talents/{slug}/`
- Robots: `https://seju.tokyo/robots.txt`

Observed facts:

- `robots.txt` has `User-agent: *` and an empty `Disallow`, plus a sitemap.
- The talent index exposes profile links under `/talents/{slug}/`.
- Profile pages expose name, birthday when available, Open Graph image, and page image tags.
- Social platforms are not used as automatic sources because they add login, rate, and platform terms risk.

## Extraction Logic

1. Fetch `robots.txt` and block if the requested URL is disallowed.
2. Fetch the official talent index.
3. Extract only same-host profile URLs matching `/talents/{slug}/`.
4. Fetch profile pages with throttling and bounded worker concurrency.
5. Extract:
   - name from `og:title` or `<title>`
   - birth date from profile/description text
   - candidate images from `og:image`, `twitter:image`, `<img>`, and image links
6. Keep only `wp-content/uploads` image URLs and reject logos/favicons/theme assets.
7. Write a JSONL manifest. Do not download images in the discovery step.
8. Mark `eligible_for_analysis=false` when age is unknown or below `--min-age` unless explicitly overridden.
9. Download only after manifest review, with `sources download`; raw images remain Git-ignored.

## Current Full Discovery Result

Command:

```powershell
python -m seju_face_lab sources discover --out data/processed/seju_sources_official_2026-06-14.jsonl --as-of 2026-06-14 --workers 4 --delay-seconds 0.5
```

Result:

- profiles: 35
- image URL candidates: 265
- eligible candidates with default adult/known-age gate: 259
- excluded/review-required: 6, all `age_unknown`

The generated manifest is ignored by Git because it is derived data and may contain portrait-image URLs that should be reviewed before use.

## Parallelization

The crawler uses a `ThreadPoolExecutor` with a shared throttled fetcher. This lets the machine overlap parsing and network waits while still enforcing a minimum delay between requests.

For larger source sets, run separate manifests per source family rather than mixing official seju, PR releases, and social sources in one pass.

## Boundaries

- Discovery is not consent clearance.
- Image URL existence is not permission to train or publish outputs.
- The default pipeline should use official profile pages first, then manually reviewed secondary sources.
- Generated faces should be evaluated as aggregate-style approximations, not as likenesses of real people.
