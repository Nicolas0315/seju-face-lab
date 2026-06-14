from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .backends import backend_help, get_vector_backend
from .embeddings import iter_image_paths, render_appearance
from .generation import build_generation_config, run_diffusers_generation, write_generation_plan
from .metrics import review_subject_directories, score_generated_images, write_scores, write_subject_reviews
from .model import build_centroid_model, load_model, save_model
from .prompting import prompt_from_descriptors
from .run_reviews import review_generation_runs, write_generation_run_reviews
from .sources import discover_sources, download_source_images, read_source_manifest, write_source_manifest
from .style import OpenClipStyleBackend, score_style_images, write_style_scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="seju-face-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="build centroid vectors from reference images")
    build_parser.add_argument("--images", type=Path, required=True)
    build_parser.add_argument("--out", type=Path, required=True)
    build_parser.add_argument("--crop", choices=["center", "none"], default="center")
    build_parser.add_argument(
        "--backend",
        default="deterministic",
        help="vector backend name; currently deterministic is built in",
    )

    prompt_parser = subparsers.add_parser("prompt", help="print a generation prompt from a built model")
    prompt_parser.add_argument("--model", type=Path, required=True)
    prompt_parser.add_argument("--kind", choices=["mean", "median"], default="median")

    generate_parser = subparsers.add_parser(
        "generate",
        help="plan or run aggregate image generation from a centroid prompt",
    )
    generate_parser.add_argument("--model", type=Path, required=True)
    generate_parser.add_argument("--out", type=Path, required=True)
    generate_parser.add_argument("--provider", choices=["dry-run", "diffusers"], default="dry-run")
    generate_parser.add_argument("--hf-model", default="runwayml/stable-diffusion-v1-5")
    generate_parser.add_argument("--count", type=int, default=4)
    generate_parser.add_argument("--seed", type=int, default=150315)
    generate_parser.add_argument("--steps", type=int, default=30)
    generate_parser.add_argument("--guidance-scale", type=float, default=7.0)
    generate_parser.add_argument("--width", type=int, default=512)
    generate_parser.add_argument("--height", type=int, default=512)
    generate_parser.add_argument("--device", default="cuda")
    generate_parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    generate_parser.add_argument(
        "--variant",
        default="auto",
        help="Diffusers model variant; auto uses fp16 with float16, none disables it",
    )
    generate_parser.add_argument("--prompt", default=None)
    generate_parser.add_argument("--negative-prompt", default=None)
    generate_parser.add_argument("--dry-run", action="store_true")

    render_parser = subparsers.add_parser("render", help="render mean or median face image again")
    render_parser.add_argument("--model", type=Path, required=True)
    render_parser.add_argument("--kind", choices=["mean", "median"], default="median")
    render_parser.add_argument("--out", type=Path, required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="score generated images against centroids")
    evaluate_parser.add_argument("--model", type=Path, required=True)
    evaluate_parser.add_argument("--images", type=Path, required=True)
    evaluate_parser.add_argument("--out", type=Path, required=True)
    evaluate_parser.add_argument("--crop", choices=["center", "none"], default="center")
    evaluate_parser.add_argument("--backend", default="deterministic")

    style_parser = subparsers.add_parser(
        "style-evaluate",
        help="rank generated images on a separate OpenCLIP style axis",
    )
    style_parser.add_argument("--model", type=Path, required=True)
    style_parser.add_argument("--images", type=Path, required=True)
    style_parser.add_argument("--out", type=Path, required=True)
    style_parser.add_argument("--clip-model", default="ViT-B-32")
    style_parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    style_parser.add_argument("--device", default="auto")

    compare_runs_parser = subparsers.add_parser(
        "compare-runs",
        help="rank evaluated generation run directories by centroid scores",
    )
    compare_runs_parser.add_argument("--runs", type=Path, nargs="+", required=True)
    compare_runs_parser.add_argument("--out", type=Path, required=True)

    review_parser = subparsers.add_parser(
        "review-subjects",
        help="rank per-person image folders against a seju centroid model",
    )
    review_parser.add_argument("--model", type=Path, required=True)
    review_parser.add_argument("--subjects", type=Path, required=True)
    review_parser.add_argument("--out", type=Path, required=True)
    review_parser.add_argument("--crop", choices=["center", "none"], default="center")
    review_parser.add_argument("--backend", default="deterministic")

    subparsers.add_parser("backends", help="list available vector backend plans")

    source_parsers = subparsers.add_parser("sources", help="discover and audit web source candidates")
    source_subparsers = source_parsers.add_subparsers(dest="sources_command", required=True)
    discover_parser = source_subparsers.add_parser("discover", help="write seju profile image URL manifest")
    discover_parser.add_argument("--index-url", default="https://seju.tokyo/talents/")
    discover_parser.add_argument("--out", type=Path, required=True)
    discover_parser.add_argument("--as-of", default=None, help="YYYY-MM-DD, defaults to today's date")
    discover_parser.add_argument("--min-age", type=int, default=18)
    discover_parser.add_argument("--include-under-min-age", action="store_true")
    discover_parser.add_argument("--max-profiles", type=int, default=None)
    discover_parser.add_argument("--workers", type=int, default=2)
    discover_parser.add_argument("--delay-seconds", type=float, default=0.5)
    discover_parser.add_argument(
        "--user-agent",
        default="seju-face-lab/0.1 (+local research; contact: local)",
    )
    download_parser = source_subparsers.add_parser(
        "download",
        help="download reviewed eligible source images from a manifest",
    )
    download_parser.add_argument("--manifest", type=Path, required=True)
    download_parser.add_argument("--out", type=Path, default=Path("data/raw/seju_official"))
    download_parser.add_argument("--max-count", type=int, default=None)
    download_parser.add_argument("--dry-run", action="store_true")
    download_parser.add_argument("--include-ineligible", action="store_true")
    download_parser.add_argument("--delay-seconds", type=float, default=0.5)
    download_parser.add_argument("--max-bytes", type=int, default=20_000_000)
    download_parser.add_argument(
        "--user-agent",
        default="seju-face-lab/0.1 (+local research; contact: local)",
    )

    scrape_handles_parser = source_subparsers.add_parser(
        "scrape-handles",
        help="re-fetch talent profile pages and extract SNS handles (Instagram/Twitter/TikTok)",
    )
    scrape_handles_parser.add_argument("--manifest", type=Path, required=True,
                                       help="existing sources manifest (e.g. data/processed/seju_sources_official_2026-06-14.jsonl)")
    scrape_handles_parser.add_argument("--out", type=Path, required=True,
                                       help="output .jsonl for SNS handles (e.g. data/processed/sns_handles.jsonl)")
    scrape_handles_parser.add_argument("--max-profiles", type=int, default=None)
    scrape_handles_parser.add_argument("--delay-seconds", type=float, default=1.0)
    scrape_handles_parser.add_argument(
        "--user-agent",
        default="seju-face-lab/0.1 (+local research; contact: local)",
    )

    fetch_eng_parser = source_subparsers.add_parser(
        "fetch-engagement",
        help="fetch SNS follower + engagement metrics for all talents in a handles manifest",
    )
    fetch_eng_parser.add_argument("--handles", type=Path, required=True,
                                   help="handles manifest from scrape-handles")
    fetch_eng_parser.add_argument("--out", type=Path, required=True,
                                   help="output .jsonl for engagement records")
    fetch_eng_parser.add_argument("--platforms", nargs="+",
                                   choices=["instagram", "twitter", "tiktok"],
                                   default=["instagram", "twitter", "tiktok"])
    fetch_eng_parser.add_argument("--delay-seconds", type=float, default=2.0)

    import_eng_parser = source_subparsers.add_parser(
        "import-engagement",
        help="import SNS engagement data from a hand-curated CSV into the engagement manifest",
    )
    import_eng_parser.add_argument("--csv", type=Path, required=True,
                                    help="CSV with columns: talent_slug,platform,handle,followers,...")
    import_eng_parser.add_argument("--out", type=Path, required=True,
                                    help="output .jsonl engagement manifest (will be created or updated)")
    import_eng_parser.add_argument("--existing", type=Path, default=None,
                                    help="existing engagement .jsonl to merge into (optional)")
    import_eng_parser.add_argument("--no-overwrite", action="store_true",
                                    help="skip rows where (talent_slug, platform) already exists in --existing")

    # analyze command group
    analyze_parser = subparsers.add_parser("analyze", help="statistical analysis of face scores and SNS engagement")
    analyze_subparsers = analyze_parser.add_subparsers(dest="analyze_command", required=True)

    corr_parser = analyze_subparsers.add_parser(
        "correlation",
        help="correlate face centroid scores against SNS engagement metrics",
    )
    corr_parser.add_argument("--face-scores", type=Path, required=True,
                              help="subject_reviews.json from review-subjects command")
    corr_parser.add_argument("--engagement", type=Path, required=True,
                              help="engagement .jsonl from sources fetch-engagement")
    corr_parser.add_argument("--out", type=Path, required=True,
                              help="output directory for correlation report")

    args = parser.parse_args(argv)
    if args.command == "build":
        return _build(args.images, args.out, args.crop, args.backend)
    if args.command == "prompt":
        return _prompt(args.model, args.kind)
    if args.command == "generate":
        return _generate(args)
    if args.command == "render":
        return _render(args.model, args.kind, args.out)
    if args.command == "evaluate":
        return _evaluate(args.model, args.images, args.out, args.crop, args.backend)
    if args.command == "style-evaluate":
        return _style_evaluate(args)
    if args.command == "compare-runs":
        return _compare_runs(args.runs, args.out)
    if args.command == "review-subjects":
        return _review_subjects(args.model, args.subjects, args.out, args.crop, args.backend)
    if args.command == "backends":
        print(backend_help())
        return 0
    if args.command == "sources" and args.sources_command == "discover":
        return _sources_discover(args)
    if args.command == "sources" and args.sources_command == "download":
        return _sources_download(args)
    if args.command == "sources" and args.sources_command == "scrape-handles":
        return _sources_scrape_handles(args)
    if args.command == "sources" and args.sources_command == "fetch-engagement":
        return _sources_fetch_engagement(args)
    if args.command == "sources" and args.sources_command == "import-engagement":
        return _sources_import_engagement(args)
    if args.command == "analyze" and args.analyze_command == "correlation":
        return _analyze_correlation(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def _build(images: Path, out: Path, crop: str, backend_name: str) -> int:
    paths = iter_image_paths(images)
    if not paths:
        raise SystemExit(f"No supported images found under {images}")

    backend = get_vector_backend(backend_name)
    vectors = []
    failed_vectors = []
    for path in paths:
        try:
            vectors.append(backend.vectorize(path, crop=crop))
        except Exception as exc:  # noqa: BLE001 - keep model builds usable with noisy source folders.
            failed_vectors.append({"path": str(path), "reason": str(exc)})
    if not vectors:
        raise SystemExit(f"No usable images found under {images}")
    embeddings = np.stack([vector.embedding for vector in vectors])
    appearances = np.stack([vector.appearance for vector in vectors])
    model = build_centroid_model(
        image_ids=[vector.image_id for vector in vectors],
        source_paths=[str(vector.path) for vector in vectors],
        embeddings=embeddings,
        appearances=appearances,
    )
    save_model(model, out)

    vector_dir = out / "vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        vector_dir / "image_vectors.npz",
        embeddings=embeddings,
        appearances=appearances,
        image_ids=np.asarray([vector.image_id for vector in vectors]),
        source_paths=np.asarray([str(vector.path) for vector in vectors]),
    )
    descriptor_payload = {vector.image_id: vector.descriptors for vector in vectors}
    (vector_dir / "image_descriptors.json").write_text(
        json.dumps(descriptor_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if failed_vectors:
        (vector_dir / "image_vector_failures.json").write_text(
            json.dumps({"failed_count": len(failed_vectors), "failures": failed_vectors}, indent=2),
            encoding="utf-8",
        )

    print(f"built model: {out}")
    print(f"images: {len(vectors)}")
    print(f"failed images: {len(failed_vectors)}")
    print(f"backend: {backend.name}")
    print(f"embedding_dim: {model.embedding_dim}")
    print(f"prompt: {out / 'prompt.txt'}")
    return 0


def _prompt(model_dir: Path, kind: str) -> int:
    model = load_model(model_dir)
    print(prompt_from_descriptors(model.descriptors[kind]))
    return 0


def _generate(args: argparse.Namespace) -> int:
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    provider = "dry-run" if args.dry_run else args.provider
    variant = _resolve_generation_variant(args.variant, args.dtype)
    config = build_generation_config(
        model_dir=args.model,
        provider=provider,
        model_id=args.hf_model,
        count=args.count,
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        device=args.device,
        dtype=args.dtype,
        variant=variant,
        prompt_override=args.prompt,
        negative_prompt_override=args.negative_prompt,
    )
    if provider == "dry-run":
        result = write_generation_plan(config, args.model, args.out)
    else:
        result = run_diffusers_generation(config, args.model, args.out)
    print(f"generation status: {result.status}")
    print(f"run manifest: {args.out / 'generation_run.json'}")
    print(f"evaluate: {result.evaluation_command}")
    return 0


def _resolve_generation_variant(variant: str, dtype: str) -> str | None:
    if variant == "none":
        return None
    if variant == "auto":
        return "fp16" if dtype == "float16" else None
    return variant


def _render(model_dir: Path, kind: str, out: Path) -> int:
    model = load_model(model_dir)
    appearance = model.mean_appearance if kind == "mean" else model.median_appearance
    render_appearance(appearance, out)
    print(f"rendered: {out}")
    return 0


def _evaluate(model_dir: Path, images: Path, out: Path, crop: str, backend_name: str) -> int:
    model = load_model(model_dir)
    backend = get_vector_backend(backend_name)
    failed_paths: list[str] = []
    scores = score_generated_images(model, images, crop=crop, backend=backend, failed_paths=failed_paths)
    write_scores(scores, out, failed_paths=failed_paths)
    print(f"evaluated images: {len(scores)}")
    print(f"failed images: {len(failed_paths)}")
    print(f"scores: {out / 'scores.csv'}")
    return 0


def _style_evaluate(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    backend = OpenClipStyleBackend(
        model_name=args.clip_model,
        pretrained=args.pretrained,
        device=args.device,
    )
    failed_paths: list[str] = []
    scores = score_style_images(model, args.images, backend=backend, failed_paths=failed_paths)
    write_style_scores(scores, args.out, failed_paths=failed_paths)
    print(f"style evaluated images: {len(scores)}")
    print(f"failed images: {len(failed_paths)}")
    print(f"style scores: {args.out / 'style_scores.csv'}")
    return 0


def _compare_runs(run_dirs: list[Path], out: Path) -> int:
    reviews = review_generation_runs(run_dirs)
    write_generation_run_reviews(reviews, out)
    print(f"compared runs: {len(reviews)}")
    print(f"reviews: {out / 'generation_run_reviews.csv'}")
    return 0


def _review_subjects(model_dir: Path, subjects: Path, out: Path, crop: str, backend_name: str) -> int:
    if not subjects.is_dir():
        raise SystemExit(f"No subject directory found: {subjects}")
    model = load_model(model_dir)
    backend = get_vector_backend(backend_name)
    reviews = review_subject_directories(model, subjects, crop=crop, backend=backend)
    write_subject_reviews(reviews, out)
    print(f"reviewed subjects: {len(reviews)}")
    print(f"reviews: {out / 'subject_reviews.csv'}")
    return 0


def _sources_discover(args: argparse.Namespace) -> int:
    candidates = discover_sources(
        index_url=args.index_url,
        out_path=args.out,
        as_of=args.as_of,
        min_age=args.min_age,
        include_under_min_age=args.include_under_min_age,
        max_profiles=args.max_profiles,
        workers=args.workers,
        delay_seconds=args.delay_seconds,
        user_agent=args.user_agent,
    )
    write_source_manifest(candidates, args.out)
    eligible = sum(1 for candidate in candidates if candidate.eligible_for_analysis)
    print(f"source candidates: {len(candidates)}")
    print(f"eligible candidates: {eligible}")
    print(f"manifest: {args.out}")
    print(f"audit: {args.out.with_suffix('.audit.md')}")
    return 0


def _sources_download(args: argparse.Namespace) -> int:
    candidates = read_source_manifest(args.manifest)
    results = download_source_images(
        candidates=candidates,
        out_dir=args.out,
        max_count=args.max_count,
        dry_run=args.dry_run,
        include_ineligible=args.include_ineligible,
        delay_seconds=args.delay_seconds,
        max_bytes=args.max_bytes,
        user_agent=args.user_agent,
    )
    downloaded = sum(1 for result in results if result.status == "downloaded")
    skipped = sum(1 for result in results if result.status == "skipped")
    failed = sum(1 for result in results if result.status == "failed")
    planned = sum(1 for result in results if result.status == "planned")
    print(f"downloaded: {downloaded}")
    print(f"planned: {planned}")
    print(f"skipped: {skipped}")
    print(f"failed: {failed}")
    print(f"out: {args.out}")
    return 1 if failed else 0


def _sources_scrape_handles(args: argparse.Namespace) -> int:
    from .sns_metrics import scrape_talent_sns_handles

    records = scrape_talent_sns_handles(
        manifest_path=args.manifest,
        out_path=args.out,
        delay_seconds=args.delay_seconds,
        max_profiles=args.max_profiles,
        user_agent=args.user_agent,
    )
    with_sns = sum(1 for r in records if r.sns_handles)
    total_handles = sum(len(r.sns_handles) for r in records)
    print(f"profiles scraped: {len(records)}")
    print(f"profiles with SNS handles: {with_sns}")
    print(f"total handles found: {total_handles}")
    print(f"manifest: {args.out}")

    # Print summary
    for r in records:
        if r.sns_handles:
            handles_str = "  ".join(f"{p}=@{h}" for p, h in r.sns_handles.items())
            print(f"  {r.talent_slug}: {handles_str}")
        else:
            print(f"  {r.talent_slug}: (no SNS found)")
    return 0


def _sources_fetch_engagement(args: argparse.Namespace) -> int:
    from .sns_metrics import fetch_all_talent_engagement

    records = fetch_all_talent_engagement(
        handles_path=args.handles,
        out_path=args.out,
        delay_between_talents=args.delay_seconds,
        platforms=args.platforms,
    )
    ok_count = sum(
        1 for r in records
        for e in r.engagements if e.fetch_status in ("ok", "partial")
    )
    blocked = sum(
        1 for r in records
        for e in r.engagements if e.fetch_status == "blocked"
    )
    print(f"talents processed: {len(records)}")
    print(f"engagements fetched (ok/partial): {ok_count}")
    print(f"blocked: {blocked}")
    print(f"output: {args.out}")
    return 0


def _sources_import_engagement(args: argparse.Namespace) -> int:
    from .sns_metrics import import_engagement_csv

    records = import_engagement_csv(
        csv_path=args.csv,
        out_path=args.out,
        existing_path=args.existing or (args.out if args.out.exists() else None),
        overwrite_platforms=not args.no_overwrite,
    )
    total_engs = sum(len(r.engagements) for r in records)
    with_followers = sum(
        1 for r in records for e in r.engagements if e.followers is not None
    )
    print(f"talents in manifest: {len(records)}")
    print(f"engagement records: {total_engs}")
    print(f"records with followers: {with_followers}")
    print(f"output: {args.out}")
    return 0


def _analyze_correlation(args: argparse.Namespace) -> int:
    from .correlation import build_correlation_dataset, compute_correlations, write_correlation_report

    print(f"loading face scores: {args.face_scores}")
    print(f"loading engagement data: {args.engagement}")
    rows = build_correlation_dataset(
        subject_reviews_json=args.face_scores,
        engagement_manifest=args.engagement,
    )
    print(f"dataset rows: {len(rows)}")
    with_face = sum(1 for r in rows if r.face_mean_centroid_score is not None)
    with_ig = sum(1 for r in rows if r.ig_followers is not None)
    with_tw = sum(1 for r in rows if r.tw_followers is not None)
    with_tk = sum(1 for r in rows if r.tk_followers is not None)
    print(f"  with face score: {with_face}")
    print(f"  with Instagram: {with_ig}")
    print(f"  with Twitter: {with_tw}")
    print(f"  with TikTok: {with_tk}")

    correlations = compute_correlations(rows)
    write_correlation_report(rows, correlations, args.out)

    print("\nTop correlations (by |Spearman r|):")
    top = sorted(
        [c for c in correlations if c.spearman_r is not None],
        key=lambda c: -abs(c.spearman_r),
    )[:5]
    for c in top:
        print(f"  {c.variable_a} × {c.variable_b}: ρ={c.spearman_r:+.3f} (n={c.n}) [{c.interpretation}]")

    print(f"\nreport: {args.out / 'correlation_report.md'}")
    return 0
