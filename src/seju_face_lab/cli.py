from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .backends import backend_help, get_vector_backend
from .agency import write_agency_average_params
from .backend_compare import compare_deepface_detectors, compare_subject_backends, compare_vector_backends
from .backend_diagnostics import write_backend_diagnostics
from .benchmark_research import write_benchmark_research
from .calibration import write_generation_calibration
from .embeddings import iter_image_paths, render_appearance
from .enhancement import write_agency_enhancement_bundle
from .face_axes import write_face_axis_report
from .generation import (
    build_generation_config,
    run_diffusers_generation,
    run_openai_image_generation,
    write_generation_plan,
)
from .ingredients import write_ingredients_report
from .metrics import review_subject_directories, score_generated_images, write_scores, write_subject_reviews
from .model import build_centroid_model, load_model, save_model
from .model_audit import centroid_stability, write_model_audit
from .pipeline import run_pipeline_config
from .prompting import prompt_from_descriptors
from .precision import write_precision_report
from .quality import review_image_quality, write_image_quality
from .run_reviews import review_generation_runs, write_generation_run_reviews
from .sources import discover_sources, download_source_images, read_source_manifest, write_source_manifest
from .style import OpenClipStyleBackend, score_style_images, write_style_scores
from .subject_vectors import vectorize_subjects, write_subject_vectors
from .vector_export import write_vector_export
from .workers import DEFAULT_DIAGNOSTIC_WORKERS, LOCAL_4090, distribute_vectorize, write_worker_diagnostics


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
    build_parser.add_argument(
        "--balance",
        choices=["image", "subject"],
        default="image",
        help="centroid aggregation mode; subject balances one folder/person as one template",
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
    generate_parser.add_argument(
        "--provider",
        choices=["dry-run", "diffusers", "openai-image"],
        default="dry-run",
    )
    generate_parser.add_argument(
        "--hf-model",
        default="runwayml/stable-diffusion-v1-5",
        help="Diffusers model id; kept for existing configs",
    )
    generate_parser.add_argument(
        "--image-model",
        default=None,
        help="image generation model id for provider=openai-image",
    )
    generate_parser.add_argument("--count", type=int, default=4)
    generate_parser.add_argument("--seed", type=int, default=150315)
    generate_parser.add_argument("--steps", type=int, default=30)
    generate_parser.add_argument("--guidance-scale", type=float, default=7.0)
    generate_parser.add_argument("--width", type=int, default=512)
    generate_parser.add_argument("--height", type=int, default=512)
    generate_parser.add_argument("--device", default="cuda")
    generate_parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    generate_parser.add_argument("--output-format", choices=["png", "jpeg", "webp"], default="png")
    generate_parser.add_argument("--quality", choices=["low", "medium", "high"], default="medium")
    generate_parser.add_argument(
        "--variant",
        default="auto",
        help="Diffusers model variant; auto uses fp16 with float16, none disables it",
    )
    generate_parser.add_argument(
        "--prompt-profile",
        choices=["balanced", "detector-friendly"],
        default="balanced",
        help="generation prompt profile; detector-friendly favors frontal visible faces",
    )
    generate_parser.add_argument(
        "--centroid-kind",
        choices=["mean", "median"],
        default="median",
        help="centroid descriptor used to build the generation prompt",
    )
    generate_parser.add_argument("--prompt", default=None)
    generate_parser.add_argument("--negative-prompt", default=None)
    generate_parser.add_argument("--dry-run", action="store_true")
    generate_parser.add_argument("--review", action="store_true", help="review generated images after a real generation run")
    generate_parser.add_argument("--review-out", type=Path, default=None)
    generate_parser.add_argument("--review-crop", choices=["center", "none"], default="center")
    generate_parser.add_argument("--review-backend", default="deterministic")

    render_parser = subparsers.add_parser("render", help="render mean or median face image again")
    render_parser.add_argument("--model", type=Path, required=True)
    render_parser.add_argument("--kind", choices=["mean", "median"], default="median")
    render_parser.add_argument("--out", type=Path, required=True)

    export_vectors_parser = subparsers.add_parser(
        "export-vectors",
        help="export mean/median centroid vectors to JSON or CSV",
    )
    export_vectors_parser.add_argument("--model", type=Path, required=True)
    export_vectors_parser.add_argument("--out", type=Path, required=True)
    export_vectors_parser.add_argument("--format", choices=["json", "csv"], default="json")
    export_vectors_parser.add_argument(
        "--include-appearance",
        action="store_true",
        help="also export flattened mean/median appearance render vectors",
    )

    audit_parser = subparsers.add_parser(
        "audit-model",
        help="audit mean/median centroid vectors and descriptor deltas for a built model",
    )
    audit_parser.add_argument("--model", type=Path, required=True)
    audit_parser.add_argument("--out", type=Path, required=True)

    ingredients_parser = subparsers.add_parser(
        "ingredients-report",
        help="decompose aggregate face ingredients from model descriptors",
    )
    ingredients_parser.add_argument("--model", type=Path, required=True)
    ingredients_parser.add_argument("--out", type=Path, required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="score generated images against centroids")
    evaluate_parser.add_argument("--model", type=Path, required=True)
    evaluate_parser.add_argument("--images", type=Path, required=True)
    evaluate_parser.add_argument("--out", type=Path, required=True)
    evaluate_parser.add_argument("--crop", choices=["center", "none"], default="center")
    evaluate_parser.add_argument("--backend", default="deterministic")

    distributed_evaluate_parser = subparsers.add_parser(
        "distributed-evaluate",
        help="score generated images through explicit worker chunks",
    )
    distributed_evaluate_parser.add_argument("--model", type=Path, required=True)
    distributed_evaluate_parser.add_argument("--images", type=Path, required=True)
    distributed_evaluate_parser.add_argument("--out", type=Path, required=True)
    distributed_evaluate_parser.add_argument("--crop", choices=["center", "none"], default="center")
    distributed_evaluate_parser.add_argument("--backend", default="deterministic")
    distributed_evaluate_parser.add_argument(
        "--include-remote",
        action="store_true",
        help="include configured SSH workers; currently requires a shared-path/sync plan",
    )

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

    face_axes_parser = subparsers.add_parser(
        "face-axes",
        help="map images onto 8 visual face axes with quadrant and cross-axis labels",
    )
    face_axes_parser.add_argument("--images", type=Path, required=True)
    face_axes_parser.add_argument("--out", type=Path, required=True)
    face_axes_parser.add_argument("--crop", choices=["center", "none"], default="center")
    face_axes_parser.add_argument("--backend", default="deterministic")

    compare_runs_parser = subparsers.add_parser(
        "compare-runs",
        help="rank evaluated generation run directories by centroid scores",
    )
    compare_runs_parser.add_argument("--runs", type=Path, nargs="+", required=True)
    compare_runs_parser.add_argument("--out", type=Path, required=True)

    precision_parser = subparsers.add_parser(
        "precision-report",
        help="summarize centroid, generated-image, QA, and subject-review evidence",
    )
    precision_parser.add_argument("--model", type=Path, required=True)
    precision_parser.add_argument("--out", type=Path, required=True)
    precision_parser.add_argument("--generation-review", type=Path, default=None)
    precision_parser.add_argument("--subject-review", type=Path, default=None)
    precision_parser.add_argument("--evaluation", type=Path, default=None)
    precision_parser.add_argument("--quality", type=Path, default=None)
    precision_parser.add_argument("--backend-comparison", type=Path, default=None)
    precision_parser.add_argument("--subject-backend-comparison", type=Path, default=None)
    precision_parser.add_argument("--correlation", type=Path, default=None)
    precision_parser.add_argument("--model-audit", type=Path, default=None)
    precision_parser.add_argument("--vector-export", type=Path, default=None)
    precision_parser.add_argument("--face-ingredients", type=Path, default=None)
    precision_parser.add_argument("--benchmark-research", type=Path, default=None)

    run_pipeline_parser = subparsers.add_parser(
        "run-pipeline",
        help="run a reproducible build/generation/review pipeline from a JSON config",
    )
    run_pipeline_parser.add_argument("--config", type=Path, required=True)
    run_pipeline_parser.add_argument("--out", type=Path, default=None)

    qa_parser = subparsers.add_parser(
        "qa-images",
        help="flag generated images that are not a single centered frontal face",
    )
    qa_parser.add_argument("--images", type=Path, required=True)
    qa_parser.add_argument("--out", type=Path, required=True)

    review_generated_parser = subparsers.add_parser(
        "review-generated",
        help="run evaluate, QA, and one-run comparison for a generated image directory",
    )
    review_generated_parser.add_argument("--model", type=Path, required=True)
    review_generated_parser.add_argument("--images", type=Path, required=True)
    review_generated_parser.add_argument("--out", type=Path, default=None)
    review_generated_parser.add_argument("--crop", choices=["center", "none"], default="center")
    review_generated_parser.add_argument("--backend", default="deterministic")

    review_parser = subparsers.add_parser(
        "review-subjects",
        help="rank per-person image folders against a seju centroid model",
    )
    review_parser.add_argument("--model", type=Path, required=True)
    review_parser.add_argument("--subjects", type=Path, required=True)
    review_parser.add_argument("--out", type=Path, required=True)
    review_parser.add_argument("--crop", choices=["center", "none"], default="center")
    review_parser.add_argument("--backend", default="deterministic")

    vectorize_subjects_parser = subparsers.add_parser(
        "vectorize-subjects",
        help="save one aggregate vector per person folder for later real agency centroids",
    )
    vectorize_subjects_parser.add_argument("--subjects", type=Path, required=True)
    vectorize_subjects_parser.add_argument("--out", type=Path, required=True)
    vectorize_subjects_parser.add_argument("--crop", choices=["center", "none"], default="center")
    vectorize_subjects_parser.add_argument("--backend", default="deterministic")
    vectorize_subjects_parser.add_argument("--workers", type=int, default=4)

    subparsers.add_parser("backends", help="list available vector backend plans")

    backend_diag_parser = subparsers.add_parser(
        "backend-diagnostics",
        help="write dependency and GPU/provider diagnostics for optional backends",
    )
    backend_diag_parser.add_argument("--out", type=Path, required=True)

    benchmark_research_parser = subparsers.add_parser(
        "benchmark-research",
        help="write face/iris benchmark and OSS adoption notes for vectorization planning",
    )
    benchmark_research_parser.add_argument("--out", type=Path, required=True)

    agency_parser = subparsers.add_parser(
        "review-agencies",
        help="build agency-level average face parameters and image-generation prompts",
    )
    agency_parser.add_argument("--model", type=Path, required=True)
    agency_parser.add_argument("--agencies", type=Path, required=True)
    agency_parser.add_argument("--out", type=Path, required=True)

    enhance_agency_parser = subparsers.add_parser(
        "enhance-agencies",
        help="fuse agency hypotheses, image centroid scores, and 8-axis observations",
    )
    enhance_agency_parser.add_argument("--model", type=Path, required=True)
    enhance_agency_parser.add_argument("--agencies", type=Path, required=True)
    enhance_agency_parser.add_argument("--images", type=Path, required=True)
    enhance_agency_parser.add_argument("--out", type=Path, required=True)
    enhance_agency_parser.add_argument("--crop", choices=["center", "none"], default="center")
    enhance_agency_parser.add_argument("--backend", default="deterministic")

    calibrate_agency_parser = subparsers.add_parser(
        "calibrate-agency-generation",
        help="turn agency enhancement measurements into refined generation prompts",
    )
    calibrate_agency_parser.add_argument("--enhancement", type=Path, required=True)
    calibrate_agency_parser.add_argument("--agency-params", type=Path, required=True)
    calibrate_agency_parser.add_argument("--out", type=Path, required=True)
    calibrate_agency_parser.add_argument("--target-image-score", type=float, default=0.35)
    calibrate_agency_parser.add_argument("--target-axis-alignment", type=float, default=0.62)
    calibrate_agency_parser.add_argument("--target-enhancement-score", type=float, default=0.76)
    calibrate_agency_parser.add_argument("--seed-start", type=int, default=260623)
    calibrate_agency_parser.add_argument("--variants-per-agency", type=int, default=3)

    worker_diag_parser = subparsers.add_parser(
        "worker-diagnostics",
        help="write local/SSH worker readiness diagnostics for GPU split-run planning",
    )
    worker_diag_parser.add_argument("--out", type=Path, required=True)
    worker_diag_parser.add_argument(
        "--include-remote",
        action="store_true",
        help="also probe the configured remote GPU worker over SSH",
    )
    worker_diag_parser.add_argument("--timeout-seconds", type=int, default=30)

    compare_backends_parser = subparsers.add_parser(
        "compare-backends",
        help="build/evaluate multiple vector backends on the same local image sets",
    )
    compare_backends_parser.add_argument("--reference-images", type=Path, required=True)
    compare_backends_parser.add_argument("--images", type=Path, required=True)
    compare_backends_parser.add_argument("--out", type=Path, required=True)
    compare_backends_parser.add_argument("--backends", nargs="+", default=["deterministic", "opencv-face"])
    compare_backends_parser.add_argument("--crop", choices=["center", "none"], default="center")

    compare_subject_backends_parser = subparsers.add_parser(
        "compare-subject-backends",
        help="review per-subject similarity rankings across multiple vector backends",
    )
    compare_subject_backends_parser.add_argument("--reference-images", type=Path, required=True)
    compare_subject_backends_parser.add_argument("--subjects", type=Path, required=True)
    compare_subject_backends_parser.add_argument("--out", type=Path, required=True)
    compare_subject_backends_parser.add_argument("--backends", nargs="+", default=["deterministic", "opencv-face"])
    compare_subject_backends_parser.add_argument("--crop", choices=["center", "none"], default="center")

    compare_deepface_parser = subparsers.add_parser(
        "compare-deepface-detectors",
        help="compare DeepFace detector backends on the same local image sets",
    )
    compare_deepface_parser.add_argument("--reference-images", type=Path, required=True)
    compare_deepface_parser.add_argument("--images", type=Path, required=True)
    compare_deepface_parser.add_argument("--out", type=Path, required=True)
    compare_deepface_parser.add_argument(
        "--detectors",
        nargs="+",
        default=["opencv", "mtcnn", "retinaface", "skip"],
        choices=[
            "opencv",
            "ssd",
            "dlib",
            "mtcnn",
            "retinaface",
            "mediapipe",
            "yolov8",
            "yunet",
            "fastmtcnn",
            "centerface",
            "skip",
        ],
    )
    compare_deepface_parser.add_argument("--model-name", default="ArcFace")
    compare_deepface_parser.add_argument("--crop", choices=["center", "none"], default="center")
    compare_deepface_parser.add_argument(
        "--max-reference-images",
        type=int,
        default=None,
        help="limit reference images for slow detector smoke audits",
    )
    compare_deepface_parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="limit target images for slow detector smoke audits",
    )
    compare_deepface_parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="reuse completed per-detector model/evaluation outputs already present under --out",
    )

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

    # explore command group
    explore_parser = subparsers.add_parser("explore", help="SNS exploration engine (multi-source, cached)")
    explore_sub = explore_parser.add_subparsers(dest="explore_command", required=True)

    ex_profile_p = explore_sub.add_parser("profile", help="fetch a single SNS profile via the router")
    ex_profile_p.add_argument("--platform", required=True, choices=["instagram", "twitter", "tiktok"])
    ex_profile_p.add_argument("--handle", required=True)
    ex_profile_p.add_argument("--cache", type=Path, default=Path("data/processed/sns_cache.db"))
    ex_profile_p.add_argument("--remote-host", default=None, help="optional SSH host for Instagram fetches")
    ex_profile_p.add_argument("--force", action="store_true", help="bypass cache and re-fetch")

    ex_batch_p = explore_sub.add_parser("batch", help="fetch all talent handles via the router (with cache)")
    ex_batch_p.add_argument("--handles", type=Path, required=True)
    ex_batch_p.add_argument("--out", type=Path, required=True, help="output engagement JSONL")
    ex_batch_p.add_argument("--cache", type=Path, default=Path("data/processed/sns_cache.db"))
    ex_batch_p.add_argument("--remote-host", default=None, help="optional SSH host for Instagram fetches")
    ex_batch_p.add_argument("--platforms", nargs="+",
                             choices=["instagram", "twitter", "tiktok"],
                             default=["instagram", "twitter", "tiktok"])
    ex_batch_p.add_argument("--delay", type=float, default=1.5)
    ex_batch_p.add_argument("--force", action="store_true")

    ex_load_p = explore_sub.add_parser(
        "load-cache",
        help="pre-populate the SNS cache from existing engagement manifests/JSON files",
    )
    ex_load_p.add_argument("--engagement", type=Path, help="existing engagement .jsonl")
    ex_load_p.add_argument("--ig-json", type=Path, default=None,
                            help="Instagram result JSON from an external batch fetch")
    ex_load_p.add_argument("--handles", type=Path, default=None,
                            help="sns_handles.jsonl for handle→slug lookup")
    ex_load_p.add_argument("--cache", type=Path, default=Path("data/processed/sns_cache.db"))

    ex_disc_p = explore_sub.add_parser("discover", help="discover SNS handles for a talent name")
    ex_disc_p.add_argument("--name", required=True, help="talent name to search (e.g. '森 香澄')")
    ex_disc_p.add_argument("--platforms", nargs="+",
                            choices=["instagram", "twitter"],
                            default=["instagram", "twitter"])

    args = parser.parse_args(argv)
    if args.command == "build":
        return _build(args.images, args.out, args.crop, args.backend, args.balance)
    if args.command == "prompt":
        return _prompt(args.model, args.kind)
    if args.command == "generate":
        return _generate(args)
    if args.command == "render":
        return _render(args.model, args.kind, args.out)
    if args.command == "export-vectors":
        return _export_vectors(args.model, args.out, args.format, args.include_appearance)
    if args.command == "audit-model":
        return _audit_model(args.model, args.out)
    if args.command == "ingredients-report":
        return _ingredients_report(args.model, args.out)
    if args.command == "evaluate":
        return _evaluate(args.model, args.images, args.out, args.crop, args.backend)
    if args.command == "distributed-evaluate":
        return _distributed_evaluate(args)
    if args.command == "style-evaluate":
        return _style_evaluate(args)
    if args.command == "face-axes":
        return _face_axes(args)
    if args.command == "compare-runs":
        return _compare_runs(args.runs, args.out)
    if args.command == "precision-report":
        return _precision_report(args)
    if args.command == "run-pipeline":
        return _run_pipeline(args)
    if args.command == "qa-images":
        return _qa_images(args.images, args.out)
    if args.command == "review-generated":
        return _review_generated(args)
    if args.command == "review-subjects":
        return _review_subjects(args.model, args.subjects, args.out, args.crop, args.backend)
    if args.command == "vectorize-subjects":
        return _vectorize_subjects(args.subjects, args.out, args.crop, args.backend, args.workers)
    if args.command == "backends":
        print(backend_help())
        return 0
    if args.command == "backend-diagnostics":
        return _backend_diagnostics(args.out)
    if args.command == "benchmark-research":
        return _benchmark_research(args.out)
    if args.command == "review-agencies":
        return _review_agencies(args)
    if args.command == "enhance-agencies":
        return _enhance_agencies(args)
    if args.command == "calibrate-agency-generation":
        return _calibrate_agency_generation(args)
    if args.command == "worker-diagnostics":
        return _worker_diagnostics(args.out, args.include_remote, args.timeout_seconds)
    if args.command == "compare-backends":
        return _compare_backends(args)
    if args.command == "compare-subject-backends":
        return _compare_subject_backends(args)
    if args.command == "compare-deepface-detectors":
        return _compare_deepface_detectors(args)
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
    if args.command == "explore" and args.explore_command == "profile":
        return _explore_profile(args)
    if args.command == "explore" and args.explore_command == "batch":
        return _explore_batch(args)
    if args.command == "explore" and args.explore_command == "discover":
        return _explore_discover(args)
    if args.command == "explore" and args.explore_command == "load-cache":
        return _explore_load_cache(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def _build(images: Path, out: Path, crop: str, backend_name: str, balance: str = "image") -> int:
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
    centroid_mode = "subject_balanced" if balance == "subject" else "image_weighted"
    subject_ids = [_subject_id_for_path(images, vector.path) for vector in vectors] if balance == "subject" else None
    model = build_centroid_model(
        image_ids=[vector.image_id for vector in vectors],
        source_paths=[str(vector.path) for vector in vectors],
        embeddings=embeddings,
        appearances=appearances,
        centroid_mode=centroid_mode,
        subject_ids=subject_ids,
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
    if subject_ids is not None:
        (vector_dir / "subject_counts.json").write_text(
            json.dumps(model.subject_counts or {}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    stability = centroid_stability(embeddings, subject_ids=subject_ids)
    (vector_dir / "centroid_stability.json").write_text(
        json.dumps(stability, ensure_ascii=False, indent=2),
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
    print(f"centroid_mode: {model.centroid_mode}")
    print(f"subjects: {model.subject_count}")
    print(f"centroid_stability: {stability.get('band')} (self_cosine_mean={stability.get('self_cosine_mean')})")
    print(f"embedding_dim: {model.embedding_dim}")
    print(f"prompt: {out / 'prompt.txt'}")
    return 0


def _subject_id_for_path(root: Path, image_path: Path) -> str:
    try:
        relative = image_path.resolve().relative_to(root.resolve())
    except ValueError:
        return image_path.parent.name or image_path.stem
    parts = relative.parts
    if len(parts) > 1:
        return parts[0]
    return image_path.stem


def _prompt(model_dir: Path, kind: str) -> int:
    model = load_model(model_dir)
    print(prompt_from_descriptors(model.descriptors[kind]))
    return 0


def _generate(args: argparse.Namespace) -> int:
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    provider = "dry-run" if args.dry_run else args.provider
    variant = _resolve_generation_variant(args.variant, args.dtype)
    width, height = _generation_dimensions(provider, args.width, args.height)
    config = build_generation_config(
        model_dir=args.model,
        provider=provider,
        model_id=_generation_model_id(args),
        count=args.count,
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=width,
        height=height,
        device=args.device,
        dtype=args.dtype,
        variant=variant,
        output_format=args.output_format,
        quality=args.quality,
        centroid_kind=args.centroid_kind,
        prompt_profile=args.prompt_profile,
        prompt_override=args.prompt,
        negative_prompt_override=args.negative_prompt,
    )
    if provider == "dry-run":
        result = write_generation_plan(config, args.model, args.out)
    elif provider == "diffusers":
        result = run_diffusers_generation(config, args.model, args.out)
    elif provider == "openai-image":
        result = run_openai_image_generation(config, args.model, args.out)
    else:
        raise SystemExit(f"Unsupported generation provider: {provider}")
    print(f"generation status: {result.status}")
    print(f"run manifest: {args.out / 'generation_run.json'}")
    print(f"evaluate: {result.evaluation_command}")
    if args.review and result.generated_images:
        _review_generated(
            argparse.Namespace(
                model=args.model,
                images=args.out,
                out=args.review_out,
                crop=args.review_crop,
                backend=args.review_backend,
            )
        )
    return 0


def _resolve_generation_variant(variant: str, dtype: str) -> str | None:
    if variant == "none":
        return None
    if variant == "auto":
        return "fp16" if dtype == "float16" else None
    return variant


def _generation_model_id(args: argparse.Namespace) -> str:
    if args.provider == "openai-image":
        return args.image_model or "gpt-image-2"
    return args.hf_model


def _generation_dimensions(provider: str, width: int, height: int) -> tuple[int, int]:
    if provider == "openai-image" and (width, height) == (512, 512):
        return 1024, 1024
    return width, height


def _render(model_dir: Path, kind: str, out: Path) -> int:
    model = load_model(model_dir)
    appearance = model.mean_appearance if kind == "mean" else model.median_appearance
    render_appearance(appearance, out)
    print(f"rendered: {out}")
    return 0


def _export_vectors(model_dir: Path, out: Path, output_format: str, include_appearance: bool) -> int:
    payload = write_vector_export(
        model_dir,
        out,
        output_format=output_format,
        include_appearance=include_appearance,
    )
    print(f"exported vectors: {out}")
    print(f"embedding_dim: {payload['embedding_dim']}")
    print(f"vectors: {len(payload['vectors'])}")
    return 0


def _audit_model(model_dir: Path, out: Path) -> int:
    audit = write_model_audit(model_dir, out)
    centroids = audit["centroids"]
    embedding_pair = centroids.get("mean_median_embedding", {})
    print(f"model audit: {out / 'model_audit.md'}")
    print(f"centroids available: {centroids.get('available', False)}")
    print(f"mean/median embedding cosine: {embedding_pair.get('cosine')}")
    return 0


def _ingredients_report(model_dir: Path, out: Path) -> int:
    report = write_ingredients_report(model_dir, out)
    ingredients = report.get("ingredients", {})
    overall = ingredients.get("overall", {}) if isinstance(ingredients, dict) else {}
    print(f"ingredients report: {out / 'face_ingredients.md'}")
    print(f"image_count: {report.get('image_count')}")
    print(f"overall: {overall.get('summary')}")
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


def _distributed_evaluate(args: argparse.Namespace) -> int:
    workers = DEFAULT_DIAGNOSTIC_WORKERS if args.include_remote else [LOCAL_4090]
    try:
        scores = distribute_vectorize(
            list(iter_image_paths(args.images)),
            args.model,
            args.out,
            backend=args.backend,
            crop=args.crop,
            workers=workers,
        )
    except RuntimeError as exc:
        print(f"distributed evaluate failed: {exc}")
        print(f"scores: {args.out / 'scores.csv'}")
        print(f"distributed scores: {args.out / 'distributed_scores.json'}")
        return 1
    print(f"distributed evaluated images: {len(scores)}")
    print(f"workers: {len(workers)}")
    print(f"scores: {args.out / 'scores.csv'}")
    print(f"distributed scores: {args.out / 'distributed_scores.json'}")
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


def _face_axes(args: argparse.Namespace) -> int:
    report = write_face_axis_report(
        images=args.images,
        out_dir=args.out,
        crop=args.crop,
        backend_name=args.backend,
    )
    print(f"face axis report: {args.out / 'face_axis_report.md'}")
    print(f"images: {report['image_count']}")
    summary = report.get("summary", {})
    distribution = summary.get("distribution", {}) if isinstance(summary, dict) else {}
    print(f"quadrant: {distribution.get('quadrant')}")
    return 0


def _compare_runs(run_dirs: list[Path], out: Path) -> int:
    reviews = review_generation_runs(run_dirs)
    write_generation_run_reviews(reviews, out)
    print(f"compared runs: {len(reviews)}")
    print(f"reviews: {out / 'generation_run_reviews.csv'}")
    return 0


def _precision_report(args: argparse.Namespace) -> int:
    report = write_precision_report(
        model_dir=args.model,
        out_dir=args.out,
        generation_review=args.generation_review,
        subject_review=args.subject_review,
        evaluation=args.evaluation,
        quality=args.quality,
        backend_comparison=args.backend_comparison,
        subject_backend_comparison=args.subject_backend_comparison,
        correlation=args.correlation,
        model_audit=args.model_audit,
        vector_export=args.vector_export,
        face_ingredients=args.face_ingredients,
        benchmark_research=args.benchmark_research,
    )
    print(f"precision report: {args.out / 'precision_report.md'}")
    print(f"model images: {report['model']['image_count']}")
    print(f"best generated score: {report['generation']['best_centroid_score']}")
    print(f"top subject: {report['subjects']['top_subject']}")
    return 0


def _run_pipeline(args: argparse.Namespace) -> int:
    handlers = {
        "build": _run_pipeline_build,
        "audit-model": _run_pipeline_audit_model,
        "export-vectors": _run_pipeline_export_vectors,
        "ingredients-report": _run_pipeline_ingredients_report,
        "benchmark-research": _run_pipeline_benchmark_research,
        "generate": _run_pipeline_generate,
        "generation-sweep": _run_pipeline_generation_sweep,
        "evaluate": _run_pipeline_evaluate,
        "style-evaluate": _run_pipeline_style_evaluate,
        "review-generated": _run_pipeline_review_generated,
        "review-subjects": _run_pipeline_review_subjects,
        "compare-backends": _run_pipeline_compare_backends,
        "compare-subject-backends": _run_pipeline_compare_subject_backends,
        "explore-batch": _run_pipeline_explore_batch,
        "analyze-correlation": _run_pipeline_analyze_correlation,
        "precision-report": _run_pipeline_precision_report,
    }
    plan = run_pipeline_config(args.config, args.out, handlers)
    failed = [step for step in plan.steps if step.status == "failed"]
    print(f"pipeline: {plan.name}")
    print(f"steps: {len(plan.steps)}")
    print(f"failed: {len(failed)}")
    return 1 if failed else 0


def _run_pipeline_build(config: dict) -> int:
    return _build(
        Path(config["reference_images"]),
        _pipeline_model(config),
        str(config.get("crop", "center")),
        str(config.get("vector_backend", config.get("backend", "deterministic"))),
        str(config.get("balance", "image")),
    )


def _run_pipeline_audit_model(config: dict) -> int:
    return _audit_model(_pipeline_model(config), _pipeline_model_audit_out(config))


def _run_pipeline_export_vectors(config: dict) -> int:
    export = _pipeline_vector_export_config(config)
    return _export_vectors(
        _pipeline_model(config),
        Path(export["out"]),
        str(export.get("format", "json")),
        bool(export.get("include_appearance", False)),
    )


def _run_pipeline_ingredients_report(config: dict) -> int:
    return _ingredients_report(_pipeline_model(config), _pipeline_ingredients_report_out(config))


def _run_pipeline_benchmark_research(config: dict) -> int:
    return _benchmark_research(_pipeline_benchmark_research_out(config))


def _run_pipeline_generate(config: dict) -> int:
    generation = _pipeline_generation_config(config)
    return _generate(_generation_namespace(config, generation, _pipeline_generated_images(config)))


def _run_pipeline_generation_sweep(config: dict) -> int:
    sweep = _pipeline_generation_sweep_config(config)
    base_out = Path(sweep["out"])
    runs = sweep.get("runs")
    if not isinstance(runs, list) or not runs:
        raise SystemExit("Pipeline generation_sweep requires at least one run")

    run_dirs: list[Path] = []
    seen_run_dirs: set[Path] = set()
    for index, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            raise SystemExit("Pipeline generation_sweep runs must be objects")
        run_config = _merge_generation_sweep_run(sweep, run)
        if bool(sweep.get("compare_runs", False)) and not _generation_run_is_reviewable(run_config):
            raise SystemExit(
                "Pipeline generation_sweep compare_runs requires non-dry-run runs with review=true"
            )
        run_dir = _generation_sweep_run_dir(base_out, run_config, index)
        run_dir_key = run_dir.resolve(strict=False)
        if run_dir_key in seen_run_dirs:
            raise SystemExit(f"Pipeline generation_sweep run output collision: {run_dir}")
        seen_run_dirs.add(run_dir_key)
        result = _generate(_generation_namespace(config, run_config, run_dir))
        if result != 0:
            return result
        run_dirs.append(run_dir)

    if bool(sweep.get("compare_runs", False)):
        review_out = Path(sweep.get("review_out") or (base_out / "run_reviews"))
        return _compare_runs(run_dirs, review_out)
    return 0


def _run_pipeline_evaluate(config: dict) -> int:
    return _evaluate(
        _pipeline_model(config),
        _pipeline_generated_images(config),
        Path(config["evaluation_out"]),
        str(config.get("crop", "center")),
        str(config.get("evaluation_backend", config.get("vector_backend", "deterministic"))),
    )


def _run_pipeline_style_evaluate(config: dict) -> int:
    style = _pipeline_style_evaluation_config(config)
    generated_images = _pipeline_generated_images(config)
    out = Path(style["out"])
    result = _style_evaluate(
        argparse.Namespace(
            model=_pipeline_model(config),
            images=Path(style.get("images") or generated_images),
            out=out,
            clip_model=str(style.get("clip_model", "ViT-B-32")),
            pretrained=str(style.get("pretrained", "laion2b_s34b_b79k")),
            device=str(style.get("device", "auto")),
        )
    )
    _mirror_style_output_for_review(out, generated_images)
    return result


def _run_pipeline_review_generated(config: dict) -> int:
    return _review_generated(
        argparse.Namespace(
            model=_pipeline_model(config),
            images=_pipeline_generated_images(config),
            out=Path(config["review_out"]),
            crop=str(config.get("crop", "center")),
            backend=str(config.get("review_backend", config.get("vector_backend", "deterministic"))),
        )
    )


def _run_pipeline_review_subjects(config: dict) -> int:
    return _review_subjects(
        _pipeline_model(config),
        Path(config["subjects"]),
        Path(config.get("subject_review_out") or config["subject_out"]),
        str(config.get("crop", "center")),
        str(config.get("subject_backend", config.get("vector_backend", "deterministic"))),
    )


def _run_pipeline_compare_backends(config: dict) -> int:
    comparison = _pipeline_backend_comparison_config(config)
    return _compare_backends(
        argparse.Namespace(
            reference_images=Path(comparison.get("reference_images") or config["reference_images"]),
            images=_pipeline_generated_images(config),
            out=Path(comparison["out"]),
            backends=[str(name) for name in comparison.get("backends", ["deterministic", "opencv-face"])],
            crop=str(comparison.get("crop", config.get("crop", "center"))),
        )
    )


def _run_pipeline_compare_subject_backends(config: dict) -> int:
    comparison = _pipeline_subject_backend_comparison_config(config)
    return _compare_subject_backends(
        argparse.Namespace(
            reference_images=Path(comparison.get("reference_images") or config["reference_images"]),
            subjects=Path(comparison.get("subjects") or config["subjects"]),
            out=Path(comparison["out"]),
            backends=[str(name) for name in comparison.get("backends", ["deterministic", "opencv-face"])],
            crop=str(comparison.get("crop", config.get("crop", "center"))),
        )
    )


def _run_pipeline_explore_batch(config: dict) -> int:
    engagement = _pipeline_sns_engagement_config(config)
    return _explore_batch(
        argparse.Namespace(
            handles=Path(engagement["handles"]),
            out=Path(engagement["out"]),
            cache=Path(engagement["cache"]) if engagement.get("cache") else None,
            remote_host=engagement.get("remote_host"),
            platforms=[str(platform) for platform in engagement.get("platforms", ["instagram", "twitter", "tiktok"])],
            delay=float(engagement.get("delay", engagement.get("delay_between", 1.5))),
            force=bool(engagement.get("force", False)),
        )
    )


def _run_pipeline_analyze_correlation(config: dict) -> int:
    correlation = _pipeline_correlation_config(config)
    return _analyze_correlation(
        argparse.Namespace(
            face_scores=Path(correlation["face_scores"]),
            engagement=Path(correlation["engagement"]),
            out=Path(correlation["out"]),
        )
    )


def _run_pipeline_precision_report(config: dict) -> int:
    return _precision_report(
        argparse.Namespace(
            model=_pipeline_model(config),
            out=Path(config["precision_out"]),
            generation_review=_pipeline_generation_review_out(config),
            subject_review=(
                Path(config.get("subject_review_out") or config["subject_out"])
                if config.get("subject_review_out") or config.get("subject_out")
                else None
            ),
            evaluation=Path(config["evaluation_out"]) if config.get("evaluation_out") else None,
            quality=Path(config["quality_out"]) if config.get("quality_out") else None,
            backend_comparison=_pipeline_backend_comparison_out(config),
            subject_backend_comparison=_pipeline_subject_backend_comparison_out(config),
            correlation=_pipeline_correlation_out(config),
            model_audit=_pipeline_model_audit_out_or_none(config),
            vector_export=_pipeline_vector_export_out(config),
            face_ingredients=_pipeline_ingredients_report_out_or_none(config),
            benchmark_research=_pipeline_benchmark_research_out_or_none(config),
        )
    )


def _pipeline_generation_config(config: dict) -> dict:
    return config["generation"] if isinstance(config.get("generation"), dict) else config


def _pipeline_generation_sweep_config(config: dict) -> dict:
    sweep = config.get("generation_sweep")
    return sweep if isinstance(sweep, dict) else {}


def _merge_generation_sweep_run(sweep: dict, run: dict) -> dict:
    ignored = {"runs", "out", "review_out", "compare_runs"}
    merged = {key: value for key, value in sweep.items() if key not in ignored}
    merged.update(run)
    return merged


def _generation_sweep_run_dir(base_out: Path, run: dict, index: int) -> Path:
    if run.get("out"):
        return Path(run["out"])
    name = str(run.get("name") or f"run_{index:02d}")
    safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name)
    return base_out / safe_name


def _generation_run_is_reviewable(generation: dict) -> bool:
    return (
        str(generation.get("provider", "dry-run")) != "dry-run"
        and not bool(generation.get("dry_run", False))
        and bool(generation.get("review", False))
    )


def _generation_namespace(config: dict, generation: dict, out: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model=_pipeline_model(config),
        out=out,
        provider=str(generation.get("provider", "dry-run")),
        hf_model=str(generation.get("hf_model", "runwayml/stable-diffusion-v1-5")),
        image_model=str(generation.get("image_model", "")) or None,
        count=int(generation.get("count", 4)),
        seed=int(generation.get("seed", 150315)),
        steps=int(generation.get("steps", 30)),
        guidance_scale=float(generation.get("guidance_scale", 7.0)),
        width=int(generation.get("width", 512)),
        height=int(generation.get("height", 512)),
        device=str(generation.get("device", "cuda")),
        dtype=str(generation.get("dtype", "float16")),
        output_format=str(generation.get("output_format", "png")),
        quality=str(generation.get("quality", "medium")),
        centroid_kind=str(generation.get("centroid_kind", "median")),
        variant=str(generation.get("variant", "auto")),
        prompt_profile=str(generation.get("prompt_profile", "balanced")),
        prompt=generation.get("prompt"),
        negative_prompt=generation.get("negative_prompt"),
        dry_run=bool(generation.get("dry_run", False)),
        review=bool(generation.get("review", False)),
        review_out=Path(generation["review_out"]) if generation.get("review_out") else None,
        review_crop=str(generation.get("review_crop", "center")),
        review_backend=str(generation.get("review_backend", "deterministic")),
    )


def _pipeline_generated_images(config: dict) -> Path:
    generation = _pipeline_generation_config(config)
    value = generation.get("out") or config.get("generated_images")
    if not value:
        raise SystemExit("Pipeline config requires generated_images or generation.out")
    return Path(value)


def _pipeline_generation_review_out(config: dict) -> Path | None:
    if config.get("review_out"):
        return Path(config["review_out"])
    sweep = _pipeline_generation_sweep_config(config)
    if sweep.get("review_out"):
        return Path(sweep["review_out"])
    if sweep.get("compare_runs"):
        return Path(sweep["out"]) / "run_reviews"
    generation = _pipeline_generation_config(config)
    if generation.get("review_out"):
        return Path(generation["review_out"])
    if generation.get("review"):
        return _pipeline_generated_images(config) / "run_review"
    return None


def _pipeline_model_audit_out(config: dict) -> Path:
    audit = config.get("model_audit")
    if isinstance(audit, dict) and audit.get("out"):
        return Path(audit["out"])
    if config.get("model_audit_out"):
        return Path(config["model_audit_out"])
    if config.get("audit_out"):
        return Path(config["audit_out"])
    if isinstance(audit, dict):
        return _pipeline_model(config) / "audit"
    raise SystemExit("Pipeline config requires model_audit.out or model_audit_out")


def _pipeline_model_audit_out_or_none(config: dict) -> Path | None:
    audit = config.get("model_audit")
    if isinstance(audit, dict) or config.get("model_audit_out") or config.get("audit_out"):
        return _pipeline_model_audit_out(config)
    return None


def _pipeline_vector_export_config(config: dict) -> dict:
    export = config.get("vector_export")
    if isinstance(export, dict):
        merged = dict(export)
        if "out" not in merged and config.get("vector_export_out"):
            merged["out"] = config["vector_export_out"]
        if "out" not in merged:
            merged["out"] = str(_pipeline_default_vector_export_path(config, merged))
        _fill_vector_export_format_from_path(merged)
        return merged
    if config.get("vector_export_out"):
        merged = {"out": config["vector_export_out"]}
        _fill_vector_export_format_from_path(merged)
        return merged
    return {}


def _pipeline_vector_export_out(config: dict) -> Path | None:
    export = _pipeline_vector_export_config(config)
    if export.get("out"):
        return Path(export["out"])
    return None


def _pipeline_ingredients_report_out(config: dict) -> Path:
    ingredients = config.get("ingredients_report")
    if isinstance(ingredients, dict) and ingredients.get("out"):
        return Path(ingredients["out"])
    if config.get("face_ingredients_out"):
        return Path(config["face_ingredients_out"])
    if config.get("ingredients_out"):
        return Path(config["ingredients_out"])
    if isinstance(ingredients, dict):
        return _pipeline_model(config) / "face_ingredients"
    raise SystemExit("Pipeline config requires ingredients_report.out or face_ingredients_out")


def _pipeline_ingredients_report_out_or_none(config: dict) -> Path | None:
    if (
        isinstance(config.get("ingredients_report"), dict)
        or config.get("face_ingredients_out")
        or config.get("ingredients_out")
    ):
        return _pipeline_ingredients_report_out(config)
    return None


def _pipeline_benchmark_research_out(config: dict) -> Path:
    research = config.get("benchmark_research")
    if isinstance(research, dict) and research.get("out"):
        return Path(research["out"])
    if config.get("benchmark_research_out"):
        return Path(config["benchmark_research_out"])
    if isinstance(research, dict):
        return Path("outputs") / "benchmark_research"
    raise SystemExit("Pipeline config requires benchmark_research.out or benchmark_research_out")


def _pipeline_benchmark_research_out_or_none(config: dict) -> Path | None:
    if isinstance(config.get("benchmark_research"), dict) or config.get("benchmark_research_out"):
        return _pipeline_benchmark_research_out(config)
    return None


def _pipeline_default_vector_export_path(config: dict, export: dict) -> Path:
    output_format = str(export.get("format", "json")).lower()
    suffix = "csv" if output_format == "csv" else "json"
    return _pipeline_model(config) / f"vectors.{suffix}"


def _fill_vector_export_format_from_path(export: dict) -> None:
    if export.get("format") or not export.get("out"):
        return
    if Path(str(export["out"])).suffix.lower() == ".csv":
        export["format"] = "csv"


def _pipeline_style_evaluation_config(config: dict) -> dict:
    style = config.get("style_evaluation")
    if isinstance(style, dict):
        merged = dict(style)
        if "out" not in merged and config.get("style_evaluation_out"):
            merged["out"] = config["style_evaluation_out"]
        if "out" not in merged:
            merged["out"] = str(_pipeline_generated_images(config) / "style_evaluation")
        return merged
    if config.get("style_evaluation_out"):
        return {"out": config["style_evaluation_out"]}
    return {}


def _mirror_style_output_for_review(out: Path, generated_images: Path) -> None:
    review_style_dir = generated_images / "style_evaluation"
    if out.resolve() == review_style_dir.resolve():
        return
    review_style_dir.mkdir(parents=True, exist_ok=True)
    for name in ("style_scores.csv", "style_summary.json", "style_evaluation.md"):
        source = out / name
        if source.exists():
            shutil.copy2(source, review_style_dir / name)


def _pipeline_model(config: dict) -> Path:
    value = config.get("model_out") or config.get("model")
    if not value:
        raise SystemExit("Pipeline config requires model_out or model")
    return Path(value)


def _pipeline_backend_comparison_config(config: dict) -> dict:
    comparison = config.get("backend_comparison")
    if isinstance(comparison, dict):
        return comparison
    if config.get("backend_comparison_out"):
        return {"out": config["backend_comparison_out"]}
    return {}


def _pipeline_backend_comparison_out(config: dict) -> Path | None:
    comparison = _pipeline_backend_comparison_config(config)
    if comparison.get("out"):
        return Path(comparison["out"])
    return None


def _pipeline_subject_backend_comparison_config(config: dict) -> dict:
    comparison = config.get("subject_backend_comparison")
    if isinstance(comparison, dict):
        merged = dict(comparison)
        if "out" not in merged and config.get("subject_backend_comparison_out"):
            merged["out"] = config["subject_backend_comparison_out"]
        return merged
    if config.get("subject_backend_comparison_out"):
        return {"out": config["subject_backend_comparison_out"]}
    return {}


def _pipeline_subject_backend_comparison_out(config: dict) -> Path | None:
    comparison = _pipeline_subject_backend_comparison_config(config)
    if comparison.get("out"):
        return Path(comparison["out"])
    return None


def _pipeline_sns_engagement_config(config: dict) -> dict:
    engagement = config.get("sns_engagement")
    if isinstance(engagement, dict):
        merged = dict(engagement)
    else:
        merged = {}
    if "handles" not in merged and config.get("sns_handles"):
        merged["handles"] = config["sns_handles"]
    if "out" not in merged and config.get("sns_engagement_out"):
        merged["out"] = config["sns_engagement_out"]
    if "cache" not in merged and config.get("sns_cache"):
        merged["cache"] = config["sns_cache"]
    if "platforms" not in merged and config.get("sns_platforms"):
        merged["platforms"] = config["sns_platforms"]
    if not merged.get("handles") or not merged.get("out"):
        raise SystemExit("Pipeline config requires sns_engagement.handles and sns_engagement.out")
    return merged


def _pipeline_correlation_config(config: dict) -> dict:
    correlation = config.get("correlation")
    if isinstance(correlation, dict):
        merged = dict(correlation)
    else:
        merged = {}
    if "face_scores" not in merged and config.get("correlation_face_scores"):
        merged["face_scores"] = config["correlation_face_scores"]
    if "face_scores" not in merged:
        subject_review_out = config.get("subject_review_out") or config.get("subject_out")
        if subject_review_out:
            merged["face_scores"] = str(Path(subject_review_out) / "subject_reviews.json")
    if "engagement" not in merged and config.get("correlation_engagement"):
        merged["engagement"] = config["correlation_engagement"]
    if "engagement" not in merged:
        sns_engagement = config.get("sns_engagement")
        if isinstance(sns_engagement, dict) and sns_engagement.get("out"):
            merged["engagement"] = sns_engagement["out"]
        elif config.get("sns_engagement_out"):
            merged["engagement"] = config["sns_engagement_out"]
    if "out" not in merged and config.get("correlation_out"):
        merged["out"] = config["correlation_out"]
    if not merged.get("face_scores") or not merged.get("engagement") or not merged.get("out"):
        raise SystemExit("Pipeline config requires correlation face_scores, engagement, and out")
    return merged


def _pipeline_correlation_out(config: dict) -> Path | None:
    correlation = config.get("correlation")
    if isinstance(correlation, dict) and correlation.get("out"):
        return Path(correlation["out"])
    if config.get("correlation_out"):
        return Path(config["correlation_out"])
    return None


def _qa_images(images: Path, out: Path) -> int:
    reviews = review_image_quality(images)
    write_image_quality(reviews, out)
    pass_count = sum(1 for review in reviews if review.qa_pass)
    print(f"reviewed images: {len(reviews)}")
    print(f"qa pass: {pass_count}")
    print(f"qa report: {out / 'image_quality.csv'}")
    return 0


def _review_generated(args: argparse.Namespace) -> int:
    review_out = args.out or (args.images / "run_review")
    evaluation_out = args.images / "evaluation"
    quality_out = args.images / "quality"
    _evaluate(args.model, args.images, evaluation_out, args.crop, args.backend)
    _qa_images(args.images, quality_out)
    _compare_runs([args.images], review_out)
    print(f"generated review: {review_out / 'generation_run_reviews.csv'}")
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


def _vectorize_subjects(subjects: Path, out: Path, crop: str, backend_name: str, workers: int) -> int:
    if not subjects.is_dir():
        raise SystemExit(f"No subject directory found: {subjects}")
    backend = get_vector_backend(backend_name)
    subject_vectors = vectorize_subjects(subjects, backend, crop=crop, workers=max(1, workers))
    manifest = write_subject_vectors(subject_vectors, out)
    print(f"subjects: {manifest['subject_count']}")
    print(
        f"vectorized: {manifest['ok_count']} "
        f"(empty {manifest['empty_count']}, failed {manifest['failed_count']})"
    )
    print(f"manifest: {out / 'manifest.json'}")
    return 0


def _backend_diagnostics(out: Path) -> int:
    report = write_backend_diagnostics(out)
    implemented = sum(1 for backend in report["backends"] if backend["implemented"])
    cuda_available = report["runtime"]["torch"]["cuda_available"]
    print(f"backend diagnostics: {out / 'backend_diagnostics.md'}")
    print(f"implemented backends: {implemented}")
    print(f"torch cuda available: {cuda_available}")
    return 0


def _benchmark_research(out: Path) -> int:
    report = write_benchmark_research(out)
    print(f"benchmark research: {out / 'benchmark_research.md'}")
    print(f"sources: {len(report['sources'])}")
    print(f"primary face embedding: {report['vectorization_strategy']['primary_face_embedding']}")
    return 0


def _review_agencies(args: argparse.Namespace) -> int:
    report = write_agency_average_params(args.model, args.agencies, args.out)
    print(f"agency average params: {args.out / 'agency_average_params.md'}")
    print(f"agencies: {len(report['agencies'])}")
    top = report["rankings"]["by_descriptor_similarity"][0] if report["agencies"] else {}
    print(f"top descriptor match: {top.get('name')} {top.get('descriptor_similarity')}")
    return 0


def _enhance_agencies(args: argparse.Namespace) -> int:
    report = write_agency_enhancement_bundle(
        model_dir=args.model,
        agencies_config=args.agencies,
        images=args.images,
        out_dir=args.out,
        crop=args.crop,
        backend_name=args.backend,
    )
    print(f"agency enhancement report: {args.out / 'agency_enhancement_report.md'}")
    print(f"agencies: {len(report['agencies'])}")
    print(f"top enhanced match: {report['summary'].get('top_slug')} {report['summary'].get('top_score')}")
    return 0


def _calibrate_agency_generation(args: argparse.Namespace) -> int:
    report = write_generation_calibration(
        enhancement_report=args.enhancement,
        agency_params=args.agency_params,
        out_dir=args.out,
        target_image_score=args.target_image_score,
        target_axis_alignment=args.target_axis_alignment,
        target_enhancement_score=args.target_enhancement_score,
        seed_start=args.seed_start,
        variants_per_agency=args.variants_per_agency,
    )
    print(f"agency generation calibration: {args.out / 'generation_calibration.md'}")
    print(f"agencies: {len(report['agencies'])}")
    print(f"regenerate first: {', '.join(report['summary']['regenerate_first'])}")
    return 0


def _worker_diagnostics(out: Path, include_remote: bool, timeout_seconds: int) -> int:
    workers = DEFAULT_DIAGNOSTIC_WORKERS if include_remote else [LOCAL_4090]
    report = write_worker_diagnostics(out, workers=workers, timeout_seconds=timeout_seconds)
    ready_count = sum(1 for worker in report["workers"] if worker["ok"])
    print(f"worker diagnostics: {out / 'worker_diagnostics.md'}")
    print(f"workers ready: {ready_count}/{report['worker_count']}")
    return 0


def _compare_backends(args: argparse.Namespace) -> int:
    report = compare_vector_backends(
        reference_images=args.reference_images,
        images=args.images,
        out_dir=args.out,
        backend_names=args.backends,
        crop=args.crop,
    )
    completed = sum(1 for run in report["runs"] if run["status"] == "completed")
    failed = sum(1 for run in report["runs"] if run["status"] == "failed")
    print(f"backend comparison: {args.out / 'backend_comparison.md'}")
    print(f"completed backends: {completed}")
    print(f"failed backends: {failed}")
    return 1 if completed == 0 else 0


def _compare_subject_backends(args: argparse.Namespace) -> int:
    report = compare_subject_backends(
        reference_images=args.reference_images,
        subjects=args.subjects,
        out_dir=args.out,
        backend_names=args.backends,
        crop=args.crop,
    )
    completed = sum(1 for run in report["runs"] if run["status"] == "completed")
    failed = sum(1 for run in report["runs"] if run["status"] == "failed")
    print(f"subject backend comparison: {args.out / 'subject_backend_comparison.md'}")
    print(f"completed backends: {completed}")
    print(f"failed backends: {failed}")
    return 1 if completed == 0 else 0


def _compare_deepface_detectors(args: argparse.Namespace) -> int:
    report = compare_deepface_detectors(
        reference_images=args.reference_images,
        images=args.images,
        out_dir=args.out,
        detector_backends=args.detectors,
        model_name=args.model_name,
        crop=args.crop,
        reuse_existing=args.reuse_existing,
        max_reference_images=args.max_reference_images,
        max_images=args.max_images,
    )
    completed = sum(1 for run in report["runs"] if run["status"] == "completed")
    failed = sum(1 for run in report["runs"] if run["status"] == "failed")
    print(f"DeepFace detector comparison: {args.out / 'deepface_detector_comparison.md'}")
    print(f"completed detectors: {completed}")
    print(f"failed detectors: {failed}")
    return 1 if completed == 0 else 0


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


def _explore_profile(args: argparse.Namespace) -> int:
    from .sns_explorer import build_router
    router = build_router(cache_path=args.cache, remote_host=args.remote_host)
    p = router.fetch(args.platform, args.handle, force=args.force)
    print(f"platform:      {p.platform}")
    print(f"handle:        @{p.handle}")
    print(f"display_name:  {p.display_name}")
    print(f"followers:     {p.followers:,}" if p.followers else "followers:     N/A")
    print(f"following:     {p.following}")
    print(f"posts:         {p.posts}")
    print(f"avg_likes:     {p.avg_likes}")
    print(f"avg_comments:  {p.avg_comments}")
    print(f"engagement_rate: {p.engagement_rate}")
    print(f"source:        {p.source}")
    print(f"status:        {p.fetch_status}")
    if p.fetch_error:
        print(f"error:         {p.fetch_error}")
    return 0 if p.fetch_status in ("ok", "partial") else 1


def _explore_batch(args: argparse.Namespace) -> int:
    from .sns_metrics import SnsEngagement, TalentEngagementRecord, write_engagement_manifest
    from .sns_explorer import build_router

    handles_path: Path = args.handles
    records_raw = []
    for line in handles_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records_raw.append(json.loads(line))

    router = build_router(cache_path=args.cache, remote_host=args.remote_host)

    # Build (platform, handle, slug) list
    items = []
    for rec in records_raw:
        slug = rec["talent_slug"]
        for plat, handle in rec.get("sns_handles", {}).items():
            if plat in args.platforms:
                items.append((plat, handle, slug))

    pair_items = [(plat, handle) for plat, handle, _ in items]
    profiles = router.fetch_batch(pair_items, delay_between=args.delay, force=args.force)

    # Reassemble into TalentEngagementRecord structure
    slug_to_name = {r["talent_slug"]: r.get("name") for r in records_raw}
    slug_engs: dict[str, list[SnsEngagement]] = {}
    for (_plat, _handle, slug), profile in zip(items, profiles):
        slug_engs.setdefault(slug, []).append(SnsEngagement(**profile.to_engagement_dict()))

    out_records: list[TalentEngagementRecord] = []
    for rec in records_raw:
        slug = rec["talent_slug"]
        engs = slug_engs.get(slug, [])
        if engs:
            out_records.append(
                TalentEngagementRecord(
                    talent_slug=slug,
                    name=slug_to_name.get(slug),
                    engagements=engs,
                )
            )
    write_engagement_manifest(out_records, args.out)

    ok = sum(1 for record in out_records for engagement in record.engagements if engagement.followers)
    total = sum(len(engs) for engs in slug_engs.values())
    print(f"talents: {len(slug_engs)}")
    print(f"engagements: {total}  (with followers: {ok})")
    print(f"output: {args.out}")
    return 0


def _explore_load_cache(args: argparse.Namespace) -> int:
    from .sns_explorer import SnsProfile, SnsStore

    store = SnsStore(args.cache)
    loaded = 0

    # Load from engagement JSONL (any platform)
    if args.engagement and args.engagement.exists():
        for line in args.engagement.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for eng in rec.get("engagements", []):
                fol = eng.get("followers")
                if fol is None:
                    continue
                p = SnsProfile(
                    platform=eng["platform"], handle=eng["handle"],
                    profile_url=eng.get("profile_url", ""),
                    followers=fol, following=eng.get("following"),
                    posts=eng.get("posts"),
                    total_engagement=eng.get("total_engagement"),
                    engagement_rate=eng.get("engagement_rate"),
                    bio=eng.get("bio"), display_name=eng.get("display_name"),
                    source="loaded_from_jsonl",
                    fetch_status=eng.get("fetch_status", "ok"),
                    retrieved_at=eng.get("retrieved_at", ""),
                )
                store.put(p)
                loaded += 1

    # Load Instagram from an external batch result JSON.
    if args.ig_json and args.ig_json.exists():
        ig_data = json.loads(args.ig_json.read_text(encoding="utf-8"))
        handle_by_slug = _instagram_handles_by_slug(args.handles) if args.handles else {}
        for key, v in ig_data.items():
            handle = v.get("handle") or handle_by_slug.get(key) or key
            fol = v.get("followers")
            if not handle or fol is None:
                continue
            p = SnsProfile(
                platform="instagram", handle=handle,
                profile_url=f"https://www.instagram.com/{handle}/",
                followers=fol, source="loaded_from_ig_json",
                fetch_status="ok",
            )
            store.put(p)
            loaded += 1

    store.close()
    print(f"Loaded {loaded} profiles into cache: {args.cache}")
    return 0


def _instagram_handles_by_slug(handles_path: Path) -> dict[str, str]:
    handle_by_slug: dict[str, str] = {}
    for line in handles_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        handle = record.get("sns_handles", {}).get("instagram")
        if handle:
            handle_by_slug[str(record["talent_slug"])] = str(handle)
    return handle_by_slug


def _explore_discover(args: argparse.Namespace) -> int:
    from .sns_explorer import discover_handles_for_talent
    results = discover_handles_for_talent(args.name, platforms=args.platforms)
    for platform, handles in results.items():
        print(f"\n{platform}:")
        if not handles:
            print("  (no results)")
        for h in handles:
            fol = f"{h.followers:,}" if h.followers else "?"
            print(f"  @{h.handle} [{h.display_name}] followers={fol} score={h.relevance_score:.2f}")
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
