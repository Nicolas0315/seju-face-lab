from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .backends import backend_help, get_vector_backend
from .embeddings import iter_image_paths, render_appearance
from .metrics import score_generated_images, write_scores
from .model import build_centroid_model, load_model, save_model
from .prompting import prompt_from_descriptors
from .sources import discover_sources, write_source_manifest


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

    args = parser.parse_args(argv)
    if args.command == "build":
        return _build(args.images, args.out, args.crop, args.backend)
    if args.command == "prompt":
        return _prompt(args.model, args.kind)
    if args.command == "render":
        return _render(args.model, args.kind, args.out)
    if args.command == "evaluate":
        return _evaluate(args.model, args.images, args.out, args.crop, args.backend)
    if args.command == "backends":
        print(backend_help())
        return 0
    if args.command == "sources" and args.sources_command == "discover":
        return _sources_discover(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def _build(images: Path, out: Path, crop: str, backend_name: str) -> int:
    paths = iter_image_paths(images)
    if not paths:
        raise SystemExit(f"No supported images found under {images}")

    backend = get_vector_backend(backend_name)
    vectors = [backend.vectorize(path, crop=crop) for path in paths]
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

    print(f"built model: {out}")
    print(f"images: {len(paths)}")
    print(f"backend: {backend.name}")
    print(f"embedding_dim: {model.embedding_dim}")
    print(f"prompt: {out / 'prompt.txt'}")
    return 0


def _prompt(model_dir: Path, kind: str) -> int:
    model = load_model(model_dir)
    print(prompt_from_descriptors(model.descriptors[kind]))
    return 0


def _render(model_dir: Path, kind: str, out: Path) -> int:
    model = load_model(model_dir)
    appearance = model.mean_appearance if kind == "mean" else model.median_appearance
    render_appearance(appearance, out)
    print(f"rendered: {out}")
    return 0


def _evaluate(model_dir: Path, images: Path, out: Path, crop: str, backend_name: str) -> int:
    model = load_model(model_dir)
    backend = get_vector_backend(backend_name)
    scores = score_generated_images(model, images, crop=crop, backend=backend)
    write_scores(scores, out)
    print(f"evaluated images: {len(scores)}")
    print(f"scores: {out / 'scores.csv'}")
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
