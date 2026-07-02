from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import backends
from .embeddings import iter_image_paths
from .metrics import (
    Score,
    SubjectReview,
    _score_vector,
    review_subject_directories,
    score_generated_images,
    write_scores,
    write_subject_reviews,
)
from .model import build_centroid_model, save_model


@dataclass(frozen=True)
class BackendRun:
    backend: str
    status: str
    model_dir: str | None
    evaluation_dir: str | None
    reference_count: int
    reference_failed_count: int
    image_count: int
    image_failed_count: int
    embedding_dim: int | None
    best_image_id: str | None
    best_centroid_score: float | None
    mean_centroid_score: float | None
    error: str | None = None


@dataclass(frozen=True)
class SubjectBackendRun:
    backend: str
    status: str
    model_dir: str | None
    subject_review_dir: str | None
    reference_count: int
    reference_failed_count: int
    subject_count: int
    embedding_dim: int | None
    top_subject: str | None
    top_subject_mean_score: float | None
    error: str | None = None


def compare_vector_backends(
    reference_images: Path,
    images: Path,
    out_dir: Path,
    backend_names: list[str],
    crop: str = "center",
) -> dict[str, Any]:
    _validate_backend_names(backend_names)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [_run_backend(reference_images, images, out_dir, name, crop) for name in backend_names]
    tables = {
        run.backend: _read_scores(Path(run.evaluation_dir) / "scores.csv")
        for run in runs
        if run.status == "completed" and run.evaluation_dir
    }
    report = {
        "reference_images": str(reference_images),
        "images": str(images),
        "crop": crop,
        "runs": [asdict(run) for run in runs],
        "rank_agreement": _rank_agreement(tables),
        "boundary": (
            "Backends are compared by same-image ranking only. Embedding dimensions and "
            "score scales are backend-specific."
        ),
    }
    (out_dir / "backend_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "backend_comparison.md").write_text(_render(report), encoding="utf-8")
    return report


def compare_subject_backends(
    reference_images: Path,
    subjects: Path,
    out_dir: Path,
    backend_names: list[str],
    crop: str = "center",
) -> dict[str, Any]:
    _validate_backend_names(backend_names)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [_run_subject_backend(reference_images, subjects, out_dir, name, crop) for name in backend_names]
    tables = {
        run.backend: _read_subject_scores(Path(run.subject_review_dir) / "subject_reviews.json")
        for run in runs
        if run.status == "completed" and run.subject_review_dir
    }
    report = {
        "reference_images": str(reference_images),
        "subjects": str(subjects),
        "crop": crop,
        "runs": [asdict(run) for run in runs],
        "rank_agreement": _subject_rank_agreement(tables),
        "boundary": (
            "Subject backends are compared by same-subject ranking only. Scores are local "
            "centroid similarities, not identity, attractiveness, ethnicity, or objective labels."
        ),
    }
    (out_dir / "subject_backend_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "subject_backend_comparison.md").write_text(_render_subject_backend_report(report), encoding="utf-8")
    return report


def compare_deepface_detectors(
    reference_images: Path,
    images: Path,
    out_dir: Path,
    detector_backends: list[str],
    model_name: str = "ArcFace",
    crop: str = "center",
    reuse_existing: bool = False,
    max_reference_images: int | None = None,
    max_images: int | None = None,
) -> dict[str, Any]:
    _validate_detector_names(detector_backends)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_paths = _limited_image_paths(reference_images, max_reference_images)
    image_paths = _limited_image_paths(images, max_images)
    runs = []
    for detector in detector_backends:
        existing = (
            _read_existing_detector_run(
                out_dir,
                detector,
                reference_images,
                images,
                model_name,
                crop,
                max_reference_images,
                max_images,
            )
            if reuse_existing
            else None
        )
        runs.append(
            existing
            or _run_deepface_detector(
                reference_images,
                images,
                out_dir,
                detector,
                model_name,
                crop,
                reference_paths,
                image_paths,
                max_reference_images,
                max_images,
            )
        )
    tables = {
        run.backend: _read_scores(Path(run.evaluation_dir) / "scores.csv")
        for run in runs
        if run.status == "completed" and run.evaluation_dir
    }
    report = {
        "reference_images": str(reference_images),
        "images": str(images),
        "crop": crop,
        "model_name": model_name,
        "detector_backends": detector_backends,
        "max_reference_images": max_reference_images,
        "max_images": max_images,
        "runs": [asdict(run) for run in runs],
        "rank_agreement": _rank_agreement(tables),
        "boundary": (
            "DeepFace detectors are compared by acceptance counts and same-image ranking only. "
            "Detector coverage and score scales are model/backend-specific."
        ),
    }
    (out_dir / "deepface_detector_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "deepface_detector_comparison.md").write_text(_render_detector_report(report), encoding="utf-8")
    return report


def _validate_backend_names(backend_names: list[str]) -> None:
    unknown = sorted({name for name in backend_names if name not in backends.BACKENDS})
    if unknown:
        choices = ", ".join(sorted(backends.BACKENDS))
        raise ValueError(f"Unknown backend(s): {', '.join(unknown)}. Choices: {choices}")


def _validate_detector_names(detector_backends: list[str]) -> None:
    if not detector_backends:
        raise ValueError("At least one DeepFace detector backend is required")
    invalid = sorted({name for name in detector_backends if not name or any(char in name for char in "/\\:")})
    if invalid:
        raise ValueError(f"Invalid DeepFace detector backend(s): {', '.join(invalid)}")


def _run_backend(reference_images: Path, images: Path, out_dir: Path, backend_name: str, crop: str) -> BackendRun:
    model_dir = out_dir / backend_name / "model"
    evaluation_dir = out_dir / backend_name / "evaluation"
    try:
        backend = backends.get_vector_backend(backend_name)
        vectors, failures = [], []
        for path in iter_image_paths(reference_images):
            try:
                vectors.append(backend.vectorize(path, crop=crop))
            except Exception as exc:  # noqa: BLE001 - compare should continue through noisy folders.
                failures.append({"path": str(path), "reason": str(exc)})
        if not vectors:
            raise RuntimeError(f"No usable reference images for backend '{backend_name}'")
        model = build_centroid_model(
            image_ids=[vector.image_id for vector in vectors],
            source_paths=[str(vector.path) for vector in vectors],
            embeddings=np.stack([vector.embedding for vector in vectors]),
            appearances=np.stack([vector.appearance for vector in vectors]),
        )
        save_model(model, model_dir)
        _write_reference_failures(model_dir, failures)
        failed_paths: list[str] = []
        scores = score_generated_images(model, images, crop=crop, backend=backend, failed_paths=failed_paths)
        write_scores(scores, evaluation_dir, failed_paths=failed_paths, model=model)
        return _completed_run(
            backend_name, model_dir, evaluation_dir, len(vectors), len(failures), model.embedding_dim, scores, len(failed_paths)
        )
    except Exception as exc:  # noqa: BLE001 - optional backend failures are report data.
        failure_dir = out_dir / backend_name
        failure_dir.mkdir(parents=True, exist_ok=True)
        (failure_dir / "backend_failure.json").write_text(
            json.dumps({"backend": backend_name, "status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return BackendRun(backend_name, "failed", None, None, 0, 0, 0, 0, None, None, None, None, str(exc))


def _run_subject_backend(
    reference_images: Path,
    subjects: Path,
    out_dir: Path,
    backend_name: str,
    crop: str,
) -> SubjectBackendRun:
    model_dir = out_dir / backend_name / "model"
    subject_review_dir = out_dir / backend_name / "subject_review"
    try:
        backend = backends.get_vector_backend(backend_name)
        vectors, failures = [], []
        for path in iter_image_paths(reference_images):
            try:
                vectors.append(backend.vectorize(path, crop=crop))
            except Exception as exc:  # noqa: BLE001 - optional backend comparison should report noisy references.
                failures.append({"path": str(path), "reason": str(exc)})
        if not vectors:
            raise RuntimeError(f"No usable reference images for backend '{backend_name}'")
        model = build_centroid_model(
            image_ids=[vector.image_id for vector in vectors],
            source_paths=[str(vector.path) for vector in vectors],
            embeddings=np.stack([vector.embedding for vector in vectors]),
            appearances=np.stack([vector.appearance for vector in vectors]),
        )
        save_model(model, model_dir)
        _write_reference_failures(model_dir, failures)
        reviews = review_subject_directories(model, subjects, crop=crop, backend=backend)
        write_subject_reviews(reviews, subject_review_dir)
        return _completed_subject_run(
            backend_name,
            model_dir,
            subject_review_dir,
            len(vectors),
            len(failures),
            model.embedding_dim,
            reviews,
        )
    except Exception as exc:  # noqa: BLE001 - optional backend failures are report data.
        failure_dir = out_dir / backend_name
        failure_dir.mkdir(parents=True, exist_ok=True)
        (failure_dir / "subject_backend_failure.json").write_text(
            json.dumps({"backend": backend_name, "status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return SubjectBackendRun(backend_name, "failed", None, None, 0, 0, 0, None, None, None, str(exc))


def _run_deepface_detector(
    reference_images: Path,
    images: Path,
    out_dir: Path,
    detector_backend: str,
    model_name: str,
    crop: str,
    reference_paths: list[Path],
    image_paths: list[Path],
    max_reference_images: int | None,
    max_images: int | None,
) -> BackendRun:
    label = f"deepface-{detector_backend}"
    model_dir = out_dir / label / "model"
    evaluation_dir = out_dir / label / "evaluation"
    try:
        backend = backends.get_deepface_backend(
            model_name=model_name,
            detector_backend=detector_backend,
        )
        vectors, failures = [], []
        for path in reference_paths:
            try:
                vectors.append(backend.vectorize(path, crop=crop))
            except Exception as exc:  # noqa: BLE001 - detector comparison is an acceptance audit.
                failures.append({"path": str(path), "reason": str(exc)})
        if not vectors:
            raise RuntimeError(f"No usable reference images for DeepFace detector '{detector_backend}'")
        model = build_centroid_model(
            image_ids=[vector.image_id for vector in vectors],
            source_paths=[str(vector.path) for vector in vectors],
            embeddings=np.stack([vector.embedding for vector in vectors]),
            appearances=np.stack([vector.appearance for vector in vectors]),
        )
        save_model(model, model_dir)
        _write_reference_failures(model_dir, failures)
        failed_paths: list[str] = []
        scores = _score_image_paths(model, image_paths, crop=crop, backend=backend, failed_paths=failed_paths)
        write_scores(scores, evaluation_dir, failed_paths=failed_paths, model=model)
        _write_detector_run_config(
            out_dir,
            detector_backend,
            reference_images,
            images,
            model_name,
            crop,
            max_reference_images,
            max_images,
        )
        return _completed_run(
            label, model_dir, evaluation_dir, len(vectors), len(failures), model.embedding_dim, scores, len(failed_paths)
        )
    except Exception as exc:  # noqa: BLE001 - optional detector failures are report data.
        failure_dir = out_dir / label
        failure_dir.mkdir(parents=True, exist_ok=True)
        (failure_dir / "backend_failure.json").write_text(
            json.dumps(
                {"backend": label, "detector_backend": detector_backend, "status": "failed", "error": str(exc)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return BackendRun(label, "failed", None, None, 0, 0, 0, 0, None, None, None, None, str(exc))


def _read_existing_detector_run(
    out_dir: Path,
    detector_backend: str,
    reference_images: Path,
    images: Path,
    model_name: str,
    crop: str,
    max_reference_images: int | None,
    max_images: int | None,
) -> BackendRun | None:
    label = f"deepface-{detector_backend}"
    run_dir = out_dir / label
    model_dir = out_dir / label / "model"
    evaluation_dir = out_dir / label / "evaluation"
    config_path = run_dir / "detector_run.json"
    profile_path = model_dir / "profile.json"
    summary_path = evaluation_dir / "summary.json"
    if not config_path.exists() or not profile_path.exists() or not summary_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if _normalized_detector_run_config(config) != _detector_run_config(
        detector_backend,
        reference_images,
        images,
        model_name,
        crop,
        max_reference_images,
        max_images,
    ):
        return None
    reference_count = int(profile.get("image_count") or 0)
    if reference_count <= 0:
        return None
    return BackendRun(
        backend=label,
        status="completed",
        model_dir=str(model_dir),
        evaluation_dir=str(evaluation_dir),
        reference_count=reference_count,
        reference_failed_count=_read_reference_failed_count(model_dir),
        image_count=int(summary.get("image_count") or 0),
        image_failed_count=int(summary.get("failed_count") or 0),
        embedding_dim=int(profile["embedding_dim"]) if profile.get("embedding_dim") else None,
        best_image_id=summary.get("best_image_id"),
        best_centroid_score=summary.get("best_centroid_score"),
        mean_centroid_score=summary.get("mean_centroid_score"),
    )


def _write_detector_run_config(
    out_dir: Path,
    detector_backend: str,
    reference_images: Path,
    images: Path,
    model_name: str,
    crop: str,
    max_reference_images: int | None,
    max_images: int | None,
) -> None:
    label = f"deepface-{detector_backend}"
    run_dir = out_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "detector_run.json").write_text(
        json.dumps(
            _detector_run_config(
                detector_backend,
                reference_images,
                images,
                model_name,
                crop,
                max_reference_images,
                max_images,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _detector_run_config(
    detector_backend: str,
    reference_images: Path,
    images: Path,
    model_name: str,
    crop: str,
    max_reference_images: int | None,
    max_images: int | None,
) -> dict[str, str]:
    return {
        "detector_backend": detector_backend,
        "reference_images": str(reference_images),
        "images": str(images),
        "model_name": model_name,
        "crop": crop,
        "max_reference_images": "" if max_reference_images is None else str(max_reference_images),
        "max_images": "" if max_images is None else str(max_images),
    }


def _normalized_detector_run_config(config: dict[str, Any]) -> dict[str, str]:
    normalized = {str(key): str(value) for key, value in config.items()}
    normalized.setdefault("max_reference_images", "")
    normalized.setdefault("max_images", "")
    return normalized


def _limited_image_paths(root: Path, limit: int | None) -> list[Path]:
    paths = iter_image_paths(root)
    if limit is None:
        return paths
    if limit <= 0:
        raise ValueError("image limits must be positive")
    return paths[:limit]


def _score_image_paths(
    model: Any,
    image_paths: list[Path],
    crop: str,
    backend: backends.VectorBackend,
    failed_paths: list[str],
) -> list[Score]:
    scores = []
    for path in image_paths:
        try:
            scores.append(_score_vector(model, backend.vectorize(path, crop=crop)))
        except Exception:  # noqa: BLE001 - detector comparison should report per-image failures.
            failed_paths.append(str(path))
    return sorted(scores, key=lambda item: item.centroid_score, reverse=True)


def _read_reference_failed_count(model_dir: Path) -> int:
    failures_path = model_dir / "vectors" / "reference_vector_failures.json"
    if not failures_path.exists():
        return 0
    try:
        payload = json.loads(failures_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    return int(payload.get("failed_count") or 0)


def _completed_run(
    backend_name: str,
    model_dir: Path,
    evaluation_dir: Path,
    reference_count: int,
    reference_failed_count: int,
    embedding_dim: int,
    scores: list[Score],
    image_failed_count: int,
) -> BackendRun:
    centroid_scores = np.asarray([score.centroid_score for score in scores], dtype=np.float32)
    best = scores[0] if scores else None
    return BackendRun(
        backend_name,
        "completed",
        str(model_dir),
        str(evaluation_dir),
        reference_count,
        reference_failed_count,
        len(scores),
        image_failed_count,
        embedding_dim,
        best.image_id if best else None,
        round(float(best.centroid_score), 6) if best else None,
        round(float(np.mean(centroid_scores)), 6) if len(centroid_scores) else None,
    )


def _completed_subject_run(
    backend_name: str,
    model_dir: Path,
    subject_review_dir: Path,
    reference_count: int,
    reference_failed_count: int,
    embedding_dim: int,
    reviews: list[SubjectReview],
) -> SubjectBackendRun:
    top = reviews[0] if reviews else None
    return SubjectBackendRun(
        backend=backend_name,
        status="completed",
        model_dir=str(model_dir),
        subject_review_dir=str(subject_review_dir),
        reference_count=reference_count,
        reference_failed_count=reference_failed_count,
        subject_count=len(reviews),
        embedding_dim=embedding_dim,
        top_subject=top.subject if top else None,
        top_subject_mean_score=_round_optional(top.mean_centroid_score) if top else None,
    )


def _write_reference_failures(model_dir: Path, failures: list[dict[str, str]]) -> None:
    vector_dir = model_dir / "vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    (vector_dir / "reference_vector_failures.json").write_text(
        json.dumps({"failed_count": len(failures), "failures": failures}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_scores(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["path"]: float(row["centroid_score"])
            for row in csv.DictReader(handle)
            if row.get("path") and row.get("centroid_score")
        }


def _read_subject_scores(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    subjects = payload.get("subjects")
    if not isinstance(subjects, list):
        return {}
    return {
        str(row["subject"]): float(row["mean_centroid_score"])
        for row in subjects
        if isinstance(row, dict) and row.get("subject") and row.get("mean_centroid_score") is not None
    }


def _rank_agreement(tables: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    names = sorted(tables)
    rows = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            common = sorted(set(tables[left]) & set(tables[right]))
            rows.append(
                {
                    "backend_a": left,
                    "backend_b": right,
                    "common_image_count": len(common),
                    "spearman_rank": _spearman(
                        [tables[left][image_id] for image_id in common],
                        [tables[right][image_id] for image_id in common],
                    ),
                }
            )
    return rows


def _subject_rank_agreement(tables: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    return [
        {
            "backend_a": row["backend_a"],
            "backend_b": row["backend_b"],
            "common_subject_count": row["common_image_count"],
            "spearman_rank": row["spearman_rank"],
        }
        for row in _rank_agreement(tables)
    ]


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_rank, right_rank = _ranks(left), _ranks(right)
    left_rank -= float(np.mean(left_rank))
    right_rank -= float(np.mean(right_rank))
    denom = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
    return None if denom < 1e-12 else round(float(np.dot(left_rank, right_rank) / denom), 6)


def _ranks(values: list[float]) -> np.ndarray:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = np.zeros(len(values), dtype=np.float32)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


def _render(report: dict[str, Any]) -> str:
    lines = ["# backend comparison", "", "| backend | status | refs | images | failed | dim | best_image | best_score |"]
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |")
    for run in report["runs"]:
        lines.append(
            f"| {run['backend']} | {run['status']} | {run['reference_count']} | {run['image_count']} | "
            f"{run['image_failed_count']} | {run['embedding_dim'] or ''} | {run['best_image_id'] or ''} | "
            f"{_optional_float(run['best_centroid_score'])} |"
        )
    lines.extend(["", "## Rank Agreement", "", "| backend_a | backend_b | common_images | spearman_rank |"])
    lines.append("| --- | --- | ---: | ---: |")
    for item in report["rank_agreement"]:
        lines.append(
            f"| {item['backend_a']} | {item['backend_b']} | {item['common_image_count']} | "
            f"{_optional_float(item['spearman_rank'])} |"
        )
    lines.extend(["", report["boundary"], ""])
    return "\n".join(lines)


def _render_detector_report(report: dict[str, Any]) -> str:
    lines = [
        "# DeepFace detector comparison",
        "",
        "| detector | status | refs | ref_failed | images | image_failed | dim | best_image | best_score |",
    ]
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
    for run in report["runs"]:
        lines.append(
            f"| {run['backend']} | {run['status']} | {run['reference_count']} | "
            f"{run['reference_failed_count']} | {run['image_count']} | {run['image_failed_count']} | "
            f"{run['embedding_dim'] or ''} | {run['best_image_id'] or ''} | "
            f"{_optional_float(run['best_centroid_score'])} |"
        )
    lines.extend(["", "## Rank Agreement", "", "| detector_a | detector_b | common_images | spearman_rank |"])
    lines.append("| --- | --- | ---: | ---: |")
    for item in report["rank_agreement"]:
        lines.append(
            f"| {item['backend_a']} | {item['backend_b']} | {item['common_image_count']} | "
            f"{_optional_float(item['spearman_rank'])} |"
        )
    lines.extend(["", report["boundary"], ""])
    return "\n".join(lines)


def _render_subject_backend_report(report: dict[str, Any]) -> str:
    lines = [
        "# subject backend comparison",
        "",
        "| backend | status | refs | ref_failed | subjects | dim | top_subject | top_mean_score |",
    ]
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |")
    for run in report["runs"]:
        lines.append(
            f"| {run['backend']} | {run['status']} | {run['reference_count']} | "
            f"{run['reference_failed_count']} | {run['subject_count']} | {run['embedding_dim'] or ''} | "
            f"{run['top_subject'] or ''} | {_optional_float(run['top_subject_mean_score'])} |"
        )
    lines.extend(["", "## Rank Agreement", "", "| backend_a | backend_b | common_subjects | spearman_rank |"])
    lines.append("| --- | --- | ---: | ---: |")
    for item in report["rank_agreement"]:
        lines.append(
            f"| {item['backend_a']} | {item['backend_b']} | {item['common_subject_count']} | "
            f"{_optional_float(item['spearman_rank'])} |"
        )
    lines.extend(["", report["boundary"], ""])
    return "\n".join(lines)


def _optional_float(value: float | None) -> str:
    return "" if value is None else f"{float(value):.6f}"


def _round_optional(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)
