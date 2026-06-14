from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np


def write_precision_report(
    model_dir: Path,
    out_dir: Path,
    generation_review: Path | None = None,
    subject_review: Path | None = None,
    evaluation: Path | None = None,
    quality: Path | None = None,
    backend_comparison: Path | None = None,
) -> dict[str, Any]:
    """Write a compact review bundle for centroid, generation, QA, and subject evidence."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_precision_report(
        model_dir=model_dir,
        generation_review=generation_review,
        subject_review=subject_review,
        evaluation=evaluation,
        quality=quality,
        backend_comparison=backend_comparison,
    )
    (out_dir / "precision_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "precision_report.md").write_text(_render_precision_report(report), encoding="utf-8")
    return report


def build_precision_report(
    model_dir: Path,
    generation_review: Path | None = None,
    subject_review: Path | None = None,
    evaluation: Path | None = None,
    quality: Path | None = None,
    backend_comparison: Path | None = None,
) -> dict[str, Any]:
    profile = _load_optional_json(model_dir / "profile.json")
    generation = _load_optional_json(_resolve_generation_review_path(generation_review))
    subjects = _load_optional_json(_resolve_subject_review_path(subject_review))
    evaluation_summary = _load_optional_json(_resolve_evaluation_path(evaluation))
    quality_summary = _load_optional_json(_resolve_quality_path(quality))
    backend_comparison_summary = _load_optional_json(_resolve_backend_comparison_path(backend_comparison))
    return {
        "model": _model_summary(model_dir, profile),
        "generation": _generation_summary(generation, evaluation_summary, quality_summary),
        "subjects": _subject_summary(subjects),
        "backend_comparison": _backend_comparison_summary(backend_comparison_summary),
        "inputs": {
            "model_dir": str(model_dir),
            "generation_review": str(generation_review) if generation_review else None,
            "subject_review": str(subject_review) if subject_review else None,
            "evaluation": str(evaluation) if evaluation else None,
            "quality": str(quality) if quality else None,
            "backend_comparison": str(backend_comparison) if backend_comparison else None,
        },
        "boundary": (
            "Approximate local precision review only. Scores are model-relative vector "
            "similarities and detector/style QA signals, not identity or objective labels."
        ),
    }


def _model_summary(model_dir: Path, profile: dict[str, Any]) -> dict[str, Any]:
    descriptors = profile.get("descriptors", {})
    centroid_path = model_dir / "centroids.npz"
    return {
        "model_dir": str(model_dir),
        "image_count": profile.get("image_count"),
        "embedding_dim": profile.get("embedding_dim"),
        "appearance_shape": profile.get("appearance_shape"),
        "has_centroid_vectors": centroid_path.exists(),
        "centroid_vectors": _centroid_vector_summary(centroid_path),
        "mean_descriptor": descriptors.get("mean", {}),
        "median_descriptor": descriptors.get("median", {}),
        "reference_outputs": {
            "mean_face": str(model_dir / "mean_face.png"),
            "median_face": str(model_dir / "median_face.png"),
            "centroid_vectors": str(centroid_path),
        },
    }


def _centroid_vector_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    if not zipfile.is_zipfile(path):
        return {"available": False, "error": "unreadable centroids.npz"}
    try:
        with np.load(path, allow_pickle=False) as data:
            return {
                "available": True,
                "mean_embedding": _array_summary(data, "mean_embedding"),
                "median_embedding": _array_summary(data, "median_embedding"),
                "mean_appearance": _array_summary(data, "mean_appearance"),
                "median_appearance": _array_summary(data, "median_appearance"),
            }
    except (OSError, ValueError, zipfile.BadZipFile):
        return {"available": False, "error": "unreadable centroids.npz"}


def _array_summary(data: np.lib.npyio.NpzFile, key: str) -> dict[str, Any]:
    if key not in data:
        return {"available": False}
    array = np.asarray(data[key], dtype=np.float32)
    flat = array.reshape(-1)
    preview = [round(float(value), 6) for value in flat[:8]]
    digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    return {
        "available": True,
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "l2_norm": round(float(np.linalg.norm(flat)), 6),
        "sha256": digest,
        "preview": preview,
    }


def _generation_summary(
    generation: dict[str, Any],
    evaluation: dict[str, Any],
    quality: dict[str, Any],
) -> dict[str, Any]:
    top_run = {}
    runs = generation.get("runs")
    if isinstance(runs, list) and runs:
        top_run = runs[0] if isinstance(runs[0], dict) else {}
    qa_pass_count = _first_present(top_run.get("qa_pass_count"), quality.get("pass_count"))
    qa_fail_count = _first_present(top_run.get("qa_fail_count"), quality.get("fail_count"))
    qa_total = None
    if isinstance(qa_pass_count, int) and isinstance(qa_fail_count, int):
        qa_total = qa_pass_count + qa_fail_count
    qa_reviewed_count = _first_present(quality.get("image_count"), qa_total)
    return {
        "reviewed_run_count": generation.get("run_count"),
        "best_run_dir": generation.get("best_run_dir"),
        "best_centroid_score": _first_present(
            generation.get("best_qa_centroid_score"),
            generation.get("best_centroid_score"),
            evaluation.get("best_centroid_score"),
        ),
        "best_image_id": _first_present(
            generation.get("best_qa_image_id"),
            top_run.get("best_image_id"),
            evaluation.get("best_image_id"),
        ),
        "best_image_path": _first_present(
            generation.get("best_qa_path"),
            _top_image_path(evaluation),
        ),
        "best_style_score": generation.get("best_style_score"),
        "best_combined_image_id": generation.get("best_combined_image_id"),
        "best_combined_image_path": generation.get("best_combined_path"),
        "best_combined_score": generation.get("best_combined_score"),
        "qa_pass_count": qa_pass_count,
        "qa_fail_count": qa_fail_count,
        "qa_reviewed_count": qa_reviewed_count,
        "qa_pass_rate": _first_present(top_run.get("qa_pass_rate"), _pass_rate(qa_pass_count, qa_total)),
        "evaluated_image_count": _first_present(top_run.get("image_count"), evaluation.get("image_count")),
        "failed_image_count": _first_present(top_run.get("failed_count"), evaluation.get("failed_count")),
    }


def _subject_summary(subjects: dict[str, Any]) -> dict[str, Any]:
    subject_rows = subjects.get("subjects")
    if not isinstance(subject_rows, list):
        subject_rows = []
    best = subject_rows[0] if subject_rows and isinstance(subject_rows[0], dict) else {}
    return {
        "subject_count": subjects.get("subject_count", len(subject_rows) if subject_rows else None),
        "top_subject": best.get("subject"),
        "top_subject_mean_score": best.get("mean_centroid_score"),
        "top_subject_best_score": best.get("best_centroid_score"),
        "top_subject_best_image_path": best.get("best_image_path"),
        "subjects": subject_rows[:10],
    }


def _backend_comparison_summary(comparison: dict[str, Any]) -> dict[str, Any]:
    runs = comparison.get("runs")
    if not isinstance(runs, list):
        runs = []
    agreement = comparison.get("rank_agreement")
    if not isinstance(agreement, list):
        agreement = []
    completed = [run for run in runs if isinstance(run, dict) and run.get("status") == "completed"]
    failed = [run for run in runs if isinstance(run, dict) and run.get("status") == "failed"]
    return {
        "run_count": len(runs) if runs else None,
        "completed_count": len(completed) if runs else None,
        "failed_count": len(failed) if runs else None,
        "completed_backends": [str(run.get("backend")) for run in completed],
        "failed_backends": [str(run.get("backend")) for run in failed],
        "rank_agreement": agreement,
    }


def _render_precision_report(report: dict[str, Any]) -> str:
    model = report["model"]
    generation = report["generation"]
    subjects = report["subjects"]
    backend_comparison = report["backend_comparison"]
    lines = [
        "# seju-face precision report",
        "",
        "## Model",
        "",
        f"- model_dir: {model['model_dir']}",
        f"- image_count: {_value(model['image_count'])}",
        f"- embedding_dim: {_value(model['embedding_dim'])}",
        f"- has_centroid_vectors: {model['has_centroid_vectors']}",
        f"- mean_face: {model['reference_outputs']['mean_face']}",
        f"- median_face: {model['reference_outputs']['median_face']}",
        f"- mean_embedding_norm: {_value(_vector_field(model, 'mean_embedding', 'l2_norm'))}",
        f"- median_embedding_norm: {_value(_vector_field(model, 'median_embedding', 'l2_norm'))}",
        f"- mean_embedding_sha256: {_value(_vector_field(model, 'mean_embedding', 'sha256'))}",
        f"- median_embedding_sha256: {_value(_vector_field(model, 'median_embedding', 'sha256'))}",
        "",
        "## Generated Image Review",
        "",
        f"- reviewed_run_count: {_value(generation['reviewed_run_count'])}",
        f"- best_run_dir: {_value(generation['best_run_dir'])}",
        f"- best_image_id: {_value(generation['best_image_id'])}",
        f"- best_centroid_score: {_value(generation['best_centroid_score'])}",
        f"- best_style_score: {_value(generation['best_style_score'])}",
        f"- best_combined_image_id: {_value(generation['best_combined_image_id'])}",
        f"- best_combined_score: {_value(generation['best_combined_score'])}",
        f"- qa_pass: {_value(generation['qa_pass_count'])}/{_value(generation['qa_reviewed_count'])}",
        "",
        "## Subject Review",
        "",
        f"- subject_count: {_value(subjects['subject_count'])}",
        f"- top_subject: {_value(subjects['top_subject'])}",
        f"- top_subject_mean_score: {_value(subjects['top_subject_mean_score'])}",
        f"- top_subject_best_score: {_value(subjects['top_subject_best_score'])}",
        "",
        "## Backend Comparison",
        "",
        f"- run_count: {_value(backend_comparison['run_count'])}",
        f"- completed_backends: {', '.join(backend_comparison['completed_backends'])}",
        f"- failed_backends: {', '.join(backend_comparison['failed_backends'])}",
    ]
    if backend_comparison["rank_agreement"]:
        lines.extend(["", "| backend_a | backend_b | common_images | spearman_rank |"])
        lines.append("| --- | --- | ---: | ---: |")
        for row in backend_comparison["rank_agreement"]:
            if isinstance(row, dict):
                lines.append(
                    f"| {_value(row.get('backend_a'))} | {_value(row.get('backend_b'))} | "
                    f"{_value(row.get('common_image_count'))} | {_value(row.get('spearman_rank'))} |"
                )
    lines.extend(
        [
        "",
        "## Boundary",
        "",
        report["boundary"],
        "",
        ]
    )
    return "\n".join(lines)


def _resolve_generation_review_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_dir():
        return path / "generation_run_reviews.json"
    return path


def _resolve_subject_review_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_dir():
        return path / "subject_reviews.json"
    return path


def _resolve_evaluation_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_dir():
        return path / "summary.json"
    return path


def _resolve_quality_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_dir():
        return path / "image_quality.json"
    return path


def _resolve_backend_comparison_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_dir():
        return path / "backend_comparison.json"
    return path


def _load_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _pass_rate(pass_count: Any, total: Any) -> float | None:
    if not isinstance(pass_count, int) or not isinstance(total, int) or total <= 0:
        return None
    return round(pass_count / total, 6)


def _top_image_path(evaluation: dict[str, Any]) -> str | None:
    top_images = evaluation.get("top_images")
    if not isinstance(top_images, list) or not top_images:
        return None
    first = top_images[0]
    if not isinstance(first, dict):
        return None
    path = first.get("path")
    return str(path) if path is not None else None


def _value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _vector_field(model: dict[str, Any], vector_name: str, field: str) -> Any:
    vectors = model.get("centroid_vectors")
    if not isinstance(vectors, dict):
        return None
    summary = vectors.get(vector_name)
    if not isinstance(summary, dict):
        return None
    return summary.get(field)
