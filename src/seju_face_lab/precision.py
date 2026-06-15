from __future__ import annotations

import csv
import hashlib
import json
import math
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
    subject_backend_comparison: Path | None = None,
    correlation: Path | None = None,
    model_audit: Path | None = None,
    vector_export: Path | None = None,
    face_ingredients: Path | None = None,
    benchmark_research: Path | None = None,
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
        subject_backend_comparison=subject_backend_comparison,
        correlation=correlation,
        model_audit=model_audit,
        vector_export=vector_export,
        face_ingredients=face_ingredients,
        benchmark_research=benchmark_research,
    )
    report = _json_safe(report)
    (out_dir / "precision_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
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
    subject_backend_comparison: Path | None = None,
    correlation: Path | None = None,
    model_audit: Path | None = None,
    vector_export: Path | None = None,
    face_ingredients: Path | None = None,
    benchmark_research: Path | None = None,
) -> dict[str, Any]:
    profile = _load_optional_json(model_dir / "profile.json")
    model_audit_summary = _load_optional_json(_resolve_model_audit_path(model_audit))
    vector_export_summary = _load_optional_vector_export(_resolve_vector_export_path(vector_export))
    generation = _load_optional_json(_resolve_generation_review_path(generation_review))
    subjects = _load_optional_json(_resolve_subject_review_path(subject_review))
    evaluation_summary = _load_optional_json(_resolve_evaluation_path(evaluation))
    evaluation_scores = _load_score_rows(_resolve_evaluation_scores_path(evaluation))
    quality_summary = _load_optional_json(_resolve_quality_path(quality))
    backend_comparison_summary = _load_optional_json(_resolve_backend_comparison_path(backend_comparison))
    subject_backend_comparison_summary = _load_optional_json(
        _resolve_subject_backend_comparison_path(subject_backend_comparison)
    )
    correlation_summary_raw = _load_optional_json(_resolve_correlation_path(correlation))
    face_ingredients_raw = _load_optional_json(_resolve_face_ingredients_path(face_ingredients))
    benchmark_research_raw = _load_optional_json(_resolve_benchmark_research_path(benchmark_research))
    model = _model_summary(model_dir, profile, model_audit_summary, vector_export_summary)
    generation_summary = _generation_summary(
        generation,
        evaluation_summary,
        quality_summary,
        evaluation_scores,
    )
    subjects_summary = _subject_summary(subjects)
    backend_summary = _backend_comparison_summary(backend_comparison_summary)
    subject_backend_summary = _backend_comparison_summary(subject_backend_comparison_summary)
    correlation_summary = _correlation_summary(correlation_summary_raw)
    face_ingredients_summary = _face_ingredients_summary(face_ingredients_raw)
    benchmark_research_summary = _benchmark_research_summary(benchmark_research_raw)
    return {
        "workflow_readiness": _workflow_readiness(
            model,
            generation_summary,
            subjects_summary,
            backend_summary,
            subject_backend_summary,
            correlation_summary,
            face_ingredients_summary,
            benchmark_research_summary,
            bool(evaluation_summary or evaluation_scores),
            bool(quality_summary),
        ),
        "model": model,
        "generation": generation_summary,
        "subjects": subjects_summary,
        "backend_comparison": backend_summary,
        "subject_backend_comparison": subject_backend_summary,
        "correlation": correlation_summary,
        "face_ingredients": face_ingredients_summary,
        "benchmark_research": benchmark_research_summary,
        "inputs": {
            "model_dir": str(model_dir),
            "generation_review": str(generation_review) if generation_review else None,
            "subject_review": str(subject_review) if subject_review else None,
            "evaluation": str(evaluation) if evaluation else None,
            "quality": str(quality) if quality else None,
            "backend_comparison": str(backend_comparison) if backend_comparison else None,
            "subject_backend_comparison": (
                str(subject_backend_comparison) if subject_backend_comparison else None
            ),
            "correlation": str(correlation) if correlation else None,
            "model_audit": str(model_audit) if model_audit else None,
            "vector_export": str(vector_export) if vector_export else None,
            "face_ingredients": str(face_ingredients) if face_ingredients else None,
            "benchmark_research": str(benchmark_research) if benchmark_research else None,
        },
        "boundary": (
            "Approximate local precision review only. Scores are model-relative vector "
            "similarities and detector/style QA signals, not identity or objective labels."
        ),
    }


def _model_summary(
    model_dir: Path,
    profile: dict[str, Any],
    model_audit: dict[str, Any],
    vector_export: dict[str, Any],
) -> dict[str, Any]:
    descriptors = profile.get("descriptors", {})
    centroid_path = model_dir / "centroids.npz"
    return {
        "model_dir": str(model_dir),
        "image_count": profile.get("image_count"),
        "embedding_dim": profile.get("embedding_dim"),
        "appearance_shape": profile.get("appearance_shape"),
        "has_centroid_vectors": centroid_path.exists(),
        "centroid_vectors": _centroid_vector_summary(centroid_path),
        "model_audit": _model_audit_summary(model_audit),
        "vector_export": _vector_export_summary(vector_export),
        "mean_descriptor": descriptors.get("mean", {}),
        "median_descriptor": descriptors.get("median", {}),
        "reference_outputs": {
            "mean_face": str(model_dir / "mean_face.png"),
            "median_face": str(model_dir / "median_face.png"),
            "centroid_vectors": str(centroid_path),
        },
    }


def _model_audit_summary(audit: dict[str, Any]) -> dict[str, Any]:
    if not audit:
        return {"available": False}
    centroids = audit.get("centroids")
    if not isinstance(centroids, dict):
        centroids = {}
    return {
        "available": True,
        "model_dir": audit.get("model_dir"),
        "image_count": audit.get("image_count"),
        "embedding_dim": audit.get("embedding_dim"),
        "appearance_shape": audit.get("appearance_shape"),
        "mean_median_embedding": _audit_pair_summary(centroids.get("mean_median_embedding")),
        "mean_median_appearance": _audit_pair_summary(centroids.get("mean_median_appearance")),
        "descriptor_delta": audit.get("descriptor_delta", {}),
    }


def _vector_export_summary(export: dict[str, Any]) -> dict[str, Any]:
    if not export:
        return {"available": False}
    vectors = export.get("vectors")
    if not isinstance(vectors, dict):
        vectors = {}
    return {
        "available": True,
        "model_dir": export.get("model_dir"),
        "image_count": export.get("image_count"),
        "embedding_dim": export.get("embedding_dim"),
        "include_appearance": export.get("include_appearance"),
        "vectors": {
            name: _exported_vector_summary(vector)
            for name, vector in vectors.items()
            if isinstance(vector, dict)
        },
    }


def _exported_vector_summary(vector: dict[str, Any]) -> dict[str, Any]:
    values = vector.get("values")
    values_count = len(values) if isinstance(values, list) else vector.get("values_count")
    return {
        "shape": vector.get("shape"),
        "dtype": vector.get("dtype"),
        "l2_norm": vector.get("l2_norm"),
        "sha256": vector.get("sha256"),
        "values_count": values_count,
    }


def _audit_pair_summary(pair: Any) -> dict[str, Any]:
    if not isinstance(pair, dict):
        return {"available": False}
    return {
        "available": bool(pair.get("available", True)),
        "cosine": pair.get("cosine"),
        "euclidean": pair.get("euclidean"),
        "mean_abs_delta": pair.get("mean_abs_delta"),
        "max_abs_delta": pair.get("max_abs_delta"),
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
    evaluation_scores: list[dict[str, Any]],
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
    best_image_id = _first_present(
        generation.get("best_qa_image_id"),
        top_run.get("best_image_id"),
        evaluation.get("best_image_id"),
    )
    best_image = _matching_image(evaluation, best_image_id, evaluation_scores)
    generation_config = generation.get("best_generation")
    if not isinstance(generation_config, dict):
        generation_config = _generation_config_from_run(top_run)
    return {
        "reviewed_run_count": generation.get("run_count"),
        "best_run_dir": generation.get("best_run_dir"),
        "provider": generation_config.get("provider"),
        "model_id": generation_config.get("model_id"),
        "status": generation_config.get("status"),
        "centroid_kind": generation_config.get("centroid_kind"),
        "prompt_profile": generation_config.get("prompt_profile"),
        "seed": generation_config.get("seed"),
        "planned_count": generation_config.get("planned_count"),
        "steps": generation_config.get("steps"),
        "guidance_scale": generation_config.get("guidance_scale"),
        "size": generation_config.get("size"),
        "device": generation_config.get("device"),
        "dtype": generation_config.get("dtype"),
        "prompt_words": generation_config.get("prompt_words"),
        "best_centroid_score": _first_present(
            generation.get("best_qa_centroid_score"),
            generation.get("best_centroid_score"),
            evaluation.get("best_centroid_score"),
        ),
        "best_image_id": best_image_id,
        "best_image_path": _first_present(
            generation.get("best_qa_path"),
            _top_image_path(best_image),
        ),
        "best_cosine_to_mean": best_image.get("cosine_to_mean"),
        "best_cosine_to_median": best_image.get("cosine_to_median"),
        "best_euclidean_to_mean": best_image.get("euclidean_to_mean"),
        "best_euclidean_to_median": best_image.get("euclidean_to_median"),
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
        "by_centroid_kind": _generation_by_centroid_kind(generation.get("by_centroid_kind")),
    }


def _generation_by_centroid_kind(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    return {
        str(kind): values
        for kind, values in summary.items()
        if isinstance(values, dict)
    }


def _generation_config_from_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": run.get("provider"),
        "model_id": run.get("model_id"),
        "status": run.get("status"),
        "centroid_kind": run.get("centroid_kind"),
        "prompt_profile": run.get("prompt_profile"),
        "seed": run.get("seed"),
        "planned_count": _first_present(run.get("planned_count"), run.get("count")),
        "steps": run.get("steps"),
        "guidance_scale": run.get("guidance_scale"),
        "size": run.get("size"),
        "device": run.get("device"),
        "dtype": run.get("dtype"),
        "prompt_words": run.get("prompt_words"),
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
        "analysis": _subject_analysis(subjects.get("analysis")),
        "subjects": subject_rows[:10],
    }


def _subject_analysis(analysis: Any) -> dict[str, Any]:
    if not isinstance(analysis, dict):
        return {}
    return {
        key: value
        for key, value in analysis.items()
        if key in {
            "score_stats",
            "top_mean_subjects",
            "top_best_subjects",
            "single_image_lift",
            "mean_vector_leaders",
            "median_vector_leaders",
        }
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


def _correlation_summary(summary: dict[str, Any]) -> dict[str, Any]:
    if not summary:
        return {"available": False}
    correlations = summary.get("correlations")
    if not isinstance(correlations, list):
        correlations = []
    ranked = sorted(
        [
            row
            for row in correlations
            if isinstance(row, dict) and _finite_float(row.get("spearman_r")) is not None
        ],
        key=lambda row: -abs(_finite_float(row.get("spearman_r")) or 0.0),
    )
    top = ranked[0] if ranked else {}
    return {
        "available": True,
        "talent_count": summary.get("talent_count"),
        "with_face_score": summary.get("with_face_score"),
        "with_ig": summary.get("with_ig"),
        "with_twitter": summary.get("with_twitter"),
        "with_tiktok": summary.get("with_tiktok"),
        "correlation_count": len(correlations),
        "top_pair": {
            "a": top.get("a"),
            "b": top.get("b"),
            "n": top.get("n"),
            "spearman_r": top.get("spearman_r"),
            "spearman_p": top.get("spearman_p"),
            "pearson_r": top.get("pearson_r"),
            "interpretation": top.get("interpretation"),
        } if top else None,
        "correlations": correlations[:10],
    }


def _face_ingredients_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {"available": False}
    ingredients = report.get("ingredients")
    if not isinstance(ingredients, dict):
        ingredients = {}
    return {
        "available": True,
        "image_count": report.get("image_count"),
        "overall": _ingredient_section_summary(ingredients.get("overall")),
        "face_parts": _ingredient_section_summary(ingredients.get("face_parts")),
        "color_tone": _ingredient_section_summary(ingredients.get("color_tone")),
        "makeup_texture": _ingredient_section_summary(ingredients.get("makeup_texture")),
        "hair": _ingredient_section_summary(ingredients.get("hair")),
        "prompt_guidance": _string_list(report.get("prompt_guidance")),
    }


def _ingredient_section_summary(section: Any) -> dict[str, Any]:
    if not isinstance(section, dict):
        return {"summary": None, "evidence": {}}
    evidence = section.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    return {
        "summary": section.get("summary"),
        "evidence": evidence,
    }


def _benchmark_research_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {"available": False}
    sources = report.get("sources")
    if not isinstance(sources, list):
        sources = []
    strategy = report.get("vectorization_strategy")
    if not isinstance(strategy, dict):
        strategy = {}
    recommendations = report.get("recommendations")
    if not isinstance(recommendations, list):
        recommendations = []
    return {
        "available": True,
        "retrieved_at": report.get("retrieved_at"),
        "source_count": len([source for source in sources if isinstance(source, dict)]),
        "source_names": [
            str(source.get("name"))
            for source in sources
            if isinstance(source, dict) and source.get("name")
        ],
        "primary_face_embedding": strategy.get("primary_face_embedding"),
        "face_analysis_axis": strategy.get("face_analysis_axis"),
        "iris_axis": strategy.get("iris_axis"),
        "recommendations": [
            str(item.get("title"))
            for item in recommendations
            if isinstance(item, dict) and item.get("title")
        ][:6],
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _workflow_readiness(
    model: dict[str, Any],
    generation: dict[str, Any],
    subjects: dict[str, Any],
    backend_comparison: dict[str, Any],
    subject_backend_comparison: dict[str, Any],
    correlation: dict[str, Any],
    face_ingredients: dict[str, Any],
    benchmark_research: dict[str, Any],
    has_evaluation: bool,
    has_quality: bool,
) -> dict[str, Any]:
    checks = [
        _readiness_check(
            "model_profile",
            model.get("image_count") is not None and model.get("embedding_dim") is not None,
            "Build a centroid model with profile.json",
        ),
        _readiness_check(
            "centroid_vectors",
            bool(model.get("has_centroid_vectors"))
            and bool(_vector_field(model, "mean_embedding", "available"))
            and bool(_vector_field(model, "median_embedding", "available")),
            "Build centroids.npz with mean and median embeddings",
        ),
        _readiness_check(
            "model_audit",
            bool(model.get("model_audit", {}).get("available")),
            "Run audit-model for mean/median vector distance evidence",
            required=False,
        ),
        _readiness_check(
            "vector_export",
            bool(model.get("vector_export", {}).get("available")),
            "Run export-vectors for portable mean/median vector evidence",
            required=False,
        ),
        _readiness_check(
            "generation_review",
            generation.get("best_image_id") is not None or generation.get("reviewed_run_count") is not None,
            "Run generate --review or review-generated on candidate portraits",
        ),
        _readiness_check(
            "evaluation",
            has_evaluation or generation.get("best_centroid_score") is not None,
            "Run evaluate against generated candidate portraits",
        ),
        _readiness_check(
            "quality_review",
            has_quality or generation.get("qa_reviewed_count") is not None,
            "Run qa-images or review-generated for detector QA",
        ),
        _readiness_check(
            "subject_review",
            subjects.get("subject_count") is not None,
            "Run review-subjects for celebrity/public-figure near-face ranking",
        ),
        _readiness_check(
            "backend_comparison",
            bool(backend_comparison.get("completed_backends")),
            "Run compare-backends for backend rank-agreement evidence",
        ),
        _readiness_check(
            "subject_backend_comparison",
            bool(subject_backend_comparison.get("completed_backends")),
            "Run compare-subject-backends for subject-ranking backend agreement",
        ),
        _readiness_check(
            "correlation_report",
            bool(correlation.get("available")),
            "Run analyze correlation for face-score/SNS engagement review",
            required=False,
        ),
        _readiness_check(
            "face_ingredients",
            bool(face_ingredients.get("available")),
            "Run ingredients-report for face-part/color/makeup/hair decomposition",
            required=False,
        ),
        _readiness_check(
            "benchmark_research",
            bool(benchmark_research.get("available")),
            "Run benchmark-research for OSS and benchmark vectorization guidance",
            required=False,
        ),
    ]
    required_checks = [check for check in checks if check["required"]]
    optional_checks = [check for check in checks if not check["required"]]
    passed = [check for check in required_checks if check["ready"]]
    optional_passed = [check for check in optional_checks if check["ready"]]
    missing = [check["name"] for check in required_checks if not check["ready"]]
    optional_missing = [check["name"] for check in optional_checks if not check["ready"]]
    return {
        "ready_count": len(passed),
        "total_count": len(required_checks),
        "ready_ratio": round(len(passed) / len(required_checks), 6) if required_checks else None,
        "optional_ready_count": len(optional_passed),
        "optional_total_count": len(optional_checks),
        "optional_missing": optional_missing,
        "missing": missing,
        "next_action": _next_readiness_action(checks),
        "checks": checks,
    }


def _readiness_check(
    name: str,
    ready: bool,
    next_action: str,
    required: bool = True,
) -> dict[str, Any]:
    return {
        "name": name,
        "required": required,
        "ready": bool(ready),
        "next_action": None if ready else next_action,
    }


def _next_readiness_action(checks: list[dict[str, Any]]) -> str | None:
    for check in checks:
        if check["required"] and not check["ready"]:
            return check["next_action"]
    return None


def _render_precision_report(report: dict[str, Any]) -> str:
    readiness = report["workflow_readiness"]
    model = report["model"]
    generation = report["generation"]
    subjects = report["subjects"]
    backend_comparison = report["backend_comparison"]
    subject_backend_comparison = report["subject_backend_comparison"]
    correlation = report["correlation"]
    face_ingredients = report["face_ingredients"]
    benchmark_research = report["benchmark_research"]
    lines = [
        "# seju-face precision report",
        "",
        "## Workflow Readiness",
        "",
        f"- ready: {readiness['ready_count']}/{readiness['total_count']}",
        f"- ready_ratio: {_value(readiness['ready_ratio'])}",
        f"- missing: {', '.join(readiness['missing'])}",
        f"- optional_ready: {readiness['optional_ready_count']}/{readiness['optional_total_count']}",
        f"- optional_missing: {', '.join(readiness['optional_missing'])}",
        f"- next_action: {_value(readiness['next_action'])}",
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
        f"- model_audit_available: {model['model_audit']['available']}",
        f"- vector_export_available: {model['vector_export']['available']}",
        f"- vector_export_mean_sha256: {_value(_export_field(model, 'mean_embedding', 'sha256'))}",
        f"- vector_export_median_sha256: {_value(_export_field(model, 'median_embedding', 'sha256'))}",
        f"- mean_median_embedding_cosine: {_value(_audit_field(model, 'mean_median_embedding', 'cosine'))}",
        f"- mean_median_embedding_euclidean: {_value(_audit_field(model, 'mean_median_embedding', 'euclidean'))}",
        f"- mean_median_appearance_cosine: {_value(_audit_field(model, 'mean_median_appearance', 'cosine'))}",
        f"- mean_median_appearance_euclidean: {_value(_audit_field(model, 'mean_median_appearance', 'euclidean'))}",
        "",
        "## Face Ingredients",
        "",
        f"- available: {face_ingredients['available']}",
        f"- image_count: {_value(face_ingredients.get('image_count'))}",
        f"- overall: {_value(face_ingredients.get('overall', {}).get('summary'))}",
        f"- face_parts: {_value(face_ingredients.get('face_parts', {}).get('summary'))}",
        f"- color_tone: {_value(face_ingredients.get('color_tone', {}).get('summary'))}",
        f"- makeup_texture: {_value(face_ingredients.get('makeup_texture', {}).get('summary'))}",
        f"- hair: {_value(face_ingredients.get('hair', {}).get('summary'))}",
        f"- prompt_guidance: {_value(', '.join(face_ingredients.get('prompt_guidance', [])))}",
        "",
        "## Benchmark Research",
        "",
        f"- available: {benchmark_research['available']}",
        f"- retrieved_at: {_value(benchmark_research.get('retrieved_at'))}",
        f"- source_count: {_value(benchmark_research.get('source_count'))}",
        f"- sources: {_value(', '.join(benchmark_research.get('source_names', [])))}",
        f"- primary_face_embedding: {_value(benchmark_research.get('primary_face_embedding'))}",
        f"- face_analysis_axis: {_value(benchmark_research.get('face_analysis_axis'))}",
        f"- iris_axis: {_value(benchmark_research.get('iris_axis'))}",
        f"- recommendations: {_value(', '.join(benchmark_research.get('recommendations', [])))}",
        "",
        "## Generated Image Review",
        "",
        f"- reviewed_run_count: {_value(generation['reviewed_run_count'])}",
        f"- best_run_dir: {_value(generation['best_run_dir'])}",
        f"- provider: {_value(generation['provider'])}",
        f"- model_id: {_value(generation['model_id'])}",
        f"- status: {_value(generation['status'])}",
        f"- centroid_kind: {_value(generation['centroid_kind'])}",
        f"- prompt_profile: {_value(generation['prompt_profile'])}",
        f"- seed: {_value(generation['seed'])}",
        f"- planned_count: {_value(generation['planned_count'])}",
        f"- steps: {_value(generation['steps'])}",
        f"- guidance_scale: {_value(generation['guidance_scale'])}",
        f"- size: {_value(generation['size'])}",
        f"- device: {_value(generation['device'])}",
        f"- dtype: {_value(generation['dtype'])}",
        f"- prompt_words: {_value(generation['prompt_words'])}",
        f"- best_image_id: {_value(generation['best_image_id'])}",
        f"- best_centroid_score: {_value(generation['best_centroid_score'])}",
        f"- best_cosine_to_mean: {_value(generation['best_cosine_to_mean'])}",
        f"- best_cosine_to_median: {_value(generation['best_cosine_to_median'])}",
        f"- best_euclidean_to_mean: {_value(generation['best_euclidean_to_mean'])}",
        f"- best_euclidean_to_median: {_value(generation['best_euclidean_to_median'])}",
        f"- best_style_score: {_value(generation['best_style_score'])}",
        f"- best_combined_image_id: {_value(generation['best_combined_image_id'])}",
        f"- best_combined_score: {_value(generation['best_combined_score'])}",
        f"- qa_pass: {_value(generation['qa_pass_count'])}/{_value(generation['qa_reviewed_count'])}",
        "",
        *_render_generation_by_centroid_kind(generation.get("by_centroid_kind")),
        "## Subject Review",
        "",
        f"- subject_count: {_value(subjects['subject_count'])}",
        f"- top_subject: {_value(subjects['top_subject'])}",
        f"- top_subject_mean_score: {_value(subjects['top_subject_mean_score'])}",
        f"- top_subject_best_score: {_value(subjects['top_subject_best_score'])}",
        "",
        *_render_subject_analysis(subjects.get("analysis")),
        "## Backend Comparison",
        "",
        f"- run_count: {_value(backend_comparison['run_count'])}",
        f"- completed_backends: {', '.join(backend_comparison['completed_backends'])}",
        f"- failed_backends: {', '.join(backend_comparison['failed_backends'])}",
        "",
        "## Subject Backend Comparison",
        "",
        f"- run_count: {_value(subject_backend_comparison['run_count'])}",
        f"- completed_backends: {', '.join(subject_backend_comparison['completed_backends'])}",
        f"- failed_backends: {', '.join(subject_backend_comparison['failed_backends'])}",
        "",
        "## Correlation Review",
        "",
        f"- available: {correlation['available']}",
        f"- talent_count: {_value(correlation.get('talent_count'))}",
        f"- with_face_score: {_value(correlation.get('with_face_score'))}",
        f"- with_instagram: {_value(correlation.get('with_ig'))}",
        f"- with_twitter: {_value(correlation.get('with_twitter'))}",
        f"- with_tiktok: {_value(correlation.get('with_tiktok'))}",
        f"- correlation_count: {_value(correlation.get('correlation_count'))}",
        f"- top_pair: {_format_correlation_pair(correlation.get('top_pair'))}",
    ]
    if readiness["checks"]:
        lines.extend(["", "## Workflow Readiness Checks", "", "| check | required | ready | next_action |"])
        lines.append("| --- | --- | --- | --- |")
        for check in readiness["checks"]:
            if isinstance(check, dict):
                lines.append(
                    f"| {_value(check.get('name'))} | {_value(check.get('required'))} | "
                    f"{_value(check.get('ready'))} | "
                    f"{_value(check.get('next_action'))} |"
                )
    if backend_comparison["rank_agreement"]:
        lines.extend(["", "| backend_a | backend_b | common_images | spearman_rank |"])
        lines.append("| --- | --- | ---: | ---: |")
        for row in backend_comparison["rank_agreement"]:
            if isinstance(row, dict):
                lines.append(
                    f"| {_value(row.get('backend_a'))} | {_value(row.get('backend_b'))} | "
                    f"{_value(row.get('common_image_count'))} | {_value(row.get('spearman_rank'))} |"
                )
    if subject_backend_comparison["rank_agreement"]:
        lines.extend(["", "| backend_a | backend_b | common_subjects | spearman_rank |"])
        lines.append("| --- | --- | ---: | ---: |")
        for row in subject_backend_comparison["rank_agreement"]:
            if isinstance(row, dict):
                lines.append(
                    f"| {_value(row.get('backend_a'))} | {_value(row.get('backend_b'))} | "
                    f"{_value(row.get('common_subject_count'))} | {_value(row.get('spearman_rank'))} |"
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


def _render_subject_analysis(analysis: Any) -> list[str]:
    if not isinstance(analysis, dict) or not analysis:
        return []
    lines: list[str] = []
    stats = analysis.get("score_stats")
    if isinstance(stats, dict):
        lines.extend(
            [
                "### Subject Vector Analysis",
                "",
                f"- reviewed_images: {_value(stats.get('reviewed_image_count'))}",
                f"- failed_images: {_value(stats.get('failed_image_count'))}",
                f"- mean_of_subject_means: {_value(stats.get('mean_of_subject_means'))}",
                f"- median_of_subject_means: {_value(stats.get('median_of_subject_means'))}",
                "",
            ]
        )
    lines.extend(_render_subject_analysis_table("Stable Mean Leaders", analysis.get("top_mean_subjects")))
    lines.extend(_render_subject_analysis_table("Peak Best Leaders", analysis.get("top_best_subjects")))
    lines.extend(_render_subject_analysis_table("Single Image Lift", analysis.get("single_image_lift")))
    lines.extend(_render_subject_analysis_table("Mean Vector Affinity", analysis.get("mean_vector_leaders")))
    lines.extend(_render_subject_analysis_table("Median Vector Affinity", analysis.get("median_vector_leaders")))
    return lines


def _render_subject_analysis_table(title: str, rows: Any) -> list[str]:
    if not isinstance(rows, list) or not rows:
        return []
    lines = [
        f"#### {title}",
        "",
        "| rank | subject | metric | mean | best | median |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        if isinstance(row, dict):
            lines.append(
                f"| {rank} | {_value(row.get('subject'))} | {_value(row.get('metric'))} | "
                f"{_value(row.get('mean_centroid_score'))} | "
                f"{_value(row.get('best_centroid_score'))} | "
                f"{_value(row.get('median_centroid_score'))} |"
            )
    lines.append("")
    return lines


def _render_generation_by_centroid_kind(summary: Any) -> list[str]:
    if not isinstance(summary, dict) or not summary:
        return []
    lines = [
        "### By Centroid Kind",
        "",
        "| centroid | runs | images | failed | qa_pass | best_face | best_qa_face | best_combined |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind, values in sorted(summary.items()):
        if isinstance(values, dict):
            lines.append(
                f"| {kind} | {_value(values.get('run_count'))} | "
                f"{_value(values.get('image_count'))} | "
                f"{_value(values.get('failed_count'))} | "
                f"{_value(values.get('qa_pass_count'))} | "
                f"{_value(values.get('best_centroid_score'))} | "
                f"{_value(values.get('best_qa_centroid_score'))} | "
                f"{_value(values.get('best_combined_score'))} |"
            )
    lines.append("")
    return lines


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
    if path.name == "scores.csv":
        return path.with_name("summary.json")
    return path


def _resolve_evaluation_scores_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_dir():
        return path / "scores.csv"
    if path.name == "scores.csv":
        return path
    return path.with_name("scores.csv")


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


def _resolve_subject_backend_comparison_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_dir():
        return path / "subject_backend_comparison.json"
    return path


def _resolve_correlation_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_dir():
        return path / "correlation_summary.json"
    return path


def _resolve_model_audit_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_dir():
        return path / "model_audit.json"
    return path


def _resolve_vector_export_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_dir():
        json_path = path / "vectors.json"
        if json_path.exists():
            return json_path
        csv_path = path / "vectors.csv"
        if csv_path.exists():
            return csv_path
        return json_path
    return path


def _resolve_face_ingredients_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_dir():
        return path / "face_ingredients.json"
    return path


def _resolve_benchmark_research_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_dir():
        return path / "benchmark_research.json"
    return path


def _load_optional_vector_export(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    if path.suffix.lower() == ".csv":
        return _load_vector_export_csv(path)
    return _load_optional_json(path)


def _load_vector_export_csv(path: Path) -> dict[str, Any]:
    vectors: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row.get("vector")
            if not name:
                continue
            vector = vectors.setdefault(
                name,
                {
                    "shape": _parse_shape(row.get("shape")),
                    "dtype": row.get("dtype") or None,
                    "l2_norm": _optional_csv_float(row.get("l2_norm")),
                    "sha256": row.get("sha256") or None,
                    "values_count": 0,
                },
            )
            vector["values_count"] += 1
    return {
        "format": "csv",
        "path": str(path),
        "vectors": vectors,
    }


def _parse_shape(value: str | None) -> list[int] | None:
    if not value:
        return None
    try:
        return [int(part) for part in value.split("x") if part]
    except ValueError:
        return None


def _load_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_score_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [_score_row_to_image(row) for row in csv.DictReader(handle)]


def _score_row_to_image(row: dict[str, str]) -> dict[str, Any]:
    return {
        "image_id": row.get("image_id"),
        "path": row.get("path"),
        "centroid_score": _optional_csv_float(row.get("centroid_score")),
        "cosine_to_mean": _optional_csv_float(row.get("cosine_to_mean")),
        "cosine_to_median": _optional_csv_float(row.get("cosine_to_median")),
        "euclidean_to_mean": _optional_csv_float(row.get("euclidean_to_mean")),
        "euclidean_to_median": _optional_csv_float(row.get("euclidean_to_median")),
    }


def _optional_csv_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _pass_rate(pass_count: Any, total: Any) -> float | None:
    if not isinstance(pass_count, int) or not isinstance(total, int) or total <= 0:
        return None
    return round(pass_count / total, 6)


def _matching_image(
    evaluation: dict[str, Any],
    image_id: Any,
    score_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    top_images = evaluation.get("top_images")
    if not isinstance(top_images, list):
        top_images = []
    if image_id is not None:
        for image in top_images:
            if isinstance(image, dict) and image.get("image_id") == image_id:
                return image
        for image in score_rows:
            if image.get("image_id") == image_id:
                return image
        return {}
    if top_images and isinstance(top_images[0], dict):
        return top_images[0]
    return score_rows[0] if score_rows else {}


def _top_image_path(top_image: dict[str, Any]) -> str | None:
    path = top_image.get("path")
    return str(path) if path is not None else None


def _value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _format_correlation_pair(pair: Any) -> str:
    if not isinstance(pair, dict):
        return ""
    return (
        f"{_value(pair.get('a'))} x {_value(pair.get('b'))} "
        f"rho={_value(pair.get('spearman_r'))} "
        f"n={_value(pair.get('n'))} "
        f"{_value(pair.get('interpretation'))}"
    ).strip()


def _vector_field(model: dict[str, Any], vector_name: str, field: str) -> Any:
    vectors = model.get("centroid_vectors")
    if not isinstance(vectors, dict):
        return None
    summary = vectors.get(vector_name)
    if not isinstance(summary, dict):
        return None
    return summary.get(field)


def _audit_field(model: dict[str, Any], pair_name: str, field: str) -> Any:
    audit = model.get("model_audit")
    if not isinstance(audit, dict):
        return None
    pair = audit.get(pair_name)
    if not isinstance(pair, dict):
        return None
    return pair.get(field)


def _export_field(model: dict[str, Any], vector_name: str, field: str) -> Any:
    export = model.get("vector_export")
    if not isinstance(export, dict):
        return None
    vectors = export.get("vectors")
    if not isinstance(vectors, dict):
        return None
    vector = vectors.get(vector_name)
    if not isinstance(vector, dict):
        return None
    return vector.get(field)
