from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import backends
from .embeddings import iter_image_paths
from .metrics import Score, score_generated_images, write_scores
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


def _validate_backend_names(backend_names: list[str]) -> None:
    unknown = sorted({name for name in backend_names if name not in backends.BACKENDS})
    if unknown:
        choices = ", ".join(sorted(backends.BACKENDS))
        raise ValueError(f"Unknown backend(s): {', '.join(unknown)}. Choices: {choices}")


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
        write_scores(scores, evaluation_dir, failed_paths=failed_paths)
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


def _write_reference_failures(model_dir: Path, failures: list[dict[str, str]]) -> None:
    if failures:
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


def _optional_float(value: float | None) -> str:
    return "" if value is None else f"{float(value):.6f}"
