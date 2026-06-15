from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .backends import VectorBackend
from .embeddings import iter_image_paths, vectorize_batch_parallel
from .model import build_centroid_model

_BOUNDARY = (
    "Per-subject vectors approximate only the local consented image folder for that subject. "
    "They are aggregate research inputs for building agency centroids and comparisons, not "
    "identity recognition, attractiveness, demographic, or personal-value labels. Raw vectors and "
    "appearance tensors stay local; only counts, dimensions, norms, and hashes are surfaced."
)


@dataclass(frozen=True)
class SubjectVector:
    subject: str
    status: str  # "ok" | "empty" | "failed"
    image_count: int
    failed_count: int
    embedding_dim: int | None = None
    appearance_shape: tuple[int, int, int] | None = None
    mean_embedding: np.ndarray | None = None
    median_embedding: np.ndarray | None = None
    mean_appearance: np.ndarray | None = None
    median_appearance: np.ndarray | None = None


def vectorize_subjects(
    subjects_dir: Path,
    backend: VectorBackend,
    crop: str = "center",
    workers: int = 4,
) -> list[SubjectVector]:
    """Build one aggregate vector per subject folder under subjects_dir.

    Each subject folder is treated as a tiny image-weighted centroid, reusing the same
    aggregation as ``build --balance subject`` so per-subject vectors stay consistent with the
    main model. Subjects with no readable images are recorded as empty/failed without aborting
    the batch.
    """
    subject_dirs = sorted(path for path in subjects_dir.iterdir() if path.is_dir())
    results: list[SubjectVector] = []
    for subject_dir in subject_dirs:
        try:
            results.append(_vectorize_subject(subject_dir, backend, crop, workers))
        except Exception:  # noqa: BLE001 - one unreadable folder must not abort the whole batch
            results.append(
                SubjectVector(subject=subject_dir.name, status="failed", image_count=0, failed_count=0)
            )
    return results


def _vectorize_subject(
    subject_dir: Path,
    backend: VectorBackend,
    crop: str,
    workers: int,
) -> SubjectVector:
    paths = iter_image_paths(subject_dir)
    if not paths:
        return SubjectVector(subject=subject_dir.name, status="empty", image_count=0, failed_count=0)

    vectors = vectorize_batch_parallel(paths, backend, crop=crop, workers=workers)
    failed_count = len(paths) - len(vectors)
    if not vectors:
        return SubjectVector(
            subject=subject_dir.name, status="failed", image_count=0, failed_count=failed_count
        )

    model = build_centroid_model(
        image_ids=[vector.image_id for vector in vectors],
        source_paths=[str(vector.path) for vector in vectors],
        embeddings=np.stack([vector.embedding for vector in vectors]),
        appearances=np.stack([vector.appearance for vector in vectors]),
        centroid_mode="image_weighted",
    )
    return SubjectVector(
        subject=subject_dir.name,
        status="ok",
        image_count=len(vectors),
        failed_count=failed_count,
        embedding_dim=model.embedding_dim,
        appearance_shape=model.appearance_shape,
        mean_embedding=model.mean_embedding,
        median_embedding=model.median_embedding,
        mean_appearance=model.mean_appearance,
        median_appearance=model.median_appearance,
    )


def write_subject_vectors(subjects: list[SubjectVector], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_write_subject(subject, out_dir) for subject in subjects]
    manifest = {
        "subject_count": len(subjects),
        "ok_count": sum(1 for subject in subjects if subject.status == "ok"),
        "empty_count": sum(1 for subject in subjects if subject.status == "empty"),
        "failed_count": sum(1 for subject in subjects if subject.status == "failed"),
        "subjects": rows,
        "boundary": _BOUNDARY,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_manifest_csv(rows, out_dir / "manifest.csv")
    return manifest


def _write_subject(subject: SubjectVector, out_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "subject": subject.subject,
        "status": subject.status,
        "image_count": subject.image_count,
        "failed_count": subject.failed_count,
        "embedding_dim": subject.embedding_dim,
        "appearance_shape": list(subject.appearance_shape) if subject.appearance_shape else None,
        "mean_embedding_l2_norm": None,
        "mean_embedding_sha256": None,
        "median_embedding_sha256": None,
        "mean_appearance_sha256": None,
        "vector_path": None,
        "profile_path": None,
    }
    if subject.status != "ok" or subject.mean_embedding is None:
        return row

    subject_dir = out_dir / subject.subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        subject_dir / "vector.npz",
        mean_embedding=subject.mean_embedding,
        median_embedding=subject.median_embedding,
        mean_appearance=subject.mean_appearance,
        median_appearance=subject.median_appearance,
    )
    (subject_dir / "profile.json").write_text(
        json.dumps(
            {
                "subject": subject.subject,
                "image_count": subject.image_count,
                "failed_count": subject.failed_count,
                "embedding_dim": subject.embedding_dim,
                "appearance_shape": list(subject.appearance_shape) if subject.appearance_shape else None,
                "boundary": _BOUNDARY,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    mean_summary = _vector_summary(subject.mean_embedding)
    row.update(
        {
            "mean_embedding_l2_norm": mean_summary["l2_norm"],
            "mean_embedding_sha256": mean_summary["sha256"],
            "median_embedding_sha256": _vector_summary(subject.median_embedding)["sha256"],
            "mean_appearance_sha256": _vector_summary(subject.mean_appearance)["sha256"],
            "vector_path": str((subject_dir / "vector.npz").relative_to(out_dir)),
            "profile_path": str((subject_dir / "profile.json").relative_to(out_dir)),
        }
    )
    return row


def _vector_summary(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array.astype(np.float32))
    flat = contiguous.reshape(-1)
    return {
        "l2_norm": round(float(np.linalg.norm(flat)), 6),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


_MANIFEST_FIELDS = [
    "subject",
    "status",
    "image_count",
    "failed_count",
    "embedding_dim",
    "mean_embedding_l2_norm",
    "mean_embedding_sha256",
    "median_embedding_sha256",
    "mean_appearance_sha256",
    "vector_path",
    "profile_path",
]


def _write_manifest_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in _MANIFEST_FIELDS})


def _csv_value(value: Any) -> Any:
    return "" if value is None else value
