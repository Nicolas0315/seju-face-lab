from __future__ import annotations

import io
import json
import tempfile
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .face_observations import FaceObservationExtractor, extract_face_observation
from .model_contract import ModelContract
from .robust_templates import unit_vector

IMAGE_PERTURBATIONS = (
    "downscale",
    "defocus_blur",
    "exposure_shift",
    "jpeg_recompression",
    "partial_crop",
)
DETECTOR_PERTURBATION = "detector_320"


def apply_image_perturbation(image: Image.Image, kind: str) -> Image.Image:
    source = ImageOps.exif_transpose(image).convert("RGB")
    if kind == "downscale":
        width, height = source.size
        small = source.resize(
            (max(32, width // 4), max(32, height // 4)), Image.Resampling.BILINEAR
        )
        return small.resize(source.size, Image.Resampling.BILINEAR)
    if kind == "defocus_blur":
        return source.filter(ImageFilter.GaussianBlur(radius=3.0))
    if kind == "exposure_shift":
        return ImageEnhance.Brightness(source).enhance(0.65)
    if kind == "jpeg_recompression":
        buffer = io.BytesIO()
        source.save(buffer, format="JPEG", quality=35, optimize=False, progressive=False)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()
    if kind == "partial_crop":
        width, height = source.size
        crop = source.crop((width // 10, height // 10, width, height))
        return crop.resize(source.size, Image.Resampling.BILINEAR)
    raise ValueError(f"unknown image perturbation: {kind}")


def summarize_perturbation_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    clean_count: int,
    min_coverage: float = 0.95,
    min_embedding_cosine: float = 0.80,
    max_b1_degradation_p90: float = 0.08,
) -> dict[str, Any]:
    if clean_count <= 0:
        raise ValueError("clean_count must be positive")
    materialized = [dict(row) for row in rows]
    kinds: dict[str, Any] = {}
    gate_pass = True
    for kind in sorted({str(row["kind"]) for row in materialized}):
        selected = [row for row in materialized if str(row["kind"]) == kind]
        accepted = [row for row in selected if bool(row.get("accepted"))]
        cosines = np.asarray(
            [float(row["embedding_cosine"]) for row in accepted], dtype=np.float64
        )
        degradation = np.asarray(
            [float(row["b1_center_degradation"]) for row in accepted], dtype=np.float64
        )
        coverage = len(accepted) / clean_count
        cosine_median = float(np.median(cosines)) if cosines.size else None
        degradation_p90 = float(np.quantile(degradation, 0.90)) if degradation.size else None
        kind_pass = bool(
            coverage >= min_coverage
            and cosine_median is not None
            and cosine_median >= min_embedding_cosine
            and degradation_p90 is not None
            and degradation_p90 <= max_b1_degradation_p90
        )
        gate_pass = gate_pass and kind_pass
        kinds[kind] = {
            "attempted": len(selected),
            "accepted": len(accepted),
            "coverage": coverage,
            "embedding_cosine_median": cosine_median,
            "b1_center_degradation_p90": degradation_p90,
            "gate_pass": kind_pass,
        }
    return {
        "schema_version": "vector_perturbation_evidence_v1",
        "clean_count": clean_count,
        "thresholds": {
            "min_coverage": min_coverage,
            "min_embedding_cosine": min_embedding_cosine,
            "max_b1_degradation_p90": max_b1_degradation_p90,
        },
        "kinds": kinds,
        "gate_pass": bool(kinds) and gate_pass,
    }


def evaluate_vector_perturbations(
    observations_dir: Path,
    model_dir: Path,
    out_dir: Path,
    extractor: FaceObservationExtractor,
    detector_change_extractor: FaceObservationExtractor,
) -> dict[str, Any]:
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    records = [
        json.loads(line)
        for line in (observations_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    neural_data = np.load(observations_dir / "neural_vectors.npz", allow_pickle=False)
    original_vectors = neural_data["vectors"]
    contract = ModelContract.from_mapping(
        json.loads((observations_dir / "model_contract.json").read_text(encoding="utf-8"))
    )
    templates = np.load(model_dir / "subject_templates.npz", allow_pickle=False)
    b1_center = unit_vector(np.mean(templates["neural_vectors"], axis=0))
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=out_dir.parent) as tmp:
        temp_root = Path(tmp)
        for index, record in enumerate(records):
            source_path = Path(str(record["path"]))
            with Image.open(source_path) as source_image:
                for kind in IMAGE_PERTURBATIONS:
                    target = temp_root / f"{record['image_id']}_{kind}.png"
                    apply_image_perturbation(source_image, kind).save(target)
                    rows.append(
                        _evaluate_one(
                            kind,
                            target,
                            str(record["subject_id"]),
                            extractor,
                            contract,
                            original_vectors[index],
                            b1_center,
                        )
                    )
            rows.append(
                _evaluate_one(
                    DETECTOR_PERTURBATION,
                    source_path,
                    str(record["subject_id"]),
                    detector_change_extractor,
                    contract,
                    original_vectors[index],
                    b1_center,
                )
            )
    summary = summarize_perturbation_rows(rows, clean_count=len(records))
    summary["runtime_seconds"] = time.perf_counter() - start
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "perturbation_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    (out_dir / "perturbation_evidence.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _evaluate_one(
    kind: str,
    path: Path,
    subject_id: str,
    extractor: FaceObservationExtractor,
    contract: ModelContract,
    original_vector: np.ndarray,
    b1_center: np.ndarray,
) -> dict[str, Any]:
    try:
        observation = extract_face_observation(path, subject_id, extractor, contract)
    except Exception as exc:  # noqa: BLE001 - each perturbation needs a failure reason.
        return {
            "kind": kind,
            "subject_id": subject_id,
            "accepted": False,
            "embedding_cosine": None,
            "b1_center_degradation": None,
            "rejection_reason": str(exc),
        }
    original = unit_vector(original_vector)
    perturbed = observation.neural_vector
    return {
        "kind": kind,
        "subject_id": subject_id,
        "accepted": True,
        "embedding_cosine": float(original @ perturbed),
        "b1_center_degradation": float(original @ b1_center - perturbed @ b1_center),
        "rejection_reason": None,
    }
