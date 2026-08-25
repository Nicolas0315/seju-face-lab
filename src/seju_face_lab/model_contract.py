from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelContract:
    schema_version: str
    source_manifest_sha256: str
    clean_manifest_sha256: str
    code_commit: str
    backend_name: str
    recognition_model_name: str
    recognition_model_sha256: str
    detector_name: str
    detector_model_sha256: str
    alignment_method: str
    landmark_schema: str
    crop_size: int
    color_space: str
    preprocessing_version: str
    embedding_dimension: int
    embedding_normalization: str
    quality_definition_version: str
    duplicate_definition_version: str
    subject_aggregation: str
    global_aggregation: str
    component_method: str
    component_count: int
    score_definition_version: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ModelContract:
        expected = {item.name for item in fields(cls)}
        supplied = set(value)
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        if missing or unknown:
            raise ValueError(f"invalid model contract fields: missing={missing}, unknown={unknown}")
        contract = cls(**dict(value))
        if contract.crop_size <= 0 or contract.embedding_dimension <= 0:
            raise ValueError("model contract dimensions must be positive")
        if contract.component_count < 0:
            raise ValueError("model contract component_count must be non-negative")
        return contract

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        payload = json.dumps(
            self.to_mapping(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def assert_compatible(left: ModelContract, right: ModelContract) -> None:
    if left.sha256() != right.sha256():
        raise ValueError(
            f"model contract mismatch: left={left.sha256()} right={right.sha256()}"
        )


def insightface_contract(
    *,
    source_manifest_sha256: str,
    clean_manifest_sha256: str,
    code_commit: str,
    model_dir: Path,
    component_count: int = 8,
) -> ModelContract:
    recognition = model_dir / "w600k_r50.onnx"
    detector = model_dir / "det_10g.onnx"
    landmarks = model_dir / "2d106det.onnx"
    missing = [str(path) for path in (recognition, detector, landmarks) if not path.is_file()]
    if missing:
        raise ValueError(f"missing InsightFace model files: {missing}")
    return ModelContract.from_mapping(
        {
            "schema_version": "1",
            "source_manifest_sha256": source_manifest_sha256,
            "clean_manifest_sha256": clean_manifest_sha256,
            "code_commit": code_commit,
            "backend_name": "insightface",
            "recognition_model_name": "buffalo_l:w600k_r50.onnx",
            "recognition_model_sha256": _sha256_file(recognition),
            "detector_name": "buffalo_l:det_10g.onnx",
            "detector_model_sha256": _sha256_file(detector),
            "alignment_method": "insightface_5point_112",
            "landmark_schema": f"insightface_2d_106_v1:{_sha256_file(landmarks)}",
            "crop_size": 112,
            "color_space": "RGB",
            "preprocessing_version": "1",
            "embedding_dimension": 512,
            "embedding_normalization": "l2_unit",
            "quality_definition_version": "confidence_area_v1",
            "duplicate_definition_version": "dhash64_ssim_v1",
            "subject_aggregation": "weighted_spherical_median_v1",
            "global_aggregation": "equal_subject_spherical_median_v1",
            "component_method": "tangent_svd_v1",
            "component_count": component_count,
            "score_definition_version": "loso_hmean_v1",
        }
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
