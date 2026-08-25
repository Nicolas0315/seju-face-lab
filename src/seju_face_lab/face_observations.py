from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .backends import _prepare_windows_torch_cuda_dlls
from .geometry_vectors import GeometryVector, geometry_vector
from .model_contract import ModelContract


class FaceObservationExtractor(Protocol):
    def detect(self, path: Path) -> list[Mapping[str, Any]]:
        ...


@dataclass(frozen=True)
class FaceObservation:
    image_id: str
    subject_id: str
    path: Path
    contract_hash: str
    detector_confidence: float
    bbox: tuple[float, float, float, float]
    relative_face_area: float
    landmarks_5: np.ndarray
    landmarks_106: np.ndarray
    neural_vector: np.ndarray
    geometry: GeometryVector
    quality: float


def extract_face_observation(
    path: Path,
    subject_id: str,
    extractor: FaceObservationExtractor,
    contract: ModelContract,
) -> FaceObservation:
    faces = extractor.detect(path)
    if len(faces) != 1:
        raise ValueError(f"expected exactly one face, got {len(faces)} for {path}")
    face = faces[0]
    embedding = np.asarray(face["embedding"], dtype=np.float64)
    if embedding.shape != (contract.embedding_dimension,) or not np.isfinite(embedding).all():
        raise ValueError("embedding does not match model contract")
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-12:
        raise ValueError("embedding norm is zero")
    embedding = embedding / norm
    landmarks_5 = np.asarray(face["landmarks_5"], dtype=np.float64)
    if landmarks_5.shape != (5, 2):
        raise ValueError("expected five alignment landmarks")
    landmarks_106 = np.asarray(face["landmarks_106"], dtype=np.float64)
    geometry = geometry_vector(landmarks_106)
    bbox_values = tuple(float(value) for value in face["bbox"])
    if len(bbox_values) != 4:
        raise ValueError("expected bbox with four values")
    confidence = float(face.get("detector_confidence", 0.0))
    relative_area = float(face.get("relative_face_area", 0.0))
    quality = float(np.clip(confidence * np.sqrt(max(relative_area, 0.0)), 0.0, 1.0))
    return FaceObservation(
        image_id=path.stem,
        subject_id=subject_id,
        path=path,
        contract_hash=contract.sha256(),
        detector_confidence=confidence,
        bbox=bbox_values,
        relative_face_area=relative_area,
        landmarks_5=landmarks_5,
        landmarks_106=landmarks_106,
        neural_vector=embedding,
        geometry=geometry,
        quality=quality,
    )


class InsightFaceObservationExtractor:
    def __init__(
        self,
        gpu_id: int = 0,
        model_pack: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        self.gpu_id = gpu_id
        self.model_pack = model_pack
        self.det_size = det_size
        self._app: Any = None

    def _get_app(self) -> Any:
        if self._app is None:
            from insightface.app import FaceAnalysis

            _prepare_windows_torch_cuda_dlls()
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.gpu_id >= 0
                else ["CPUExecutionProvider"]
            )
            self._app = FaceAnalysis(
                name=self.model_pack,
                providers=providers,
                allowed_modules=["detection", "recognition", "landmark_2d_106"],
            )
            self._app.prepare(ctx_id=self.gpu_id, det_size=self.det_size)
        return self._app

    def detect(self, path: Path) -> list[Mapping[str, Any]]:
        import cv2

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"could not load image: {path}")
        height, width = image.shape[:2]
        image_area = float(max(height * width, 1))
        observations: list[Mapping[str, Any]] = []
        for face in self._get_app().get(image):
            bbox = np.asarray(face.bbox, dtype=np.float64)
            area = max(float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])), 0.0)
            landmarks_106 = getattr(face, "landmark_2d_106", None)
            if landmarks_106 is None:
                landmarks_106 = face.get("landmark_2d_106")
            observations.append(
                {
                    "embedding": np.asarray(face.normed_embedding, dtype=np.float64),
                    "bbox": bbox,
                    "detector_confidence": float(face.det_score),
                    "relative_face_area": area / image_area,
                    "landmarks_5": np.asarray(face.kps, dtype=np.float64),
                    "landmarks_106": np.asarray(landmarks_106, dtype=np.float64),
                }
            )
        return observations


def build_face_observations(
    rows: Iterable[Mapping[str, Any]],
    extractor: FaceObservationExtractor,
    contract: ModelContract,
    out_dir: Path,
) -> dict[str, Any]:
    accepted: list[FaceObservation] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        path = Path(str(row["resolved_path"]))
        subject_id = str(row["subject_id"])
        try:
            accepted.append(extract_face_observation(path, subject_id, extractor, contract))
        except Exception as exc:  # noqa: BLE001 - every noisy image needs a recorded reason.
            rejected.append(
                {
                    "image_id": path.stem,
                    "subject_id": subject_id,
                    "path": str(path),
                    "rejection_reason": str(exc),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    image_ids = np.asarray([item.image_id for item in accepted])
    subject_ids = np.asarray([item.subject_id for item in accepted])
    neural = (
        np.stack([item.neural_vector for item in accepted])
        if accepted
        else np.empty((0, contract.embedding_dimension), dtype=np.float64)
    )
    geometry = (
        np.stack([item.geometry.values for item in accepted])
        if accepted
        else np.empty((0, 213), dtype=np.float64)
    )
    np.savez_compressed(
        out_dir / "neural_vectors.npz",
        image_ids=image_ids,
        subject_ids=subject_ids,
        vectors=neural,
        contract_hash=np.asarray([contract.sha256()]),
    )
    np.savez_compressed(
        out_dir / "geometry_vectors.npz",
        image_ids=image_ids,
        subject_ids=subject_ids,
        vectors=geometry,
        schema_version=np.asarray(["insightface_2d_106_shape_v1"]),
    )
    manifest_rows = [
        {
            "image_id": item.image_id,
            "subject_id": item.subject_id,
            "path": str(item.path),
            "contract_hash": item.contract_hash,
            "detector_confidence": item.detector_confidence,
            "bbox": list(item.bbox),
            "relative_face_area": item.relative_face_area,
            "quality": item.quality,
            "geometry_schema": item.geometry.schema_version,
            "accepted": True,
        }
        for item in accepted
    ]
    (out_dir / "observations.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    (out_dir / "rejected_observations.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rejected),
        encoding="utf-8",
    )
    (out_dir / "model_contract.json").write_text(
        json.dumps(contract.to_mapping(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": "face_observations_v1",
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "subject_count": len({item.subject_id for item in accepted}),
        "contract_hash": contract.sha256(),
    }
    (out_dir / "observations_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return summary
