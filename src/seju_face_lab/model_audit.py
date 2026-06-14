from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np


def write_model_audit(model_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = build_model_audit(model_dir)
    (out_dir / "model_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "model_audit.md").write_text(_render_model_audit(audit), encoding="utf-8")
    return audit


def build_model_audit(model_dir: Path) -> dict[str, Any]:
    profile = _load_json(model_dir / "profile.json")
    centroids = _load_centroids(model_dir / "centroids.npz")
    return {
        "model_dir": str(model_dir),
        "image_count": profile.get("image_count"),
        "embedding_dim": profile.get("embedding_dim"),
        "appearance_shape": profile.get("appearance_shape"),
        "mean_face": str(model_dir / "mean_face.png"),
        "median_face": str(model_dir / "median_face.png"),
        "centroids": centroids,
        "descriptor_delta": _descriptor_delta(profile.get("descriptors", {})),
        "boundary": (
            "Model audit describes local aggregate vectors and rendered centroid appearances. "
            "It is not identity, attractiveness, ethnicity, or an objective face-type label."
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_centroids(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    if not zipfile.is_zipfile(path):
        return {"available": False, "error": "unreadable centroids.npz"}
    try:
        with np.load(path, allow_pickle=False) as data:
            mean_embedding = _array(data, "mean_embedding")
            median_embedding = _array(data, "median_embedding")
            mean_appearance = _array(data, "mean_appearance")
            median_appearance = _array(data, "median_appearance")
            return {
                "available": True,
                "mean_embedding": _array_summary(mean_embedding),
                "median_embedding": _array_summary(median_embedding),
                "mean_appearance": _array_summary(mean_appearance),
                "median_appearance": _array_summary(median_appearance),
                "mean_median_embedding": _pair_summary(mean_embedding, median_embedding),
                "mean_median_appearance": _pair_summary(mean_appearance, median_appearance),
            }
    except (OSError, ValueError, zipfile.BadZipFile):
        return {"available": False, "error": "unreadable centroids.npz"}


def _array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data:
        return None
    return np.asarray(data[key], dtype=np.float32)


def _array_summary(array: np.ndarray | None) -> dict[str, Any]:
    if array is None:
        return {"available": False}
    flat = array.reshape(-1)
    return {
        "available": True,
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "l2_norm": round(float(np.linalg.norm(flat)), 6),
        "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
        "preview": [round(float(value), 6) for value in flat[:8]],
    }


def _pair_summary(left: np.ndarray | None, right: np.ndarray | None) -> dict[str, Any]:
    if left is None or right is None:
        return {"available": False}
    if left.shape != right.shape:
        return {
            "available": False,
            "error": "shape mismatch",
            "left_shape": [int(value) for value in left.shape],
            "right_shape": [int(value) for value in right.shape],
        }
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    diff = left_flat - right_flat
    return {
        "available": True,
        "cosine": round(_cosine(left_flat, right_flat), 6),
        "euclidean": round(float(np.linalg.norm(diff)), 6),
        "mean_abs_delta": round(float(np.mean(np.abs(diff))), 6),
        "max_abs_delta": round(float(np.max(np.abs(diff))), 6),
    }


def _descriptor_delta(descriptors: Any) -> dict[str, Any]:
    if not isinstance(descriptors, dict):
        return {}
    mean = descriptors.get("mean")
    median = descriptors.get("median")
    if not isinstance(mean, dict) or not isinstance(median, dict):
        return {}
    delta: dict[str, Any] = {}
    for key in sorted(set(mean) | set(median)):
        left = mean.get(key)
        right = median.get(key)
        if isinstance(left, int | float) and isinstance(right, int | float):
            delta[key] = round(float(left) - float(right), 6)
    return delta


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def _render_model_audit(audit: dict[str, Any]) -> str:
    centroids = audit["centroids"]
    embedding_pair = centroids.get("mean_median_embedding", {})
    appearance_pair = centroids.get("mean_median_appearance", {})
    lines = [
        "# seju-face model audit",
        "",
        f"- model_dir: {audit['model_dir']}",
        f"- image_count: {_value(audit['image_count'])}",
        f"- embedding_dim: {_value(audit['embedding_dim'])}",
        f"- mean_face: {audit['mean_face']}",
        f"- median_face: {audit['median_face']}",
        f"- centroids_available: {centroids.get('available', False)}",
        f"- mean_embedding_sha256: {_nested(centroids, 'mean_embedding', 'sha256')}",
        f"- median_embedding_sha256: {_nested(centroids, 'median_embedding', 'sha256')}",
        f"- mean_median_embedding_cosine: {_value(embedding_pair.get('cosine'))}",
        f"- mean_median_embedding_euclidean: {_value(embedding_pair.get('euclidean'))}",
        f"- mean_median_appearance_cosine: {_value(appearance_pair.get('cosine'))}",
        f"- mean_median_appearance_euclidean: {_value(appearance_pair.get('euclidean'))}",
        "",
        "## Descriptor Delta",
        "",
    ]
    descriptor_delta = audit.get("descriptor_delta", {})
    if descriptor_delta:
        lines.extend(f"- {key}: {value}" for key, value in descriptor_delta.items())
    else:
        lines.append("- no numeric descriptor delta available")
    lines.extend(["", "## Boundary", "", audit["boundary"], ""])
    return "\n".join(lines)


def _nested(root: dict[str, Any], key: str, field: str) -> str:
    value = root.get(key)
    if not isinstance(value, dict):
        return ""
    return _value(value.get(field))


def _value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
