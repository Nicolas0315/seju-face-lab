from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .model import CentroidModel, load_model


def write_vector_export(
    model_dir: Path,
    out_path: Path,
    *,
    output_format: str,
    include_appearance: bool = False,
) -> dict[str, Any]:
    model = load_model(model_dir)
    payload = build_vector_export(model_dir, model, include_appearance=include_appearance)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    elif output_format == "csv":
        _write_vector_csv(payload, out_path)
    else:
        raise ValueError(f"Unsupported vector export format: {output_format}")
    return payload


def build_vector_export(
    model_dir: Path,
    model: CentroidModel,
    *,
    include_appearance: bool = False,
) -> dict[str, Any]:
    vectors = {
        "mean_embedding": _vector_payload(model.mean_embedding),
        "median_embedding": _vector_payload(model.median_embedding),
    }
    if include_appearance:
        vectors["mean_appearance"] = _vector_payload(model.mean_appearance)
        vectors["median_appearance"] = _vector_payload(model.median_appearance)
    return {
        "model_dir": str(model_dir),
        "image_count": len(model.image_ids),
        "embedding_dim": model.embedding_dim,
        "appearance_shape": list(model.appearance_shape),
        "include_appearance": include_appearance,
        "vectors": vectors,
        "boundary": (
            "Exported vectors summarize only the local reviewed image set. They are approximate "
            "model inputs for generation and scoring, not identity templates or objective labels."
        ),
    }


def _vector_payload(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array.astype(np.float32))
    flat = contiguous.reshape(-1)
    return {
        "shape": [int(value) for value in contiguous.shape],
        "dtype": str(contiguous.dtype),
        "l2_norm": round(float(np.linalg.norm(flat)), 6),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        "values": [float(value) for value in flat],
    }


def _write_vector_csv(payload: dict[str, Any], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "vector",
                "index",
                "value",
                "shape",
                "dtype",
                "l2_norm",
                "sha256",
            ]
        )
        for vector_name, vector in payload["vectors"].items():
            shape = "x".join(str(value) for value in vector["shape"])
            for index, value in enumerate(vector["values"]):
                writer.writerow(
                    [
                        vector_name,
                        index,
                        repr(float(value)),
                        shape,
                        vector["dtype"],
                        vector["l2_norm"],
                        vector["sha256"],
                    ]
                )
