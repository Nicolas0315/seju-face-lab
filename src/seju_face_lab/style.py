from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageOps

from .embeddings import iter_image_paths
from .model import CentroidModel


class ImageStyleBackend(Protocol):
    name: str
    description: str

    def encode_path(self, path: Path) -> np.ndarray:
        ...

    def encode_pil(self, image: Image.Image) -> np.ndarray:
        ...


@dataclass(frozen=True)
class StyleScore:
    image_id: str
    path: str
    cosine_to_mean_style: float
    cosine_to_median_style: float
    style_score: float


class OpenClipStyleBackend:
    """OpenCLIP image embeddings for style/photographic similarity scoring."""

    name: str = "clip-style"
    description: str = (
        "OpenCLIP image embeddings for style similarity. Use as a separate axis from "
        "face-geometry centroid scores."
    )

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self._torch: Any = None
        self._model: Any = None
        self._preprocess: Any = None
        self._resolved_device: str | None = None

    def encode_path(self, path: Path) -> np.ndarray:
        image = Image.open(path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        return self.encode_pil(image)

    def encode_pil(self, image: Image.Image) -> np.ndarray:
        torch, model, preprocess, device = self._load()
        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            if str(device).startswith("cuda"):
                with torch.autocast(device_type="cuda"):
                    features = model.encode_image(tensor)
            else:
                features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.detach().cpu().numpy()[0].astype(np.float32)

    def _load(self) -> tuple[Any, Any, Any, str]:
        if self._model is not None and self._preprocess is not None and self._resolved_device:
            return self._torch, self._model, self._preprocess, self._resolved_device

        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "OpenCLIP is not installed. Install the optional clip extra before using "
                "style-evaluate: pip install -e '.[clip]'"
            ) from exc

        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
        )
        model.eval()
        model.to(device)

        self._torch = torch
        self._model = model
        self._preprocess = preprocess
        self._resolved_device = device
        return torch, model, preprocess, device


def score_style_images(
    model: CentroidModel,
    images_dir: Path,
    backend: ImageStyleBackend | None = None,
    failed_paths: list[str] | None = None,
) -> list[StyleScore]:
    active_backend = backend or OpenClipStyleBackend()
    mean_style = active_backend.encode_pil(_appearance_to_image(model.mean_appearance))
    median_style = active_backend.encode_pil(_appearance_to_image(model.median_appearance))

    scores: list[StyleScore] = []
    for path in iter_image_paths(images_dir):
        try:
            vector = active_backend.encode_path(path)
        except Exception:  # noqa: BLE001 - keep batch evaluation running and report failures.
            if failed_paths is None:
                raise
            failed_paths.append(str(path))
            continue
        scores.append(_score_style_vector(path, vector, mean_style, median_style))
    return sorted(scores, key=lambda item: item.style_score, reverse=True)


def write_style_scores(
    scores: list[StyleScore],
    out_dir: Path,
    failed_paths: list[str] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = failed_paths or []
    csv_lines = [
        "image_id,path,cosine_to_mean_style,cosine_to_median_style,style_score"
    ]
    for score in scores:
        csv_lines.append(
            ",".join(
                [
                    _csv(score.image_id),
                    _csv(score.path),
                    f"{score.cosine_to_mean_style:.6f}",
                    f"{score.cosine_to_median_style:.6f}",
                    f"{score.style_score:.6f}",
                ]
            )
        )
    (out_dir / "style_scores.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8-sig")
    (out_dir / "style_evaluation.md").write_text(
        _render_style_scores(scores, failed),
        encoding="utf-8",
    )
    (out_dir / "style_summary.json").write_text(
        json.dumps(_style_summary(scores, failed), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _score_style_vector(
    path: Path,
    vector: np.ndarray,
    mean_style: np.ndarray,
    median_style: np.ndarray,
) -> StyleScore:
    cosine_mean = _cosine(vector, mean_style)
    cosine_median = _cosine(vector, median_style)
    return StyleScore(
        image_id=path.stem,
        path=str(path),
        cosine_to_mean_style=cosine_mean,
        cosine_to_median_style=cosine_median,
        style_score=(cosine_mean + cosine_median) / 2.0,
    )


def _appearance_to_image(appearance: np.ndarray) -> Image.Image:
    pixels = np.clip(appearance * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _render_style_scores(scores: list[StyleScore], failed_paths: list[str]) -> str:
    lines = ["# generated-image style evaluation", ""]
    if not scores:
        lines.extend(["No generated images found.", ""])
    else:
        lines.append("| rank | image_id | style_score | cosine_mean_style | cosine_median_style |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for rank, score in enumerate(scores, start=1):
            lines.append(
                f"| {rank} | {score.image_id} | {score.style_score:.4f} | "
                f"{score.cosine_to_mean_style:.4f} | {score.cosine_to_median_style:.4f} |"
            )
        lines.append("")
    if failed_paths:
        lines.append("## Failed Images")
        lines.append("")
        lines.extend(f"- {path}" for path in failed_paths)
        lines.append("")
    lines.extend(
        [
            "Scores are approximate OpenCLIP image-style similarity against the local centroid renderings.",
            "They are a style/photographic axis, not face geometry or identity similarity.",
            "",
        ]
    )
    return "\n".join(lines)


def _style_summary(scores: list[StyleScore], failed_paths: list[str]) -> dict:
    if not scores:
        return {
            "image_count": 0,
            "failed_count": len(failed_paths),
            "failed_paths": failed_paths[:20],
            "best_image_id": None,
            "best_style_score": None,
            "mean_style_score": None,
            "median_style_score": None,
            "top_images": [],
            "boundary": "Approximate style similarity only; not face geometry.",
        }
    style_scores = np.asarray([score.style_score for score in scores], dtype=np.float32)
    best = scores[0]
    return {
        "image_count": len(scores),
        "failed_count": len(failed_paths),
        "failed_paths": failed_paths[:20],
        "best_image_id": best.image_id,
        "best_style_score": round(float(best.style_score), 6),
        "mean_style_score": round(float(np.mean(style_scores)), 6),
        "median_style_score": round(float(np.median(style_scores)), 6),
        "top_images": [
            {
                "image_id": score.image_id,
                "path": score.path,
                "style_score": round(float(score.style_score), 6),
                "cosine_to_mean_style": round(float(score.cosine_to_mean_style), 6),
                "cosine_to_median_style": round(float(score.cosine_to_median_style), 6),
            }
            for score in scores[:5]
        ],
        "boundary": "Approximate style similarity only; not face geometry.",
    }


def _csv(value: str) -> str:
    escaped = value.replace('"', '""')
    return f'"{escaped}"'
