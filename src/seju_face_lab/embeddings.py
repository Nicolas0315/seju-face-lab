from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
APPEARANCE_SIZE = 64
EMBED_SIZE = 32


@dataclass(frozen=True)
class ImageVector:
    image_id: str
    path: Path
    embedding: np.ndarray
    appearance: np.ndarray
    descriptors: dict[str, float]


def iter_image_paths(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() in IMAGE_SUFFIXES:
        return [root]
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_normalized_image(path: Path, crop: str = "center") -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    if crop == "center":
        image = _center_square_crop(image)
    elif crop != "none":
        raise ValueError(f"Unsupported crop mode: {crop}")
    return image.resize((APPEARANCE_SIZE, APPEARANCE_SIZE), Image.Resampling.LANCZOS)


def vectorize_image(path: Path, crop: str = "center") -> ImageVector:
    image = load_normalized_image(path, crop=crop)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = np.asarray(image.convert("L").resize((EMBED_SIZE, EMBED_SIZE)), dtype=np.float32) / 255.0

    embedding_parts = [
        _standardize(gray).reshape(-1),
        _color_histogram(rgb, bins=8),
        _edge_features(gray),
        _region_features(gray, rgb),
    ]
    embedding = np.concatenate(embedding_parts).astype(np.float32)
    embedding = _l2_normalize(embedding)
    descriptors = describe_image_arrays(gray, rgb)
    image_id = path.stem
    return ImageVector(
        image_id=image_id,
        path=path,
        embedding=embedding,
        appearance=rgb.astype(np.float32),
        descriptors=descriptors,
    )


def describe_image_arrays(gray: np.ndarray, rgb: np.ndarray) -> dict[str, float]:
    left = gray[:, : gray.shape[1] // 2]
    right = np.fliplr(gray[:, gray.shape[1] // 2 :])
    min_width = min(left.shape[1], right.shape[1])
    symmetry_error = float(np.mean(np.abs(left[:, :min_width] - right[:, :min_width])))

    gy, gx = np.gradient(gray)
    edges = np.sqrt(gx * gx + gy * gy)

    upper = gray[int(gray.shape[0] * 0.20) : int(gray.shape[0] * 0.48), :]
    middle = gray[int(gray.shape[0] * 0.32) : int(gray.shape[0] * 0.68), :]
    lower = gray[int(gray.shape[0] * 0.62) :, :]

    warmth = float(np.mean(rgb[:, :, 0]) - np.mean(rgb[:, :, 2]))
    saturation_proxy = float(np.mean(np.max(rgb, axis=2) - np.min(rgb, axis=2)))

    return {
        "luminance": float(np.mean(gray)),
        "contrast": float(np.std(gray)),
        "warmth": warmth,
        "saturation": saturation_proxy,
        "edge_density": float(np.mean(edges)),
        "symmetry": float(1.0 - symmetry_error),
        "upper_band_darkness": float(1.0 - np.mean(upper)),
        "middle_luminance": float(np.mean(middle)),
        "lower_luminance": float(np.mean(lower)),
    }


def descriptors_from_appearance(appearance: np.ndarray) -> dict[str, float]:
    rgb = np.clip(appearance, 0.0, 1.0)
    gray = np.mean(rgb, axis=2)
    return describe_image_arrays(gray.astype(np.float32), rgb.astype(np.float32))


def render_appearance(appearance: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.clip(appearance * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(out_path)


def _center_square_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def _standardize(gray: np.ndarray) -> np.ndarray:
    std = float(np.std(gray))
    if std < 1e-6:
        return gray - float(np.mean(gray))
    return (gray - float(np.mean(gray))) / std


def _color_histogram(rgb: np.ndarray, bins: int) -> np.ndarray:
    histograms = []
    for channel in range(3):
        hist, _ = np.histogram(rgb[:, :, channel], bins=bins, range=(0.0, 1.0), density=True)
        histograms.append(hist.astype(np.float32))
    return _l2_normalize(np.concatenate(histograms))


def _edge_features(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray)
    magnitude = np.sqrt(gx * gx + gy * gy)
    angle = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
    hist, _ = np.histogram(angle, bins=8, range=(0.0, 1.0), weights=magnitude)
    stats = np.array(
        [
            float(np.mean(magnitude)),
            float(np.std(magnitude)),
            float(np.percentile(magnitude, 75)),
            float(np.percentile(magnitude, 90)),
        ],
        dtype=np.float32,
    )
    return _l2_normalize(np.concatenate([hist.astype(np.float32), stats]))


def _region_features(gray: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    regions = [
        gray[: h // 3, :],
        gray[h // 3 : 2 * h // 3, :],
        gray[2 * h // 3 :, :],
        gray[:, : w // 2],
        gray[:, w // 2 :],
    ]
    values: list[float] = []
    for region in regions:
        values.extend([float(np.mean(region)), float(np.std(region))])
    values.extend(
        [
            float(np.mean(rgb[:, :, 0])),
            float(np.mean(rgb[:, :, 1])),
            float(np.mean(rgb[:, :, 2])),
        ]
    )
    return np.asarray(values, dtype=np.float32)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)
