from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .embeddings import ImageVector, vectorize_image


class VectorBackend(Protocol):
    name: str
    description: str

    def vectorize(self, path: Path, crop: str = "center") -> ImageVector:
        ...


@dataclass(frozen=True)
class DeterministicBackend:
    name: str = "deterministic"
    description: str = (
        "Built-in 1073D image vector from normalized grayscale pixels, color histograms, "
        "edge features, and region descriptors. No neural model required."
    )

    def vectorize(self, path: Path, crop: str = "center") -> ImageVector:
        return vectorize_image(path, crop=crop)


@dataclass(frozen=True)
class PlannedBackend:
    name: str
    description: str
    extra: str
    notes: str

    def vectorize(self, path: Path, crop: str = "center") -> ImageVector:
        raise RuntimeError(
            f"Backend '{self.name}' is a design placeholder. Install optional extra "
            f"'{self.extra}' and implement its adapter before using it."
        )


BACKENDS: dict[str, VectorBackend] = {
    "deterministic": DeterministicBackend(),
    "opencv-face": PlannedBackend(
        name="opencv-face",
        extra="vision",
        description="OpenCV Haar/DNN face crop normalization before deterministic vectors.",
        notes="Good first upgrade for local CPU/GPU preprocessing and face-box QA.",
    ),
    "insightface": PlannedBackend(
        name="insightface",
        extra="face",
        description="InsightFace/ONNXRuntime face embeddings for identity-agnostic centroid geometry.",
        notes="Best candidate for robust face embeddings on RTX machines; keep raw images local.",
    ),
    "clip-style": PlannedBackend(
        name="clip-style",
        extra="clip",
        description="OpenCLIP image embeddings for style/photographic similarity scoring.",
        notes="Use as a secondary style axis, not as the primary face-geometry vector.",
    ),
}


def get_vector_backend(name: str) -> VectorBackend:
    try:
        backend = BACKENDS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(BACKENDS))
        raise ValueError(f"Unknown backend '{name}'. Choices: {choices}") from exc
    if isinstance(backend, PlannedBackend):
        raise RuntimeError(
            f"Backend '{name}' is not implemented yet. Planned optional extra: {backend.extra}. "
            f"Notes: {backend.notes}"
        )
    return backend


def backend_help() -> str:
    lines = ["Available vector backends:", ""]
    for name in sorted(BACKENDS):
        backend = BACKENDS[name]
        state = "ready" if not isinstance(backend, PlannedBackend) else f"planned extra={backend.extra}"
        lines.append(f"- {name}: {state}")
        lines.append(f"  {backend.description}")
        if isinstance(backend, PlannedBackend):
            lines.append(f"  {backend.notes}")
    return "\n".join(lines)
