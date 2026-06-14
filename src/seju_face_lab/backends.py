from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from typing import Protocol

import numpy as np
from PIL import Image, ImageOps

from .embeddings import ImageVector, normalize_image, vectorize_image, vectorize_normalized_image


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
class OpenCVFaceBackend:
    name: str = "opencv-face"
    extra: str = "vision"
    description: str = "OpenCV Haar face crop normalization before deterministic vectors."

    def vectorize(self, path: Path, crop: str = "center") -> ImageVector:
        if crop != "center":
            raise ValueError("opencv-face backend only supports crop='center'")
        cv2 = _import_cv2()
        image = Image.open(path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        face_crop = _opencv_face_crop(cv2, image, path)
        normalized = normalize_image(face_crop, crop="none")
        return vectorize_normalized_image(path, normalized)


class InsightFaceBackend:
    """InsightFace ArcFace 512D embeddings via ONNX Runtime.

    Returns 512D normed embeddings. Centroid models built with this backend
    are incompatible with the deterministic 1073D backend — build and evaluate
    must always use the same backend.
    Falls back to deterministic vectorization when no face is detected.
    """

    name: str = "insightface"
    description: str = (
        "InsightFace ArcFace 512D face embeddings via ONNX Runtime; attempts CUDA when available. "
        "Requires: pip install 'seju-face-lab[face]'"
    )

    def __init__(self, gpu_id: int = 0, model_pack: str = "buffalo_l") -> None:
        self.gpu_id = gpu_id
        self.model_pack = model_pack
        self._app: Any = None
        self._lock: Lock = Lock()

    def _get_app(self) -> Any:
        if self._app is not None:
            return self._app
        with self._lock:
            if self._app is not None:
                return self._app
            try:
                from insightface.app import FaceAnalysis
            except ImportError as exc:
                raise RuntimeError(
                    "insightface is not installed. "
                    "Run: pip install insightface onnxruntime-gpu"
                ) from exc
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.gpu_id >= 0
                else ["CPUExecutionProvider"]
            )
            app = FaceAnalysis(
                name=self.model_pack,
                providers=providers,
                allowed_modules=["detection", "recognition"],
            )
            app.prepare(ctx_id=self.gpu_id)
            self._app = app
        return self._app

    def vectorize(self, path: Path, crop: str = "center") -> ImageVector:
        cv2 = _import_cv2()
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise ValueError(f"Could not load image: {path}")

        app = self._get_app()
        faces = app.get(img_bgr)

        if not faces:
            raise ValueError(f"No face detected in {path}")

        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        embedding = face.normed_embedding.astype(np.float32)

        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(img_bgr.shape[1], x2)
        y2 = min(img_bgr.shape[0], y2)
        if x2 > x1 and y2 > y1:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(img_rgb[y1:y2, x1:x2]).resize(
                (64, 64), Image.Resampling.LANCZOS
            )
            appearance = np.asarray(face_pil, dtype=np.float32) / 255.0
        else:
            appearance = np.zeros((64, 64, 3), dtype=np.float32)

        return ImageVector(
            image_id=path.stem,
            path=path,
            embedding=embedding,
            appearance=appearance,
            descriptors={},
        )


class DeepFaceBackend:
    """DeepFace-family face embeddings through the DeepFace.represent API."""

    name: str = "deepface"
    description: str = (
        "DeepFace-family OSS embeddings via DeepFace.represent; defaults to ArcFace. "
        "Requires: pip install 'seju-face-lab[deepface]'"
    )

    def __init__(
        self,
        model_name: str = "ArcFace",
        detector_backend: str = "opencv",
        enforce_detection: bool = True,
        align: bool = True,
    ) -> None:
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection
        self.align = align

    def vectorize(self, path: Path, crop: str = "center") -> ImageVector:
        deepface = _import_deepface()
        reps = deepface.represent(
            img_path=str(path),
            model_name=self.model_name,
            detector_backend=self.detector_backend,
            enforce_detection=self.enforce_detection,
            align=self.align,
        )
        if isinstance(reps, dict):
            reps = [reps]
        if not reps:
            raise ValueError(f"No face detected in {path}")

        rep = max(reps, key=_deepface_area_size)
        raw_embedding = rep.get("embedding")
        if raw_embedding is None:
            raise ValueError(f"DeepFace did not return an embedding for {path}")
        embedding = np.asarray(raw_embedding, dtype=np.float32)
        if embedding.ndim != 1 or embedding.size == 0:
            raise ValueError(f"DeepFace returned an invalid embedding for {path}")
        embedding = _l2_normalize(embedding)

        appearance = _deepface_appearance(path, rep, crop)
        return ImageVector(
            image_id=path.stem,
            path=path,
            embedding=embedding,
            appearance=appearance,
            descriptors={},
        )


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


def _make_insightface_backend() -> InsightFaceBackend | PlannedBackend:
    """Return InsightFaceBackend if onnxruntime-gpu is installed, else PlannedBackend."""
    try:
        import onnxruntime  # noqa: F401
        import insightface  # noqa: F401
        return InsightFaceBackend()
    except ImportError:
        return PlannedBackend(
            name="insightface",
            extra="face",
            description="InsightFace ArcFace 512D GPU face embeddings (install: pip install insightface onnxruntime-gpu).",
            notes="Best candidate for robust face embeddings on RTX machines; keep raw images local.",
        )


def _make_deepface_backend() -> DeepFaceBackend | PlannedBackend:
    """Return DeepFaceBackend if the optional package is installed."""
    if importlib.util.find_spec("deepface") is not None:
        return DeepFaceBackend()
    return PlannedBackend(
        name="deepface",
        extra="deepface",
        description="DeepFace-family OSS adapters for model comparison and face QA.",
        notes="Use for cross-checking embeddings; keep it optional because dependencies are heavy.",
    )


BACKENDS: dict[str, VectorBackend] = {
    "deterministic": DeterministicBackend(),
    "opencv-face": OpenCVFaceBackend(),
    "insightface": _make_insightface_backend(),
    "deepface": _make_deepface_backend(),
    "diffusion-generation": PlannedBackend(
        name="diffusion-generation",
        extra="generation",
        description="Diffusers/ComfyUI prompt and candidate-generation loop.",
        notes="Use RTX machines for batches, then score outputs with evaluate/review commands.",
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


def get_insightface_backend(gpu_id: int = 0, model_pack: str = "buffalo_l") -> InsightFaceBackend:
    """Create a configured InsightFaceBackend instance (not from the shared registry)."""
    return InsightFaceBackend(gpu_id=gpu_id, model_pack=model_pack)


def get_deepface_backend(
    model_name: str = "ArcFace",
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
) -> DeepFaceBackend:
    """Create a configured DeepFaceBackend instance (not from the shared registry)."""
    return DeepFaceBackend(
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )


def backend_help() -> str:
    lines = ["Available vector backends:", ""]
    for name in sorted(BACKENDS):
        backend = BACKENDS[name]
        if isinstance(backend, PlannedBackend):
            state = f"planned extra={backend.extra}"
        elif isinstance(backend, InsightFaceBackend):
            state = "implemented extra=face"
        elif isinstance(backend, DeepFaceBackend):
            state = "implemented extra=deepface"
        elif isinstance(backend, OpenCVFaceBackend):
            state = f"implemented extra={backend.extra}"
        else:
            state = "ready"
        lines.append(f"- {name}: {state}")
        lines.append(f"  {backend.description}")
        if isinstance(backend, PlannedBackend):
            lines.append(f"  {backend.notes}")
    lines.extend(
        [
            "",
            "Available style axes:",
            "",
            "- clip-style: implemented extra=clip",
            "  OpenCLIP image embeddings for style/photographic similarity scoring.",
            "  Use via: python -m seju_face_lab style-evaluate --model MODEL --images IMAGES --out OUT",
        ]
    )
    return "\n".join(lines)


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Install the optional vision extra before using "
            "backend=opencv-face."
        ) from exc
    return cv2


def _import_deepface() -> Any:
    try:
        from deepface import DeepFace
    except ImportError as exc:
        raise RuntimeError(
            "DeepFace is not installed. Install the optional deepface extra before using "
            "backend=deepface."
        ) from exc
    return DeepFace


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _deepface_area_size(rep: dict[str, Any]) -> int:
    area = rep.get("facial_area") or {}
    width = area.get("w") or area.get("width") or 0
    height = area.get("h") or area.get("height") or 0
    try:
        return int(width) * int(height)
    except (TypeError, ValueError):
        return 0


def _deepface_appearance(path: Path, rep: dict[str, Any], crop: str) -> np.ndarray:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    area = rep.get("facial_area") or {}
    try:
        x = int(area["x"])
        y = int(area["y"])
        width = int(area["w"])
        height = int(area["h"])
    except (KeyError, TypeError, ValueError):
        normalized = normalize_image(image, crop=crop)
        return np.asarray(normalized, dtype=np.float32) / 255.0

    face_crop = image.crop(_square_bounds(x, y, width, height, image.width, image.height))
    face_crop = face_crop.resize((64, 64), Image.Resampling.LANCZOS)
    return np.asarray(face_crop, dtype=np.float32) / 255.0


def _opencv_face_crop(cv2: Any, image: Image.Image, path: Path) -> Image.Image:
    rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"OpenCV face cascade could not be loaded: {cascade_path}")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
    if len(faces) == 0:
        raise ValueError(f"No face detected in {path}")
    x, y, w, h = max((tuple(int(value) for value in face) for face in faces), key=lambda face: face[2] * face[3])
    return image.crop(_square_bounds(x, y, w, h, image.width, image.height))


def _square_bounds(x: int, y: int, width: int, height: int, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    margin = int(round(max(width, height) * 0.25))
    center_x = x + width // 2
    center_y = y + height // 2
    side = max(width, height) + margin * 2
    left = center_x - side // 2
    top = center_y - side // 2
    right = left + side
    bottom = top + side

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > image_width:
        left -= right - image_width
        right = image_width
    if bottom > image_height:
        top -= bottom - image_height
        bottom = image_height

    left = max(0, left)
    top = max(0, top)
    right = min(image_width, right)
    bottom = min(image_height, bottom)
    return left, top, right, bottom
