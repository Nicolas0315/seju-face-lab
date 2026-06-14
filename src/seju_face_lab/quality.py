from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
import numpy as np

from .backends import _import_cv2
from .embeddings import iter_image_paths


@dataclass(frozen=True)
class ImageQuality:
    image_id: str
    path: str
    face_count: int
    largest_face_area_ratio: float | None
    center_offset: float | None
    qa_pass: bool
    reason: str


def review_image_quality(images_dir: Path) -> list[ImageQuality]:
    cv2 = _import_cv2()
    reviews: list[ImageQuality] = []
    for path in iter_image_paths(images_dir):
        try:
            reviews.append(_review_one_image(cv2, path))
        except Exception as exc:  # noqa: BLE001 - keep generated-batch QA running per candidate.
            reviews.append(
                ImageQuality(
                    image_id=path.stem,
                    path=str(path),
                    face_count=0,
                    largest_face_area_ratio=None,
                    center_offset=None,
                    qa_pass=False,
                    reason=f"quality review failed: {exc}",
                )
            )
    return reviews


def write_image_quality(reviews: list[ImageQuality], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "image_quality.csv").write_text(_render_quality_csv(reviews), encoding="utf-8-sig")
    (out_dir / "image_quality.md").write_text(_render_quality_md(reviews), encoding="utf-8")
    (out_dir / "image_quality.json").write_text(
        json.dumps(_quality_summary(reviews), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def judge_face_quality(
    face_count: int,
    largest_face_area_ratio: float | None,
    center_offset: float | None,
) -> tuple[bool, str]:
    if face_count != 1:
        return False, "requires exactly one detected face"
    if largest_face_area_ratio is None or largest_face_area_ratio < 0.08:
        return False, "detected face is too small"
    if largest_face_area_ratio > 0.60:
        return False, "detected face is too close or cropped"
    if center_offset is None or center_offset > 0.24:
        return False, "detected face is off center"
    return True, "single centered face"


def _review_one_image(cv2: Any, path: Path) -> ImageQuality:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    faces = _detect_faces(cv2, image)
    if not faces:
        qa_pass, reason = judge_face_quality(0, None, None)
        return ImageQuality(path.stem, str(path), 0, None, None, qa_pass, reason)

    largest = max(faces, key=lambda face: face[2] * face[3])
    area_ratio = (largest[2] * largest[3]) / float(image.width * image.height)
    center_offset = _center_offset(largest, image.width, image.height)
    qa_pass, reason = judge_face_quality(len(faces), area_ratio, center_offset)
    return ImageQuality(
        image_id=path.stem,
        path=str(path),
        face_count=len(faces),
        largest_face_area_ratio=area_ratio,
        center_offset=center_offset,
        qa_pass=qa_pass,
        reason=reason,
    )


def _detect_faces(cv2: Any, image: Image.Image) -> list[tuple[int, int, int, int]]:
    rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"OpenCV face cascade could not be loaded: {cascade_path}")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
    return [tuple(int(value) for value in face) for face in faces]


def _center_offset(face: tuple[int, int, int, int], image_width: int, image_height: int) -> float:
    x, y, width, height = face
    face_center_x = x + width / 2.0
    face_center_y = y + height / 2.0
    offset_x = abs(face_center_x - image_width / 2.0) / image_width
    offset_y = abs(face_center_y - image_height / 2.0) / image_height
    return max(offset_x, offset_y)


def _render_quality_csv(reviews: list[ImageQuality]) -> str:
    lines = ["image_id,path,face_count,largest_face_area_ratio,center_offset,qa_pass,reason"]
    for review in reviews:
        lines.append(
            ",".join(
                [
                    _csv(review.image_id),
                    _csv(review.path),
                    str(review.face_count),
                    _format_optional(review.largest_face_area_ratio),
                    _format_optional(review.center_offset),
                    "true" if review.qa_pass else "false",
                    _csv(review.reason),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _render_quality_md(reviews: list[ImageQuality]) -> str:
    lines = ["# generated-image face quality", ""]
    if not reviews:
        lines.extend(["No generated images found.", ""])
        return "\n".join(lines)
    lines.append("| image | pass | faces | area_ratio | center_offset | reason |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for review in reviews:
        lines.append(
            f"| {review.image_id} | {'yes' if review.qa_pass else 'no'} | "
            f"{review.face_count} | {_format_optional(review.largest_face_area_ratio)} | "
            f"{_format_optional(review.center_offset)} | {review.reason} |"
        )
    lines.extend(
        [
            "",
            "QA is an OpenCV frontal-face gate for generated-image triage only.",
            "",
        ]
    )
    return "\n".join(lines)


def _quality_summary(reviews: list[ImageQuality]) -> dict:
    passed = [review for review in reviews if review.qa_pass]
    return {
        "image_count": len(reviews),
        "pass_count": len(passed),
        "fail_count": len(reviews) - len(passed),
        "pass_rate": round(len(passed) / len(reviews), 6) if reviews else None,
        "images": [
            {
                **asdict(review),
                "largest_face_area_ratio": _round_optional(review.largest_face_area_ratio),
                "center_offset": _round_optional(review.center_offset),
            }
            for review in reviews
        ],
        "boundary": "OpenCV face-quality triage only; not identity or attractiveness analysis.",
    }


def _format_optional(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _csv(value: str) -> str:
    escaped = value.replace('"', '""')
    return f'"{escaped}"'
