from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class DatasetAudit:
    clean: list[dict[str, Any]]
    rejected: list[dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        reasons: dict[str, int] = {}
        for row in self.rejected:
            reason = str(row["rejection_reason"])
            reasons[reason] = reasons.get(reason, 0) + 1
        return {
            "schema_version": "face_dataset_audit_v1",
            "clean_count": len(self.clean),
            "rejected_count": len(self.rejected),
            "rejection_reasons": dict(sorted(reasons.items())),
            "subject_count": len({str(row["subject_id"]) for row in self.clean}),
        }


def gate_download_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    allowed_hosts: set[str],
) -> DatasetAudit:
    clean: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    normalized_hosts = {host.lower() for host in allowed_hosts}
    for source_row in rows:
        row = dict(source_row)
        row["subject_id"] = str(row.get("subject_id") or row.get("talent_slug") or "")
        reason = _row_rejection_reason(row, normalized_hosts)
        if reason is None:
            clean.append(row)
        else:
            row["rejection_reason"] = reason
            rejected.append(row)
    return DatasetAudit(clean=clean, rejected=rejected)


def audit_face_dataset(
    source_manifest: Path,
    download_manifest: Path,
    image_root: Path,
    out_dir: Path,
    *,
    allowed_hosts: set[str] | None = None,
) -> DatasetAudit:
    allowed = allowed_hosts or {"seju.tokyo"}
    source_rows = _read_jsonl(source_manifest)
    source_by_url = {str(row.get("image_url")): row for row in source_rows}
    merged: list[dict[str, Any]] = []
    for download in _read_jsonl(download_manifest):
        row = dict(download)
        source = source_by_url.get(str(row.get("image_url")))
        row["source_manifest_present"] = source is not None
        row["eligible_for_analysis"] = bool(source and source.get("eligible_for_analysis"))
        row["subject_id"] = str(row.get("talent_slug") or "")
        local_path = image_root / Path(str(row.get("path", ""))).name
        row["resolved_path"] = str(local_path.resolve())
        row["file_exists"] = local_path.is_file()
        if local_path.is_file():
            row["actual_sha256"] = _sha256_file(local_path)
            row["sha256_matches"] = row["actual_sha256"] == row.get("sha256")
        else:
            row["actual_sha256"] = None
            row["sha256_matches"] = False
        merged.append(row)

    audit = gate_download_rows(merged, allowed_hosts=allowed)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_manifest_path = out_dir / "clean_manifest.jsonl"
    _write_jsonl(clean_manifest_path, audit.clean)
    _write_jsonl(out_dir / "rejected_manifest.jsonl", audit.rejected)
    summary = audit.summary()
    summary.update(
        {
            "source_manifest_sha256": _sha256_file(source_manifest),
            "download_manifest_sha256": _sha256_file(download_manifest),
            "clean_manifest_sha256": _sha256_file(clean_manifest_path),
        }
    )
    (out_dir / "dataset_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def perceptual_hash(path: Path, size: int = 8) -> int:
    image = ImageOps.exif_transpose(Image.open(path)).convert("L")
    pixels = np.asarray(image.resize((size + 1, size), Image.Resampling.LANCZOS))
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def cluster_near_duplicates(
    rows: Iterable[Mapping[str, Any]],
    *,
    hamming_threshold: int = 6,
    ssim_threshold: float = 0.92,
) -> list[dict[str, Any]]:
    indexed = [dict(row) for row in rows]
    clusters: list[dict[str, Any]] = []
    for subject_id in sorted({str(row["subject_id"]) for row in indexed}):
        subject_rows = [row for row in indexed if str(row["subject_id"]) == subject_id]
        hashes = [perceptual_hash(Path(str(row["path"]))) for row in subject_rows]
        pixels = [_duplicate_pixels(Path(str(row["path"]))) for row in subject_rows]
        parents = list(range(len(subject_rows)))

        for left in range(len(subject_rows)):
            for right in range(left + 1, len(subject_rows)):
                if hamming_distance(hashes[left], hashes[right]) > hamming_threshold:
                    continue
                if structural_similarity(pixels[left], pixels[right]) >= ssim_threshold:
                    _union_parents(parents, left, right)

        groups: dict[int, list[dict[str, Any]]] = {}
        for index, row in enumerate(subject_rows):
            groups.setdefault(_find_parent(parents, index), []).append(row)
        for group_index, members in enumerate(groups.values(), start=1):
            ordered = sorted(
                members,
                key=lambda row: (-float(row.get("quality", 0.0)), str(row["image_id"])),
            )
            clusters.append(
                {
                    "cluster_id": f"{subject_id}:{group_index:04d}",
                    "subject_id": subject_id,
                    "representative_id": str(ordered[0]["image_id"]),
                    "suppressed_ids": sorted(str(row["image_id"]) for row in ordered[1:]),
                    "member_ids": sorted(str(row["image_id"]) for row in members),
                    "method_version": "dhash64_ssim_v1",
                }
            )
    return clusters


def structural_similarity(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("SSIM inputs must have the same non-empty shape")
    dynamic_range = 255.0 if max(float(a.max()), float(b.max())) > 1.0 else 1.0
    c1 = (0.01 * dynamic_range) ** 2
    c2 = (0.03 * dynamic_range) ** 2
    mean_a, mean_b = float(a.mean()), float(b.mean())
    var_a, var_b = float(a.var()), float(b.var())
    covariance = float(((a - mean_a) * (b - mean_b)).mean())
    numerator = (2 * mean_a * mean_b + c1) * (2 * covariance + c2)
    denominator = (mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2)
    return float(numerator / denominator) if denominator else 1.0


def _duplicate_pixels(path: Path) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert("L")
    return np.asarray(image.resize((64, 64), Image.Resampling.LANCZOS), dtype=np.float64)


def _find_parent(parents: list[int], index: int) -> int:
    while parents[index] != index:
        parents[index] = parents[parents[index]]
        index = parents[index]
    return index


def _union_parents(parents: list[int], left: int, right: int) -> None:
    root_left = _find_parent(parents, left)
    root_right = _find_parent(parents, right)
    if root_left != root_right:
        parents[root_right] = root_left


def _row_rejection_reason(row: Mapping[str, Any], allowed_hosts: set[str]) -> str | None:
    if not row.get("source_manifest_present", True):
        return "missing_source_manifest"
    if not row.get("eligible_for_analysis", True):
        return "ineligible_source"
    for field in ("profile_url", "image_url"):
        host = (urlparse(str(row.get(field, ""))).hostname or "").lower()
        if host not in allowed_hosts:
            return f"disallowed_{field}_host"
    image_path = urlparse(str(row.get("image_url", ""))).path.lower()
    filename = Path(image_path).name
    if re.search(r"(^|[_-])sign(?:ature)?(?:[_\-.]|$)", filename):
        return "signature_asset"
    if any(token in filename for token in ("logo", "favicon", "icon")):
        return "non_portrait_asset"
    if row.get("file_exists") is False:
        return "missing_file"
    if row.get("sha256_matches") is False:
        return "sha256_mismatch"
    if not str(row.get("subject_id", "")).strip():
        return "missing_subject_id"
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
