from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import descriptors_from_appearance, render_appearance
from .prompting import PROMPT_PROFILES, negative_prompt_for_profile, prompt_from_descriptors


@dataclass(frozen=True)
class CentroidModel:
    image_ids: list[str]
    source_paths: list[str]
    mean_embedding: np.ndarray
    median_embedding: np.ndarray
    mean_appearance: np.ndarray
    median_appearance: np.ndarray
    descriptors: dict[str, dict[str, float]]
    embedding_dim: int
    appearance_shape: tuple[int, int, int]


def build_centroid_model(image_ids: list[str], source_paths: list[str], embeddings: np.ndarray, appearances: np.ndarray) -> CentroidModel:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if appearances.ndim != 4:
        raise ValueError("appearances must be a 4D array")
    if embeddings.shape[0] == 0:
        raise ValueError("at least one image is required")

    mean_embedding = _l2_normalize(np.mean(embeddings, axis=0))
    median_embedding = _l2_normalize(np.median(embeddings, axis=0))
    mean_appearance = np.mean(appearances, axis=0)
    median_appearance = np.median(appearances, axis=0)
    descriptors = {
        "mean": descriptors_from_appearance(mean_appearance),
        "median": descriptors_from_appearance(median_appearance),
    }
    return CentroidModel(
        image_ids=image_ids,
        source_paths=source_paths,
        mean_embedding=mean_embedding,
        median_embedding=median_embedding,
        mean_appearance=mean_appearance.astype(np.float32),
        median_appearance=median_appearance.astype(np.float32),
        descriptors=descriptors,
        embedding_dim=int(embeddings.shape[1]),
        appearance_shape=tuple(int(x) for x in appearances.shape[1:]),
    )


def save_model(model: CentroidModel, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "centroids.npz",
        mean_embedding=model.mean_embedding,
        median_embedding=model.median_embedding,
        mean_appearance=model.mean_appearance,
        median_appearance=model.median_appearance,
        image_ids=np.asarray(model.image_ids),
        source_paths=np.asarray(model.source_paths),
    )
    render_appearance(model.mean_appearance, out_dir / "mean_face.png")
    render_appearance(model.median_appearance, out_dir / "median_face.png")

    profile = {
        "image_count": len(model.image_ids),
        "embedding_dim": model.embedding_dim,
        "appearance_shape": list(model.appearance_shape),
        "descriptors": model.descriptors,
        "notes": [
            "Centroids summarize only the provided image set.",
            "Default crop mode is center crop; manually cropped face images improve signal.",
            "Generated-image scores are approximate vector similarity, not ground truth.",
        ],
    }
    (out_dir / "profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    prompt = prompt_from_descriptors(model.descriptors["median"])
    prompt_profiles = {
        profile: prompt_from_descriptors(model.descriptors["median"], profile=profile)
        for profile in PROMPT_PROFILES
    }
    negative_prompt_profiles = {}
    for profile_name in PROMPT_PROFILES:
        profile_negative = negative_prompt_for_profile(profile_name)
        if profile_negative:
            negative_prompt_profiles[profile_name] = profile_negative
    (out_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    manifest = {
        "prompt": prompt,
        "prompt_profiles": prompt_profiles,
        "negative_prompt": (
            "specific celebrity likeness, copied identity, distorted face, extra eyes, "
            "asymmetry artifact, hair covering face, obscured eyes, harsh shadows, "
            "illustration, doll, mannequin, painted skin, low detail, watermark, text"
        ),
        "negative_prompt_profiles": negative_prompt_profiles,
        "reference_outputs": {
            "mean_face": "mean_face.png",
            "median_face": "median_face.png",
            "centroid_vectors": "centroids.npz",
        },
        "recommended_generation": {
            "batch_size": 8,
            "vary_seed": True,
            "keep_prompt_constant": True,
        },
        "evaluation_command": (
            "python -m seju_face_lab evaluate --model outputs/seju_model "
            "--images outputs/generated --out outputs/evaluation"
        ),
        "boundary": "Aggregate-trait synthesis only; do not copy or identify a real person.",
    }
    (out_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(_render_report(profile, prompt), encoding="utf-8")


def load_model(model_dir: Path) -> CentroidModel:
    data = np.load(model_dir / "centroids.npz", allow_pickle=False)
    profile = json.loads((model_dir / "profile.json").read_text(encoding="utf-8"))
    return CentroidModel(
        image_ids=[str(x) for x in data["image_ids"].tolist()],
        source_paths=[str(x) for x in data["source_paths"].tolist()],
        mean_embedding=data["mean_embedding"].astype(np.float32),
        median_embedding=data["median_embedding"].astype(np.float32),
        mean_appearance=data["mean_appearance"].astype(np.float32),
        median_appearance=data["median_appearance"].astype(np.float32),
        descriptors=profile["descriptors"],
        embedding_dim=int(profile["embedding_dim"]),
        appearance_shape=tuple(int(x) for x in profile["appearance_shape"]),
    )


def _render_report(profile: dict, prompt: str) -> str:
    lines = [
        "# seju-face centroid report",
        "",
        f"- image_count: {profile['image_count']}",
        f"- embedding_dim: {profile['embedding_dim']}",
        f"- appearance_shape: {profile['appearance_shape']}",
        "",
        "## Median Descriptor",
        "",
    ]
    for key, value in profile["descriptors"]["median"].items():
        lines.append(f"- {key}: {value:.4f}")
    lines.extend(["", "## Generation Prompt", "", prompt, "", "## Boundaries", ""])
    lines.extend(f"- {note}" for note in profile["notes"])
    lines.append("")
    return "\n".join(lines)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)
