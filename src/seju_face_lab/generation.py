from __future__ import annotations

import json
import base64
import subprocess
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .prompting import PROMPT_PROFILES, negative_prompt_for_profile, prompt_from_descriptors


@dataclass(frozen=True)
class GenerationConfig:
    provider: str
    model_id: str
    centroid_kind: str
    prompt_profile: str
    prompt: str
    negative_prompt: str
    count: int
    seed: int
    steps: int
    guidance_scale: float
    width: int
    height: int
    device: str
    dtype: str
    variant: str | None
    output_format: str
    quality: str


@dataclass(frozen=True)
class GenerationResult:
    status: str
    provider: str
    output_dir: str
    planned_count: int
    generated_images: list[str]
    evaluation_command: str
    evaluation_argv: list[str]


def build_generation_config(
    model_dir: Path,
    provider: str,
    model_id: str,
    count: int,
    seed: int,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    device: str,
    dtype: str,
    variant: str | None,
    output_format: str = "png",
    quality: str = "medium",
    centroid_kind: str = "median",
    prompt_profile: str = "balanced",
    prompt_override: str | None = None,
    negative_prompt_override: str | None = None,
) -> GenerationConfig:
    manifest = json.loads((model_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    if prompt_profile not in PROMPT_PROFILES:
        raise ValueError(f"Unsupported prompt profile: {prompt_profile}")
    if centroid_kind not in {"mean", "median"}:
        raise ValueError(f"Unsupported centroid kind: {centroid_kind}")
    prompt = _prompt_for_profile(model_dir, manifest, centroid_kind, prompt_profile, prompt_override)
    negative_prompt = _negative_prompt_for_profile(manifest, prompt_profile, negative_prompt_override)
    return GenerationConfig(
        provider=provider,
        model_id=model_id,
        centroid_kind=centroid_kind,
        prompt_profile=prompt_profile,
        prompt=prompt,
        negative_prompt=negative_prompt,
        count=count,
        seed=seed,
        steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        device=device,
        dtype=dtype,
        variant=variant,
        output_format=output_format,
        quality=quality,
    )


def _prompt_for_profile(
    model_dir: Path,
    manifest: dict[str, Any],
    centroid_kind: str,
    prompt_profile: str,
    prompt_override: str | None,
) -> str:
    if prompt_override is not None:
        return prompt_override
    if centroid_kind == "median" and prompt_profile == "balanced":
        return str(manifest["prompt"])
    centroid_profiles = manifest.get("centroid_prompt_profiles", {})
    if isinstance(centroid_profiles, dict):
        kind_profiles = centroid_profiles.get(centroid_kind, {})
        if isinstance(kind_profiles, dict) and prompt_profile in kind_profiles:
            return str(kind_profiles[prompt_profile])
    prompt_profiles = manifest.get("prompt_profiles", {})
    if centroid_kind == "median" and isinstance(prompt_profiles, dict) and prompt_profile in prompt_profiles:
        return str(prompt_profiles[prompt_profile])
    profile = json.loads((model_dir / "profile.json").read_text(encoding="utf-8"))
    return prompt_from_descriptors(profile["descriptors"][centroid_kind], profile=prompt_profile)


def _negative_prompt_for_profile(
    manifest: dict[str, Any],
    prompt_profile: str,
    negative_prompt_override: str | None,
) -> str:
    if negative_prompt_override is not None:
        return negative_prompt_override
    profile_negatives = manifest.get("negative_prompt_profiles", {})
    profile_negative = ""
    if isinstance(profile_negatives, dict):
        profile_negative = str(profile_negatives.get(prompt_profile, "")).strip()
    if not profile_negative:
        profile_negative = negative_prompt_for_profile(prompt_profile)
    if prompt_profile == "detector-friendly":
        return _detector_friendly_negative_prompt(profile_negative)
    parts = [str(manifest.get("negative_prompt", "")).strip()]
    parts.append(profile_negative)
    return ", ".join(part for part in parts if part)


def _detector_friendly_negative_prompt(profile_negative: str) -> str:
    base_terms = [
        "specific celebrity likeness",
        "copied identity",
        "distorted face",
        "extra eyes",
        "hair covering face",
        "obscured eyes",
        "low detail",
        "watermark",
        "text",
    ]
    profile_terms = [term.strip() for term in profile_negative.split(",") if term.strip()]
    compact_terms = []
    seen = set()
    for term in [*base_terms, *profile_terms]:
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        compact_terms.append(term)
    return ", ".join(compact_terms)


def write_generation_plan(config: GenerationConfig, model_dir: Path, out_dir: Path) -> GenerationResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = GenerationResult(
        status="planned",
        provider=config.provider,
        output_dir=str(out_dir),
        planned_count=config.count,
        generated_images=[],
        evaluation_command=_evaluation_command(model_dir, out_dir),
        evaluation_argv=_evaluation_argv(model_dir, out_dir),
    )
    _write_generation_artifacts(config, result, out_dir)
    return result


def run_diffusers_generation(config: GenerationConfig, model_dir: Path, out_dir: Path) -> GenerationResult:
    if config.provider != "diffusers":
        raise ValueError(f"Unsupported generation provider: {config.provider}")

    torch = _import_torch()
    diffusion_pipeline = _import_diffusion_pipeline()
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = _torch_dtype(torch, config.dtype)
    load_kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": True,
    }
    if config.variant:
        load_kwargs["variant"] = config.variant
    pipe = diffusion_pipeline.from_pretrained(config.model_id, **load_kwargs)
    pipe = pipe.to(config.device)

    generated: list[str] = []
    for index in range(config.count):
        image_seed = config.seed + index
        generator = torch.Generator(device=config.device).manual_seed(image_seed)
        output = pipe(
            prompt=config.prompt,
            negative_prompt=config.negative_prompt or None,
            num_inference_steps=config.steps,
            guidance_scale=config.guidance_scale,
            width=config.width,
            height=config.height,
            generator=generator,
        )
        image = output.images[0]
        image_path = out_dir / f"candidate_{index + 1:04d}_seed_{image_seed}.png"
        image.save(image_path)
        generated.append(str(image_path))

    result = GenerationResult(
        status="generated",
        provider=config.provider,
        output_dir=str(out_dir),
        planned_count=config.count,
        generated_images=generated,
        evaluation_command=_evaluation_command(model_dir, out_dir),
        evaluation_argv=_evaluation_argv(model_dir, out_dir),
    )
    _write_generation_artifacts(config, result, out_dir)
    return result


def run_openai_image_generation(config: GenerationConfig, model_dir: Path, out_dir: Path) -> GenerationResult:
    if config.provider != "openai-image":
        raise ValueError(f"Unsupported generation provider: {config.provider}")
    if config.negative_prompt:
        prompt = f"{config.prompt}\n\nAvoid: {config.negative_prompt}"
    else:
        prompt = config.prompt

    client = _openai_client()
    response = client.images.generate(
        model=config.model_id,
        prompt=prompt,
        n=config.count,
        size=f"{config.width}x{config.height}",
        quality=config.quality,
        output_format=config.output_format,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    for index, item in enumerate(getattr(response, "data", []) or [], start=1):
        image_path = out_dir / f"candidate_{index:04d}_openai.{config.output_format}"
        _write_openai_image_item(item, image_path)
        generated.append(str(image_path))

    result = GenerationResult(
        status="generated",
        provider=config.provider,
        output_dir=str(out_dir),
        planned_count=config.count,
        generated_images=generated,
        evaluation_command=_evaluation_command(model_dir, out_dir),
        evaluation_argv=_evaluation_argv(model_dir, out_dir),
    )
    _write_generation_artifacts(config, result, out_dir)
    return result


def _write_generation_artifacts(
    config: GenerationConfig,
    result: GenerationResult,
    out_dir: Path,
) -> None:
    payload = {
        "config": asdict(config),
        "result": asdict(result),
        "boundary": "Aggregate seju-face prompt generation only; do not copy a specific real person.",
    }
    (out_dir / "generation_run.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "generation_run.md").write_text(_render_generation_run(config, result), encoding="utf-8")


def _render_generation_run(config: GenerationConfig, result: GenerationResult) -> str:
    lines = [
        "# generation run",
        "",
        f"- status: {result.status}",
        f"- provider: {config.provider}",
        f"- model_id: {config.model_id}",
        f"- centroid_kind: {config.centroid_kind}",
        f"- prompt_profile: {config.prompt_profile}",
        f"- count: {config.count}",
        f"- seed: {config.seed}",
        f"- steps: {config.steps}",
        f"- guidance_scale: {config.guidance_scale}",
        f"- size: {config.width}x{config.height}",
        f"- device: {config.device}",
        f"- dtype: {config.dtype}",
        f"- variant: {config.variant or 'none'}",
        f"- output_format: {config.output_format}",
        f"- quality: {config.quality}",
        "",
        "## Prompt",
        "",
        config.prompt,
        "",
        "## Negative Prompt",
        "",
        config.negative_prompt,
        "",
        "## Evaluate",
        "",
        f"```powershell\n{result.evaluation_command}\n```",
        "",
    ]
    if result.generated_images:
        lines.extend(["## Generated Images", ""])
        lines.extend(f"- {path}" for path in result.generated_images)
        lines.append("")
    return "\n".join(lines)


def _evaluation_command(model_dir: Path, out_dir: Path) -> str:
    return subprocess.list2cmdline(_evaluation_argv(model_dir, out_dir))


def _evaluation_argv(model_dir: Path, out_dir: Path) -> list[str]:
    return [
        "python",
        "-m",
        "seju_face_lab",
        "evaluate",
        "--model",
        str(model_dir),
        "--images",
        str(out_dir),
        "--out",
        str(out_dir / "evaluation"),
    ]


def _import_diffusion_pipeline() -> Any:
    try:
        from diffusers import DiffusionPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Diffusers is not installed. Install the optional generation extra before running "
            "provider=diffusers."
        ) from exc
    return DiffusionPipeline


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install a CUDA-enabled torch build before running "
            "provider=diffusers."
        ) from exc
    return torch


def _openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI Python SDK is not installed. Install the optional openai extra before running "
            "provider=openai-image."
        ) from exc
    return OpenAI()


def _write_openai_image_item(item: Any, image_path: Path) -> None:
    b64_json = getattr(item, "b64_json", None)
    if b64_json:
        image_path.write_bytes(base64.b64decode(b64_json))
        return
    url = getattr(item, "url", None)
    if url:
        with urllib.request.urlopen(url, timeout=60) as response:
            image_path.write_bytes(response.read())
        return
    raise RuntimeError("OpenAI image response did not include b64_json or url data.")


def _torch_dtype(torch: Any, dtype: str) -> Any:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")
