from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerationConfig:
    provider: str
    model_id: str
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
    prompt_override: str | None = None,
    negative_prompt_override: str | None = None,
) -> GenerationConfig:
    manifest = json.loads((model_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    return GenerationConfig(
        provider=provider,
        model_id=model_id,
        prompt=prompt_override if prompt_override is not None else manifest["prompt"],
        negative_prompt=(
            negative_prompt_override
            if negative_prompt_override is not None
            else manifest.get("negative_prompt", "")
        ),
        count=count,
        seed=seed,
        steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        device=device,
        dtype=dtype,
        variant=variant,
    )


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
        f"- count: {config.count}",
        f"- seed: {config.seed}",
        f"- steps: {config.steps}",
        f"- guidance_scale: {config.guidance_scale}",
        f"- size: {config.width}x{config.height}",
        f"- device: {config.device}",
        f"- dtype: {config.dtype}",
        f"- variant: {config.variant or 'none'}",
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


def _torch_dtype(torch: Any, dtype: str) -> Any:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")
