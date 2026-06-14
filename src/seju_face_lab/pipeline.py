from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class PipelineStep:
    name: str
    status: str
    path: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class PipelinePlan:
    name: str
    config_path: str
    steps: list[PipelineStep]


def load_pipeline_config(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def write_pipeline_run(plan: PipelinePlan, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": plan.name,
        "config_path": plan.config_path,
        "steps": [asdict(step) for step in plan.steps],
        "boundary": (
            "Pipeline run manifests record local command orchestration only. Scores and QA "
            "remain approximate local measurements."
        ),
    }
    (out_dir / "pipeline_run.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "pipeline_run.md").write_text(_render_pipeline_run(payload), encoding="utf-8")


def build_pipeline_plan(config: dict[str, Any], config_path: Path) -> PipelinePlan:
    name = str(config.get("name") or config_path.stem)
    steps: list[PipelineStep] = []
    build_model_out = _path_value(config, "model_out")
    model_out = build_model_out or _path_value(config, "model")
    reference_images = _path_value(config, "reference_images")
    if reference_images and build_model_out:
        steps.append(PipelineStep("build", "planned", str(build_model_out)))

    generation = _generation_config(config)
    generation_out = _path_value(generation, "out") or _path_value(config, "generated_images")
    if generation and model_out and generation_out:
        steps.append(PipelineStep("generate", "planned", str(generation_out)))

    generated_images = _path_value(config, "generated_images") or generation_out
    evaluation_out = _path_value(config, "evaluation_out")
    if model_out and generated_images and evaluation_out:
        steps.append(PipelineStep("evaluate", "planned", str(evaluation_out)))

    review_out = _path_value(config, "review_out")
    if model_out and generated_images and review_out:
        steps.append(PipelineStep("review-generated", "planned", str(review_out)))

    subject_review = _subject_review_config(config)
    if subject_review:
        steps.append(PipelineStep("review-subjects", "planned", str(subject_review["out"])))

    backend_comparison = _backend_comparison_config(config)
    if backend_comparison and reference_images and generated_images:
        steps.append(PipelineStep("compare-backends", "planned", str(backend_comparison["out"])))

    precision_out = _path_value(config, "precision_out")
    if model_out and precision_out:
        steps.append(PipelineStep("precision-report", "planned", str(precision_out)))
    return PipelinePlan(name=name, config_path=str(config_path), steps=steps)


def run_pipeline_config(
    config_path: Path,
    out_dir: Path | None,
    handlers: dict[str, Callable[[dict[str, Any]], int]],
) -> PipelinePlan:
    config = load_pipeline_config(config_path)
    plan = build_pipeline_plan(config, config_path)
    completed_steps: list[PipelineStep] = []
    for step in plan.steps:
        handler = handlers.get(step.name)
        if handler is None:
            completed_steps.append(
                PipelineStep(step.name, "skipped", step.path, "no handler registered")
            )
            continue
        try:
            result = handler(config)
            status = "completed" if result == 0 else "failed"
            message = None if result == 0 else f"exit code {result}"
        except SystemExit as exc:
            status = "failed"
            message = str(exc)
        except Exception as exc:  # noqa: BLE001 - keep the pipeline run manifest writable.
            status = "failed"
            message = str(exc)
        completed_steps.append(PipelineStep(step.name, status, step.path, message))
        if status == "failed":
            break
    completed = PipelinePlan(plan.name, plan.config_path, completed_steps)
    write_pipeline_run(completed, out_dir or _default_out_dir(config, config_path))
    return completed


def _render_pipeline_run(payload: dict[str, Any]) -> str:
    lines = ["# seju-face pipeline run", "", f"- name: {payload['name']}", ""]
    lines.append("| step | status | path | message |")
    lines.append("| --- | --- | --- | --- |")
    for step in payload["steps"]:
        lines.append(
            f"| {step['name']} | {step['status']} | {step.get('path') or ''} | "
            f"{step.get('message') or ''} |"
        )
    lines.extend(["", payload["boundary"], ""])
    return "\n".join(lines)


def _generation_config(config: dict[str, Any]) -> dict[str, Any]:
    generation = config.get("generation")
    if isinstance(generation, dict):
        return generation
    if "provider" in config or "hf_model" in config:
        return config
    return {}


def _subject_review_config(config: dict[str, Any]) -> dict[str, Path] | None:
    subjects = _path_value(config, "subjects")
    out = _path_value(config, "subject_review_out") or _path_value(config, "subject_out")
    model = _path_value(config, "model_out") or _path_value(config, "model")
    if not subjects or not out or not model:
        return None
    return {"subjects": subjects, "out": out, "model": model}


def _backend_comparison_config(config: dict[str, Any]) -> dict[str, Path] | None:
    comparison = config.get("backend_comparison")
    out = None
    if isinstance(comparison, dict):
        out = _path_value(comparison, "out")
    out = out or _path_value(config, "backend_comparison_out")
    if not out:
        return None
    return {"out": out}


def _default_out_dir(config: dict[str, Any], config_path: Path) -> Path:
    if out := _path_value(config, "pipeline_out"):
        return out
    if out := _path_value(config, "precision_out"):
        return out.parent / "pipeline_run"
    return Path("outputs") / f"{config_path.stem}_pipeline"


def _path_value(config: dict[str, Any], key: str) -> Path | None:
    value = config.get(key)
    if value is None or value == "":
        return None
    return Path(str(value))
