from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from .backends import (
    BACKENDS,
    GENERATION_PROVIDERS,
    DeepFaceBackend,
    InsightFaceBackend,
    OpenCVFaceBackend,
    PlannedBackend,
)


def collect_backend_diagnostics() -> dict[str, Any]:
    report = {
        "runtime": {
            "torch": _torch_status(),
            "onnxruntime_providers": _onnxruntime_providers(),
        },
        "backends": [_backend_status(name) for name in sorted(BACKENDS)],
        "generation_providers": [_generation_provider_status(name) for name in sorted(GENERATION_PROVIDERS)],
        "boundary": (
            "Dependency/provider visibility only. A backend is validated after it vectorizes "
            "the target local image set. A generation provider is validated after it writes "
            "a generation_run.json and its generated images can be reviewed."
        ),
    }
    return report


def write_backend_diagnostics(out_dir: Path) -> dict[str, Any]:
    report = collect_backend_diagnostics()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "backend_diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "backend_diagnostics.md").write_text(_render(report), encoding="utf-8")
    return report


def _backend_status(name: str) -> dict[str, Any]:
    backend = BACKENDS[name]
    status: dict[str, Any] = {"name": name, "description": backend.description}
    if isinstance(backend, PlannedBackend):
        return status | {"implemented": False, "state": "planned", "extra": backend.extra, "notes": backend.notes}
    if isinstance(backend, InsightFaceBackend):
        providers = _onnxruntime_providers()
        return status | {
            "implemented": True,
            "state": "implemented",
            "extra": "face",
            "dependencies": _dependencies("insightface", "onnxruntime", "cv2"),
            "onnxruntime_providers": providers,
            "cuda_provider_available": "CUDAExecutionProvider" in providers,
            "model_pack": backend.model_pack,
            "gpu_id": backend.gpu_id,
        }
    if isinstance(backend, DeepFaceBackend):
        return status | {
            "implemented": True,
            "state": "implemented",
            "extra": "deepface",
            "dependencies": _dependencies("deepface", "tensorflow", "tf_keras", "cv2"),
            "model_name": backend.model_name,
            "detector_backend": backend.detector_backend,
        }
    if isinstance(backend, OpenCVFaceBackend):
        return status | {
            "implemented": True,
            "state": "implemented",
            "extra": backend.extra,
            "dependencies": _dependencies("cv2"),
        }
    return status | {"implemented": True, "state": "ready", "extra": None, "dependencies": {}}


def _generation_provider_status(name: str) -> dict[str, Any]:
    provider = GENERATION_PROVIDERS[name]
    dependencies = _dependencies("diffusers", "transformers", "accelerate", "safetensors", "torch")
    if provider.name == "dry-run":
        dependencies = {}
    return {
        "name": provider.name,
        "state": provider.state,
        "implemented": True,
        "extra": provider.extra,
        "description": provider.description,
        "dependencies": dependencies,
    }


def _dependencies(*names: str) -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in names}


def _torch_status() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"installed": False, "cuda_available": False, "device_count": 0, "devices": []}
    import torch

    cuda_available = bool(torch.cuda.is_available())
    devices = []
    for index in range(int(torch.cuda.device_count()) if cuda_available else 0):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "total_memory_gb": round(float(props.total_memory) / (1024**3), 2),
                "capability": f"{props.major}.{props.minor}",
            }
        )
    return {
        "installed": True,
        "version": str(torch.__version__),
        "cuda_available": cuda_available,
        "device_count": len(devices),
        "devices": devices,
    }


def _onnxruntime_providers() -> list[str]:
    if importlib.util.find_spec("onnxruntime") is None:
        return []
    import onnxruntime

    return [str(provider) for provider in onnxruntime.get_available_providers()]


def _render(report: dict[str, Any]) -> str:
    torch = report["runtime"]["torch"]
    lines = [
        "# backend diagnostics",
        "",
        f"- torch_cuda_available: {torch['cuda_available']}",
        f"- torch_cuda_device_count: {torch['device_count']}",
        f"- onnxruntime_providers: {', '.join(report['runtime']['onnxruntime_providers']) or 'none'}",
        "",
        "| backend | state | extra | runtime |",
        "| --- | --- | --- | --- |",
    ]
    for backend in report["backends"]:
        runtime = _runtime_summary(backend)
        lines.append(f"| {backend['name']} | {backend['state']} | {backend.get('extra') or ''} | {runtime} |")
    lines.extend(["", "## Generation Providers", "", "| provider | state | extra | runtime |"])
    lines.append("| --- | --- | --- | --- |")
    for provider in report["generation_providers"]:
        runtime = _runtime_summary(provider)
        lines.append(f"| {provider['name']} | {provider['state']} | {provider.get('extra') or ''} | {runtime} |")
    lines.extend(["", report["boundary"], ""])
    return "\n".join(lines)


def _runtime_summary(backend: dict[str, Any]) -> str:
    dependencies = backend.get("dependencies") or {}
    parts = [f"{key}={value}" for key, value in dependencies.items()]
    if backend["name"] == "insightface":
        parts.insert(0, f"cuda_provider={backend.get('cuda_provider_available', False)}")
    return ", ".join(parts) or backend.get("notes", "")
