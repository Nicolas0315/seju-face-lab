"""GPU worker distribution for seju-face-lab.

Supports local RTX 4090 (ultra2025) plus configured SSH GPU workers such as
nicolas2025 for parallel vectorization and evaluation workloads.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class WorkerConfig:
    name: str
    python: str
    project_dir: str
    gpu_id: int = 0
    remote_host: str | None = None
    remote_user: str | None = None


# Fleet defaults
LOCAL_4090 = WorkerConfig(
    name="ultra2025-4090",
    python=r"C:\Users\ogosh\work\seju-face-lab\.venv\Scripts\python.exe",
    project_dir=r"C:\Users\ogosh\work\seju-face-lab",
    gpu_id=0,
    remote_host=None,
)

REMOTE_NICOLAS_GPU = WorkerConfig(
    name="nicolas2025-remote-gpu",
    python=r"C:\Users\ogosh\work\seju-face-lab\.venv\Scripts\python.exe",
    project_dir=r"C:\Users\ogosh\work\seju-face-lab",
    gpu_id=0,
    remote_host="nicolas2025",
    remote_user=None,
)

REMOTE_5060TI = REMOTE_NICOLAS_GPU
DEFAULT_DIAGNOSTIC_WORKERS = [LOCAL_4090, REMOTE_NICOLAS_GPU]

_DIAGNOSTIC_SCRIPT = r"""
import json
import os
import platform
import socket
import sys

report = {
    "hostname": socket.gethostname(),
    "platform": platform.platform(),
    "python": sys.executable,
    "cwd": os.getcwd(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "configured_project_dir": os.environ.get("SEJU_WORKER_PROJECT_DIR"),
    "configured_project_exists": None,
    "configured_python": os.environ.get("SEJU_WORKER_CONFIGURED_PYTHON"),
    "configured_python_exists": None,
    "seju_face_lab_importable": False,
    "torch_importable": False,
    "torch_cuda_available": None,
    "torch_cuda_device_count": None,
    "torch_cuda_device_name": None,
}
if report["configured_project_dir"]:
    report["configured_project_exists"] = os.path.isdir(report["configured_project_dir"])
if report["configured_python"]:
    report["configured_python_exists"] = os.path.exists(report["configured_python"])
try:
    import seju_face_lab  # noqa: F401
    report["seju_face_lab_importable"] = True
except Exception as exc:
    report["seju_face_lab_error"] = f"{type(exc).__name__}: {exc}"
try:
    import torch
    report["torch_importable"] = True
    report["torch_cuda_available"] = bool(torch.cuda.is_available())
    report["torch_cuda_device_count"] = int(torch.cuda.device_count())
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        report["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
except Exception as exc:
    report["torch_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(report, ensure_ascii=False))
""".strip()


def distribute_vectorize(
    image_paths: list[Path],
    model_dir: Path,
    out_dir: Path,
    backend: str = "insightface",
    crop: str = "center",
    workers: list[WorkerConfig] | None = None,
    tmp_dir: Path | None = None,
) -> list[dict]:
    """Split images across available workers and collect evaluation results.

    Local workers score only their assigned image-path subset. Remote workers
    currently require an explicit shared-path/sync implementation before use.
    Returns merged list of score dicts.
    """
    active_workers = workers or [LOCAL_4090]
    if not image_paths:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "distributed_scores.json").write_text("[]\n", encoding="utf-8")
        _write_distributed_scores([], 0, [], [], out_dir)
        return []
    remote_workers = [worker.name for worker in active_workers if worker.remote_host is not None]
    if remote_workers:
        raise NotImplementedError(
            "remote worker subset evaluation needs an explicit shared-path or sync manifest: "
            + ", ".join(remote_workers)
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = tmp_dir or out_dir / ".worker_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Split image list across workers
    chunks = _split_paths(image_paths, len(active_workers))
    futures_map: dict[Any, tuple[WorkerConfig, Path]] = {}

    with ThreadPoolExecutor(max_workers=len(active_workers)) as executor:
        for worker, chunk in zip(active_workers, chunks):
            if not chunk:
                continue
            worker_out = tmp_root / worker.name
            worker_out.mkdir(parents=True, exist_ok=True)
            f = executor.submit(
                _run_worker_evaluate,
                worker, chunk, model_dir, worker_out, backend, crop, tmp_root,
            )
            futures_map[f] = (worker, worker_out)

        all_scores: list[dict] = []
        failed_count = 0
        failed_paths: list[str] = []
        worker_failures: list[dict[str, str]] = []
        for future in as_completed(futures_map):
            worker_cfg, worker_out = futures_map[future]
            try:
                scores = future.result()
                all_scores.extend(scores)
                worker_failed_count, worker_failed_paths = _worker_failure_summary(worker_out)
                failed_count += worker_failed_count
                failed_paths.extend(worker_failed_paths)
                print(f"  [{worker_cfg.name}] {len(scores)} scores collected")
            except Exception as exc:  # noqa: BLE001
                worker_failures.append({"worker": worker_cfg.name, "error": str(exc)})
                print(f"  [{worker_cfg.name}] failed: {exc}")

    all_scores.sort(key=lambda s: s.get("centroid_score", 0.0), reverse=True)
    (out_dir / "distributed_scores.json").write_text(
        json.dumps(all_scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_distributed_scores(all_scores, failed_count, failed_paths, worker_failures, out_dir)
    if worker_failures:
        names = ", ".join(failure["worker"] for failure in worker_failures)
        raise RuntimeError(f"distributed worker failures: {names}")
    return all_scores


def run_local_evaluate(
    image_paths: list[Path],
    model_dir: Path,
    out_dir: Path,
    backend: str = "insightface",
    crop: str = "center",
) -> list[dict]:
    """Run evaluation for an explicit image-path subset on the local machine."""
    from .backends import get_vector_backend
    from .metrics import _score_vector, write_scores
    from .model import load_model

    model = load_model(model_dir)
    backend_obj = get_vector_backend(backend)
    out_dir.mkdir(parents=True, exist_ok=True)

    vectors_by_index = {}
    failed_indices = []
    max_workers = max(1, min(4, len(image_paths)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(backend_obj.vectorize, path, crop): index
            for index, path in enumerate(image_paths)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                vectors_by_index[index] = future.result()
            except Exception:  # noqa: BLE001 - record per-path worker failures in the summary.
                failed_indices.append(index)
    vectors = [vectors_by_index[index] for index in sorted(vectors_by_index)]
    failed_paths = [str(image_paths[index]) for index in sorted(failed_indices)]
    scores = sorted(
        [_score_vector(model, vector) for vector in vectors],
        key=lambda item: item.centroid_score,
        reverse=True,
    )
    write_scores(scores, out_dir, failed_paths=failed_paths, model=model)
    return [
        {
            "image_id": s.image_id, "path": s.path,
            "centroid_score": s.centroid_score,
            "cosine_to_mean": s.cosine_to_mean, "cosine_to_median": s.cosine_to_median,
            "euclidean_to_mean": s.euclidean_to_mean,
            "euclidean_to_median": s.euclidean_to_median,
        }
        for s in scores
    ]


def check_remote_worker(worker: WorkerConfig) -> bool:
    """Ping remote host via SSH and check seju-face-lab is installed."""
    if worker.remote_host is None:
        return True  # local worker is always available
    try:
        cmd = _ssh_cmd(worker, f'"{worker.python}" -m seju_face_lab backends')
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def write_worker_diagnostics(
    out_dir: Path,
    workers: list[WorkerConfig] | None = None,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Write local/SSH worker readiness diagnostics without mutating remote state."""
    active_workers = workers or [LOCAL_4090]
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "worker_count": len(active_workers),
        "workers": [_diagnose_worker(worker, timeout_seconds) for worker in active_workers],
        "boundary": (
            "Diagnostics only. Remote workers are probed over SSH but no files are copied, "
            "and distributed remote evaluation still requires an explicit sync/shared-path plan."
        ),
    }
    (out_dir / "worker_diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "worker_diagnostics.md").write_text(_render_worker_diagnostics(report), encoding="utf-8")
    return report


# ─── internal ────────────────────────────────────────────────────────────────

def _run_worker_evaluate(
    worker: WorkerConfig,
    image_paths: list[Path],
    model_dir: Path,
    out_dir: Path,
    backend: str,
    crop: str,
    tmp_root: Path,
) -> list[dict]:
    # Persist the assignment as an audit artifact even for local direct evaluation.
    list_file = tmp_root / f"{worker.name}_images.txt"
    list_file.write_text(
        "\n".join(str(p) for p in image_paths), encoding="utf-8"
    )

    if worker.remote_host is None:
        return run_local_evaluate(image_paths, model_dir, out_dir, backend, crop)
    return _run_remote_subprocess(worker, image_paths, model_dir, out_dir, backend, crop)


def _run_remote_subprocess(
    worker: WorkerConfig,
    image_paths: list[Path],
    model_dir: Path,
    out_dir: Path,
    backend: str,
    crop: str,
) -> list[dict]:
    """Placeholder for a future shared-path remote evaluator."""
    raise NotImplementedError(
        "remote worker subset evaluation needs an explicit shared-path or sync manifest; "
        f"not running unrelated default paths on {worker.name}"
    )


def _diagnose_worker(worker: WorkerConfig, timeout_seconds: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(worker.gpu_id)
    env["SEJU_WORKER_PROJECT_DIR"] = worker.project_dir
    env["SEJU_WORKER_CONFIGURED_PYTHON"] = worker.python
    if worker.remote_host is None:
        command = [worker.python, "-c", _DIAGNOSTIC_SCRIPT]
        cwd = worker.project_dir if worker.project_dir else None
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics must record failures.
            return _worker_probe_failure(worker, type(exc).__name__, str(exc))
        return _worker_probe_result(worker, result)

    remote_command = _remote_diagnostic_command(worker)
    try:
        result = subprocess.run(
            _ssh_cmd(worker, remote_command),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must record failures.
        return _worker_probe_failure(worker, type(exc).__name__, str(exc))
    return _worker_probe_result(worker, result)


def _worker_probe_result(worker: WorkerConfig, result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    parsed = _parse_probe_json(result.stdout)
    ready = (
        result.returncode == 0
        and bool(parsed.get("seju_face_lab_importable"))
        and bool(parsed.get("torch_importable"))
        and bool(parsed.get("torch_cuda_available"))
        and int(parsed.get("torch_cuda_device_count") or 0) > 0
        and bool(parsed.get("configured_project_exists"))
        and bool(parsed.get("configured_python_exists"))
    )
    return {
        "name": worker.name,
        "remote_host": worker.remote_host,
        "gpu_id": worker.gpu_id,
        "project_dir": worker.project_dir,
        "python": worker.python,
        "ok": ready,
        "returncode": result.returncode,
        "probe": parsed,
        "stdout_tail": _tail(result.stdout),
        "stderr_tail": _tail(result.stderr),
    }


def _worker_probe_failure(worker: WorkerConfig, error_type: str, message: str) -> dict[str, Any]:
    return {
        "name": worker.name,
        "remote_host": worker.remote_host,
        "gpu_id": worker.gpu_id,
        "project_dir": worker.project_dir,
        "python": worker.python,
        "ok": False,
        "error": f"{error_type}: {message}",
        "probe": {},
        "stdout_tail": "",
        "stderr_tail": "",
    }


def _worker_failure_summary(worker_out: Path) -> tuple[int, list[str]]:
    summary_path = worker_out / "summary.json"
    if not summary_path.exists():
        return 0, []
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0, []
    failed_count = int(summary.get("failed_count") or 0)
    failed = summary.get("failed_paths")
    failed_paths = [str(path) for path in failed] if isinstance(failed, list) else []
    return failed_count, failed_paths


def _write_distributed_scores(
    scores: list[dict],
    failed_count: int,
    failed_paths: list[str],
    worker_failures: list[dict[str, str]],
    out_dir: Path,
) -> None:
    lines = [
        "image_id,path,cosine_to_mean,cosine_to_median,euclidean_to_mean,euclidean_to_median,centroid_score"
    ]
    for score in scores:
        lines.append(
            ",".join(
                [
                    _csv(str(score.get("image_id", ""))),
                    _csv(str(score.get("path", ""))),
                    _format_score(score.get("cosine_to_mean")),
                    _format_score(score.get("cosine_to_median")),
                    _format_score(score.get("euclidean_to_mean")),
                    _format_score(score.get("euclidean_to_median")),
                    _format_score(score.get("centroid_score")),
                ]
            )
        )
    (out_dir / "scores.csv").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    summary = {
        "image_count": len(scores),
        "failed_count": failed_count,
        "failed_paths": failed_paths[:20],
        "worker_failures": worker_failures,
        "best_image_id": scores[0].get("image_id") if scores else None,
        "best_centroid_score": _round_score(scores[0].get("centroid_score")) if scores else None,
        "top_images": [
            {
                "image_id": score.get("image_id"),
                "path": score.get("path"),
                "centroid_score": _round_score(score.get("centroid_score")),
                "cosine_to_mean": _round_score(score.get("cosine_to_mean")),
                "cosine_to_median": _round_score(score.get("cosine_to_median")),
                "euclidean_to_mean": _round_score(score.get("euclidean_to_mean")),
                "euclidean_to_median": _round_score(score.get("euclidean_to_median")),
            }
            for score in scores[:5]
        ],
        "boundary": "Distributed worker scores are approximate local centroid measurements only.",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _format_score(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _round_score(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _csv(value: str) -> str:
    escaped = value.replace('"', '""')
    return f'"{escaped}"'


def _parse_probe_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _remote_diagnostic_command(worker: WorkerConfig) -> str:
    encoded = base64.b64encode(_DIAGNOSTIC_SCRIPT.encode("utf-8")).decode("ascii")
    project_dir = worker.project_dir.replace("'", "''")
    python = worker.python.replace("'", "''")
    python_code = f"exec(__import__('base64').b64decode('{encoded}').decode('utf-8'))"
    ps_script = (
        f"$project='{project_dir}'; "
        f"$python='{python}'; "
        "$env:SEJU_WORKER_PROJECT_DIR=$project; "
        "$env:SEJU_WORKER_CONFIGURED_PYTHON=$python; "
        "if (Test-Path -LiteralPath $project) { Set-Location -LiteralPath $project }; "
        f"$env:CUDA_VISIBLE_DEVICES='{worker.gpu_id}'; "
        "$activePython = if (Test-Path -LiteralPath $python) { $python } else { 'python' }; "
        f"& $activePython -c \"{python_code}\""
    )
    encoded_ps = base64.b64encode(ps_script.encode("utf-16le")).decode("ascii")
    return f"powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand {encoded_ps}"


def _render_worker_diagnostics(report: dict[str, Any]) -> str:
    lines = ["# worker diagnostics", ""]
    for worker in report["workers"]:
        probe = worker.get("probe", {})
        lines.extend(
            [
                f"## {worker['name']}",
                "",
                f"- remote_host: {worker.get('remote_host') or 'local'}",
                f"- ok: {worker.get('ok')}",
                f"- returncode: {worker.get('returncode', '')}",
                f"- hostname: {probe.get('hostname', '')}",
                f"- seju_face_lab_importable: {probe.get('seju_face_lab_importable', '')}",
                f"- torch_importable: {probe.get('torch_importable', '')}",
                f"- torch_cuda_available: {probe.get('torch_cuda_available', '')}",
                f"- torch_cuda_device_count: {probe.get('torch_cuda_device_count', '')}",
                f"- torch_cuda_device_name: {probe.get('torch_cuda_device_name', '')}",
                "",
            ]
        )
        if worker.get("stderr_tail"):
            lines.extend(["### stderr tail", "", "```text", worker["stderr_tail"], "```", ""])
    lines.extend(["## Boundary", "", str(report["boundary"]), ""])
    return "\n".join(lines)


def _tail(value: str, max_chars: int = 2000) -> str:
    return value[-max_chars:] if len(value) > max_chars else value


def _ssh_cmd(worker: WorkerConfig, remote_cmd: str) -> list[str]:
    host = worker.remote_host
    if worker.remote_user:
        host = f"{worker.remote_user}@{host}"
    return ["ssh", host, remote_cmd]


def _split_paths(paths: list[Path], n: int) -> list[list[Path]]:
    chunk_size = max(1, (len(paths) + n - 1) // n)
    return [paths[i : i + chunk_size] for i in range(0, len(paths), chunk_size)]
