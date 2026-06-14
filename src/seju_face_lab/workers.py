"""GPU worker distribution for seju-face-lab.

Supports local RTX 4090 (ultra2025) and remote RTX 5060 Ti (nicolas2025)
for parallel vectorization and evaluation workloads.
"""

from __future__ import annotations

import json
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

REMOTE_5060TI = WorkerConfig(
    name="nicolas2025-5060ti",
    python=r"C:\Users\nicolas\work\seju-face-lab\.venv\Scripts\python.exe",
    project_dir=r"C:\Users\nicolas\work\seju-face-lab",
    gpu_id=0,
    remote_host="nicolas2025",
    remote_user=None,
)


def distribute_vectorize(
    image_paths: list[Path],
    model_dir: Path,
    out_dir: Path,
    backend: str = "insightface",
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
                worker, chunk, model_dir, worker_out, backend, tmp_root,
            )
            futures_map[f] = (worker, worker_out)

        all_scores: list[dict] = []
        for future in as_completed(futures_map):
            worker_cfg, worker_out = futures_map[future]
            try:
                scores = future.result()
                all_scores.extend(scores)
                print(f"  [{worker_cfg.name}] {len(scores)} scores collected")
            except Exception as exc:  # noqa: BLE001
                print(f"  [{worker_cfg.name}] failed: {exc}")

    all_scores.sort(key=lambda s: s.get("centroid_score", 0.0), reverse=True)
    (out_dir / "distributed_scores.json").write_text(
        json.dumps(all_scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return all_scores


def run_local_evaluate(
    image_paths: list[Path],
    model_dir: Path,
    out_dir: Path,
    backend: str = "insightface",
) -> list[dict]:
    """Run evaluation for an explicit image-path subset on the local machine."""
    from .backends import get_vector_backend
    from .metrics import _score_vector, write_scores
    from .model import load_model

    model = load_model(model_dir)
    backend_obj = get_vector_backend(backend)
    out_dir.mkdir(parents=True, exist_ok=True)

    vectors_by_index = {}
    failed_paths = []
    max_workers = max(1, min(4, len(image_paths)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(backend_obj.vectorize, path, "center"): index
            for index, path in enumerate(image_paths)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            path = image_paths[index]
            try:
                vectors_by_index[index] = future.result()
            except Exception:  # noqa: BLE001 - record per-path worker failures in the summary.
                failed_paths.append(str(path))
    vectors = [vectors_by_index[index] for index in sorted(vectors_by_index)]
    scores = sorted(
        [_score_vector(model, vector) for vector in vectors],
        key=lambda item: item.centroid_score,
        reverse=True,
    )
    write_scores(scores, out_dir, failed_paths=failed_paths)
    return [
        {
            "image_id": s.image_id, "path": s.path,
            "centroid_score": s.centroid_score,
            "cosine_to_mean": s.cosine_to_mean, "cosine_to_median": s.cosine_to_median,
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


# ─── internal ────────────────────────────────────────────────────────────────

def _run_worker_evaluate(
    worker: WorkerConfig,
    image_paths: list[Path],
    model_dir: Path,
    out_dir: Path,
    backend: str,
    tmp_root: Path,
) -> list[dict]:
    # Persist the assignment as an audit artifact even for local direct evaluation.
    list_file = tmp_root / f"{worker.name}_images.txt"
    list_file.write_text(
        "\n".join(str(p) for p in image_paths), encoding="utf-8"
    )

    if worker.remote_host is None:
        return run_local_evaluate(image_paths, model_dir, out_dir, backend)
    return _run_remote_subprocess(worker, image_paths, model_dir, out_dir, backend)


def _run_remote_subprocess(
    worker: WorkerConfig,
    image_paths: list[Path],
    model_dir: Path,
    out_dir: Path,
    backend: str,
) -> list[dict]:
    """Placeholder for a future shared-path remote evaluator."""
    raise NotImplementedError(
        "remote worker subset evaluation needs an explicit shared-path or sync manifest; "
        f"not running unrelated default paths on {worker.name}"
    )


def _ssh_cmd(worker: WorkerConfig, remote_cmd: str) -> list[str]:
    host = worker.remote_host
    if worker.remote_user:
        host = f"{worker.remote_user}@{host}"
    return ["ssh", host, remote_cmd]


def _split_paths(paths: list[Path], n: int) -> list[list[Path]]:
    chunk_size = max(1, (len(paths) + n - 1) // n)
    return [paths[i : i + chunk_size] for i in range(0, len(paths), chunk_size)]
