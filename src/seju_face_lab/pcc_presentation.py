"""Aggregate privacy-preserving PCC-NH presentation observations.

This module deliberately keeps FFC prompt text, image locations, face vectors,
and person attributes out of Seju. It reports automated triage only; blinded
human review remains mandatory before any publication decision.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


CONTRACT_VERSION = "pcc-seju-presentation/v1"
REQUIRED_PRIVACY_FIELDS = (
    "contains_raw_images",
    "contains_image_locations",
    "contains_face_embeddings",
    "contains_prompt_text",
    "contains_personal_attributes",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"observations line {number} is not valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"observations line {number} must be a JSON object")
        rows.append(value)
    return rows


def load_pcc_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("contract must be a JSON object")
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(f"unsupported contract_version: {contract.get('contract_version')!r}")
    privacy = contract.get("privacy_boundary")
    if not isinstance(privacy, dict) or any(privacy.get(field) is not False for field in REQUIRED_PRIVACY_FIELDS):
        raise ValueError("contract privacy_boundary must explicitly exclude private image and person artifacts")
    tasks = contract.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("contract tasks must be a non-empty list")
    task_ids = [task.get("task_id") for task in tasks if isinstance(task, dict)]
    if len(task_ids) != len(tasks) or not all(isinstance(task_id, str) and task_id for task_id in task_ids):
        raise ValueError("every contract task must have a non-empty task_id")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("contract has duplicate task_id values")
    return contract


def build_pcc_presentation_review(contract: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = {str(task["task_id"]): task for task in contract["tasks"]}
    records: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for observation in observations:
        task_id = observation.get("task_id")
        image_id = observation.get("image_id")
        face_qa_pass = observation.get("face_qa_pass")
        flags = observation.get("presentation_flags")
        if not isinstance(task_id, str) or task_id not in tasks:
            raise ValueError(f"observation references unknown task_id: {task_id!r}")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("observation image_id must be a non-empty string")
        if not isinstance(face_qa_pass, bool):
            raise ValueError(f"observation {image_id!r} face_qa_pass must be boolean")
        if not isinstance(flags, list) or not all(isinstance(flag, str) and flag for flag in flags):
            raise ValueError(f"observation {image_id!r} presentation_flags must be a list of strings")
        pair = (task_id, image_id)
        if pair in seen_pairs:
            raise ValueError(f"duplicate observation for task/image pair: {task_id}/{image_id}")
        seen_pairs.add(pair)
        task = tasks[task_id]
        records.append(
            {
                "task_id": task_id,
                "image_id": image_id,
                "stage": task.get("stage"),
                "model_family": task.get("model_family"),
                "condition_id": task.get("condition_id"),
                "face_qa_pass": face_qa_pass,
                "presentation_flags": sorted(set(flags)),
            }
        )

    observed_task_ids = {record["task_id"] for record in records}
    clear = [record for record in records if record["face_qa_pass"] and not record["presentation_flags"]]
    flag_counts = Counter(flag for record in records for flag in record["presentation_flags"])
    return {
        "contract_version": CONTRACT_VERSION,
        "study_id": contract.get("study_id"),
        "task_count": len(tasks),
        "mapped_count": len(records),
        "missing_task_count": len(tasks) - len(observed_task_ids),
        "face_qa_pass_count": sum(record["face_qa_pass"] for record in records),
        "presentation_clear_count": len(clear),
        "presentation_flag_counts": dict(sorted(flag_counts.items())),
        "promotion_ready": False,
        "promotion_reason": "requires blinded human review; automated presentation QA cannot promote candidates",
        "records": records,
        "boundary": (
            "Task metadata and anonymous presentation QA only; no raw images, image locations, "
            "face embeddings, identity, demographic attributes, or attractiveness scoring."
        ),
    }


def write_pcc_presentation_review(
    contract_path: Path, observations_path: Path | None, out_dir: Path
) -> dict[str, Any]:
    contract = load_pcc_contract(contract_path)
    observations = _read_jsonl(observations_path) if observations_path is not None else []
    review = build_pcc_presentation_review(contract, observations)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pcc_presentation_review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# PCC-NH presentation review",
        "",
        f"- study_id: {review['study_id']}",
        f"- mapped: {review['mapped_count']} / {review['task_count']}",
        f"- face QA pass: {review['face_qa_pass_count']}",
        f"- presentation clear: {review['presentation_clear_count']}",
        f"- promotion: no ({review['promotion_reason']})",
        "",
        review["boundary"],
        "",
    ]
    (out_dir / "pcc_presentation_review.md").write_text("\n".join(lines), encoding="utf-8")
    return review
