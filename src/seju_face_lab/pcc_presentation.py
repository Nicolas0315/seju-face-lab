"""Aggregate privacy-preserving PCC-NH presentation observations.

This module deliberately keeps FFC prompt text, image locations, face vectors,
and person attributes out of Seju. It reports automated triage only; blinded
human review remains mandatory before any publication decision.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


CONTRACT_VERSION = "pcc-seju-presentation/v1"
CONTRACT_FIELDS = {
    "contract_version",
    "study_id",
    "study_version",
    "presentation_modes",
    "privacy_boundary",
    "tasks",
}
REQUIRED_PRIVACY_FIELDS = (
    "contains_raw_images",
    "contains_image_locations",
    "contains_face_embeddings",
    "contains_prompt_text",
    "contains_personal_attributes",
)
TASK_FIELDS = {
    "task_id",
    "stage",
    "model_family",
    "culture_cell",
    "language_cell",
    "condition_id",
    "seed_requested",
    "condition_type",
    "factors",
    "prompt_sha256",
}
OBSERVATION_FIELDS = {"task_id", "image_id", "face_qa_pass", "presentation_flags"}
TASK_ID_PATTERN = re.compile(r"GEN_[0-9a-f]{12}$")
PROMPT_HASH_PATTERN = re.compile(r"[0-9a-f]{64}$")
ANONYMOUS_IMAGE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
SEED_PATTERN = re.compile(r"(?:0|[1-9][0-9]{0,8})$")
ALLOWED_MODEL_FAMILIES = {
    "imagen_family",
    "gpt_image_family",
    "midjourney_family",
    "stable_diffusion_family",
    "flux_family",
}
ALLOWED_CULTURE_CELLS = {"japan", "korea", "china", "india", "indonesia"}
ALLOWED_LANGUAGE_CELLS = {"english", "native_language"}
ALLOWED_CONDITION_TYPES = {"factorial", "control"}
ALLOWED_FACTOR_NAMES = {
    "task_directed_agency",
    "individual_identity_anchors",
    "harmony_based_beauty",
    "coupled_eye_skin_optics",
    "lived_in_context",
}
ALLOWED_PRESENTATION_FLAGS = {
    "face_count_invalid",
    "face_too_small",
    "face_too_close_or_cropped",
    "face_off_center",
    "low_global_contrast",
    "excessive_highlight_clipping",
    "excessive_shadow_clipping",
    "low_detail_signal",
}


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _load_json(raw: str, source: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} is not valid JSON") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = _load_json(line, f"observations line {number}")
        if not isinstance(value, dict):
            raise ValueError(f"observations line {number} must be a JSON object")
        rows.append(value)
    return rows


def validate_pcc_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless a contract contains only approved aggregate metadata."""
    if not isinstance(contract, dict):
        raise ValueError("contract must be a JSON object")
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(f"unsupported contract_version: {contract.get('contract_version')!r}")
    unknown_contract_fields = sorted(set(contract) - CONTRACT_FIELDS)
    if unknown_contract_fields:
        raise ValueError(f"contract has unsupported fields: {', '.join(unknown_contract_fields)}")
    if contract.get("study_id") != "PCC-NH-01":
        raise ValueError("contract study_id must be PCC-NH-01")
    if not isinstance(contract.get("study_version"), str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", contract["study_version"]):
        raise ValueError("contract study_version must be a semantic version")
    modes = contract.get("presentation_modes")
    if modes != ["full", "face_crop", "face_hidden"]:
        raise ValueError("contract presentation_modes must be full, face_crop, face_hidden in that order")
    privacy = contract.get("privacy_boundary")
    if not isinstance(privacy, dict) or set(privacy) != set(REQUIRED_PRIVACY_FIELDS):
        raise ValueError("contract privacy_boundary schema is invalid")
    if any(privacy.get(field) is not False for field in REQUIRED_PRIVACY_FIELDS):
        raise ValueError("contract privacy_boundary must explicitly exclude private image and person artifacts")
    tasks = contract.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("contract tasks must be a non-empty list")
    task_ids: list[str] = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("every contract task must be a JSON object")
        unknown_task_fields = sorted(set(task) - TASK_FIELDS)
        missing_task_fields = sorted(TASK_FIELDS - set(task))
        if unknown_task_fields or missing_task_fields:
            raise ValueError(
                "contract task schema mismatch: "
                f"unknown={','.join(unknown_task_fields) or 'none'} "
                f"missing={','.join(missing_task_fields) or 'none'}"
            )
        task_id = task["task_id"]
        if not isinstance(task_id, str) or not TASK_ID_PATTERN.fullmatch(task_id):
            raise ValueError("contract task_id must be a generated anonymous task identifier")
        if task["stage"] not in {"A", "B"}:
            raise ValueError(f"contract task {task_id} has invalid stage")
        if task["model_family"] not in ALLOWED_MODEL_FAMILIES:
            raise ValueError(f"contract task {task_id} has invalid model_family")
        if task["culture_cell"] not in ALLOWED_CULTURE_CELLS:
            raise ValueError(f"contract task {task_id} has invalid culture_cell")
        if task["language_cell"] not in ALLOWED_LANGUAGE_CELLS:
            raise ValueError(f"contract task {task_id} has invalid language_cell")
        if not isinstance(task["condition_id"], str) or not re.fullmatch(r"(?:F[01]{5}|C_[A-Z_]+)", task["condition_id"]):
            raise ValueError(f"contract task {task_id} has invalid condition_id")
        if task["condition_type"] not in ALLOWED_CONDITION_TYPES:
            raise ValueError(f"contract task {task_id} has invalid condition_type")
        if not isinstance(task["seed_requested"], str) or not SEED_PATTERN.fullmatch(task["seed_requested"]):
            raise ValueError(f"contract task {task_id} has invalid seed_requested")
        if not isinstance(task["prompt_sha256"], str) or not PROMPT_HASH_PATTERN.fullmatch(task["prompt_sha256"]):
            raise ValueError(f"contract task {task_id} has invalid prompt_sha256")
        factors = task["factors"]
        if not isinstance(factors, dict) or set(factors) - ALLOWED_FACTOR_NAMES:
            raise ValueError(f"contract task {task_id} has invalid factors")
        if any(isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1} for value in factors.values()):
            raise ValueError(f"contract task {task_id} has non-binary factors")
        task_ids.append(task_id)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("contract has duplicate task_id values")
    return contract


def load_pcc_contract(path: Path) -> dict[str, Any]:
    contract = _load_json(path.read_text(encoding="utf-8"), "contract")
    return validate_pcc_contract(contract)


def build_pcc_presentation_review(contract: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any]:
    contract = validate_pcc_contract(contract)
    tasks = {str(task["task_id"]): task for task in contract["tasks"]}
    records: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for observation in observations:
        unknown_observation_fields = sorted(set(observation) - OBSERVATION_FIELDS)
        missing_observation_fields = sorted(OBSERVATION_FIELDS - set(observation))
        if unknown_observation_fields or missing_observation_fields:
            raise ValueError(
                "observation schema mismatch: "
                f"unknown={','.join(unknown_observation_fields) or 'none'} "
                f"missing={','.join(missing_observation_fields) or 'none'}"
            )
        task_id = observation.get("task_id")
        image_id = observation.get("image_id")
        face_qa_pass = observation.get("face_qa_pass")
        flags = observation.get("presentation_flags")
        if not isinstance(task_id, str) or task_id not in tasks:
            raise ValueError(f"observation references unknown task_id: {task_id!r}")
        if not isinstance(image_id, str) or not ANONYMOUS_IMAGE_ID_PATTERN.fullmatch(image_id):
            raise ValueError("observation image_id must be an anonymous token, not a path")
        if not isinstance(face_qa_pass, bool):
            raise ValueError(f"observation {image_id!r} face_qa_pass must be boolean")
        if not isinstance(flags, list) or not all(isinstance(flag, str) and flag in ALLOWED_PRESENTATION_FLAGS for flag in flags):
            raise ValueError(f"observation {image_id!r} has unsupported presentation_flags")
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
