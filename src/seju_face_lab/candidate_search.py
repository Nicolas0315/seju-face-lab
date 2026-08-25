from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def plan_candidates(
    evaluation: Mapping[str, Any],
    measured_candidates: Iterable[Mapping[str, Any]],
    *,
    score_band: tuple[float, float] = (70.0, 90.0),
) -> dict[str, Any]:
    promotion = evaluation.get("promotion")
    if not isinstance(promotion, Mapping) or promotion.get("status") != "promoted":
        raise ValueError("promoted model required before fictional candidate planning")
    low, high = score_band
    if not 0.0 <= low <= high <= 100.0:
        raise ValueError("invalid approximation score band")
    eligible = []
    rejected = []
    for source in measured_candidates:
        row = dict(source)
        score = float(row.get("approximation_score", -1.0))
        reason = None
        if not low <= score <= high:
            reason = "outside_score_band"
        elif bool(row.get("out_of_support", True)):
            reason = "out_of_component_support"
        elif not bool(row.get("qa_pass", False)):
            reason = "qa_failed"
        elif not bool(row.get("synthetic", False)):
            reason = "synthetic_provenance_required"
        if reason:
            row["rejection_reason"] = reason
            rejected.append(row)
        else:
            eligible.append(row)
    _add_candidate_diversity(eligible)
    frontier = [row for row in eligible if not _is_dominated(row, eligible)]
    return {
        "schema_version": "fictional_candidate_frontier_v1",
        "score_band": [low, high],
        "eligible_count": len(eligible),
        "rejected_count": len(rejected),
        "frontier": sorted(frontier, key=lambda row: str(row["candidate_id"])),
        "rejected": rejected,
        "boundary": "fictional candidates only; preference is separate from approximation",
    }


def _add_candidate_diversity(candidates: list[dict[str, Any]]) -> None:
    for row in candidates:
        coordinates = np.asarray(row.get("component_coordinates", []), dtype=np.float64)
        distances = []
        for other in candidates:
            if other is row:
                continue
            other_coordinates = np.asarray(
                other.get("component_coordinates", []), dtype=np.float64
            )
            if coordinates.shape != other_coordinates.shape:
                raise ValueError("candidate component coordinates must have matching shape")
            distances.append(float(np.linalg.norm(coordinates - other_coordinates)))
        row["candidate_diversity"] = min(distances) if distances else 0.0


def _is_dominated(candidate: Mapping[str, Any], candidates: list[dict[str, Any]]) -> bool:
    score = float(candidate["approximation_score"])
    diversity = float(candidate["candidate_diversity"])
    for other in candidates:
        if other is candidate:
            continue
        other_score = float(other["approximation_score"])
        other_diversity = float(other["candidate_diversity"])
        if (
            other_score >= score
            and other_diversity >= diversity
            and (other_score > score or other_diversity > diversity)
        ):
            return True
    return False
