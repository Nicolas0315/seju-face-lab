from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from typing import Any

QUESTIONS = ("印象に残る", "応援したい", "SNSで見たい")


def make_blind_bundle(
    candidates: Iterable[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    rows = list(candidates)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    items = []
    for display_index, source_index in enumerate(order, start=1):
        row = rows[source_index]
        items.append(
            {
                "review_id": f"R{display_index:03d}",
                "image_path": str(row["image_path"]),
                "synthetic_label": str(row.get("synthetic_label", "架空生成候補")),
            }
        )
    return {
        "schema_version": "blind_fictional_review_v1",
        "items": items,
        "questions": list(QUESTIONS),
        "choices": ["left", "right", "tie", "abstain"],
    }


def make_private_review_mapping(
    candidates: Iterable[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, str]:
    rows = list(candidates)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return {
        f"R{display_index:03d}": str(rows[source_index]["candidate_id"])
        for display_index, source_index in enumerate(order, start=1)
    }


def aggregate_pairwise(
    responses: Iterable[Mapping[str, Any]],
    *,
    panel_scope: str,
) -> dict[str, Any]:
    by_question: dict[str, dict[str, int]] = {
        question: {"left": 0, "right": 0, "tie": 0, "abstain": 0}
        for question in QUESTIONS
    }
    for row in responses:
        question = str(row["question"])
        choice = str(row["choice"])
        if question not in by_question or choice not in by_question[question]:
            raise ValueError("invalid blinded review response")
        by_question[question][choice] += 1
    summaries = {}
    for question, counts in by_question.items():
        decisive = counts["left"] + counts["right"]
        wins = counts["left"]
        interval = _wilson_interval(wins, decisive) if decisive else [0.0, 1.0]
        summaries[question] = {
            "counts": counts,
            "left_win_rate": wins / decisive if decisive else None,
            "wilson_95_interval": interval,
        }
    return {
        "schema_version": "blind_preference_summary_v1",
        "panel_scope": panel_scope,
        "questions": summaries,
        "boundary": "panel preference only; not inherent or public popularity",
    }


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(probability * (1.0 - probability) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [center - margin, center + margin]
