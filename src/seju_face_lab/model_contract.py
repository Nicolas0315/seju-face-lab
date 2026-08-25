from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelContract:
    schema_version: str
    source_manifest_sha256: str
    clean_manifest_sha256: str
    code_commit: str
    backend_name: str
    recognition_model_name: str
    recognition_model_sha256: str
    detector_name: str
    detector_model_sha256: str
    alignment_method: str
    landmark_schema: str
    crop_size: int
    color_space: str
    preprocessing_version: str
    embedding_dimension: int
    embedding_normalization: str
    quality_definition_version: str
    duplicate_definition_version: str
    subject_aggregation: str
    global_aggregation: str
    component_method: str
    component_count: int
    score_definition_version: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ModelContract:
        expected = {item.name for item in fields(cls)}
        supplied = set(value)
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        if missing or unknown:
            raise ValueError(f"invalid model contract fields: missing={missing}, unknown={unknown}")
        contract = cls(**dict(value))
        if contract.crop_size <= 0 or contract.embedding_dimension <= 0:
            raise ValueError("model contract dimensions must be positive")
        if contract.component_count < 0:
            raise ValueError("model contract component_count must be non-negative")
        return contract

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        payload = json.dumps(
            self.to_mapping(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def assert_compatible(left: ModelContract, right: ModelContract) -> None:
    if left.sha256() != right.sha256():
        raise ValueError(
            f"model contract mismatch: left={left.sha256()} right={right.sha256()}"
        )
