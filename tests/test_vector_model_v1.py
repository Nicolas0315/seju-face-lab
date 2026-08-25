from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import bootstrap  # noqa: F401
from PIL import Image

from scripts.verify_mermaid_blocks import extract_mermaid_blocks
from seju_face_lab.data_gate import cluster_near_duplicates, gate_download_rows
from seju_face_lab.model_contract import ModelContract, assert_compatible


def contract_fixture(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "1",
        "source_manifest_sha256": "1" * 64,
        "clean_manifest_sha256": "2" * 64,
        "code_commit": "abc123",
        "backend_name": "insightface",
        "recognition_model_name": "buffalo_l",
        "recognition_model_sha256": "3" * 64,
        "detector_name": "retinaface",
        "detector_model_sha256": "4" * 64,
        "alignment_method": "insightface_5point_112",
        "landmark_schema": "insightface_2d_106_v1",
        "crop_size": 112,
        "color_space": "RGB",
        "preprocessing_version": "1",
        "embedding_dimension": 512,
        "embedding_normalization": "l2_unit",
        "quality_definition_version": "1",
        "duplicate_definition_version": "1",
        "subject_aggregation": "weighted_spherical_median_v1",
        "global_aggregation": "equal_subject_spherical_median_v1",
        "component_method": "tangent_svd_v1",
        "component_count": 8,
        "score_definition_version": "loso_hmean_v1",
    }
    value.update(overrides)
    return value


class VectorModelV1Tests(unittest.TestCase):
    def test_extract_mermaid_blocks_rejects_unclosed_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "bad.md"
            source.write_text("```mermaid\nflowchart TD\nA-->B\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unclosed mermaid block"):
                extract_mermaid_blocks(source, root / "out")

    def test_same_dimension_different_model_hash_is_incompatible(self) -> None:
        left = ModelContract.from_mapping(
            contract_fixture(recognition_model_sha256="a" * 64)
        )
        right = ModelContract.from_mapping(
            contract_fixture(recognition_model_sha256="b" * 64)
        )

        with self.assertRaisesRegex(ValueError, "model contract mismatch"):
            assert_compatible(left, right)

    def test_dataset_gate_rejects_signature_before_face_detection(self) -> None:
        audit = gate_download_rows(
            [
                {
                    "profile_url": "https://seju.tokyo/talents/example/",
                    "image_url": "https://seju.tokyo/uploads/example_sign.png",
                    "talent_slug": "example",
                    "path": "example_sign.png",
                    "sha256": "a" * 64,
                }
            ],
            allowed_hosts={"seju.tokyo"},
        )

        self.assertEqual(audit.rejected[0]["rejection_reason"], "signature_asset")
        self.assertEqual(audit.clean, [])

    def test_dataset_gate_rejects_signature_prefix_and_version_suffix(self) -> None:
        rows = []
        for filename in ("sign_example.png", "example_sign_2503.png"):
            rows.append(
                {
                    "profile_url": "https://seju.tokyo/talents/example/",
                    "image_url": f"https://seju.tokyo/uploads/{filename}",
                    "talent_slug": "example",
                    "path": filename,
                    "sha256": "a" * 64,
                }
            )

        audit = gate_download_rows(rows, allowed_hosts={"seju.tokyo"})

        self.assertEqual(len(audit.rejected), 2)
        self.assertEqual(
            {row["rejection_reason"] for row in audit.rejected}, {"signature_asset"}
        )

    def test_near_duplicate_gate_keeps_highest_quality_within_subject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pixels = Image.new("RGB", (96, 96), (220, 180, 150))
            high = root / "high.png"
            low = root / "low.jpg"
            pixels.save(high)
            pixels.resize((48, 48)).save(low, quality=70)
            rows = [
                {"subject_id": "a", "image_id": "high", "path": str(high), "quality": 0.9},
                {"subject_id": "a", "image_id": "low", "path": str(low), "quality": 0.4},
            ]

            clusters = cluster_near_duplicates(rows)

            self.assertEqual(len(clusters), 1)
            self.assertEqual(clusters[0]["representative_id"], "high")
            self.assertEqual(clusters[0]["suppressed_ids"], ["low"])


if __name__ == "__main__":
    unittest.main()
