from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import bootstrap  # noqa: F401
import numpy as np
from PIL import Image

from scripts.verify_mermaid_blocks import extract_mermaid_blocks
from seju_face_lab.data_gate import cluster_near_duplicates, gate_download_rows
from seju_face_lab.component_model import fit_component_model, project_component
from seju_face_lab.face_observations import build_face_observations, extract_face_observation
from seju_face_lab.geometry_vectors import geometry_vector
from seju_face_lab.model_contract import ModelContract, assert_compatible
from seju_face_lab.robust_templates import SubjectTemplate, build_global_center


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


def landmark_fixture_106() -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, 106, endpoint=False)
    return np.column_stack((np.cos(angles) * 40.0, np.sin(angles) * 55.0))


class FakeExtractor:
    def __init__(self, faces: list[dict[str, object]]) -> None:
        self.faces = faces

    def detect(self, path: Path) -> list[dict[str, object]]:
        return self.faces


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

    def test_face_observation_requires_exactly_one_face(self) -> None:
        contract = ModelContract.from_mapping(contract_fixture())
        with self.assertRaisesRegex(ValueError, "expected exactly one face"):
            extract_face_observation(
                Path("fixture.png"),
                "subject-a",
                FakeExtractor(faces=[]),
                contract,
            )

    def test_build_face_observations_writes_separate_vector_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "face.png"
            Image.new("RGB", (32, 32), (200, 160, 140)).save(image)
            face = {
                "embedding": np.array([3.0, 4.0]),
                "bbox": np.array([2.0, 2.0, 30.0, 30.0]),
                "detector_confidence": 0.95,
                "relative_face_area": 0.75,
                "landmarks_5": landmark_fixture_106()[:5],
                "landmarks_106": landmark_fixture_106(),
            }
            contract = ModelContract.from_mapping(
                contract_fixture(embedding_dimension=2, component_count=1)
            )

            summary = build_face_observations(
                [{"subject_id": "a", "resolved_path": str(image)}],
                FakeExtractor([face]),
                contract,
                root / "out",
            )

            self.assertEqual(summary["accepted_count"], 1)
            self.assertTrue((root / "out" / "neural_vectors.npz").exists())
            self.assertTrue((root / "out" / "geometry_vectors.npz").exists())

    def test_geometry_vector_is_translation_and_scale_invariant(self) -> None:
        points = landmark_fixture_106()

        actual = geometry_vector(points * 3.0 + np.array([90.0, -40.0])).values

        np.testing.assert_allclose(actual, geometry_vector(points).values, atol=1e-8)

    def test_global_center_weights_subjects_equally(self) -> None:
        templates = [
            SubjectTemplate("a", np.array([1.0, 0.0]), np.array([0.0]), 20, 0.0),
            SubjectTemplate("b", np.array([0.0, 1.0]), np.array([0.0]), 3, 0.0),
        ]

        center = build_global_center(templates)

        np.testing.assert_allclose(center, np.array([1.0, 1.0]) / np.sqrt(2.0), atol=1e-6)

    def test_component_model_caps_components_and_projects_center_to_zero(self) -> None:
        vectors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.98, 0.2, 0.0],
                [0.98, -0.2, 0.0],
                [0.98, 0.0, 0.2],
                [0.98, 0.0, -0.2],
            ]
        )

        model = fit_component_model(vectors, max_components=2, variance_threshold=0.8)

        self.assertLessEqual(model.components.shape[0], 2)
        np.testing.assert_allclose(project_component(model, model.center), 0.0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
