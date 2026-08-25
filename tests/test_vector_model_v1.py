from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bootstrap  # noqa: F401
import numpy as np
from PIL import Image

from scripts.verify_mermaid_blocks import extract_mermaid_blocks
from seju_face_lab.approximation_score import ScoreCalibration, combine_percentiles, score_vector
from seju_face_lab.candidate_search import plan_candidates
from seju_face_lab.component_model import (
    ComponentModel,
    bootstrap_component_stability,
    fit_component_model,
    fit_component_model_at_center,
    project_component,
)
from seju_face_lab.data_gate import cluster_near_duplicates, gate_download_rows
from seju_face_lab.face_observations import (
    InsightFaceObservationExtractor,
    build_face_observations,
    extract_face_observation,
)
from seju_face_lab.geometry_vectors import geometry_vector
from seju_face_lab.model_contract import ModelContract, assert_compatible
from seju_face_lab.perturbations import (
    apply_image_perturbation,
    summarize_perturbation_rows,
)
from seju_face_lab.preference_review import make_blind_bundle
from seju_face_lab.robust_templates import SubjectTemplate, build_global_center
from seju_face_lab.score_face import ScoreBundle, load_score_bundle, score_face_image
from seju_face_lab.vector_evaluation import (
    build_loso_folds,
    decide_promotion,
    evaluate_vector_model_dir,
)
from seju_face_lab.vector_pipeline import build_vector_model_records


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


class FakeFaceAnalysis:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.prepared_with: int | None = None

    def prepare(self, ctx_id: int, det_size: tuple[int, int]) -> None:
        self.prepared_with = ctx_id
        self.det_size = det_size


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

    def test_insightface_observation_extractor_prepares_windows_cuda_dlls(self) -> None:
        extractor = InsightFaceObservationExtractor(gpu_id=0)
        with (
            patch("insightface.app.FaceAnalysis", FakeFaceAnalysis),
            patch(
                "seju_face_lab.face_observations._prepare_windows_torch_cuda_dlls"
            ) as prepare_dlls,
        ):
            app = extractor._get_app()

        prepare_dlls.assert_called_once_with()
        self.assertEqual(app.prepared_with, 0)

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

    def test_component_bootstrap_is_deterministic(self) -> None:
        vectors = np.eye(5) + 0.1

        first = bootstrap_component_stability(vectors, seed=7, samples=20)
        second = bootstrap_component_stability(vectors, seed=7, samples=20)

        self.assertEqual(first, second)
        self.assertEqual(first["bootstrap_samples"], 20)

    def test_b1_component_model_keeps_arithmetic_subject_center(self) -> None:
        vectors = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
        expected = np.mean(vectors, axis=0)
        expected /= np.linalg.norm(expected)

        model = fit_component_model_at_center(vectors, expected, max_components=1)

        np.testing.assert_allclose(model.center, expected, atol=1e-8)

    def test_loso_fold_never_contains_held_out_subject_in_fit(self) -> None:
        for fold in build_loso_folds(["a", "b", "c"]):
            self.assertNotIn(fold.held_out_subject_id, fold.fit_subject_ids)
            self.assertEqual(len(fold.fit_subject_ids), 2)

    def test_approximation_is_harmonic_mean_of_two_percentiles(self) -> None:
        self.assertEqual(
            combine_percentiles(center_percentile=80.0, manifold_percentile=20.0),
            32.0,
        )

    def test_center_vector_gets_full_manifold_percentile(self) -> None:
        model = ComponentModel(
            center=np.array([1.0, 0.0]),
            components=np.array([[0.0, 1.0]]),
            explained_variance_ratio=np.array([1.0]),
            support_low=np.array([-1.0]),
            support_high=np.array([1.0]),
        )
        calibration = ScoreCalibration(
            center_agreements=np.array([0.5, 0.8]),
            reconstruction_residuals=np.array([0.0, 0.1]),
        )

        result = score_vector(np.array([1.0, 0.0]), model, calibration)

        self.assertEqual(result.manifold_percentile, 100.0)
        self.assertEqual(result.seju_approximation, 100.0)

    def test_vector_model_with_too_few_subjects_is_diagnostic(self) -> None:
        records = []
        neural = []
        geometry = []
        for subject_index, subject_id in enumerate(("a", "b")):
            for image_index in range(3):
                records.append(
                    {
                        "image_id": f"{subject_id}-{image_index}",
                        "subject_id": subject_id,
                        "quality": 1.0,
                    }
                )
                neural.append(
                    np.array([1.0, 0.1 * subject_index, 0.01 * image_index])
                )
                geometry.append(np.array([float(subject_index), float(image_index)]))

        with tempfile.TemporaryDirectory() as tmp:
            summary = build_vector_model_records(
                records,
                np.asarray(neural),
                np.asarray(geometry),
                ModelContract.from_mapping(
                    contract_fixture(embedding_dimension=3, component_count=1)
                ),
                Path(tmp),
            )

        self.assertEqual(summary["status"], "diagnostic")
        self.assertEqual(summary["valid_subject_count"], 2)

    def test_promotion_defers_without_perturbation_and_component_stability(self) -> None:
        decision = decide_promotion(
            {
                "minimum_evidence": True,
                "loso_complete": True,
                "contract_valid": True,
                "positive_improvement_lower_bound": True,
                "worst_decile_within_tolerance": True,
                "detector_coverage_within_tolerance": True,
                "reproducible": True,
                "component_stability_available": False,
                "perturbation_evidence_available": False,
            }
        )

        self.assertEqual(decision["status"], "deferred")
        self.assertEqual(decision["failed_gates"], ["perturbation_evidence_available"])
        self.assertIsNone(decision["selected_algorithm"])

    def test_promotion_falls_back_to_b1_when_c1_does_not_improve(self) -> None:
        decision = decide_promotion(
            {
                "minimum_evidence": True,
                "loso_complete": True,
                "contract_valid": True,
                "positive_improvement_lower_bound": False,
                "worst_decile_within_tolerance": True,
                "detector_coverage_within_tolerance": True,
                "component_stability_available": False,
                "perturbation_evidence_available": True,
                "reproducible": True,
            }
        )

        self.assertEqual(decision["status"], "promoted")
        self.assertEqual(decision["selected_algorithm"], "B1")

    def test_image_perturbation_is_deterministic(self) -> None:
        image = Image.new("RGB", (64, 64), (180, 140, 120))

        first = np.asarray(apply_image_perturbation(image, "jpeg_recompression"))
        second = np.asarray(apply_image_perturbation(image, "jpeg_recompression"))

        np.testing.assert_array_equal(first, second)

    def test_perturbation_summary_requires_coverage_and_stability(self) -> None:
        rows = [
            {
                "kind": "downscale",
                "accepted": True,
                "embedding_cosine": 0.95,
                "b1_center_degradation": 0.01,
            },
            {
                "kind": "downscale",
                "accepted": False,
                "embedding_cosine": None,
                "b1_center_degradation": None,
            },
        ]

        summary = summarize_perturbation_rows(rows, clean_count=2)

        self.assertFalse(summary["gate_pass"])
        self.assertEqual(summary["kinds"]["downscale"]["coverage"], 0.5)

    def test_vector_model_evaluation_writes_all_loso_folds(self) -> None:
        records = []
        neural = []
        geometry = []
        for subject_index, subject_id in enumerate(("a", "b", "c", "d")):
            for image_index in range(3):
                records.append(
                    {
                        "image_id": f"{subject_id}-{image_index}",
                        "subject_id": subject_id,
                        "quality": 1.0,
                    }
                )
                base = np.zeros(4)
                base[subject_index] = 1.0
                base[(subject_index + 1) % 4] = 0.02 * image_index
                neural.append(base)
                geometry.append(np.array([float(subject_index), float(image_index)]))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_vector_model_records(
                records,
                np.asarray(neural),
                np.asarray(geometry),
                ModelContract.from_mapping(
                    contract_fixture(embedding_dimension=4, component_count=2)
                ),
                root / "model",
                min_subjects=4,
            )

            report = evaluate_vector_model_dir(root / "model", root / "evaluation", seed=7)

            self.assertEqual(report["loso_fold_count"], 4)
            self.assertEqual(report["promotion"]["status"], "deferred")
            self.assertTrue((root / "evaluation" / "loso_scores.jsonl").exists())

    def test_candidate_planner_rejects_unpromoted_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "promoted model required"):
            plan_candidates({"promotion": {"status": "deferred"}}, [])

    def test_blind_bundle_hides_score_seed_generator_and_component_target(self) -> None:
        candidate = {
            "candidate_id": "candidate-a",
            "image_path": "candidate-a.png",
            "synthetic_label": "架空生成候補",
            "approximation_score": 80.0,
            "generator": "local",
            "seed": 7,
            "component_target": [0.1, 0.2],
        }

        serialized = json.dumps(make_blind_bundle([candidate], seed=9))

        for forbidden in ("approximation_score", "generator", "seed", "component_target"):
            self.assertNotIn(forbidden, serialized)

    def test_score_bundle_rejects_unpromoted_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "evaluation.json").write_text(
                json.dumps({"promotion": {"status": "deferred"}}), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "promoted evaluation required"):
                load_score_bundle(root)

    def test_score_face_emits_component_coordinates_for_local_candidate_planning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "candidate.png"
            Image.new("RGB", (32, 32), (200, 160, 140)).save(image)
            bundle = ScoreBundle(
                evaluation={"score_algorithm": "B1"},
                component_model=ComponentModel(
                    center=np.array([1.0, 0.0]),
                    components=np.array([[0.0, 1.0]]),
                    explained_variance_ratio=np.array([1.0]),
                    support_low=np.array([-2.0]),
                    support_high=np.array([2.0]),
                ),
                calibration=ScoreCalibration(
                    center_agreements=np.array([0.0, 0.5]),
                    reconstruction_residuals=np.array([0.0, 1.0]),
                    definition_version="loso_hmean_v1",
                ),
                contract=ModelContract.from_mapping(
                    contract_fixture(embedding_dimension=2, component_count=1)
                ),
            )
            face = {
                "embedding": np.array([0.0, 1.0]),
                "bbox": np.array([2.0, 2.0, 30.0, 30.0]),
                "detector_confidence": 0.95,
                "relative_face_area": 0.75,
                "landmarks_5": landmark_fixture_106()[:5],
                "landmarks_106": landmark_fixture_106(),
            }

            with patch("seju_face_lab.score_face.load_score_bundle", return_value=bundle):
                result = score_face_image(image, root, FakeExtractor([face]))

            self.assertEqual(len(result["component_coordinates"]), 1)
            self.assertAlmostEqual(result["component_coordinates"][0], np.pi / 2.0)


if __name__ == "__main__":
    unittest.main()
