from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

import bootstrap  # noqa: F401
from seju_face_lab import backends
from seju_face_lab.cli import _sources_download, _subject_id_for_path, main
from seju_face_lab.model import load_model
from seju_face_lab.pipeline import build_pipeline_plan, load_pipeline_config
from seju_face_lab.sources import DownloadResult


class PipelineTests(unittest.TestCase):
    def test_full_retinaface_review_example_plans_complete_review_path(self) -> None:
        config_path = Path("configs/pipelines/full-retinaface-review.example.json")
        config = load_pipeline_config(config_path)

        plan = build_pipeline_plan(config, config_path)

        self.assertEqual(config["name"], "full_retinaface_review")
        self.assertEqual(
            [step.name for step in plan.steps],
            [
                "build",
                "audit-model",
                "export-vectors",
                "ingredients-report",
                "benchmark-research",
                "evaluate",
                "review-generated",
                "review-subjects",
                "compare-backends",
                "precision-report",
            ],
        )
        self.assertIn("deepface-retinaface", config["backend_comparison"]["backends"])
        self.assertEqual(config["backend_comparison"]["crop"], "center")

    def test_build_prompt_render_and_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            generated = root / "generated"
            generated_alt = root / "generated_alt"
            generated_empty = root / "generated_empty"
            subjects = root / "subjects"
            model_dir = root / "model space"
            eval_dir = root / "eval"
            style_eval_dir = generated / "style_evaluation"
            generation_dir = root / "generated faces"
            run_compare_dir = root / "run_compare"
            empty_run_compare_dir = root / "empty_run_compare"
            review_dir = root / "review"
            raw.mkdir()
            generated.mkdir()
            generated_alt.mkdir()
            generated_empty.mkdir()
            (subjects / "near_subject").mkdir(parents=True)
            (subjects / "far_subject").mkdir(parents=True)

            _write_face_like_image(raw / "a.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(raw / "b.png", (225, 198, 184), eye_offset=2)
            _write_face_like_image(raw / "c.png", (240, 210, 196), eye_offset=-2)
            (raw / "broken.jpg").write_text("not an image", encoding="utf-8")
            _write_face_like_image(generated / "candidate.png", (232, 202, 188), eye_offset=1)
            (generated / "broken.jpg").write_text("not an image", encoding="utf-8")
            _write_face_like_image(generated_alt / "candidate_alt.png", (170, 145, 130), eye_offset=7)
            _write_face_like_image(subjects / "near_subject" / "near.png", (232, 202, 188), eye_offset=1)
            _write_face_like_image(subjects / "far_subject" / "far.png", (170, 145, 130), eye_offset=7)
            (subjects / "far_subject" / "broken.jpg").write_text("not an image", encoding="utf-8")

            self.assertEqual(main(["build", "--images", str(raw), "--out", str(model_dir)]), 0)
            self.assertEqual(
                main(["build", "--images", str(raw), "--out", str(root / "model_backend"), "--backend", "deterministic"]),
                0,
            )
            with patch("seju_face_lab.backends._import_cv2", return_value=_FakeCV2()):
                self.assertEqual(
                    main(
                        [
                            "build",
                            "--images",
                            str(raw),
                            "--out",
                            str(root / "model_opencv"),
                            "--backend",
                            "opencv-face",
                        ]
                    ),
                    0,
                )
            self.assertTrue((root / "model_opencv" / "centroids.npz").exists())
            model = load_model(model_dir)
            self.assertEqual(len(model.image_ids), 3)
            self.assertGreater(model.embedding_dim, 100)
            self.assertTrue((model_dir / "mean_face.png").exists())
            self.assertTrue((model_dir / "median_face.png").exists())
            prompt_text = (model_dir / "prompt.txt").read_text(encoding="utf-8")
            self.assertIn("Aggregate traits", prompt_text)
            self.assertIn("new fictional person", prompt_text)
            self.assertLessEqual(len(prompt_text.split()), 55)
            self.assertTrue((model_dir / "generation_manifest.json").exists())
            generation_manifest = json.loads(
                (model_dir / "generation_manifest.json").read_text(encoding="utf-8")
            )
            self.assertIn("hair covering face", generation_manifest["negative_prompt"])
            self.assertIn("illustration", generation_manifest["negative_prompt"])
            self.assertIn("mean", generation_manifest["centroid_prompt_profiles"])
            self.assertIn("median", generation_manifest["centroid_prompt_profiles"])
            self.assertIn(
                "Aggregate traits",
                generation_manifest["centroid_prompt_profiles"]["mean"]["balanced"],
            )
            self.assertIn("detector-friendly", generation_manifest["prompt_profiles"])
            self.assertIn("passport headshot", generation_manifest["prompt_profiles"]["detector-friendly"])
            self.assertIn("side profile", generation_manifest["negative_prompt_profiles"]["detector-friendly"])
            vector_failures = json.loads(
                (model_dir / "vectors" / "image_vector_failures.json").read_text(encoding="utf-8")
            )
            self.assertEqual(vector_failures["failed_count"], 1)

            self.assertEqual(
                main(["render", "--model", str(model_dir), "--kind", "mean", "--out", str(root / "mean.png")]),
                0,
            )
            self.assertTrue((root / "mean.png").exists())

            self.assertEqual(
                main(
                    [
                        "evaluate",
                        "--model",
                        str(model_dir),
                        "--images",
                        str(generated),
                        "--out",
                        str(eval_dir),
                        "--backend",
                        "deterministic",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "evaluate",
                        "--model",
                        str(model_dir),
                        "--images",
                        str(generated_empty),
                        "--out",
                        str(generated_empty / "evaluation"),
                    ]
                ),
                0,
            )
            scores = (eval_dir / "scores.csv").read_text(encoding="utf-8-sig")
            self.assertIn("candidate", scores)
            self.assertIn("centroid_score", scores)
            self.assertTrue((eval_dir / "summary.json").exists())
            eval_summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(eval_summary["image_count"], 1)
            self.assertEqual(eval_summary["failed_count"], 1)
            self.assertIn("euclidean_to_mean", eval_summary["top_images"][0])
            self.assertIn("euclidean_to_median", eval_summary["top_images"][0])

            with patch("seju_face_lab.cli.OpenClipStyleBackend", return_value=_FakeStyleBackend()):
                self.assertEqual(
                    main(
                        [
                            "style-evaluate",
                            "--model",
                            str(model_dir),
                            "--images",
                            str(generated),
                            "--out",
                            str(style_eval_dir),
                        ]
                    ),
                    0,
                )
            style_scores = (style_eval_dir / "style_scores.csv").read_text(encoding="utf-8-sig")
            self.assertIn("candidate", style_scores)
            self.assertIn("style_score", style_scores)
            style_summary = json.loads((style_eval_dir / "style_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(style_summary["image_count"], 1)
            self.assertEqual(style_summary["failed_count"], 1)
            self.assertIn("not face geometry", style_summary["boundary"])

            self.assertEqual(
                main(
                    [
                        "evaluate",
                        "--model",
                        str(model_dir),
                        "--images",
                        str(generated),
                        "--out",
                        str(generated / "evaluation"),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "evaluate",
                        "--model",
                        str(model_dir),
                        "--images",
                        str(generated_alt),
                        "--out",
                        str(generated_alt / "evaluation"),
                    ]
                ),
                0,
            )

            self.assertEqual(
                main(
                    [
                        "generate",
                        "--model",
                        str(model_dir),
                        "--out",
                        str(generation_dir),
                        "--provider",
                        "diffusers",
                        "--dry-run",
                        "--count",
                        "2",
                        "--seed",
                        "42",
                        "--prompt",
                        "",
                        "--negative-prompt",
                        "copied identity",
                        "--centroid-kind",
                        "mean",
                    ]
                ),
                0,
            )
            generation_run = json.loads((generation_dir / "generation_run.json").read_text(encoding="utf-8"))
            self.assertEqual(generation_run["result"]["status"], "planned")
            self.assertEqual(generation_run["config"]["provider"], "dry-run")
            self.assertEqual(generation_run["config"]["count"], 2)
            self.assertEqual(generation_run["config"]["centroid_kind"], "mean")
            self.assertEqual(generation_run["config"]["prompt_profile"], "balanced")
            self.assertEqual(generation_run["config"]["prompt"], "")
            self.assertEqual(generation_run["config"]["variant"], "fp16")
            self.assertEqual(generation_run["config"]["negative_prompt"], "copied identity")
            self.assertIn("evaluate", generation_run["result"]["evaluation_command"])
            self.assertIn('"', generation_run["result"]["evaluation_command"])
            self.assertEqual(generation_run["result"]["evaluation_argv"][5], str(model_dir))
            detector_generation_dir = root / "generated detector"
            self.assertEqual(
                main(
                    [
                        "generate",
                        "--model",
                        str(model_dir),
                        "--out",
                        str(detector_generation_dir),
                        "--provider",
                        "dry-run",
                        "--count",
                        "1",
                        "--prompt-profile",
                        "detector-friendly",
                    ]
                ),
                0,
            )
            detector_run = json.loads(
                (detector_generation_dir / "generation_run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(detector_run["config"]["prompt_profile"], "detector-friendly")
            self.assertIn("passport headshot", detector_run["config"]["prompt"])
            self.assertIn("side profile", detector_run["config"]["negative_prompt"])
            self.assertIn("copied identity", detector_run["config"]["negative_prompt"])
            self.assertNotIn("illustration", detector_run["config"]["negative_prompt"])
            self.assertEqual(
                main(
                    [
                        "compare-runs",
                        "--runs",
                        str(generated),
                        str(generated_alt),
                        "--out",
                        str(run_compare_dir),
                    ]
                ),
                0,
            )
            run_compare = json.loads(
                (run_compare_dir / "generation_run_reviews.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_compare["run_count"], 2)
            self.assertEqual(run_compare["best_run_dir"], str(generated))
            self.assertEqual(run_compare["runs"][0]["failed_count"], 1)
            self.assertIsNotNone(run_compare["runs"][0]["best_style_score"])
            self.assertIsNotNone(run_compare["runs"][0]["best_combined_score"])
            self.assertIn("style", (run_compare_dir / "generation_run_reviews.csv").read_text(encoding="utf-8-sig"))
            self.assertTrue((run_compare_dir / "generation_run_reviews.csv").exists())
            self.assertEqual(
                main(
                    [
                        "compare-runs",
                        "--runs",
                        str(generated_empty / "evaluation"),
                        "--out",
                        str(empty_run_compare_dir),
                    ]
                ),
                0,
            )
            empty_run_compare = json.loads(
                (empty_run_compare_dir / "generation_run_reviews.json").read_text(encoding="utf-8")
            )
            self.assertEqual(empty_run_compare["run_count"], 1)
            self.assertIsNone(empty_run_compare["best_run_dir"])
            self.assertIsNone(empty_run_compare["best_centroid_score"])

            self.assertEqual(
                main(
                    [
                        "review-subjects",
                        "--model",
                        str(model_dir),
                        "--subjects",
                        str(subjects),
                        "--out",
                        str(review_dir),
                    ]
                ),
                0,
            )
            reviews = (review_dir / "subject_reviews.csv").read_text(encoding="utf-8-sig")
            self.assertIn("near_subject", reviews)
            self.assertIn("failed_count", reviews)
            self.assertIn("mean_centroid_score", reviews)
            review_json = json.loads((review_dir / "subject_reviews.json").read_text(encoding="utf-8"))
            review_html = (review_dir / "subject_reviews.html").read_text(encoding="utf-8")
            self.assertTrue((review_dir / "subject_reviews.html").exists())
            far_subject = next(item for item in review_json["subjects"] if item["subject"] == "far_subject")
            self.assertEqual(far_subject["failed_count"], 1)
            near_subject = next(item for item in review_json["subjects"] if item["subject"] == "near_subject")
            self.assertEqual(near_subject["top_images"][0]["image_id"], "near")
            self.assertEqual(review_json["analysis"]["top_mean_subjects"][0]["subject"], "near_subject")
            self.assertEqual(review_json["analysis"]["top_best_subjects"][0]["subject"], "near_subject")
            self.assertIn("single_image_lift", review_json["analysis"])
            self.assertIn("Vector Analysis", (review_dir / "subject_reviews.md").read_text(encoding="utf-8"))
            self.assertIn(
                "Stable Mean Leaders",
                (review_dir / "subject_reviews.md").read_text(encoding="utf-8"),
            )
            self.assertIn("near_subject", review_html)
            self.assertIn("<img", review_html)
            self.assertIn("Vector Analysis", review_html)
            self.assertIn("Stable Mean Leaders", review_html)
            self.assertIn((subjects / "near_subject" / "near.png").resolve(strict=False).as_uri(), review_html)

    def test_build_supports_subject_balanced_centroid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            (raw / "subject-a").mkdir(parents=True)
            (raw / "subject-b").mkdir(parents=True)
            model_dir = root / "model"

            _write_face_like_image(raw / "subject-a" / "a1.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(raw / "subject-a" / "a2.png", (230, 200, 188), eye_offset=1)
            _write_face_like_image(raw / "subject-b" / "b1.png", (175, 150, 132), eye_offset=7)

            result = main(["build", "--images", str(raw), "--out", str(model_dir), "--balance", "subject"])

            self.assertEqual(result, 0)
            model = load_model(model_dir)
            profile = json.loads((model_dir / "profile.json").read_text(encoding="utf-8"))
            subject_counts = json.loads((model_dir / "vectors" / "subject_counts.json").read_text(encoding="utf-8"))
            self.assertEqual(model.centroid_mode, "subject_balanced")
            self.assertEqual(model.subject_count, 2)
            self.assertEqual(profile["subject_counts"], {"subject-a": 2, "subject-b": 1})
            self.assertEqual(subject_counts, {"subject-a": 2, "subject-b": 1})

    def test_build_records_centroid_stability_and_audit_surfaces_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir(parents=True)
            model_dir = root / "model"
            audit_dir = root / "audit"

            _write_face_like_image(raw / "a.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(raw / "b.png", (180, 150, 130), eye_offset=6)
            _write_face_like_image(raw / "c.png", (210, 180, 160), eye_offset=3)

            self.assertEqual(main(["build", "--images", str(raw), "--out", str(model_dir)]), 0)

            stability = json.loads((model_dir / "vectors" / "centroid_stability.json").read_text(encoding="utf-8"))
            self.assertTrue(stability["available"])
            self.assertEqual(stability["unit"], "image")
            self.assertEqual(stability["unit_count"], 3)
            self.assertIn(stability["band"], {"stable", "moderate", "unstable"})

            self.assertEqual(main(["audit-model", "--model", str(model_dir), "--out", str(audit_dir)]), 0)
            audit = json.loads((audit_dir / "model_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(audit["stability"]["available"])
            self.assertEqual(audit["stability"]["unit_count"], 3)
            self.assertIn("## Centroid Stability", (audit_dir / "model_audit.md").read_text(encoding="utf-8"))

    def test_vectorize_subjects_writes_per_person_vectors_and_safe_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subjects = root / "subjects"
            (subjects / "alice").mkdir(parents=True)
            (subjects / "bob").mkdir(parents=True)
            out = root / "subject_vectors"

            _write_face_like_image(subjects / "alice" / "a1.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(subjects / "alice" / "a2.png", (230, 200, 188), eye_offset=1)
            _write_face_like_image(subjects / "bob" / "b1.png", (175, 150, 132), eye_offset=7)

            self.assertEqual(
                main(["vectorize-subjects", "--subjects", str(subjects), "--out", str(out), "--workers", "1"]),
                0,
            )

            with np.load(out / "alice" / "vector.npz") as data:
                mean_embedding = data["mean_embedding"]
            self.assertEqual(mean_embedding.shape, (1073,))
            self.assertAlmostEqual(float(np.linalg.norm(mean_embedding)), 1.0, places=4)

            profile = json.loads((out / "alice" / "profile.json").read_text(encoding="utf-8"))
            self.assertEqual(profile["image_count"], 2)
            self.assertEqual(profile["embedding_dim"], 1073)
            # profile.json is a metadata sidecar: no raw vectors and no appearance-derived descriptors.
            self.assertEqual(
                set(profile),
                {"subject", "image_count", "failed_count", "embedding_dim", "appearance_shape", "boundary"},
            )

            manifest_text = (out / "manifest.json").read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)
            self.assertEqual(manifest["subject_count"], 2)
            self.assertEqual(manifest["ok_count"], 2)
            by_subject = {row["subject"]: row for row in manifest["subjects"]}
            self.assertEqual(by_subject["alice"]["status"], "ok")
            self.assertEqual(by_subject["alice"]["image_count"], 2)
            self.assertEqual(len(by_subject["alice"]["mean_embedding_sha256"]), 64)
            self.assertNotIn("values", manifest_text)
            self.assertTrue((out / "manifest.csv").exists())
            # Boundary: manifest rows surface only safe metadata keys, never raw arrays.
            allowed_keys = {
                "subject", "status", "image_count", "failed_count", "embedding_dim",
                "appearance_shape", "mean_embedding_l2_norm", "mean_embedding_sha256",
                "median_embedding_sha256", "mean_appearance_sha256", "vector_path", "profile_path",
            }
            for row in manifest["subjects"]:
                self.assertLessEqual(set(row), allowed_keys)

    def test_vectorize_subjects_marks_empty_subject_without_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subjects = root / "subjects"
            (subjects / "alice").mkdir(parents=True)
            (subjects / "empty").mkdir(parents=True)
            out = root / "subject_vectors"

            _write_face_like_image(subjects / "alice" / "a1.png", (235, 205, 190), eye_offset=0)

            self.assertEqual(
                main(["vectorize-subjects", "--subjects", str(subjects), "--out", str(out), "--workers", "1"]),
                0,
            )

            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["empty_count"], 1)
            by_subject = {row["subject"]: row for row in manifest["subjects"]}
            self.assertEqual(by_subject["empty"]["status"], "empty")
            self.assertFalse((out / "empty" / "vector.npz").exists())
            self.assertTrue((out / "alice" / "vector.npz").exists())

    def test_vectorize_subjects_records_failed_subject_without_aborting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subjects = root / "subjects"
            (subjects / "alice").mkdir(parents=True)
            (subjects / "broken").mkdir(parents=True)
            out = root / "subject_vectors"

            _write_face_like_image(subjects / "alice" / "a1.png", (235, 205, 190), eye_offset=0)
            (subjects / "broken" / "bad.png").write_bytes(b"not a real png")

            self.assertEqual(
                main(["vectorize-subjects", "--subjects", str(subjects), "--out", str(out), "--workers", "1"]),
                0,
            )

            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            by_subject = {row["subject"]: row for row in manifest["subjects"]}
            self.assertEqual(by_subject["broken"]["status"], "failed")
            self.assertEqual(manifest["failed_count"], 1)
            self.assertTrue((out / "alice" / "vector.npz").exists())
            self.assertFalse((out / "broken" / "vector.npz").exists())

    def test_subject_id_fallback_uses_parent_for_external_paths(self) -> None:
        root = Path("dataset")
        external = Path("external") / "subject-x" / "face.png"

        self.assertEqual(_subject_id_for_path(root, external), "subject-x")

    def test_run_pipeline_config_builds_reviews_and_precision_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            generated = root / "generated"
            subjects = root / "subjects"
            model_dir = root / "model"
            eval_dir = root / "evaluation"
            audit_dir = root / "model_audit"
            vector_export = root / "vectors.csv"
            ingredients_dir = root / "face_ingredients"
            benchmark_research_dir = root / "benchmark_research"
            style_eval_dir = root / "style_eval_custom"
            review_dir = generated / "review"
            subject_review_dir = root / "subject_review"
            backend_compare_dir = root / "backend_compare"
            subject_backend_compare_dir = root / "subject_backend_compare"
            precision_dir = root / "precision"
            pipeline_dir = root / "pipeline_run"
            config_path = root / "pipeline.json"
            raw.mkdir()
            generated.mkdir()
            (subjects / "near_subject").mkdir(parents=True)
            _write_face_like_image(raw / "a.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(raw / "b.png", (225, 198, 184), eye_offset=2)
            _write_face_like_image(generated / "candidate.png", (232, 202, 188), eye_offset=1)
            _write_face_like_image(subjects / "near_subject" / "near.png", (232, 202, 188), eye_offset=1)
            config_path.write_text(
                json.dumps(
                    {
                        "name": "test_pipeline",
                        "reference_images": str(raw),
                        "model_out": str(model_dir),
                        "model_audit_out": str(audit_dir),
                        "vector_export_out": str(vector_export),
                        "face_ingredients_out": str(ingredients_dir),
                        "benchmark_research_out": str(benchmark_research_dir),
                        "generated_images": str(generated),
                        "evaluation_out": str(eval_dir),
                        "style_evaluation": {
                            "out": str(style_eval_dir),
                            "device": "cpu",
                        },
                        "review_out": str(review_dir),
                        "subjects": str(subjects),
                        "subject_review_out": str(subject_review_dir),
                        "backend_comparison": {
                            "out": str(backend_compare_dir),
                            "backends": ["deterministic"],
                        },
                        "subject_backend_comparison": {
                            "out": str(subject_backend_compare_dir),
                            "backends": ["deterministic"],
                        },
                        "precision_out": str(precision_dir),
                        "vector_backend": "deterministic",
                    }
                ),
                encoding="utf-8",
            )

            with patch("seju_face_lab.cli.OpenClipStyleBackend", return_value=_FakeStyleBackend()):
                self.assertEqual(
                    main(["run-pipeline", "--config", str(config_path), "--out", str(pipeline_dir)]),
                    0,
                )

            pipeline_run = json.loads((pipeline_dir / "pipeline_run.json").read_text(encoding="utf-8"))
            self.assertEqual([step["status"] for step in pipeline_run["steps"]], ["completed"] * 12)
            self.assertEqual(
                [step["name"] for step in pipeline_run["steps"]],
                [
                    "build",
                    "audit-model",
                    "export-vectors",
                    "ingredients-report",
                    "benchmark-research",
                    "evaluate",
                    "style-evaluate",
                    "review-generated",
                    "review-subjects",
                    "compare-backends",
                    "compare-subject-backends",
                    "precision-report",
                ],
            )
            self.assertTrue((model_dir / "centroids.npz").exists())
            self.assertTrue((audit_dir / "model_audit.json").exists())
            self.assertTrue(vector_export.exists())
            self.assertTrue((ingredients_dir / "face_ingredients.json").exists())
            self.assertTrue((benchmark_research_dir / "benchmark_research.json").exists())
            self.assertTrue((eval_dir / "summary.json").exists())
            self.assertTrue((style_eval_dir / "style_summary.json").exists())
            self.assertTrue((generated / "style_evaluation" / "style_summary.json").exists())
            self.assertTrue((review_dir / "generation_run_reviews.json").exists())
            self.assertTrue((review_dir / "generation_run_reviews.html").exists())
            self.assertTrue((subject_review_dir / "subject_reviews.json").exists())
            self.assertTrue((subject_review_dir / "subject_reviews.html").exists())
            self.assertTrue((backend_compare_dir / "backend_comparison.json").exists())
            self.assertTrue((subject_backend_compare_dir / "subject_backend_comparison.json").exists())
            precision = json.loads((precision_dir / "precision_report.json").read_text(encoding="utf-8"))
            self.assertEqual(precision["model"]["image_count"], 2)
            self.assertTrue(precision["model"]["model_audit"]["available"])
            self.assertTrue(precision["model"]["vector_export"]["available"])
            self.assertEqual(precision["model"]["vector_export"]["vectors"]["mean_embedding"]["shape"], [1073])
            self.assertIsNotNone(
                precision["model"]["model_audit"]["mean_median_embedding"]["cosine"]
            )
            self.assertEqual(precision["subjects"]["top_subject"], "near_subject")
            self.assertEqual(precision["backend_comparison"]["completed_backends"], ["deterministic"])
            self.assertEqual(precision["subject_backend_comparison"]["completed_backends"], ["deterministic"])
            self.assertIsNotNone(precision["generation"]["best_style_score"])
            self.assertIsNotNone(precision["generation"]["best_combined_score"])
            self.assertTrue(precision["face_ingredients"]["available"])
            self.assertIn("overall", precision["face_ingredients"])
            self.assertTrue(precision["benchmark_research"]["available"])
            self.assertIn("InsightFace", precision["benchmark_research"]["source_names"])

    def test_run_pipeline_uses_nested_generation_output_for_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            model_dir = root / "model"
            generated = root / "generated nested"
            eval_dir = root / "evaluation"
            precision_dir = root / "precision"
            pipeline_dir = root / "pipeline_run"
            config_path = root / "pipeline.json"
            raw.mkdir()
            _write_face_like_image(raw / "a.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(raw / "b.png", (225, 198, 184), eye_offset=2)
            config_path.write_text(
                json.dumps(
                    {
                        "name": "nested_generation_pipeline",
                        "reference_images": str(raw),
                        "model_out": str(model_dir),
                        "generation": {
                            "out": str(generated),
                            "provider": "dry-run",
                            "count": 1,
                        },
                        "evaluation_out": str(eval_dir),
                        "precision_out": str(precision_dir),
                        "vector_backend": "deterministic",
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                main(["run-pipeline", "--config", str(config_path), "--out", str(pipeline_dir)]),
                0,
            )

            self.assertTrue((generated / "generation_run.json").exists())
            self.assertTrue((eval_dir / "summary.json").exists())
            pipeline_run = json.loads((pipeline_dir / "pipeline_run.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [step["name"] for step in pipeline_run["steps"]],
                ["build", "generate", "evaluate", "precision-report"],
            )
            self.assertEqual([step["status"] for step in pipeline_run["steps"]], ["completed"] * 4)

    def test_run_pipeline_generation_sweep_writes_per_run_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            model_dir = root / "model"
            sweep_dir = root / "generation_sweep"
            precision_dir = root / "precision"
            pipeline_dir = root / "pipeline_run"
            config_path = root / "pipeline.json"
            raw.mkdir()
            _write_face_like_image(raw / "a.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(raw / "b.png", (225, 198, 184), eye_offset=2)
            config_path.write_text(
                json.dumps(
                    {
                        "name": "generation_sweep_pipeline",
                        "reference_images": str(raw),
                        "model_out": str(model_dir),
                        "generation_sweep": {
                            "out": str(sweep_dir),
                            "provider": "dry-run",
                            "count": 1,
                            "steps": 12,
                            "runs": [
                                {
                                    "name": "balanced_seed_42",
                                    "centroid_kind": "mean",
                                    "prompt_profile": "balanced",
                                    "seed": 42,
                                },
                                {
                                    "name": "detector_seed_43",
                                    "prompt_profile": "detector-friendly",
                                    "seed": 43,
                                },
                            ],
                        },
                        "precision_out": str(precision_dir),
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                main(["run-pipeline", "--config", str(config_path), "--out", str(pipeline_dir)]),
                0,
            )

            first = json.loads(
                (sweep_dir / "balanced_seed_42" / "generation_run.json").read_text(encoding="utf-8")
            )
            second = json.loads(
                (sweep_dir / "detector_seed_43" / "generation_run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first["config"]["seed"], 42)
            self.assertEqual(first["config"]["steps"], 12)
            self.assertEqual(first["config"]["centroid_kind"], "mean")
            self.assertEqual(first["config"]["prompt_profile"], "balanced")
            self.assertEqual(second["config"]["seed"], 43)
            self.assertEqual(second["config"]["centroid_kind"], "median")
            self.assertEqual(second["config"]["prompt_profile"], "detector-friendly")
            pipeline_run = json.loads((pipeline_dir / "pipeline_run.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [step["name"] for step in pipeline_run["steps"]],
                ["build", "generation-sweep", "precision-report"],
            )
            self.assertEqual([step["status"] for step in pipeline_run["steps"]], ["completed"] * 3)

    def test_run_pipeline_generation_sweep_compare_requires_reviewable_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            model_dir = root / "model"
            sweep_dir = root / "generation_sweep"
            pipeline_dir = root / "pipeline_run"
            config_path = root / "pipeline.json"
            raw.mkdir()
            _write_face_like_image(raw / "a.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(raw / "b.png", (225, 198, 184), eye_offset=2)
            config_path.write_text(
                json.dumps(
                    {
                        "name": "invalid_generation_sweep_pipeline",
                        "reference_images": str(raw),
                        "model_out": str(model_dir),
                        "generation_sweep": {
                            "out": str(sweep_dir),
                            "provider": "dry-run",
                            "compare_runs": True,
                            "runs": [{"name": "planned_only", "seed": 42}],
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                main(["run-pipeline", "--config", str(config_path), "--out", str(pipeline_dir)]),
                1,
            )

            pipeline_run = json.loads((pipeline_dir / "pipeline_run.json").read_text(encoding="utf-8"))
            self.assertEqual(pipeline_run["steps"][1]["name"], "generation-sweep")
            self.assertEqual(pipeline_run["steps"][1]["status"], "failed")
            self.assertIn("compare_runs requires", pipeline_run["steps"][1]["message"])

    def test_run_pipeline_generation_sweep_rejects_output_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            model_dir = root / "model"
            sweep_dir = root / "generation_sweep"
            pipeline_dir = root / "pipeline_run"
            config_path = root / "pipeline.json"
            raw.mkdir()
            _write_face_like_image(raw / "a.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(raw / "b.png", (225, 198, 184), eye_offset=2)
            config_path.write_text(
                json.dumps(
                    {
                        "name": "colliding_generation_sweep_pipeline",
                        "reference_images": str(raw),
                        "model_out": str(model_dir),
                        "generation_sweep": {
                            "out": str(sweep_dir),
                            "provider": "dry-run",
                            "runs": [
                                {"name": "seed/a", "seed": 42},
                                {"name": "seed:a", "seed": 43},
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                main(["run-pipeline", "--config", str(config_path), "--out", str(pipeline_dir)]),
                1,
            )

            pipeline_run = json.loads((pipeline_dir / "pipeline_run.json").read_text(encoding="utf-8"))
            self.assertEqual(pipeline_run["steps"][1]["name"], "generation-sweep")
            self.assertEqual(pipeline_run["steps"][1]["status"], "failed")
            self.assertIn("output collision", pipeline_run["steps"][1]["message"])

    def test_run_pipeline_compares_existing_model_without_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            generated = root / "generated"
            model_dir = root / "model"
            backend_compare_dir = root / "backend_compare"
            precision_dir = root / "precision"
            pipeline_dir = root / "pipeline_run"
            config_path = root / "pipeline.json"
            raw.mkdir()
            generated.mkdir()
            _write_face_like_image(raw / "a.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(raw / "b.png", (225, 198, 184), eye_offset=2)
            _write_face_like_image(generated / "candidate.png", (232, 202, 188), eye_offset=1)

            self.assertEqual(main(["build", "--images", str(raw), "--out", str(model_dir)]), 0)
            config_path.write_text(
                json.dumps(
                    {
                        "name": "existing_model_backend_compare",
                        "reference_images": str(raw),
                        "model": str(model_dir),
                        "generated_images": str(generated),
                        "backend_comparison": {
                            "out": str(backend_compare_dir),
                            "backends": ["deterministic"],
                        },
                        "precision_out": str(precision_dir),
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                main(["run-pipeline", "--config", str(config_path), "--out", str(pipeline_dir)]),
                0,
            )

            pipeline_run = json.loads((pipeline_dir / "pipeline_run.json").read_text(encoding="utf-8"))
            self.assertEqual([step["name"] for step in pipeline_run["steps"]], ["compare-backends", "precision-report"])
            precision = json.loads((precision_dir / "precision_report.json").read_text(encoding="utf-8"))
            self.assertEqual(precision["backend_comparison"]["completed_backends"], ["deterministic"])

    def test_run_pipeline_executes_sns_engagement_and_correlation(self) -> None:
        from seju_face_lab.sns_explorer import SnsProfile

        class FakeRouter:
            def fetch_batch(self, items, delay_between=1.5, force=False):
                self.items = items
                self.delay_between = delay_between
                self.force = force
                return [
                    SnsProfile(
                        platform=platform,
                        handle=handle,
                        profile_url=f"https://example.test/{platform}/{handle}",
                        followers=10_000,
                        posts=120,
                        total_engagement=1_200,
                        engagement_rate=0.001,
                        source="test",
                    )
                    for platform, handle in items
                ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            subjects = root / "subjects"
            model_dir = root / "model"
            subject_review_dir = root / "subject_review"
            handles = root / "sns_handles.jsonl"
            engagement = root / "sns_engagement.jsonl"
            correlation = root / "correlation"
            precision = root / "precision"
            pipeline_dir = root / "pipeline_run"
            config_path = root / "pipeline.json"
            raw.mkdir()
            (subjects / "near_subject").mkdir(parents=True)
            _write_face_like_image(raw / "a.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(raw / "b.png", (225, 198, 184), eye_offset=2)
            _write_face_like_image(subjects / "near_subject" / "near.png", (232, 202, 188), eye_offset=1)
            handles.write_text(
                json.dumps(
                    {
                        "talent_slug": "near_subject",
                        "name": "Near Subject",
                        "profile_url": "https://example.test/near_subject",
                        "sns_handles": {"instagram": "near_subject_ig"},
                        "retrieved_at": "2026-06-15T00:00:00+00:00",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps(
                    {
                        "name": "sns_correlation_pipeline",
                        "reference_images": str(raw),
                        "model_out": str(model_dir),
                        "subjects": str(subjects),
                        "subject_review_out": str(subject_review_dir),
                        "sns_engagement": {
                            "handles": str(handles),
                            "out": str(engagement),
                            "platforms": ["instagram"],
                            "delay": 0.0,
                            "force": True,
                        },
                        "correlation": {"out": str(correlation)},
                        "precision_out": str(precision),
                        "vector_backend": "deterministic",
                    }
                ),
                encoding="utf-8",
            )

            fake_router = FakeRouter()
            with patch("seju_face_lab.sns_explorer.build_router", return_value=fake_router):
                self.assertEqual(
                    main(["run-pipeline", "--config", str(config_path), "--out", str(pipeline_dir)]),
                    0,
                )

            pipeline_run = json.loads((pipeline_dir / "pipeline_run.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [step["name"] for step in pipeline_run["steps"]],
                ["build", "review-subjects", "explore-batch", "analyze-correlation", "precision-report"],
            )
            self.assertEqual([step["status"] for step in pipeline_run["steps"]], ["completed"] * 5)
            self.assertEqual(fake_router.items, [("instagram", "near_subject_ig")])
            self.assertEqual(fake_router.delay_between, 0.0)
            self.assertTrue(fake_router.force)
            self.assertTrue(engagement.exists())
            self.assertTrue((correlation / "correlation_summary.json").exists())
            summary = json.loads((correlation / "correlation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["talent_count"], 1)
            self.assertEqual(summary["with_face_score"], 1)
            self.assertEqual(summary["with_ig"], 1)
            precision_summary = json.loads((precision / "precision_report.json").read_text(encoding="utf-8"))
            self.assertTrue(precision_summary["correlation"]["available"])
            self.assertEqual(precision_summary["correlation"]["talent_count"], 1)

    def test_run_pipeline_writes_manifest_when_step_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "empty_raw"
            model_dir = root / "model"
            pipeline_dir = root / "pipeline_run"
            config_path = root / "pipeline.json"
            raw.mkdir()
            config_path.write_text(
                json.dumps(
                    {
                        "name": "failing_pipeline",
                        "reference_images": str(raw),
                        "model_out": str(model_dir),
                        "vector_backend": "deterministic",
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                main(["run-pipeline", "--config", str(config_path), "--out", str(pipeline_dir)]),
                1,
            )

            pipeline_run = json.loads((pipeline_dir / "pipeline_run.json").read_text(encoding="utf-8"))
            self.assertEqual(pipeline_run["steps"][0]["name"], "build")
            self.assertEqual(pipeline_run["steps"][0]["status"], "failed")
            self.assertIn("No supported images", pipeline_run["steps"][0]["message"])

    def test_run_pipeline_precision_uses_nested_generation_review_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            generated = root / "generated"
            review = root / "nested_review"
            model_dir = root / "model"
            precision_dir = root / "precision"
            pipeline_dir = root / "pipeline_run"
            config_path = root / "pipeline.json"
            raw.mkdir()
            generated.mkdir()
            review.mkdir()
            _write_face_like_image(raw / "a.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(raw / "b.png", (225, 198, 184), eye_offset=2)
            (review / "generation_run_reviews.json").write_text(
                json.dumps(
                    {
                        "run_count": 1,
                        "best_run_dir": str(generated),
                        "best_centroid_score": 0.25,
                        "runs": [
                            {
                                "image_count": 0,
                                "failed_count": 0,
                                "best_image_id": "planned_candidate",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps(
                    {
                        "name": "nested_review_pipeline",
                        "reference_images": str(raw),
                        "model_out": str(model_dir),
                        "generation": {
                            "out": str(generated),
                            "provider": "dry-run",
                            "review": True,
                            "review_out": str(review),
                        },
                        "precision_out": str(precision_dir),
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                main(["run-pipeline", "--config", str(config_path), "--out", str(pipeline_dir)]),
                0,
            )

            precision = json.loads((precision_dir / "precision_report.json").read_text(encoding="utf-8"))
            self.assertEqual(precision["generation"]["best_centroid_score"], 0.25)
            self.assertEqual(precision["generation"]["best_image_id"], "planned_candidate")

    def test_run_pipeline_plans_nested_subject_backend_subjects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            subjects = root / "nested_subjects"
            model_dir = root / "model"
            subject_backend_compare_dir = root / "subject_backend_compare"
            pipeline_dir = root / "pipeline_run"
            config_path = root / "pipeline.json"
            raw.mkdir()
            (subjects / "near_subject").mkdir(parents=True)
            _write_face_like_image(raw / "a.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(raw / "b.png", (225, 198, 184), eye_offset=2)
            _write_face_like_image(subjects / "near_subject" / "near.png", (232, 202, 188), eye_offset=1)
            config_path.write_text(
                json.dumps(
                    {
                        "name": "nested_subject_backend_pipeline",
                        "reference_images": str(raw),
                        "model_out": str(model_dir),
                        "subject_backend_comparison": {
                            "subjects": str(subjects),
                            "out": str(subject_backend_compare_dir),
                            "backends": ["deterministic"],
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                main(["run-pipeline", "--config", str(config_path), "--out", str(pipeline_dir)]),
                0,
            )

            pipeline_run = json.loads((pipeline_dir / "pipeline_run.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [step["name"] for step in pipeline_run["steps"]],
                ["build", "compare-subject-backends"],
            )
            self.assertTrue((subject_backend_compare_dir / "subject_backend_comparison.json").exists())

    def test_run_pipeline_merges_subject_backend_top_level_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            subjects = root / "nested_subjects"
            model_dir = root / "model"
            subject_backend_compare_dir = root / "subject_backend_compare"
            pipeline_dir = root / "pipeline_run"
            config_path = root / "pipeline.json"
            raw.mkdir()
            (subjects / "near_subject").mkdir(parents=True)
            _write_face_like_image(raw / "a.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(raw / "b.png", (225, 198, 184), eye_offset=2)
            _write_face_like_image(subjects / "near_subject" / "near.png", (232, 202, 188), eye_offset=1)
            config_path.write_text(
                json.dumps(
                    {
                        "name": "mixed_subject_backend_pipeline",
                        "reference_images": str(raw),
                        "model_out": str(model_dir),
                        "subject_backend_comparison_out": str(subject_backend_compare_dir),
                        "subject_backend_comparison": {
                            "subjects": str(subjects),
                            "backends": ["deterministic"],
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                main(["run-pipeline", "--config", str(config_path), "--out", str(pipeline_dir)]),
                0,
            )

            pipeline_run = json.loads((pipeline_dir / "pipeline_run.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [step["name"] for step in pipeline_run["steps"]],
                ["build", "compare-subject-backends"],
            )
            self.assertTrue((subject_backend_compare_dir / "subject_backend_comparison.json").exists())

    def test_backends_command_lists_generation_providers(self) -> None:
        self.assertEqual(main(["backends"]), 0)
        self.assertNotIn("diffusion-generation", backends.BACKENDS)
        self.assertIn("diffusers", backends.GENERATION_PROVIDERS)

    def test_backend_diagnostics_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "diagnostics"

            self.assertEqual(main(["backend-diagnostics", "--out", str(out)]), 0)

            report = json.loads((out / "backend_diagnostics.json").read_text(encoding="utf-8"))
            self.assertIn("backends", report)
            self.assertIn("generation_providers", report)
            self.assertIn("runtime", report)
            self.assertTrue((out / "backend_diagnostics.md").exists())
            self.assertIn("insightface", {item["name"] for item in report["backends"]})
            self.assertIn("diffusers", {item["name"] for item in report["generation_providers"]})

    def test_benchmark_research_writes_vectorization_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "benchmark_research"

            self.assertEqual(main(["benchmark-research", "--out", str(out)]), 0)

            report = json.loads((out / "benchmark_research.json").read_text(encoding="utf-8"))
            source_names = {item["name"] for item in report["sources"]}
            self.assertIn("NIST FRTE/FATE", source_names)
            self.assertIn("IJB-C", source_names)
            self.assertIn("MegaFace", source_names)
            self.assertIn("AdaFace", source_names)
            self.assertIn("InsightFace", source_names)
            self.assertIn("Worldcoin Open IRIS", source_names)
            self.assertIn("subject-balanced", report["vectorization_strategy"]["centroid_aggregation"])
            self.assertEqual(report["vectorization_strategy"]["iris_axis"], "out of scope for face-vector scoring; separate modality only")
            self.assertTrue((out / "benchmark_research.md").exists())

    def test_review_agencies_writes_average_params_and_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            agencies = root / "agencies.json"
            out = root / "agency_review"
            model.mkdir()
            np.savez_compressed(
                model / "centroids.npz",
                mean_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                median_embedding=np.asarray([0.9, 0.1], dtype=np.float32),
                mean_appearance=np.zeros((2, 2, 1), dtype=np.float32),
                median_appearance=np.ones((2, 2, 1), dtype=np.float32),
                image_ids=np.asarray(["a"]),
                source_paths=np.asarray(["a.png"]),
            )
            (model / "profile.json").write_text(
                json.dumps(
                    {
                        "image_count": 1,
                        "embedding_dim": 2,
                        "appearance_shape": [2, 2, 1],
                        "descriptors": {
                            "median": {
                                "luminance": 0.7,
                                "contrast": 0.08,
                                "saturation": 0.08,
                                "warmth": 0.0,
                                "symmetry": 0.96,
                                "edge_density": 0.02,
                                "upper_band_darkness": 0.2,
                                "middle_luminance": 0.72,
                                "lower_luminance": 0.68,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            agencies.write_text(
                json.dumps(
                    {
                        "retrieved_at": "2026-06-15",
                        "agencies": [
                            {
                                "slug": "seju",
                                "name": "seju",
                                "official_sources": [{"url": "https://seju.tokyo/"}],
                                "public_examples": ["example"],
                                "positioning": ["SNS-native"],
                                "descriptor_offsets": {},
                            },
                            {
                                "slug": "styled",
                                "name": "Styled Agency",
                                "official_sources": [{"url": "https://example.com/"}],
                                "public_examples": ["example"],
                                "positioning": ["model"],
                                "descriptor_offsets": {"contrast": 0.04, "saturation": 0.02},
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                main(["review-agencies", "--model", str(model), "--agencies", str(agencies), "--out", str(out)]),
                0,
            )

            report = json.loads((out / "agency_average_params.json").read_text(encoding="utf-8"))
            self.assertEqual(report["rankings"]["by_descriptor_similarity"][0]["slug"], "seju")
            self.assertEqual(len(report["agencies"]), 2)
            self.assertTrue((out / "prompts" / "seju.txt").exists())
            self.assertIn("fictional young adult", (out / "prompts" / "styled.txt").read_text(encoding="utf-8"))
            self.assertIn("Analysis Logic", (out / "agency_average_params.md").read_text(encoding="utf-8"))
            self.assertIn("axis_vector", report["agencies"][0])
            self.assertIn("axis_distribution", report["agencies"][0])

    def test_face_axes_maps_images_to_8_axis_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            out = root / "axis"
            images.mkdir()
            _write_face_like_image(images / "bright.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(images / "defined.png", (170, 145, 130), eye_offset=7)

            self.assertEqual(main(["face-axes", "--images", str(images), "--out", str(out)]), 0)

            report = json.loads((out / "face_axis_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["image_count"], 2)
            self.assertEqual(set(report["summary"]["axis_means"]), {
                "soft_defined",
                "cool_warm",
                "deep_bright",
                "natural_styled",
                "muted_vivid",
                "soft_crisp",
                "light_dark_hair",
                "dynamic_symmetric",
            })
            self.assertIn(report["summary"]["distribution"]["quadrant"], {
                "defined_bright",
                "soft_bright",
                "soft_deep",
                "defined_deep",
            })
            self.assertIn("outlier_score", report["summary"]["distribution"])
            self.assertIn("presentation_flags", report["summary"]["distribution"])
            self.assertTrue((out / "face_axis_scores.csv").exists())
            self.assertIn("Axis Definitions", (out / "face_axis_report.md").read_text(encoding="utf-8"))
            self.assertIn("presentation_flags", (out / "face_axis_scores.csv").read_text(encoding="utf-8-sig"))

    def test_enhance_agencies_fuses_hypothesis_scores_and_image_axes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            images = root / "generated"
            model = root / "model"
            agencies = root / "agencies.json"
            out = root / "enhanced"
            raw.mkdir()
            images.mkdir()
            _write_face_like_image(raw / "ref_a.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(raw / "ref_b.png", (225, 198, 184), eye_offset=2)
            _write_face_like_image(images / "seju.png", (232, 202, 188), eye_offset=1)
            _write_face_like_image(images / "styled.png", (170, 145, 130), eye_offset=7)
            agencies.write_text(
                json.dumps(
                    {
                        "retrieved_at": "2026-06-15",
                        "agencies": [
                            {
                                "slug": "seju",
                                "name": "seju",
                                "official_sources": [{"url": "https://seju.tokyo/"}],
                                "public_examples": ["example"],
                                "positioning": ["SNS-native"],
                                "descriptor_offsets": {},
                            },
                            {
                                "slug": "styled",
                                "name": "Styled Agency",
                                "official_sources": [{"url": "https://example.com/"}],
                                "public_examples": ["example"],
                                "positioning": ["model"],
                                "descriptor_offsets": {"contrast": 0.04, "saturation": 0.02},
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(main(["build", "--images", str(raw), "--out", str(model)]), 0)
            self.assertEqual(
                main(
                    [
                        "enhance-agencies",
                        "--model",
                        str(model),
                        "--agencies",
                        str(agencies),
                        "--images",
                        str(images),
                        "--out",
                        str(out),
                    ]
                ),
                0,
            )

            report = json.loads((out / "agency_enhancement_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["image_count"], 2)
            self.assertEqual(report["summary"]["measured_count"], 2)
            self.assertEqual(len(report["agencies"]), 2)
            self.assertEqual(report["agencies"][0]["confidence"], "measured")
            self.assertIn("axis_alignment", report["agencies"][0]["components"])
            self.assertIn("improvement_actions", report["agencies"][0])
            self.assertTrue((out / "image_scores" / "scores.csv").exists())
            self.assertTrue((out / "face_axes" / "face_axis_report.json").exists())
            self.assertIn("agency enhancement report", (out / "agency_enhancement_report.md").read_text(encoding="utf-8"))

    def test_agency_site_renders_8_axis_map(self) -> None:
        spec = importlib.util.spec_from_file_location("build_agency_site", Path("scripts/build_agency_site.py"))
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        site = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(site)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "site"
            images = root / "images"
            images.mkdir()
            config = {
                "retrieved_at": "2026-06-15",
                "agencies": [
                    {
                        "slug": "seju",
                        "public_examples": ["sample"],
                        "positioning": ["sample positioning"],
                        "official_sources": [{"name": "official", "url": "https://example.com"}],
                    }
                ],
            }
            params = {"agencies": [{"slug": "seju", "average_descriptors": {}, "axis_vector": {}}]}
            enhancement = {
                "summary": {"top_slug": "seju", "top_score": 0.8},
                "agencies": [
                    {
                        "slug": "seju",
                        "name": "seju",
                        "rank": 1,
                        "enhancement_score": 0.8,
                        "confidence": "measured",
                        "components": {
                            "descriptor_similarity": 1.0,
                            "image_centroid_score": 0.4,
                            "axis_alignment": 0.6,
                        },
                        "observed_distribution": {"quadrant": "defined_bright"},
                        "observed_axis_vector": {
                            "soft_defined": 0.7,
                            "cool_warm": 0.3,
                            "deep_bright": 0.5,
                            "natural_styled": 0.2,
                            "muted_vivid": -0.2,
                            "soft_crisp": 0.4,
                            "light_dark_hair": 0.5,
                            "dynamic_symmetric": 0.6,
                        },
                        "hypothesis_axis_vector": {},
                        "presentation_flags": [],
                        "improvement_actions": [],
                    }
                ],
            }
            data_quality = {
                "summary": {"risk_level": "medium"},
                "agency_evidence": {
                    "agencies": [
                        {
                            "slug": "seju",
                            "evidence_type": "real_centroid_baseline",
                            "real_image_count": 3,
                            "generated_image_count": 1,
                        }
                    ]
                },
            }

            site.build_site(config, params, enhancement, {}, data_quality, images, out)

            html = (out / "index.html").read_text(encoding="utf-8")
            data = json.loads((out / "data.json").read_text(encoding="utf-8"))
            self.assertIn("8軸方向性マップ", html)
            self.assertIn('class="axis-map"', html)
            self.assertIn("soft defined", html)
            self.assertIn("real centroid baseline", html)
            self.assertEqual(data["data_quality_summary"]["risk_level"], "medium")
            self.assertEqual(data["agencies"][0]["evidence"]["real_image_count"], 3)
            self.assertEqual(data["agencies"][0]["observed_axis_vector"]["dynamic_symmetric"], 0.6)

    def test_data_quality_audit_flags_hypothesis_agencies_and_image_quality(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "audit_research_data_quality",
            Path("scripts/audit_research_data_quality.py"),
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seju = root / "seju"
            (seju / "subject-a").mkdir(parents=True)
            (seju / "subject-b").mkdir()
            Image.new("RGB", (320, 500), (230, 200, 190)).save(seju / "subject-a" / "a.png")
            Image.new("RGB", (320, 500), (230, 200, 190)).save(seju / "subject-a" / "b.png")
            Image.new("RGB", (800, 800), (230, 200, 190)).save(seju / "subject-b" / "c.png")
            agencies = root / "agencies.json"
            agencies.write_text(
                json.dumps(
                    {
                        "agencies": [
                            {"slug": "seju", "name": "seju"},
                            {"slug": "contrast", "name": "contrast"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            enhancement = root / "enhancement.json"
            enhancement.write_text(
                json.dumps(
                    {
                        "agencies": [
                            {
                                "slug": "seju",
                                "enhancement_score": 0.8,
                                "components": {"image_centroid_score": 0.4, "axis_alignment": 0.5},
                                "observed_distribution": {"quadrant": "defined_bright"},
                            },
                            {
                                "slug": "contrast",
                                "enhancement_score": 0.6,
                                "components": {"image_centroid_score": 0.2, "axis_alignment": 0.3},
                                "observed_distribution": {"quadrant": "defined_bright"},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = audit.build_report(
                seju_subjects=seju,
                agencies=agencies,
                agency_real_root=root / "agency-real",
                agency_enhancement=enhancement,
                subject_review=root / "missing.json",
            )

            codes = {item["code"] for item in report["issues"]}
            self.assertIn("non_seju_real_data_missing", codes)
            self.assertIn("small_images_present", codes)
            self.assertEqual(report["agency_evidence"]["agencies"][0]["real_image_count"], 3)
            self.assertEqual(report["summary"]["risk_level"], "needs_real_data_before_strong_claims")

    def test_calibrate_agency_generation_writes_refined_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            enhancement = root / "enhancement.json"
            params = root / "params.json"
            out = root / "calibration"
            enhancement.write_text(
                json.dumps(
                    {
                        "agencies": [
                            {
                                "slug": "seju",
                                "name": "seju",
                                "rank": 1,
                                "enhancement_score": 0.42,
                                "components": {
                                    "image_centroid_score": 0.05,
                                    "axis_alignment": 0.38,
                                },
                                "hypothesis_axis_vector": {
                                    "soft_defined": -0.8,
                                    "deep_bright": 0.8,
                                    "dynamic_symmetric": 0.9,
                                },
                                "observed_axis_vector": {
                                    "soft_defined": 0.8,
                                    "deep_bright": -0.4,
                                    "dynamic_symmetric": -0.2,
                                },
                                "presentation_flags": [
                                    "dark_or_underlit_image",
                                    "high_texture_or_messy_edges",
                                ],
                                "improvement_actions": [
                                    "regenerate_with_detector_friendly_prompt",
                                    "increase_even_front_lighting",
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            params.write_text(
                json.dumps(
                    {
                        "agencies": [
                            {
                                "slug": "seju",
                                "imagegen_prompt": "Base fictional aggregate prompt.",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                main(
                    [
                        "calibrate-agency-generation",
                        "--enhancement",
                        str(enhancement),
                        "--agency-params",
                        str(params),
                        "--out",
                        str(out),
                    ]
                ),
                0,
            )

            report = json.loads((out / "generation_calibration.json").read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["priority_counts"]["regenerate"], 1)
            self.assertEqual(report["agencies"][0]["priority"], "regenerate")
            self.assertIn("Axis corrections", report["agencies"][0]["calibrated_prompt"])
            self.assertIn("underexposed face", report["agencies"][0]["negative_prompt"])
            self.assertTrue((out / "prompts" / "seju_calibrated.txt").exists())
            self.assertTrue((out / "generation_calibration.csv").exists())

    def test_compare_backends_writes_rank_agreement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            generated = root / "generated"
            out = root / "backend_compare"
            raw.mkdir()
            generated.mkdir()
            _write_face_like_image(raw / "a.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(raw / "b.png", (225, 198, 184), eye_offset=2)
            _write_face_like_image(generated / "candidate_a.png", (232, 202, 188), eye_offset=1)
            _write_face_like_image(generated / "candidate_b.png", (170, 145, 130), eye_offset=7)

            with patch("seju_face_lab.backends._import_cv2", return_value=_FakeCV2()):
                self.assertEqual(
                    main(
                        [
                            "compare-backends",
                            "--reference-images",
                            str(raw),
                            "--images",
                            str(generated),
                            "--out",
                            str(out),
                            "--backends",
                            "deterministic",
                            "opencv-face",
                        ]
                    ),
                    0,
                )

            report = json.loads((out / "backend_comparison.json").read_text(encoding="utf-8"))
            self.assertEqual([run["status"] for run in report["runs"]], ["completed", "completed"])
            self.assertEqual(len(report["rank_agreement"]), 1)
            self.assertEqual(report["rank_agreement"][0]["common_image_count"], 2)
            self.assertTrue((out / "deterministic" / "evaluation" / "scores.csv").exists())
            self.assertTrue((out / "opencv-face" / "model" / "centroids.npz").exists())

    def test_compare_backends_rank_agreement_uses_unique_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            generated = root / "generated"
            out = root / "backend_compare"
            raw.mkdir()
            (generated / "a").mkdir(parents=True)
            (generated / "b").mkdir(parents=True)
            _write_face_like_image(raw / "reference.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(generated / "a" / "same.png", (232, 202, 188), eye_offset=1)
            _write_face_like_image(generated / "b" / "same.png", (170, 145, 130), eye_offset=7)

            with patch("seju_face_lab.backends._import_cv2", return_value=_FakeCV2()):
                self.assertEqual(
                    main(
                        [
                            "compare-backends",
                            "--reference-images",
                            str(raw),
                            "--images",
                            str(generated),
                            "--out",
                            str(out),
                            "--backends",
                            "deterministic",
                            "opencv-face",
                        ]
                    ),
                    0,
                )

            report = json.loads((out / "backend_comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(report["rank_agreement"][0]["common_image_count"], 2)

    def test_compare_subject_backends_writes_subject_rank_agreement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            subjects = root / "subjects"
            out = root / "subject_backend_compare"
            raw.mkdir()
            (subjects / "near_subject").mkdir(parents=True)
            (subjects / "far_subject").mkdir(parents=True)
            _write_face_like_image(raw / "a.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(raw / "b.png", (225, 198, 184), eye_offset=2)
            _write_face_like_image(subjects / "near_subject" / "near.png", (232, 202, 188), eye_offset=1)
            _write_face_like_image(subjects / "far_subject" / "far.png", (170, 145, 130), eye_offset=7)

            with patch("seju_face_lab.backends._import_cv2", return_value=_FakeCV2()):
                self.assertEqual(
                    main(
                        [
                            "compare-subject-backends",
                            "--reference-images",
                            str(raw),
                            "--subjects",
                            str(subjects),
                            "--out",
                            str(out),
                            "--backends",
                            "deterministic",
                            "opencv-face",
                        ]
                    ),
                    0,
                )

            report = json.loads((out / "subject_backend_comparison.json").read_text(encoding="utf-8"))
            self.assertEqual([run["status"] for run in report["runs"]], ["completed", "completed"])
            self.assertEqual(report["runs"][0]["top_subject"], "near_subject")
            self.assertEqual(len(report["rank_agreement"]), 1)
            self.assertEqual(report["rank_agreement"][0]["common_subject_count"], 2)
            self.assertTrue((out / "deterministic" / "subject_review" / "subject_reviews.json").exists())
            self.assertIn(
                "subject backend comparison",
                (out / "subject_backend_comparison.md").read_text(encoding="utf-8"),
            )

    def test_compare_backends_rejects_unknown_backend_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            generated = root / "generated"
            raw.mkdir()
            generated.mkdir()
            _write_face_like_image(raw / "a.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(generated / "candidate_a.png", (232, 202, 188), eye_offset=1)

            with self.assertRaisesRegex(ValueError, "Unknown backend"):
                main(
                    [
                        "compare-backends",
                        "--reference-images",
                        str(raw),
                        "--images",
                        str(generated),
                        "--out",
                        str(root / "backend_compare"),
                        "--backends",
                        "deterministic",
                        "opencvface",
                    ]
                )

    def test_review_subjects_reports_missing_directory(self) -> None:
        with self.assertRaisesRegex(SystemExit, "No subject directory found"):
            main(
                [
                    "review-subjects",
                    "--model",
                    "unused",
                    "--subjects",
                    "missing-subjects",
                    "--out",
                    "unused",
                ]
            )

    def test_sources_download_returns_nonzero_on_failed_rows(self) -> None:
        failed = DownloadResult(
            profile_url="https://seju.tokyo/talents/example/",
            talent_slug="example",
            image_url="https://seju.tokyo/wp-content/uploads/example.jpg",
            status="failed",
            path=None,
            sha256=None,
            bytes=0,
            reason="network error",
        )
        args = Namespace(
            manifest=Path("manifest.jsonl"),
            out=Path("data/raw/example"),
            max_count=None,
            dry_run=False,
            include_ineligible=False,
            delay_seconds=0.0,
            max_bytes=20_000_000,
            user_agent="test-agent",
        )
        with (
            patch("seju_face_lab.cli.read_source_manifest", return_value=[]),
            patch("seju_face_lab.cli.download_source_images", return_value=[failed]),
        ):
            self.assertEqual(_sources_download(args), 1)


def _write_face_like_image(path: Path, skin: tuple[int, int, int], eye_offset: int) -> None:
    image = Image.new("RGB", (96, 96), (245, 242, 238))
    draw = ImageDraw.Draw(image)
    draw.ellipse((22, 14, 74, 82), fill=skin)
    draw.ellipse((36 + eye_offset, 40, 40 + eye_offset, 44), fill=(40, 35, 34))
    draw.ellipse((56 + eye_offset, 40, 60 + eye_offset, 44), fill=(40, 35, 34))
    draw.arc((40, 52, 58, 66), start=0, end=180, fill=(150, 80, 82), width=2)
    image.save(path)


class _FakeCV2:
    COLOR_RGB2GRAY = 1

    class data:
        haarcascades = "."

    @staticmethod
    def cvtColor(rgb: np.ndarray, _code: int) -> np.ndarray:
        return np.mean(rgb, axis=2).astype(np.uint8)

    class CascadeClassifier:
        def __init__(self, _path: str) -> None:
            pass

        def empty(self) -> bool:
            return False

        def detectMultiScale(
            self,
            _gray: np.ndarray,
            scaleFactor: float,
            minNeighbors: int,
            minSize: tuple[int, int],
        ) -> np.ndarray:
            return np.asarray([[24, 16, 52, 60]], dtype=np.int32)


class _FakeStyleBackend:
    name = "fake-style"
    description = "test style backend"

    def encode_path(self, path: Path) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        return self.encode_pil(image)

    def encode_pil(self, image: Image.Image) -> np.ndarray:
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        vector = np.asarray(
            [
                float(np.mean(rgb[:, :, 0])),
                float(np.mean(rgb[:, :, 1])),
                float(np.mean(rgb[:, :, 2])),
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm


if __name__ == "__main__":
    unittest.main()
