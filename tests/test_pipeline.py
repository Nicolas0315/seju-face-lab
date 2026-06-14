from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

import bootstrap  # noqa: F401
from seju_face_lab.cli import _sources_download, main
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
                    ]
                ),
                0,
            )
            generation_run = json.loads((generation_dir / "generation_run.json").read_text(encoding="utf-8"))
            self.assertEqual(generation_run["result"]["status"], "planned")
            self.assertEqual(generation_run["config"]["provider"], "dry-run")
            self.assertEqual(generation_run["config"]["count"], 2)
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
            far_subject = next(item for item in review_json["subjects"] if item["subject"] == "far_subject")
            self.assertEqual(far_subject["failed_count"], 1)

    def test_run_pipeline_config_builds_reviews_and_precision_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            generated = root / "generated"
            subjects = root / "subjects"
            model_dir = root / "model"
            eval_dir = root / "evaluation"
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
            self.assertEqual([step["status"] for step in pipeline_run["steps"]], ["completed"] * 8)
            self.assertEqual(
                [step["name"] for step in pipeline_run["steps"]],
                [
                    "build",
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
            self.assertTrue((eval_dir / "summary.json").exists())
            self.assertTrue((style_eval_dir / "style_summary.json").exists())
            self.assertTrue((generated / "style_evaluation" / "style_summary.json").exists())
            self.assertTrue((review_dir / "generation_run_reviews.json").exists())
            self.assertTrue((review_dir / "generation_run_reviews.html").exists())
            self.assertTrue((subject_review_dir / "subject_reviews.json").exists())
            self.assertTrue((backend_compare_dir / "backend_comparison.json").exists())
            self.assertTrue((subject_backend_compare_dir / "subject_backend_comparison.json").exists())
            precision = json.loads((precision_dir / "precision_report.json").read_text(encoding="utf-8"))
            self.assertEqual(precision["model"]["image_count"], 2)
            self.assertEqual(precision["subjects"]["top_subject"], "near_subject")
            self.assertEqual(precision["backend_comparison"]["completed_backends"], ["deterministic"])
            self.assertEqual(precision["subject_backend_comparison"]["completed_backends"], ["deterministic"])
            self.assertIsNotNone(precision["generation"]["best_style_score"])
            self.assertIsNotNone(precision["generation"]["best_combined_score"])

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

    def test_backends_command_lists_planned_backends(self) -> None:
        self.assertEqual(main(["backends"]), 0)

    def test_backend_diagnostics_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "diagnostics"

            self.assertEqual(main(["backend-diagnostics", "--out", str(out)]), 0)

            report = json.loads((out / "backend_diagnostics.json").read_text(encoding="utf-8"))
            self.assertIn("backends", report)
            self.assertIn("runtime", report)
            self.assertTrue((out / "backend_diagnostics.md").exists())
            self.assertIn("insightface", {item["name"] for item in report["backends"]})

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
