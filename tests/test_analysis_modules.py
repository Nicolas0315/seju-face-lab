from __future__ import annotations

import csv
import base64
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bootstrap  # noqa: F401
import numpy as np
from PIL import Image
from scripts.organize_by_talent import _talent_slug_from_filename
from seju_face_lab import backends as backends_module
from seju_face_lab.backends import DeepFaceBackend, InsightFaceBackend
from seju_face_lab.cli import main
from seju_face_lab.correlation import (
    build_correlation_dataset,
    compute_correlations,
    write_correlation_report,
)
from seju_face_lab.generation import build_generation_config, run_openai_image_generation
from seju_face_lab.precision import write_precision_report
from seju_face_lab.model_audit import write_model_audit
from seju_face_lab.quality import ImageQuality, judge_face_quality, review_image_quality
from seju_face_lab.run_reviews import review_generation_runs, write_generation_run_reviews
from seju_face_lab.sns_metrics import (
    SnsEngagement,
    SnsHandleRecord,
    TalentEngagementRecord,
    extract_sns_handles_from_links,
    fetch_instagram_engagement,
    fetch_tiktok_engagement,
    fetch_twitter_engagement,
    read_engagement_manifest,
    read_handles_manifest,
    import_engagement_csv,
    write_engagement_manifest,
    write_handles_manifest,
)
from seju_face_lab.workers import (
    WorkerConfig,
    _split_paths,
    distribute_vectorize,
    run_local_evaluate,
    write_worker_diagnostics,
)


class AnalysisModuleTests(unittest.TestCase):
    def test_generated_image_quality_gate_flags_collage_and_cropping(self) -> None:
        self.assertEqual(judge_face_quality(1, 0.20, 0.10), (True, "single centered face"))
        self.assertEqual(judge_face_quality(4, 0.20, 0.10), (False, "requires exactly one detected face"))
        self.assertEqual(judge_face_quality(1, 0.70, 0.10), (False, "detected face is too close or cropped"))
        self.assertEqual(judge_face_quality(1, 0.20, 0.40), (False, "detected face is off center"))

    def test_cli_qa_images_writes_quality_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "quality"
            with patch(
                "seju_face_lab.cli.review_image_quality",
                return_value=[
                    ImageQuality(
                        image_id="candidate",
                        path="candidate.png",
                        face_count=1,
                        largest_face_area_ratio=0.2,
                        center_offset=0.1,
                        qa_pass=True,
                        reason="single centered face",
                    )
                ],
            ):
                self.assertEqual(main(["qa-images", "--images", str(root), "--out", str(out)]), 0)

            summary = json.loads((out / "image_quality.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["pass_count"], 1)

    def test_cli_review_generated_runs_evaluation_quality_and_compare(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            images = root / "generated"
            out = root / "review"
            images.mkdir()

            with patch("seju_face_lab.cli._evaluate") as evaluate:
                with patch("seju_face_lab.cli._qa_images") as qa_images:
                    with patch("seju_face_lab.cli._compare_runs") as compare_runs:
                        self.assertEqual(
                            main(
                                [
                                    "review-generated",
                                    "--model",
                                    str(model),
                                    "--images",
                                    str(images),
                                    "--out",
                                    str(out),
                                ]
                            ),
                            0,
                        )

            evaluate.assert_called_once_with(model, images, images / "evaluation", "center", "deterministic")
            qa_images.assert_called_once_with(images, images / "quality")
            compare_runs.assert_called_once_with([images], out)

    def test_cli_generate_review_runs_after_real_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            out = root / "generated"
            review_out = root / "review"
            config = SimpleNamespace(provider="diffusers")
            result = SimpleNamespace(
                status="generated",
                evaluation_command="python -m seju_face_lab evaluate",
                generated_images=[str(out / "candidate.png")],
            )

            with patch("seju_face_lab.cli.build_generation_config", return_value=config):
                with patch("seju_face_lab.cli.run_diffusers_generation", return_value=result):
                    with patch("seju_face_lab.cli._review_generated") as review_generated:
                        self.assertEqual(
                            main(
                                [
                                    "generate",
                                    "--model",
                                    str(model),
                                    "--out",
                                    str(out),
                                    "--provider",
                                    "diffusers",
                                    "--review",
                                    "--review-out",
                                    str(review_out),
                                ]
                            ),
                            0,
                        )

            review_generated.assert_called_once()
            review_args = review_generated.call_args.args[0]
            self.assertEqual(review_args.model, model)
            self.assertEqual(review_args.images, out)
            self.assertEqual(review_args.out, review_out)
            self.assertEqual(review_args.backend, "deterministic")

    def test_cli_generate_review_runs_after_openai_image_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            out = root / "generated"
            review_out = root / "review"
            config = SimpleNamespace(provider="openai-image")
            result = SimpleNamespace(
                status="generated",
                evaluation_command="python -m seju_face_lab evaluate",
                generated_images=[str(out / "candidate.png")],
            )

            with patch("seju_face_lab.cli.build_generation_config", return_value=config) as build_config:
                with patch("seju_face_lab.cli.run_openai_image_generation", return_value=result):
                    with patch("seju_face_lab.cli._review_generated") as review_generated:
                        self.assertEqual(
                            main(
                                [
                                    "generate",
                                    "--model",
                                    str(model),
                                    "--out",
                                    str(out),
                                    "--provider",
                                    "openai-image",
                                    "--review",
                                    "--review-out",
                                    str(review_out),
                                ]
                            ),
                            0,
                        )

            review_generated.assert_called_once()
            review_args = review_generated.call_args.args[0]
            self.assertEqual(review_args.model, model)
            self.assertEqual(review_args.images, out)
            self.assertEqual(review_args.out, review_out)
            config_kwargs = build_config.call_args.kwargs
            self.assertEqual(config_kwargs["model_id"], "gpt-image-2")
            self.assertEqual(config_kwargs["width"], 1024)
            self.assertEqual(config_kwargs["height"], 1024)

    def test_openai_image_generation_writes_b64_images_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            out = root / "generated"
            model.mkdir()
            (model / "generation_manifest.json").write_text(
                json.dumps(
                    {
                        "prompt": "aggregate face prompt",
                        "negative_prompt": "copied identity",
                        "prompt_profiles": {},
                        "negative_prompt_profiles": {},
                    }
                ),
                encoding="utf-8",
            )
            image_bytes = base64.b64encode(b"fake-png-bytes").decode("ascii")
            response = SimpleNamespace(data=[SimpleNamespace(b64_json=image_bytes)])
            client = SimpleNamespace(images=SimpleNamespace(generate=lambda **_kwargs: response))
            config = build_generation_config(
                model_dir=model,
                provider="openai-image",
                model_id="gpt-image-2",
                count=1,
                seed=42,
                steps=1,
                guidance_scale=1.0,
                width=1024,
                height=1024,
                device="api",
                dtype="api",
                variant=None,
                output_format="png",
                quality="low",
            )

            with patch("seju_face_lab.generation._openai_client", return_value=client):
                result = run_openai_image_generation(config, model, out)

            self.assertEqual(result.status, "generated")
            self.assertEqual(result.provider, "openai-image")
            self.assertEqual((out / "candidate_0001_openai.png").read_bytes(), b"fake-png-bytes")
            run = json.loads((out / "generation_run.json").read_text(encoding="utf-8"))
            self.assertEqual(run["config"]["model_id"], "gpt-image-2")
            self.assertEqual(run["config"]["output_format"], "png")
            self.assertEqual(run["config"]["quality"], "low")

    def test_cli_generate_review_does_not_run_for_dry_run_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            out = root / "generated"
            config = SimpleNamespace(provider="dry-run")
            result = SimpleNamespace(
                status="planned",
                evaluation_command="python -m seju_face_lab evaluate",
                generated_images=[],
            )

            with patch("seju_face_lab.cli.build_generation_config", return_value=config):
                with patch("seju_face_lab.cli.write_generation_plan", return_value=result):
                    with patch("seju_face_lab.cli._review_generated") as review_generated:
                        self.assertEqual(
                            main(
                                [
                                    "generate",
                                    "--model",
                                    str(model),
                                    "--out",
                                    str(out),
                                    "--provider",
                                    "dry-run",
                                    "--review",
                                ]
                            ),
                            0,
                        )

            review_generated.assert_not_called()

    def test_precision_report_combines_model_generation_and_subject_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            generation = root / "generation_review"
            evaluation = root / "evaluation"
            subjects = root / "subject_review"
            backend_comparison = root / "backend_compare"
            subject_backend_comparison = root / "subject_backend_compare"
            correlation = root / "correlation"
            model_audit = root / "model_audit"
            vector_export = root / "vectors.json"
            out = root / "precision"
            model.mkdir()
            generation.mkdir()
            evaluation.mkdir()
            subjects.mkdir()
            backend_comparison.mkdir()
            subject_backend_comparison.mkdir()
            correlation.mkdir()
            model_audit.mkdir()
            np.savez_compressed(
                model / "centroids.npz",
                mean_embedding=np.asarray([0.6, 0.8], dtype=np.float32),
                median_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                mean_appearance=np.zeros((2, 2, 1), dtype=np.float32),
                median_appearance=np.ones((2, 2, 1), dtype=np.float32),
            )
            (model / "profile.json").write_text(
                json.dumps(
                    {
                        "image_count": 3,
                        "embedding_dim": 2,
                        "appearance_shape": [2, 2, 1],
                        "descriptors": {
                            "mean": {"brightness": 0.5},
                            "median": {"brightness": 0.6},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (generation / "generation_run_reviews.json").write_text(
                json.dumps(
                    {
                        "run_count": 1,
                        "best_run_dir": "outputs/generated",
                        "best_generation": {
                            "provider": "diffusers",
                            "model_id": "local/test-model",
                            "status": "generated",
                            "centroid_kind": "mean",
                            "prompt_profile": "detector-friendly",
                            "seed": 260623,
                            "planned_count": 2,
                            "steps": 20,
                            "guidance_scale": 7.0,
                            "size": "512x512",
                            "device": "cuda",
                            "dtype": "float16",
                            "prompt_words": 42,
                        },
                        "best_centroid_score": 0.4,
                        "best_qa_centroid_score": 0.45,
                        "best_qa_image_id": "candidate",
                        "best_qa_path": "candidate.png",
                        "runs": [
                            {
                                "image_count": 2,
                                "failed_count": 0,
                                "qa_pass_count": 1,
                                "qa_fail_count": 1,
                                "qa_pass_rate": 0.5,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (evaluation / "summary.json").write_text(
                json.dumps(
                    {
                        "best_centroid_score": 0.42,
                        "best_image_id": "candidate",
                        "top_images": [
                            {
                                "image_id": "distractor",
                                "path": "distractor.png",
                                "centroid_score": 0.5,
                                "cosine_to_mean": 0.99,
                                "cosine_to_median": 0.98,
                                "euclidean_to_mean": 0.1,
                                "euclidean_to_median": 0.2,
                            },
                            {
                                "image_id": "candidate",
                                "path": "candidate.png",
                                "centroid_score": 0.42,
                                "cosine_to_mean": 0.41,
                                "cosine_to_median": 0.43,
                                "euclidean_to_mean": 0.9,
                                "euclidean_to_median": 0.8,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (subjects / "subject_reviews.json").write_text(
                json.dumps(
                    {
                        "subject_count": 1,
                        "subjects": [
                            {
                                "subject": "near_subject",
                                "mean_centroid_score": 0.7,
                                "best_centroid_score": 0.8,
                                "best_image_path": "near.png",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (backend_comparison / "backend_comparison.json").write_text(
                json.dumps(
                    {
                        "runs": [
                            {"backend": "deterministic", "status": "completed"},
                            {"backend": "deepface", "status": "failed"},
                        ],
                        "rank_agreement": [
                            {
                                "backend_a": "deterministic",
                                "backend_b": "deepface",
                                "common_image_count": 2,
                                "spearman_rank": 0.5,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (subject_backend_comparison / "subject_backend_comparison.json").write_text(
                json.dumps(
                    {
                        "runs": [
                            {"backend": "deterministic", "status": "completed"},
                            {"backend": "opencv-face", "status": "completed"},
                        ],
                        "rank_agreement": [
                            {
                                "backend_a": "deterministic",
                                "backend_b": "opencv-face",
                                "common_subject_count": 1,
                                "spearman_rank": None,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (correlation / "correlation_summary.json").write_text(
                json.dumps(
                    {
                        "talent_count": 3,
                        "with_face_score": 3,
                        "with_ig": 2,
                        "with_twitter": 1,
                        "with_tiktok": 1,
                        "correlations": [
                            {
                                "a": "face_mean_centroid_score",
                                "b": "ig_followers",
                                "n": 3,
                                "spearman_r": 0.75,
                                "spearman_p": 0.1,
                                "pearson_r": 0.7,
                                "pearson_p": 0.12,
                                "interpretation": "strong_positive",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (model_audit / "model_audit.json").write_text(
                json.dumps(
                    {
                        "model_dir": str(model),
                        "image_count": 3,
                        "embedding_dim": 2,
                        "appearance_shape": [2, 2, 1],
                        "centroids": {
                            "mean_median_embedding": {
                                "available": True,
                                "cosine": 0.6,
                                "euclidean": 0.9,
                                "mean_abs_delta": 0.2,
                                "max_abs_delta": 0.4,
                            },
                            "mean_median_appearance": {
                                "available": True,
                                "cosine": 0.7,
                                "euclidean": 1.1,
                                "mean_abs_delta": 0.3,
                                "max_abs_delta": 0.5,
                            },
                        },
                        "descriptor_delta": {"brightness": 0.1},
                    }
                ),
                encoding="utf-8",
            )
            (vector_export).write_text(
                json.dumps(
                    {
                        "model_dir": str(model),
                        "image_count": 3,
                        "embedding_dim": 2,
                        "include_appearance": False,
                        "vectors": {
                            "mean_embedding": {
                                "shape": [2],
                                "dtype": "float32",
                                "l2_norm": 1.0,
                                "sha256": "a" * 64,
                                "values": [0.6, 0.8],
                            },
                            "median_embedding": {
                                "shape": [2],
                                "dtype": "float32",
                                "l2_norm": 1.0,
                                "sha256": "b" * 64,
                                "values": [1.0, 0.0],
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = write_precision_report(
                model_dir=model,
                out_dir=out,
                generation_review=generation,
                evaluation=evaluation,
                subject_review=subjects,
                backend_comparison=backend_comparison,
                subject_backend_comparison=subject_backend_comparison,
                correlation=correlation,
                model_audit=model_audit,
                vector_export=vector_export,
            )

            self.assertEqual(report["model"]["image_count"], 3)
            self.assertEqual(report["model"]["centroid_vectors"]["mean_embedding"]["shape"], [2])
            self.assertEqual(report["model"]["centroid_vectors"]["mean_embedding"]["l2_norm"], 1.0)
            self.assertEqual(len(report["model"]["centroid_vectors"]["mean_embedding"]["sha256"]), 64)
            self.assertTrue(report["model"]["model_audit"]["available"])
            self.assertTrue(report["model"]["vector_export"]["available"])
            self.assertEqual(report["model"]["vector_export"]["vectors"]["mean_embedding"]["sha256"], "a" * 64)
            self.assertEqual(report["model"]["vector_export"]["vectors"]["mean_embedding"]["values_count"], 2)
            self.assertEqual(report["model"]["model_audit"]["mean_median_embedding"]["cosine"], 0.6)
            self.assertEqual(report["model"]["model_audit"]["mean_median_appearance"]["euclidean"], 1.1)
            self.assertEqual(report["model"]["model_audit"]["descriptor_delta"]["brightness"], 0.1)
            self.assertEqual(report["workflow_readiness"]["ready_count"], 8)
            self.assertEqual(report["workflow_readiness"]["total_count"], 8)
            self.assertEqual(report["workflow_readiness"]["optional_ready_count"], 3)
            self.assertEqual(report["workflow_readiness"]["optional_total_count"], 3)
            self.assertEqual(report["workflow_readiness"]["missing"], [])
            self.assertEqual(report["workflow_readiness"]["optional_missing"], [])
            self.assertIsNone(report["workflow_readiness"]["next_action"])
            self.assertEqual(report["generation"]["provider"], "diffusers")
            self.assertEqual(report["generation"]["model_id"], "local/test-model")
            self.assertEqual(report["generation"]["centroid_kind"], "mean")
            self.assertEqual(report["generation"]["prompt_profile"], "detector-friendly")
            self.assertEqual(report["generation"]["seed"], 260623)
            self.assertEqual(report["generation"]["planned_count"], 2)
            self.assertEqual(report["generation"]["steps"], 20)
            self.assertEqual(report["generation"]["guidance_scale"], 7.0)
            self.assertEqual(report["generation"]["size"], "512x512")
            self.assertEqual(report["generation"]["device"], "cuda")
            self.assertEqual(report["generation"]["dtype"], "float16")
            self.assertEqual(report["generation"]["prompt_words"], 42)
            self.assertEqual(report["generation"]["best_centroid_score"], 0.45)
            self.assertEqual(report["generation"]["best_cosine_to_mean"], 0.41)
            self.assertEqual(report["generation"]["best_cosine_to_median"], 0.43)
            self.assertEqual(report["generation"]["best_euclidean_to_mean"], 0.9)
            self.assertEqual(report["generation"]["best_euclidean_to_median"], 0.8)
            self.assertEqual(report["generation"]["qa_pass_count"], 1)
            self.assertEqual(report["subjects"]["top_subject"], "near_subject")
            self.assertEqual(report["backend_comparison"]["completed_backends"], ["deterministic"])
            self.assertEqual(report["backend_comparison"]["failed_backends"], ["deepface"])
            self.assertEqual(report["backend_comparison"]["rank_agreement"][0]["spearman_rank"], 0.5)
            self.assertEqual(
                report["subject_backend_comparison"]["completed_backends"],
                ["deterministic", "opencv-face"],
            )
            self.assertEqual(
                report["subject_backend_comparison"]["rank_agreement"][0]["common_subject_count"],
                1,
            )
            self.assertTrue(report["correlation"]["available"])
            self.assertEqual(report["correlation"]["talent_count"], 3)
            self.assertEqual(report["correlation"]["top_pair"]["b"], "ig_followers")
            self.assertEqual(report["correlation"]["top_pair"]["spearman_r"], 0.75)
            self.assertTrue((out / "precision_report.json").exists())
            self.assertIn(
                "Backend Comparison",
                (out / "precision_report.md").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "Workflow Readiness",
                (out / "precision_report.md").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "ready: 8/8",
                (out / "precision_report.md").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "Subject Backend Comparison",
                (out / "precision_report.md").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "vector_export_mean_sha256: " + "a" * 64,
                (out / "precision_report.md").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "mean_embedding_sha256",
                (out / "precision_report.md").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "best_cosine_to_mean: 0.41",
                (out / "precision_report.md").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "prompt_profile: detector-friendly",
                (out / "precision_report.md").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "centroid_kind: mean",
                (out / "precision_report.md").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "mean_median_embedding_cosine: 0.6",
                (out / "precision_report.md").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "Correlation Review",
                (out / "precision_report.md").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "top_pair: face_mean_centroid_score x ig_followers rho=0.75 n=3 strong_positive",
                (out / "precision_report.md").read_text(encoding="utf-8"),
            )

    def test_precision_report_reads_csv_vector_export_from_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            out = root / "precision"
            model.mkdir()
            (model / "vectors.csv").write_text(
                "\n".join(
                    [
                        "vector,index,value,shape,dtype,l2_norm,sha256",
                        f"mean_embedding,0,0.6,2,float32,1.0,{'a' * 64}",
                        f"mean_embedding,1,0.8,2,float32,1.0,{'a' * 64}",
                        f"median_embedding,0,1.0,2,float32,1.0,{'b' * 64}",
                        f"median_embedding,1,0.0,2,float32,1.0,{'b' * 64}",
                    ]
                ),
                encoding="utf-8-sig",
            )

            report = write_precision_report(model_dir=model, out_dir=out, vector_export=model)

            self.assertTrue(report["model"]["vector_export"]["available"])
            self.assertEqual(report["model"]["vector_export"]["vectors"]["mean_embedding"]["sha256"], "a" * 64)
            self.assertEqual(report["model"]["vector_export"]["vectors"]["mean_embedding"]["values_count"], 2)
            self.assertIn(
                "vector_export_mean_sha256: " + "a" * 64,
                (out / "precision_report.md").read_text(encoding="utf-8"),
            )

    def test_precision_report_ignores_nan_correlation_top_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            correlation = root / "correlation"
            out = root / "precision"
            model.mkdir()
            correlation.mkdir()
            (correlation / "correlation_summary.json").write_text(
                json.dumps(
                    {
                        "talent_count": 4,
                        "with_face_score": 4,
                        "with_ig": 4,
                        "with_twitter": 4,
                        "with_tiktok": 0,
                        "correlations": [
                            {
                                "a": "face_mean_centroid_score",
                                "b": "ig_followers",
                                "n": 4,
                                "spearman_r": float("nan"),
                                "interpretation": "negligible_positive",
                            },
                            {
                                "a": "face_mean_centroid_score",
                                "b": "tw_followers",
                                "n": 4,
                                "spearman_r": 0.5,
                                "interpretation": "moderate_positive",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = write_precision_report(model_dir=model, out_dir=out, correlation=correlation)

            self.assertEqual(report["correlation"]["top_pair"]["b"], "tw_followers")
            self.assertEqual(report["correlation"]["top_pair"]["spearman_r"], 0.5)
            report_text = (out / "precision_report.json").read_text(encoding="utf-8")
            self.assertNotIn("NaN", report_text)
            persisted = json.loads(report_text)
            self.assertIsNone(persisted["correlation"]["correlations"][0]["spearman_r"])

    def test_model_audit_reports_mean_median_vector_distance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            out = root / "audit"
            model.mkdir()
            np.savez(
                model / "centroids.npz",
                mean_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                median_embedding=np.asarray([0.0, 1.0], dtype=np.float32),
                mean_appearance=np.asarray([[[0.2, 0.4, 0.6]]], dtype=np.float32),
                median_appearance=np.asarray([[[0.2, 0.4, 0.2]]], dtype=np.float32),
            )
            (model / "profile.json").write_text(
                json.dumps(
                    {
                        "image_count": 2,
                        "embedding_dim": 2,
                        "appearance_shape": [1, 1, 3],
                        "descriptors": {
                            "mean": {"brightness": 0.7, "contrast": 0.3},
                            "median": {"brightness": 0.5, "contrast": 0.4},
                        },
                    }
                ),
                encoding="utf-8",
            )

            audit = write_model_audit(model, out)

            self.assertEqual(audit["image_count"], 2)
            self.assertEqual(audit["centroids"]["mean_median_embedding"]["cosine"], 0.0)
            self.assertEqual(audit["centroids"]["mean_median_embedding"]["euclidean"], 1.414214)
            self.assertEqual(audit["descriptor_delta"]["brightness"], 0.2)
            self.assertTrue((out / "model_audit.json").exists())
            self.assertIn(
                "mean_median_embedding_cosine: 0.0",
                (out / "model_audit.md").read_text(encoding="utf-8"),
            )

    def test_cli_audit_model_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            out = root / "audit"
            model.mkdir()
            np.savez(
                model / "centroids.npz",
                mean_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                median_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                mean_appearance=np.asarray([[[0.1, 0.1, 0.1]]], dtype=np.float32),
                median_appearance=np.asarray([[[0.1, 0.1, 0.1]]], dtype=np.float32),
            )

            self.assertEqual(main(["audit-model", "--model", str(model), "--out", str(out)]), 0)

            audit = json.loads((out / "model_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["centroids"]["mean_median_embedding"]["cosine"], 1.0)

    def test_export_vectors_writes_mean_and_median_embedding_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            json_out = root / "vectors.json"
            appearance_json_out = root / "vectors_with_appearance.json"
            csv_out = root / "vectors.csv"
            model.mkdir()
            np.savez_compressed(
                model / "centroids.npz",
                mean_embedding=np.asarray([0.6, 0.8], dtype=np.float32),
                median_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                mean_appearance=np.ones((2, 2, 3), dtype=np.float32),
                median_appearance=np.zeros((2, 2, 3), dtype=np.float32),
                image_ids=np.asarray(["a", "b"]),
                source_paths=np.asarray(["a.png", "b.png"]),
            )
            (model / "profile.json").write_text(
                json.dumps(
                    {
                        "image_count": 2,
                        "embedding_dim": 2,
                        "appearance_shape": [2, 2, 3],
                        "descriptors": {"mean": {}, "median": {}},
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                main(["export-vectors", "--model", str(model), "--out", str(json_out)]),
                0,
            )
            payload = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["vectors"]["mean_embedding"]["values"],
                [0.6000000238418579, 0.800000011920929],
            )
            self.assertEqual(payload["vectors"]["median_embedding"]["values"], [1.0, 0.0])
            self.assertEqual(payload["vectors"]["mean_embedding"]["shape"], [2])
            self.assertEqual(len(payload["vectors"]["mean_embedding"]["sha256"]), 64)
            self.assertNotIn("mean_appearance", payload["vectors"])

            self.assertEqual(
                main(
                    [
                        "export-vectors",
                        "--model",
                        str(model),
                        "--out",
                        str(appearance_json_out),
                        "--include-appearance",
                    ]
                ),
                0,
            )
            appearance_payload = json.loads(appearance_json_out.read_text(encoding="utf-8"))
            self.assertEqual(appearance_payload["vectors"]["mean_appearance"]["shape"], [2, 2, 3])
            self.assertEqual(len(appearance_payload["vectors"]["mean_appearance"]["values"]), 12)

            self.assertEqual(
                main(["export-vectors", "--model", str(model), "--out", str(csv_out), "--format", "csv"]),
                0,
            )
            with csv_out.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["vector"], "mean_embedding")
            self.assertEqual(rows[0]["index"], "0")
            self.assertEqual(rows[2]["vector"], "median_embedding")

    def test_precision_report_keeps_centroid_only_best_image_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            generation = root / "generation_review"
            out = root / "precision"
            model.mkdir()
            generation.mkdir()
            (model / "profile.json").write_text(
                json.dumps(
                    {
                        "image_count": 2,
                        "embedding_dim": 8,
                        "appearance_shape": [2, 2, 1],
                        "descriptors": {"mean": {}, "median": {}},
                    }
                ),
                encoding="utf-8",
            )
            (generation / "generation_run_reviews.json").write_text(
                json.dumps(
                    {
                        "run_count": 1,
                        "best_run_dir": "outputs/generated_centroid_only",
                        "best_centroid_score": 0.33,
                        "runs": [
                            {
                                "image_count": 1,
                                "failed_count": 0,
                                "best_image_id": "candidate_centroid",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = write_precision_report(
                model_dir=model,
                out_dir=out,
                generation_review=generation,
            )

            self.assertEqual(report["generation"]["best_centroid_score"], 0.33)
            self.assertEqual(report["generation"]["best_image_id"], "candidate_centroid")

    def test_precision_report_handles_malformed_centroid_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            out = root / "precision"
            model.mkdir()
            (model / "centroids.npz").write_bytes(b"PK\x03\x04broken")
            (model / "profile.json").write_text(
                json.dumps(
                    {
                        "image_count": 1,
                        "embedding_dim": 2,
                        "appearance_shape": [1, 1, 1],
                        "descriptors": {"mean": {}, "median": {}},
                    }
                ),
                encoding="utf-8",
            )

            report = write_precision_report(model_dir=model, out_dir=out)

        self.assertTrue(report["model"]["has_centroid_vectors"])
        self.assertFalse(report["model"]["centroid_vectors"]["available"])
        self.assertEqual(report["model"]["centroid_vectors"]["error"], "unreadable centroids.npz")
        self.assertIn("centroid_vectors", report["workflow_readiness"]["missing"])

    def test_precision_report_does_not_mix_components_from_nonmatching_best_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            generation = root / "generation_review"
            evaluation = root / "evaluation"
            out = root / "precision"
            model.mkdir()
            generation.mkdir()
            evaluation.mkdir()
            (model / "profile.json").write_text(
                json.dumps(
                    {
                        "image_count": 1,
                        "embedding_dim": 2,
                        "appearance_shape": [1, 1, 1],
                        "descriptors": {"mean": {}, "median": {}},
                    }
                ),
                encoding="utf-8",
            )
            (generation / "generation_run_reviews.json").write_text(
                json.dumps(
                    {
                        "run_count": 1,
                        "best_qa_image_id": "qa_winner_not_in_top_images",
                        "best_qa_centroid_score": 0.4,
                    }
                ),
                encoding="utf-8",
            )
            (evaluation / "summary.json").write_text(
                json.dumps(
                    {
                        "top_images": [
                            {
                                "image_id": "raw_top",
                                "cosine_to_mean": 0.99,
                                "cosine_to_median": 0.98,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = write_precision_report(
                model_dir=model,
                out_dir=out,
                generation_review=generation,
                evaluation=evaluation,
            )

        self.assertEqual(report["generation"]["best_image_id"], "qa_winner_not_in_top_images")
        self.assertIsNone(report["generation"]["best_cosine_to_mean"])
        self.assertIsNone(report["generation"]["best_cosine_to_median"])

    def test_precision_report_reads_components_from_scores_csv_for_qa_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            generation = root / "generation_review"
            evaluation = root / "evaluation"
            out = root / "precision"
            model.mkdir()
            generation.mkdir()
            evaluation.mkdir()
            (model / "profile.json").write_text(
                json.dumps(
                    {
                        "image_count": 1,
                        "embedding_dim": 2,
                        "appearance_shape": [1, 1, 1],
                        "descriptors": {"mean": {}, "median": {}},
                    }
                ),
                encoding="utf-8",
            )
            (generation / "generation_run_reviews.json").write_text(
                json.dumps(
                    {
                        "run_count": 1,
                        "best_qa_image_id": "qa_winner_below_top_five",
                        "best_qa_centroid_score": 0.44,
                    }
                ),
                encoding="utf-8",
            )
            (evaluation / "summary.json").write_text(
                json.dumps(
                    {
                        "top_images": [
                            {
                                "image_id": "raw_top",
                                "cosine_to_mean": 0.99,
                                "cosine_to_median": 0.98,
                                "euclidean_to_mean": 0.1,
                                "euclidean_to_median": 0.2,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (evaluation / "scores.csv").write_text(
                "\n".join(
                    [
                        "image_id,path,cosine_to_mean,cosine_to_median,euclidean_to_mean,"
                        "euclidean_to_median,centroid_score",
                        "raw_top,raw.png,0.990000,0.980000,0.100000,0.200000,0.985000",
                        "qa_winner_below_top_five,qa.png,0.410000,0.430000,0.900000,0.800000,0.420000",
                    ]
                )
                + "\n",
                encoding="utf-8-sig",
            )

            report = write_precision_report(
                model_dir=model,
                out_dir=out,
                generation_review=generation,
                evaluation=evaluation,
            )

        self.assertEqual(report["generation"]["best_image_id"], "qa_winner_below_top_five")
        self.assertEqual(report["generation"]["best_image_path"], "qa.png")
        self.assertEqual(report["generation"]["best_cosine_to_mean"], 0.41)
        self.assertEqual(report["generation"]["best_cosine_to_median"], 0.43)
        self.assertEqual(report["generation"]["best_euclidean_to_mean"], 0.9)
        self.assertEqual(report["generation"]["best_euclidean_to_median"], 0.8)

    def test_precision_report_accepts_direct_scores_csv_evaluation_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            generation = root / "generation_review"
            evaluation = root / "evaluation"
            out = root / "precision"
            model.mkdir()
            generation.mkdir()
            evaluation.mkdir()
            (model / "profile.json").write_text(
                json.dumps(
                    {
                        "image_count": 1,
                        "embedding_dim": 2,
                        "appearance_shape": [1, 1, 1],
                        "descriptors": {"mean": {}, "median": {}},
                    }
                ),
                encoding="utf-8",
            )
            (generation / "generation_run_reviews.json").write_text(
                json.dumps(
                    {
                        "run_count": 1,
                        "best_qa_image_id": "candidate",
                        "best_qa_centroid_score": 0.44,
                    }
                ),
                encoding="utf-8",
            )
            (evaluation / "summary.json").write_text(
                json.dumps({"image_count": 1, "failed_count": 0, "top_images": []}),
                encoding="utf-8",
            )
            (evaluation / "scores.csv").write_text(
                "\n".join(
                    [
                        "image_id,path,cosine_to_mean,cosine_to_median,euclidean_to_mean,"
                        "euclidean_to_median,centroid_score",
                        "candidate,candidate.png,0.410000,0.430000,0.900000,0.800000,0.420000",
                    ]
                )
                + "\n",
                encoding="utf-8-sig",
            )

            report = write_precision_report(
                model_dir=model,
                out_dir=out,
                generation_review=generation,
                evaluation=evaluation / "scores.csv",
            )

        self.assertEqual(report["generation"]["evaluated_image_count"], 1)
        self.assertEqual(report["generation"]["failed_image_count"], 0)
        self.assertEqual(report["generation"]["best_image_path"], "candidate.png")
        self.assertEqual(report["generation"]["best_cosine_to_mean"], 0.41)

    def test_precision_report_keeps_combined_winner_separate_from_centroid_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            generation = root / "generation_review"
            out = root / "precision"
            model.mkdir()
            generation.mkdir()
            (model / "profile.json").write_text(
                json.dumps(
                    {
                        "image_count": 2,
                        "embedding_dim": 8,
                        "appearance_shape": [2, 2, 1],
                        "descriptors": {"mean": {}, "median": {}},
                    }
                ),
                encoding="utf-8",
            )
            (generation / "generation_run_reviews.json").write_text(
                json.dumps(
                    {
                        "run_count": 1,
                        "best_run_dir": "outputs/generated_style",
                        "best_centroid_score": 0.5,
                        "best_combined_image_id": "candidate_combined",
                        "best_combined_path": "combined.png",
                        "best_combined_score": 0.6,
                        "runs": [
                            {
                                "image_count": 2,
                                "failed_count": 0,
                                "best_image_id": "candidate_centroid",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = write_precision_report(
                model_dir=model,
                out_dir=out,
                generation_review=generation,
            )

            self.assertEqual(report["generation"]["best_image_id"], "candidate_centroid")
            self.assertEqual(
                report["generation"]["best_combined_image_id"],
                "candidate_combined",
            )
            self.assertEqual(report["generation"]["best_combined_score"], 0.6)

    def test_precision_report_uses_quality_reviewed_count_for_qa_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            quality = root / "quality"
            out = root / "precision"
            model.mkdir()
            quality.mkdir()
            (model / "profile.json").write_text(
                json.dumps(
                    {
                        "image_count": 2,
                        "embedding_dim": 8,
                        "appearance_shape": [2, 2, 1],
                        "descriptors": {"mean": {}, "median": {}},
                    }
                ),
                encoding="utf-8",
            )
            (quality / "image_quality.json").write_text(
                json.dumps(
                    {
                        "image_count": 2,
                        "pass_count": 1,
                        "fail_count": 1,
                        "pass_rate": 0.5,
                    }
                ),
                encoding="utf-8",
            )

            report = write_precision_report(model_dir=model, out_dir=out, quality=quality)

            self.assertEqual(report["generation"]["qa_pass_count"], 1)
            self.assertEqual(report["generation"]["qa_fail_count"], 1)
            self.assertEqual(report["generation"]["qa_reviewed_count"], 2)
            self.assertIn("qa_pass: 1/2", (out / "precision_report.md").read_text(encoding="utf-8"))

    def test_cli_precision_report_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            audit = root / "audit"
            out = root / "precision"
            model.mkdir()
            audit.mkdir()
            (model / "profile.json").write_text(
                json.dumps(
                    {
                        "image_count": 1,
                        "embedding_dim": 4,
                        "appearance_shape": [2, 2, 1],
                        "descriptors": {"mean": {}, "median": {}},
                    }
                ),
                encoding="utf-8",
            )
            (audit / "model_audit.json").write_text(
                json.dumps(
                    {
                        "centroids": {
                            "mean_median_embedding": {
                                "available": True,
                                "cosine": 0.99,
                                "euclidean": 0.01,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                main(
                    [
                        "precision-report",
                        "--model",
                        str(model),
                        "--out",
                        str(out),
                        "--model-audit",
                        str(audit),
                    ]
                ),
                0,
            )

            report = json.loads((out / "precision_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["model"]["embedding_dim"], 4)
            self.assertEqual(report["model"]["model_audit"]["mean_median_embedding"]["cosine"], 0.99)

    def test_image_quality_records_per_file_failures(self) -> None:
        with patch("seju_face_lab.quality._import_cv2", return_value=object()):
            with patch(
                "seju_face_lab.quality.iter_image_paths",
                return_value=[Path("bad.png"), Path("good.png")],
            ):
                with patch(
                    "seju_face_lab.quality._review_one_image",
                    side_effect=[
                        ValueError("broken image"),
                        ImageQuality("good", "good.png", 1, 0.2, 0.1, True, "single centered face"),
                    ],
                ):
                    reviews = review_image_quality(Path("images"))

        self.assertEqual(len(reviews), 2)
        self.assertFalse(reviews[0].qa_pass)
        self.assertIn("broken image", reviews[0].reason)
        self.assertTrue(reviews[1].qa_pass)

    def test_extract_sns_handles_ignores_navigation_paths(self) -> None:
        handles = extract_sns_handles_from_links(
            [
                ("https://www.instagram.com/p/abc123/", None),
                ("https://www.instagram.com/seju_account/", None),
                ("https://x.com/search?q=seju", None),
                ("https://x.com/seju_talent", None),
                ("https://www.tiktok.com/@seju.talent", None),
            ]
        )

        self.assertEqual(
            handles,
            {
                "instagram": "seju_account",
                "twitter": "seju_talent",
                "tiktok": "seju.talent",
            },
        )

    def test_sns_manifest_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            handles_path = root / "handles.jsonl"
            engagement_path = root / "engagement.jsonl"

            write_handles_manifest(
                [
                    SnsHandleRecord(
                        talent_slug="talent_a",
                        name="Talent A",
                        profile_url="https://seju.tokyo/talents/a/",
                        sns_handles={"instagram": "talent_a"},
                        retrieved_at="2026-06-15T00:00:00+00:00",
                    )
                ],
                handles_path,
            )
            self.assertEqual(read_handles_manifest(handles_path)[0].sns_handles["instagram"], "talent_a")

            write_engagement_manifest(
                [
                    TalentEngagementRecord(
                        talent_slug="talent_a",
                        name="Talent A",
                        engagements=[
                            _engagement(
                                platform="instagram",
                                handle="talent_a",
                                followers=1000,
                                total_engagement=120,
                            )
                        ],
                    )
                ],
                engagement_path,
            )
            loaded = read_engagement_manifest(engagement_path)
            self.assertEqual(loaded[0].engagements[0].followers, 1000)

    def test_sns_explorer_store_roundtrip_and_default_local_router(self) -> None:
        from seju_face_lab.sns_explorer import SnsProfile, SnsStore, build_router

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "sns_cache.db"
            store = SnsStore(db_path)
            store.put(
                SnsProfile(
                    platform="instagram",
                    handle="talent_a",
                    profile_url="https://www.instagram.com/talent_a/",
                    followers=1200,
                    source="test",
                )
            )

            cached = store.get("instagram", "TALENT_A")
            self.assertIsNotNone(cached)
            self.assertEqual(cached.followers, 1200)
            store.close()

            router = build_router(cache_path=None)
            self.assertIsNone(router._remote_ig)

    def test_remote_instagram_fetcher_uses_stdin_script_once(self) -> None:
        from seju_face_lab.sns_explorer import RemoteInstagramFetcher

        result = subprocess.CompletedProcess(
            args=["ssh"],
            returncode=0,
            stdout='noise\n{"talent_a": {"followers": 1234, "status": "ok"}}\n',
            stderr="",
        )
        with patch("seju_face_lab.sns_explorer.subprocess.run", return_value=result) as run:
            fetched = RemoteInstagramFetcher("home-mac-main").fetch_batch(["talent_a"])

        self.assertEqual(fetched["talent_a"]["followers"], 1234)
        self.assertIn("input", run.call_args.kwargs)
        self.assertNotIn("stdin", run.call_args.kwargs)

    def test_sns_router_batch_skips_remote_fetch_for_cached_instagram(self) -> None:
        from seju_face_lab.sns_explorer import SnsProfile, SnsRouter, SnsStore

        with tempfile.TemporaryDirectory() as tmp:
            store = SnsStore(Path(tmp) / "sns_cache.db")
            store.put(
                SnsProfile(
                    platform="instagram",
                    handle="cached",
                    profile_url="https://www.instagram.com/cached/",
                    followers=1000,
                    source="cached",
                )
            )
            router = SnsRouter(store=store, remote_host="home-mac-main")
            router._remote_available = True
            assert router._remote_ig is not None
            with patch.object(
                router._remote_ig,
                "fetch_batch",
                return_value={"fresh": {"followers": 2000, "status": "ok"}},
            ) as fetch_batch:
                profiles = router.fetch_batch([("instagram", "cached"), ("instagram", "fresh")], delay_between=0)

            store.close()
            fetch_batch.assert_called_once_with(["fresh"])
            self.assertEqual([profile.handle for profile in profiles], ["cached", "fresh"])
            self.assertEqual([profile.followers for profile in profiles], [1000, 2000])

    def test_sns_router_batch_falls_back_when_remote_instagram_has_no_followers(self) -> None:
        from seju_face_lab.sns_explorer import SnsProfile, SnsRouter

        router = SnsRouter(remote_host="home-mac-main")
        router._remote_available = True
        assert router._remote_ig is not None
        with patch.object(
            router._remote_ig,
            "fetch_batch",
            return_value={"talent_a": {"followers": None, "status": "ssh_error"}},
        ):
            with patch(
                "seju_face_lab.sns_explorer._fetch_instagram_local",
                return_value=SnsProfile(
                    platform="instagram",
                    handle="talent_a",
                    profile_url="https://www.instagram.com/talent_a/",
                    followers=1500,
                    source="local_ig",
                ),
            ) as local_fetch:
                profiles = router.fetch_batch([("instagram", "talent_a")], delay_between=0)

        local_fetch.assert_called_once_with("talent_a")
        self.assertEqual(profiles[0].followers, 1500)
        self.assertEqual(profiles[0].source, "local_ig")

    def test_cli_explore_batch_writes_engagement_manifest(self) -> None:
        from seju_face_lab.sns_explorer import SnsProfile

        class FakeRouter:
            def __init__(self) -> None:
                self.force: bool | None = None

            def fetch_batch(self, items: list[tuple[str, str]], delay_between: float, force: bool) -> list[SnsProfile]:
                self.force = force
                self.delay_between = delay_between
                return [
                    SnsProfile(
                        platform=platform,
                        handle=handle,
                        profile_url=f"https://example.test/{handle}",
                        followers=1000,
                        total_engagement=120,
                        engagement_rate=0.12,
                        source="fake",
                    )
                    for platform, handle in items
                ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            handles = root / "handles.jsonl"
            out = root / "engagement.jsonl"
            fake_router = FakeRouter()
            handles.write_text(
                json.dumps(
                    {
                        "talent_slug": "talent_a",
                        "name": "Talent A",
                        "sns_handles": {"instagram": "talent_a", "twitter": "talent_x"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("seju_face_lab.sns_explorer.build_router", return_value=fake_router):
                self.assertEqual(
                    main(
                        [
                            "explore",
                            "batch",
                            "--handles",
                            str(handles),
                            "--out",
                            str(out),
                            "--platforms",
                            "instagram",
                            "--force",
                            "--delay",
                            "0",
                        ]
                    ),
                    0,
                )

            loaded = read_engagement_manifest(out)
            self.assertTrue(fake_router.force)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].talent_slug, "talent_a")
            self.assertEqual(loaded[0].engagements[0].platform, "instagram")
            self.assertEqual(loaded[0].engagements[0].followers, 1000)

    def test_cli_explore_load_cache_reads_handle_keyed_ig_json(self) -> None:
        from seju_face_lab.sns_explorer import SnsStore

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ig_json = root / "ig_results.json"
            cache = root / "sns_cache.db"
            ig_json.write_text(
                json.dumps({"talent_a": {"followers": 1234, "status": "ok"}}),
                encoding="utf-8",
            )

            self.assertEqual(
                main(
                    [
                        "explore",
                        "load-cache",
                        "--ig-json",
                        str(ig_json),
                        "--cache",
                        str(cache),
                    ]
                ),
                0,
            )

            store = SnsStore(cache)
            cached = store.get("instagram", "talent_a")
            store.close()
            self.assertIsNotNone(cached)
            self.assertEqual(cached.followers, 1234)

    def test_cli_explore_load_cache_maps_slug_keyed_ig_json_through_handles(self) -> None:
        from seju_face_lab.sns_explorer import SnsStore

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            handles = root / "handles.jsonl"
            ig_json = root / "ig_results.json"
            cache = root / "sns_cache.db"
            handles.write_text(
                json.dumps(
                    {
                        "talent_slug": "talent_a",
                        "name": "Talent A",
                        "sns_handles": {"instagram": "real_handle"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            ig_json.write_text(
                json.dumps({"talent_a": {"followers": 2468, "status": "ok"}}),
                encoding="utf-8",
            )

            self.assertEqual(
                main(
                    [
                        "explore",
                        "load-cache",
                        "--ig-json",
                        str(ig_json),
                        "--handles",
                        str(handles),
                        "--cache",
                        str(cache),
                    ]
                ),
                0,
            )

            store = SnsStore(cache)
            cached = store.get("instagram", "real_handle")
            missed_slug = store.get("instagram", "talent_a")
            store.close()
            self.assertIsNotNone(cached)
            self.assertEqual(cached.followers, 2468)
            self.assertIsNone(missed_slug)

    def test_import_engagement_csv_merges_manual_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "manual.csv"
            out_path = root / "engagement.jsonl"
            csv_path.write_text(
                "\ufefftalent_slug,platform,handle,followers,following,posts,total_engagement,engagement_rate,display_name,bio\n"
                "talent_a,instagram,@talent_a,\"1,200\",100,20,240,,Talent A,\n",
                encoding="utf-8",
            )

            records = import_engagement_csv(csv_path, out_path)

            self.assertEqual(records[0].talent_slug, "talent_a")
            self.assertEqual(records[0].engagements[0].followers, 1200)
            self.assertEqual(records[0].engagements[0].engagement_rate, 0.01)
            self.assertTrue(out_path.exists())

    def test_cli_import_engagement_merges_existing_output_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "manual.csv"
            out_path = root / "engagement.jsonl"
            write_engagement_manifest(
                [
                    TalentEngagementRecord(
                        talent_slug="talent_a",
                        name="Talent A",
                        engagements=[_engagement("twitter", "talent_a", 500, 20)],
                    )
                ],
                out_path,
            )
            csv_path.write_text(
                "talent_slug,platform,handle,followers\n"
                "talent_a,instagram,talent_a,1200\n",
                encoding="utf-8",
            )

            self.assertEqual(
                main(["sources", "import-engagement", "--csv", str(csv_path), "--out", str(out_path)]),
                0,
            )
            loaded = read_engagement_manifest(out_path)
            platforms = sorted(e.platform for e in loaded[0].engagements)
            self.assertEqual(platforms, ["instagram", "twitter"])

    def test_tiktok_sigi_fallback_preserves_post_stats(self) -> None:
        html = (
            '<script id="SIGI_STATE">'
            '{"UserPage":{"userInfo":{"user":{"nickname":"Talent","signature":"bio"},'
            '"stats":{"followerCount":1000,"followingCount":25,"videoCount":10,"heartCount":200}}}}'
            "</script>"
        )
        with patch(
            "seju_face_lab.sns_metrics._Fetcher.fetch_text",
            side_effect=[RuntimeError("api blocked"), html],
        ):
            engagement = fetch_tiktok_engagement("talent")

        self.assertEqual(engagement.following, 25)
        self.assertEqual(engagement.posts, 10)
        self.assertEqual(engagement.engagement_rate, 0.02)

    def test_instagram_falls_back_when_requests_is_missing(self) -> None:
        html = (
            '<meta property="og:description" content="1,234 Followers, 50 Following, 10 Posts">'
            '<meta property="og:title" content="Talent (@talent)">'
        )
        with patch.dict("sys.modules", {"requests": None}):
            with patch(
                "seju_face_lab.sns_metrics._Fetcher.fetch_text",
                side_effect=[RuntimeError("api blocked"), html],
            ):
                engagement = fetch_instagram_engagement("talent")

        self.assertEqual(engagement.fetch_status, "partial")
        self.assertEqual(engagement.followers, 1234)

    def test_instagram_api_404_still_allows_page_fallback(self) -> None:
        html = (
            '<meta property="og:description" content="2,345 Followers">'
            '<meta property="og:title" content="Talent (@talent)">'
        )
        http_404 = urllib.error.HTTPError(
            "https://www.instagram.com/api/v1/users/web_profile_info/?username=talent",
            404,
            "not found",
            hdrs=None,
            fp=None,
        )
        with patch.dict("sys.modules", {"requests": None}):
            with patch(
                "seju_face_lab.sns_metrics._Fetcher.fetch_text",
                side_effect=[http_404, html],
            ):
                engagement = fetch_instagram_engagement("talent")

        self.assertEqual(engagement.fetch_status, "partial")
        self.assertEqual(engagement.followers, 2345)

    def test_twitter_falls_back_to_x_meta_after_public_api_failures(self) -> None:
        html = (
            '<meta property="og:description" content="1.2K Followers">'
            '<meta property="og:title" content="Talent (@talent)">'
        )
        http_404 = urllib.error.HTTPError(
            "https://api.fxtwitter.com/talent",
            404,
            "not found",
            hdrs=None,
            fp=None,
        )
        with patch("urllib.request.urlopen", side_effect=http_404):
            with patch(
                "seju_face_lab.sns_metrics._Fetcher.fetch_text",
                side_effect=[
                    RuntimeError("nitter 1"),
                    RuntimeError("nitter 2"),
                    RuntimeError("nitter 3"),
                    html,
                ],
            ):
                engagement = fetch_twitter_engagement("talent")

        self.assertEqual(engagement.fetch_status, "partial")
        self.assertEqual(engagement.followers, 1200)

    def test_correlation_report_joins_face_scores_and_engagement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            face_scores = root / "subject_reviews.json"
            engagement = root / "engagement.jsonl"
            out = root / "correlation"

            face_scores.write_text(
                json.dumps(
                    {
                        "subjects": [
                            {"subject": "a", "mean_centroid_score": 0.9, "best_centroid_score": 0.95, "image_count": 2},
                            {"subject": "b", "mean_centroid_score": 0.6, "best_centroid_score": 0.7, "image_count": 1},
                            {"subject": "c", "mean_centroid_score": 0.3, "best_centroid_score": 0.4, "image_count": 1},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            write_engagement_manifest(
                [
                    TalentEngagementRecord("a", "A", [_engagement("instagram", "a", 900, 120)]),
                    TalentEngagementRecord("b", "B", [_engagement("instagram", "b", 600, 50)]),
                    TalentEngagementRecord("c", "C", [_engagement("instagram", "c", 300, 10)]),
                ],
                engagement,
            )

            rows = build_correlation_dataset(face_scores, engagement)
            correlations = compute_correlations(rows)
            write_correlation_report(rows, correlations, out)

            self.assertEqual(len(rows), 3)
            self.assertTrue((out / "correlation_summary.json").exists())
            self.assertIn("face score", (out / "correlation_report.md").read_text(encoding="utf-8"))

    def test_generation_run_reviews_treat_negative_scores_as_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "run_a"
            run_b = root / "run_b"
            out = root / "compare"
            run_a.mkdir()
            run_b.mkdir()
            (run_a / "summary.json").write_text(
                json.dumps(
                    {
                        "image_count": 1,
                        "failed_count": 0,
                        "best_image_id": "a",
                        "best_centroid_score": -0.2,
                        "mean_centroid_score": -0.2,
                        "median_centroid_score": -0.2,
                    }
                ),
                encoding="utf-8",
            )
            (run_b / "summary.json").write_text(
                json.dumps(
                    {
                        "image_count": 1,
                        "failed_count": 0,
                        "best_image_id": "b",
                        "best_centroid_score": -0.8,
                        "mean_centroid_score": -0.8,
                        "median_centroid_score": -0.8,
                    }
                ),
                encoding="utf-8",
            )

            reviews = review_generation_runs([run_b, run_a])
            write_generation_run_reviews(reviews, out)
            summary = json.loads((out / "generation_run_reviews.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["best_run_dir"], str(run_a))
            self.assertEqual(summary["best_centroid_score"], -0.2)

    def test_generation_run_reviews_compute_combined_score_per_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            out = root / "compare"
            run.mkdir()
            (run / "summary.json").write_text(
                json.dumps(
                    {
                        "image_count": 3,
                        "failed_count": 0,
                        "best_image_id": "face_only",
                        "best_centroid_score": 1.0,
                        "mean_centroid_score": 0.566667,
                        "median_centroid_score": 0.7,
                    }
                ),
                encoding="utf-8",
            )
            (run / "generation_run.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "provider": "diffusers",
                            "model_id": "local/test-model",
                            "centroid_kind": "mean",
                            "prompt_profile": "detector-friendly",
                            "prompt": "aggregate face prompt",
                            "count": 3,
                            "seed": 42,
                            "steps": 20,
                            "guidance_scale": 7.0,
                            "width": 512,
                            "height": 512,
                            "device": "cuda",
                            "dtype": "float16",
                        },
                        "result": {"status": "generated"},
                    }
                ),
                encoding="utf-8",
            )
            (run / "style_summary.json").write_text(
                json.dumps(
                    {
                        "image_count": 3,
                        "failed_count": 0,
                        "best_image_id": "style_only",
                        "best_style_score": 1.0,
                        "mean_style_score": 0.566667,
                        "median_style_score": 0.7,
                    }
                ),
                encoding="utf-8",
            )
            (run / "scores.csv").write_text(
                "\n".join(
                    [
                        "image_id,path,centroid_score",
                        '"face_only","face.png",1.000000',
                        '"style_only","style.png",0.000000',
                        '"balanced","balanced.png",0.700000',
                    ]
                )
                + "\n",
                encoding="utf-8-sig",
            )
            (run / "style_scores.csv").write_text(
                "\n".join(
                    [
                        "image_id,path,style_score",
                        '"face_only","face.png",0.000000',
                        '"style_only","style.png",1.000000',
                        '"balanced","balanced.png",0.700000',
                    ]
                )
                + "\n",
                encoding="utf-8-sig",
            )

            reviews = review_generation_runs([run])
            write_generation_run_reviews(reviews, out)
            summary = json.loads((out / "generation_run_reviews.json").read_text(encoding="utf-8"))
            csv_text = (out / "generation_run_reviews.csv").read_text(encoding="utf-8-sig")

            self.assertEqual(summary["best_combined_image_id"], "balanced")
            self.assertEqual(summary["best_combined_path"], "balanced.png")
            self.assertEqual(summary["best_combined_score"], 0.7)
            self.assertEqual(summary["best_generation"]["provider"], "diffusers")
            self.assertEqual(summary["best_generation"]["centroid_kind"], "mean")
            self.assertEqual(summary["best_generation"]["prompt_profile"], "detector-friendly")
            self.assertEqual(summary["best_generation"]["seed"], 42)
            self.assertEqual(summary["best_generation"]["planned_count"], 3)
            self.assertEqual(summary["best_generation"]["size"], "512x512")
            self.assertIn("centroid_kind", csv_text)
            self.assertIn("prompt_profile,seed,planned_count,steps", csv_text)
            self.assertEqual(summary["runs"][0]["best_combined_image_id"], "balanced")
            self.assertEqual(summary["runs"][0]["best_combined_path"], "balanced.png")
            self.assertEqual(summary["runs"][0]["centroid_kind"], "mean")
            self.assertEqual(summary["runs"][0]["prompt_profile"], "detector-friendly")

    def test_generation_run_reviews_prefer_quality_passed_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "run_a"
            run_b = root / "run_b"
            out = root / "compare"
            for run in [run_a, run_b]:
                (run / "evaluation").mkdir(parents=True)
                (run / "quality").mkdir()

            (run_a / "evaluation" / "summary.json").write_text(
                json.dumps(
                    {
                        "image_count": 2,
                        "failed_count": 0,
                        "best_image_id": "collage",
                        "best_centroid_score": 1.0,
                        "mean_centroid_score": 0.8,
                        "median_centroid_score": 0.8,
                    }
                ),
                encoding="utf-8",
            )
            (run_a / "evaluation" / "scores.csv").write_text(
                "\n".join(
                    [
                        "image_id,path,centroid_score",
                        '"collage","collage.png",1.000000',
                        '"single_low","single_low.png",0.400000',
                    ]
                )
                + "\n",
                encoding="utf-8-sig",
            )
            (run_a / "quality" / "image_quality.csv").write_text(
                "\n".join(
                    [
                        "image_id,path,qa_pass",
                        '"collage","collage.png",false',
                        '"single_low","single_low.png",true',
                    ]
                )
                + "\n",
                encoding="utf-8-sig",
            )
            (run_a / "quality" / "image_quality.json").write_text(
                json.dumps({"pass_count": 1, "fail_count": 1, "pass_rate": 0.5}),
                encoding="utf-8",
            )

            (run_b / "evaluation" / "summary.json").write_text(
                json.dumps(
                    {
                        "image_count": 1,
                        "failed_count": 0,
                        "best_image_id": "single_high",
                        "best_centroid_score": 0.6,
                        "mean_centroid_score": 0.6,
                        "median_centroid_score": 0.6,
                    }
                ),
                encoding="utf-8",
            )
            (run_b / "evaluation" / "scores.csv").write_text(
                "\n".join(
                    [
                        "image_id,path,centroid_score",
                        '"single_high","single_high.png",0.600000',
                    ]
                )
                + "\n",
                encoding="utf-8-sig",
            )
            (run_b / "quality" / "image_quality.csv").write_text(
                "\n".join(
                    [
                        "image_id,path,qa_pass",
                        '"single_high","single_high.png",true',
                    ]
                )
                + "\n",
                encoding="utf-8-sig",
            )
            (run_b / "quality" / "image_quality.json").write_text(
                json.dumps({"pass_count": 1, "fail_count": 0, "pass_rate": 1.0}),
                encoding="utf-8",
            )

            reviews = review_generation_runs([run_a, run_b])
            write_generation_run_reviews(reviews, out)
            summary = json.loads((out / "generation_run_reviews.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["best_run_dir"], str(run_b))
            self.assertEqual(summary["best_qa_image_id"], "single_high")
            self.assertEqual(summary["best_qa_centroid_score"], 0.6)
            self.assertEqual(summary["runs"][0]["qa_pass_count"], 1)
            self.assertEqual(summary["runs"][0]["qa_pass_rate"], 1.0)

    def test_generation_run_reviews_join_combined_scores_by_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            out = root / "compare"
            run.mkdir()
            (run / "summary.json").write_text(
                json.dumps(
                    {
                        "image_count": 2,
                        "failed_count": 0,
                        "best_image_id": "dup",
                        "best_centroid_score": 1.0,
                        "mean_centroid_score": 0.5,
                        "median_centroid_score": 0.5,
                    }
                ),
                encoding="utf-8",
            )
            (run / "style_summary.json").write_text(
                json.dumps(
                    {
                        "image_count": 2,
                        "failed_count": 0,
                        "best_image_id": "dup",
                        "best_style_score": 1.0,
                        "mean_style_score": 0.5,
                        "median_style_score": 0.5,
                    }
                ),
                encoding="utf-8",
            )
            (run / "scores.csv").write_text(
                "\n".join(
                    [
                        "image_id,path,centroid_score",
                        '"dup","a/dup.png",1.000000',
                        '"dup","b/dup.png",0.000000',
                    ]
                )
                + "\n",
                encoding="utf-8-sig",
            )
            (run / "style_scores.csv").write_text(
                "\n".join(
                    [
                        "image_id,path,style_score",
                        '"dup","b/dup.png",1.000000',
                        '"dup","a/dup.png",0.000000',
                    ]
                )
                + "\n",
                encoding="utf-8-sig",
            )

            reviews = review_generation_runs([run])
            write_generation_run_reviews(reviews, out)
            summary = json.loads((out / "generation_run_reviews.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["best_combined_path"], "a/dup.png")
            self.assertEqual(summary["best_combined_score"], 0.5)

    def test_generation_run_reviews_normalize_paths_before_combining(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            out = root / "compare"
            run.mkdir()
            relative_image = Path("outputs/generated/a.png")
            absolute_image = relative_image.resolve(strict=False)
            (run / "summary.json").write_text(
                json.dumps(
                    {
                        "image_count": 1,
                        "failed_count": 0,
                        "best_image_id": "a",
                        "best_centroid_score": 0.9,
                        "mean_centroid_score": 0.9,
                        "median_centroid_score": 0.9,
                    }
                ),
                encoding="utf-8",
            )
            (run / "style_summary.json").write_text(
                json.dumps(
                    {
                        "image_count": 1,
                        "failed_count": 0,
                        "best_image_id": "a",
                        "best_style_score": 0.7,
                        "mean_style_score": 0.7,
                        "median_style_score": 0.7,
                    }
                ),
                encoding="utf-8",
            )
            (run / "scores.csv").write_text(
                "\n".join(
                    [
                        "image_id,path,centroid_score",
                        f'"a","{relative_image}",0.900000',
                    ]
                )
                + "\n",
                encoding="utf-8-sig",
            )
            (run / "style_scores.csv").write_text(
                "\n".join(
                    [
                        "image_id,path,style_score",
                        f'"a","{absolute_image}",0.700000',
                    ]
                )
                + "\n",
                encoding="utf-8-sig",
            )

            reviews = review_generation_runs([run])
            write_generation_run_reviews(reviews, out)
            summary = json.loads((out / "generation_run_reviews.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["best_combined_path"], str(relative_image))
            self.assertEqual(summary["best_combined_score"], 0.8)

    def test_generation_run_reviews_resolve_relative_paths_from_run_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "generated"
            out = root / "compare"
            (run / "evaluation").mkdir(parents=True)
            (run / "style_evaluation").mkdir()
            image_path = run / "candidate.png"
            (run / "evaluation" / "summary.json").write_text(
                json.dumps(
                    {
                        "image_count": 1,
                        "failed_count": 0,
                        "best_image_id": "candidate",
                        "best_centroid_score": 0.6,
                        "mean_centroid_score": 0.6,
                        "median_centroid_score": 0.6,
                    }
                ),
                encoding="utf-8",
            )
            (run / "style_evaluation" / "style_summary.json").write_text(
                json.dumps(
                    {
                        "image_count": 1,
                        "failed_count": 0,
                        "best_image_id": "candidate",
                        "best_style_score": 1.0,
                        "mean_style_score": 1.0,
                        "median_style_score": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            (run / "evaluation" / "scores.csv").write_text(
                "\n".join(
                    [
                        "image_id,path,centroid_score",
                        '"candidate","candidate.png",0.600000',
                    ]
                )
                + "\n",
                encoding="utf-8-sig",
            )
            (run / "style_evaluation" / "style_scores.csv").write_text(
                "\n".join(
                    [
                        "image_id,path,style_score",
                        f'"candidate","{image_path.resolve(strict=False)}",1.000000',
                    ]
                )
                + "\n",
                encoding="utf-8-sig",
            )

            reviews = review_generation_runs([run])
            write_generation_run_reviews(reviews, out)
            summary = json.loads((out / "generation_run_reviews.json").read_text(encoding="utf-8"))
            html = (out / "generation_run_reviews.html").read_text(encoding="utf-8")

            self.assertEqual(summary["best_combined_path"], "candidate.png")
            self.assertEqual(summary["best_combined_score"], 0.8)
            self.assertIn("candidate", html)
            self.assertIn("combined: 0.800000", html)
            self.assertIn("<img", html)
            self.assertEqual(html.count('<article class="candidate">'), 1)
            self.assertIn(image_path.resolve(strict=False).as_uri(), html)

    def test_generation_run_review_html_uses_generation_root_for_evaluation_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "generated"
            evaluation = run / "evaluation"
            out = root / "compare"
            evaluation.mkdir(parents=True)
            image_path = run / "candidate.png"
            image_path.write_bytes(b"not a real png for this html-only test")
            (evaluation / "summary.json").write_text(
                json.dumps(
                    {
                        "image_count": 1,
                        "failed_count": 0,
                        "best_image_id": "candidate",
                        "best_centroid_score": 0.6,
                        "mean_centroid_score": 0.6,
                        "median_centroid_score": 0.6,
                    }
                ),
                encoding="utf-8",
            )
            (evaluation / "scores.csv").write_text(
                "\n".join(
                    [
                        "image_id,path,centroid_score",
                        '"candidate","candidate.png",0.600000',
                    ]
                )
                + "\n",
                encoding="utf-8-sig",
            )

            reviews = review_generation_runs([evaluation])
            write_generation_run_reviews(reviews, out)
            html = (out / "generation_run_reviews.html").read_text(encoding="utf-8")

            self.assertIn(image_path.resolve(strict=False).as_uri(), html)
            self.assertNotIn((evaluation / "candidate.png").resolve(strict=False).as_uri(), html)

    def test_generation_run_reviews_do_not_share_parent_style_between_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "generated_a"
            run_b = root / "generated_b"
            out = root / "compare"
            (run_a / "evaluation").mkdir(parents=True)
            (run_b / "evaluation").mkdir(parents=True)
            (root / "style_evaluation").mkdir()
            for run, image_id, score in [(run_a, "a", 0.9), (run_b, "b", 0.8)]:
                (run / "evaluation" / "summary.json").write_text(
                    json.dumps(
                        {
                            "image_count": 1,
                            "failed_count": 0,
                            "best_image_id": image_id,
                            "best_centroid_score": score,
                            "mean_centroid_score": score,
                            "median_centroid_score": score,
                        }
                    ),
                    encoding="utf-8",
                )
                (run / "evaluation" / "scores.csv").write_text(
                    "\n".join(
                        [
                            "image_id,path,centroid_score",
                            f'"{image_id}","{run / (image_id + ".png")}",{score:.6f}',
                        ]
                    )
                    + "\n",
                    encoding="utf-8-sig",
                )
            (root / "style_evaluation" / "style_summary.json").write_text(
                json.dumps(
                    {
                        "image_count": 1,
                        "failed_count": 0,
                        "best_image_id": "a",
                        "best_style_score": 1.0,
                        "mean_style_score": 1.0,
                        "median_style_score": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            (root / "style_evaluation" / "style_scores.csv").write_text(
                "\n".join(
                    [
                        "image_id,path,style_score",
                        f'"a","{run_a / "a.png"}",1.000000',
                    ]
                )
                + "\n",
                encoding="utf-8-sig",
            )

            reviews = review_generation_runs([run_a, run_b])
            write_generation_run_reviews(reviews, out)
            summary = json.loads((out / "generation_run_reviews.json").read_text(encoding="utf-8"))

            self.assertIsNone(summary["runs"][0]["best_style_score"])
            self.assertIsNone(summary["runs"][1]["best_style_score"])
            self.assertIsNone(summary["runs"][0]["best_combined_score"])
            self.assertIsNone(summary["runs"][1]["best_combined_score"])

    def test_split_paths_chunks_by_worker_count(self) -> None:
        chunks = _split_paths([Path(f"{idx}.png") for idx in range(5)], 2)
        self.assertEqual([len(chunk) for chunk in chunks], [3, 2])

    def test_insightface_no_face_raises_instead_of_mixing_dimensions(self) -> None:
        backend = InsightFaceBackend()
        backend._app = _NoFaceApp()  # noqa: SLF001 - direct injection keeps this test offline.

        with patch("seju_face_lab.backends._import_cv2", return_value=_FakeCV2()):
            with self.assertRaisesRegex(ValueError, "No face detected"):
                backend.vectorize(Path("missing-face.jpg"))

    def test_prepare_windows_torch_cuda_dlls_adds_torch_lib_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch_init = root / "torch" / "__init__.py"
            dll_dir = torch_init.parent / "lib"
            dll_dir.mkdir(parents=True)
            torch_init.write_text("", encoding="utf-8")
            (dll_dir / "cublasLt64_12.dll").write_text("", encoding="utf-8")
            fake_torch = SimpleNamespace(__file__=str(torch_init))

            backends_module._ADDED_DLL_DIRS.clear()  # noqa: SLF001
            backends_module._DLL_DIRECTORY_HANDLES.clear()  # noqa: SLF001
            with (
                patch("seju_face_lab.backends._is_windows", return_value=True),
                patch("seju_face_lab.backends.importlib.util.find_spec", return_value=object()),
                patch.dict("sys.modules", {"torch": fake_torch}),
                patch.dict(os.environ, {"PATH": "C:\\base"}),
                patch("seju_face_lab.backends._add_windows_dll_directory") as add_dll_directory,
            ):
                first = backends_module._prepare_windows_torch_cuda_dlls()  # noqa: SLF001
                second = backends_module._prepare_windows_torch_cuda_dlls()  # noqa: SLF001

            self.assertEqual(first, dll_dir.resolve())
            self.assertEqual(second, dll_dir.resolve())
            self.assertEqual(add_dll_directory.call_count, 1)

    def test_prepare_windows_torch_cuda_dlls_ignores_non_windows(self) -> None:
        with patch("seju_face_lab.backends._is_windows", return_value=False):
            self.assertIsNone(backends_module._prepare_windows_torch_cuda_dlls())  # noqa: SLF001

    def test_prepare_windows_torch_cuda_dlls_ignores_broken_torch_import(self) -> None:
        real_import = __import__

        def broken_torch_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "torch":
                raise OSError("broken torch")
            return real_import(name, *args, **kwargs)

        with (
            patch("seju_face_lab.backends._is_windows", return_value=True),
            patch("seju_face_lab.backends.importlib.util.find_spec", return_value=object()),
            patch("builtins.__import__", side_effect=broken_torch_import),
        ):
            self.assertIsNone(backends_module._prepare_windows_torch_cuda_dlls())  # noqa: SLF001

    def test_prepare_utf8_console_for_deepface_reconfigures_windows_streams(self) -> None:
        fake_stdout = _FakeReconfigurableStream()
        fake_stderr = _FakeReconfigurableStream()
        with (
            patch("seju_face_lab.backends.os.name", "nt"),
            patch.object(sys, "stdout", fake_stdout),
            patch.object(sys, "stderr", fake_stderr),
            patch.dict(os.environ, {}, clear=True),
        ):
            backends_module._prepare_utf8_console_for_deepface()  # noqa: SLF001
            self.assertEqual(os.environ["PYTHONIOENCODING"], "utf-8")

        self.assertEqual(fake_stdout.calls, [{"encoding": "utf-8", "errors": "replace"}])
        self.assertEqual(fake_stderr.calls, [{"encoding": "utf-8", "errors": "replace"}])

    def test_deepface_backend_vectorizes_largest_detected_face(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "face.png"
            Image.new("RGB", (96, 64), (220, 200, 190)).save(image_path)
            backend = DeepFaceBackend()

            with patch(
                "seju_face_lab.backends._import_deepface",
                return_value=_FakeDeepFace(
                    [
                        {
                            "embedding": [1.0, 0.0, 0.0],
                            "facial_area": {"x": 0, "y": 0, "w": 8, "h": 8},
                        },
                        {
                            "embedding": [2.0, 2.0, 0.0],
                            "facial_area": {"x": 20, "y": 10, "w": 32, "h": 32},
                        },
                    ]
                ),
            ):
                vector = backend.vectorize(image_path)

        self.assertEqual(vector.image_id, "face")
        self.assertEqual(vector.embedding.shape, (3,))
        self.assertAlmostEqual(float(np.linalg.norm(vector.embedding)), 1.0, places=6)
        self.assertTrue(np.allclose(vector.embedding, np.asarray([0.70710677, 0.70710677, 0.0])))
        self.assertEqual(vector.appearance.shape, (64, 64, 3))

    def test_deepface_backend_raises_when_no_faces_are_returned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "blank.png"
            Image.new("RGB", (32, 32), (0, 0, 0)).save(image_path)
            backend = DeepFaceBackend()

            with patch("seju_face_lab.backends._import_deepface", return_value=_FakeDeepFace([])):
                with self.assertRaisesRegex(ValueError, "No face detected"):
                    backend.vectorize(image_path)

    def test_deepface_retinaface_backend_factory_uses_retinaface_detector(self) -> None:
        with patch("seju_face_lab.backends.importlib.util.find_spec", return_value=object()):
            backend = backends_module._make_deepface_backend("retinaface", name="deepface-retinaface")  # noqa: SLF001

        self.assertIsInstance(backend, DeepFaceBackend)
        self.assertEqual(backend.name, "deepface-retinaface")
        self.assertEqual(backend.detector_backend, "retinaface")
        self.assertIn("detector=retinaface", backend.description)

    def test_deepface_retinaface_backend_factory_stays_dependency_gated(self) -> None:
        with patch("seju_face_lab.backends.importlib.util.find_spec", return_value=None):
            backend = backends_module._make_deepface_backend("retinaface", name="deepface-retinaface")  # noqa: SLF001

        self.assertIsInstance(backend, backends_module.PlannedBackend)
        self.assertEqual(backend.name, "deepface-retinaface")
        self.assertIn("detector=retinaface", backend.description)
        self.assertIn("detector_backend='retinaface'", backend.notes)

    def test_compare_deepface_detectors_reports_detector_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            generated = root / "generated"
            out = root / "deepface_detectors"
            raw.mkdir()
            generated.mkdir()
            _write_image(raw / "a.png", (235, 205, 190))
            _write_image(raw / "b.png", (225, 198, 184))
            _write_image(generated / "candidate_a.png", (232, 202, 188))
            _write_image(generated / "candidate_b.png", (170, 145, 130))

            with patch("seju_face_lab.backends._import_deepface", return_value=_DetectorAwareDeepFace()):
                self.assertEqual(
                    main(
                        [
                            "compare-deepface-detectors",
                            "--reference-images",
                            str(raw),
                            "--images",
                            str(generated),
                            "--out",
                            str(out),
                            "--detectors",
                            "opencv",
                            "retinaface",
                        ]
                    ),
                    0,
                )

            report = json.loads((out / "deepface_detector_comparison.json").read_text(encoding="utf-8"))
            markdown = (out / "deepface_detector_comparison.md").read_text(encoding="utf-8")
            self.assertEqual([run["backend"] for run in report["runs"]], ["deepface-opencv", "deepface-retinaface"])
            self.assertEqual(report["runs"][0]["reference_count"], 1)
            self.assertEqual(report["runs"][0]["reference_failed_count"], 1)
            self.assertEqual(report["runs"][1]["reference_count"], 2)
            self.assertEqual(report["rank_agreement"][0]["common_image_count"], 1)
            self.assertIn("ref_failed", markdown)
            self.assertIn("| deepface-opencv | completed | 1 | 1 |", markdown)
            self.assertTrue((out / "deepface-opencv" / "model" / "vectors" / "reference_vector_failures.json").exists())

    def test_compare_deepface_detectors_can_reuse_existing_detector_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            generated = root / "generated"
            out = root / "deepface_detectors"
            raw.mkdir()
            generated.mkdir()
            _write_image(raw / "a.png", (235, 205, 190))
            _write_image(generated / "candidate_a.png", (232, 202, 188))

            with patch("seju_face_lab.backends._import_deepface", return_value=_DetectorAwareDeepFace()):
                self.assertEqual(
                    main(
                        [
                            "compare-deepface-detectors",
                            "--reference-images",
                            str(raw),
                            "--images",
                            str(generated),
                            "--out",
                            str(out),
                            "--detectors",
                            "opencv",
                        ]
                    ),
                    0,
                )

            with patch("seju_face_lab.backends._import_deepface", side_effect=AssertionError("should not run")):
                self.assertEqual(
                    main(
                        [
                            "compare-deepface-detectors",
                            "--reference-images",
                            str(raw),
                            "--images",
                            str(generated),
                            "--out",
                            str(out),
                            "--detectors",
                            "opencv",
                            "--reuse-existing",
                        ]
                    ),
                    0,
                )

            report = json.loads((out / "deepface_detector_comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(report["runs"][0]["reference_count"], 1)
            self.assertEqual(report["runs"][0]["image_count"], 1)

            run_config_path = out / "deepface-opencv" / "detector_run.json"
            legacy_run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            legacy_run_config.pop("max_reference_images")
            legacy_run_config.pop("max_images")
            run_config_path.write_text(json.dumps(legacy_run_config), encoding="utf-8")
            with patch("seju_face_lab.backends._import_deepface", side_effect=AssertionError("should not run")):
                self.assertEqual(
                    main(
                        [
                            "compare-deepface-detectors",
                            "--reference-images",
                            str(raw),
                            "--images",
                            str(generated),
                            "--out",
                            str(out),
                            "--detectors",
                            "opencv",
                            "--reuse-existing",
                        ]
                    ),
                    0,
                )

            with patch("seju_face_lab.backends._import_deepface", return_value=_DetectorAwareDeepFace()):
                self.assertEqual(
                    main(
                        [
                            "compare-deepface-detectors",
                            "--reference-images",
                            str(raw),
                            "--images",
                            str(generated),
                            "--out",
                            str(out),
                            "--detectors",
                            "opencv",
                            "--crop",
                            "none",
                            "--reuse-existing",
                        ]
                    ),
                    0,
                )

            run_config = json.loads((out / "deepface-opencv" / "detector_run.json").read_text(encoding="utf-8"))
            self.assertEqual(run_config["crop"], "none")

    def test_compare_deepface_detectors_clears_stale_reference_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            generated = root / "generated"
            out = root / "deepface_detectors"
            raw.mkdir()
            generated.mkdir()
            _write_image(raw / "a.png", (235, 205, 190))
            stale_failure = raw / "b.png"
            _write_image(stale_failure, (225, 198, 184))
            _write_image(generated / "candidate_a.png", (232, 202, 188))

            with patch("seju_face_lab.backends._import_deepface", return_value=_DetectorAwareDeepFace()):
                self.assertEqual(
                    main(
                        [
                            "compare-deepface-detectors",
                            "--reference-images",
                            str(raw),
                            "--images",
                            str(generated),
                            "--out",
                            str(out),
                            "--detectors",
                            "opencv",
                        ]
                    ),
                    0,
                )

            stale_failure.unlink()
            with patch("seju_face_lab.backends._import_deepface", return_value=_DetectorAwareDeepFace()):
                self.assertEqual(
                    main(
                        [
                            "compare-deepface-detectors",
                            "--reference-images",
                            str(raw),
                            "--images",
                            str(generated),
                            "--out",
                            str(out),
                            "--detectors",
                            "opencv",
                        ]
                    ),
                    0,
                )

            with patch("seju_face_lab.backends._import_deepface", side_effect=AssertionError("should not run")):
                self.assertEqual(
                    main(
                        [
                            "compare-deepface-detectors",
                            "--reference-images",
                            str(raw),
                            "--images",
                            str(generated),
                            "--out",
                            str(out),
                            "--detectors",
                            "opencv",
                            "--reuse-existing",
                        ]
                    ),
                    0,
                )

            report = json.loads((out / "deepface_detector_comparison.json").read_text(encoding="utf-8"))
            failures = json.loads(
                (out / "deepface-opencv" / "model" / "vectors" / "reference_vector_failures.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["runs"][0]["reference_failed_count"], 0)
            self.assertEqual(failures["failed_count"], 0)

    def test_compare_deepface_detectors_can_limit_slow_detector_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            generated = root / "generated"
            out = root / "deepface_detectors"
            raw.mkdir()
            generated.mkdir()
            _write_image(raw / "a.png", (235, 205, 190))
            _write_image(raw / "b.png", (225, 198, 184))
            _write_image(raw / "c.png", (215, 188, 174))
            _write_image(generated / "candidate_a.png", (232, 202, 188))
            _write_image(generated / "candidate_b.png", (170, 145, 130))

            with patch("seju_face_lab.backends._import_deepface", return_value=_DetectorAwareDeepFace()):
                self.assertEqual(
                    main(
                        [
                            "compare-deepface-detectors",
                            "--reference-images",
                            str(raw),
                            "--images",
                            str(generated),
                            "--out",
                            str(out),
                            "--detectors",
                            "retinaface",
                            "--max-reference-images",
                            "2",
                            "--max-images",
                            "1",
                        ]
                    ),
                    0,
                )

            report = json.loads((out / "deepface_detector_comparison.json").read_text(encoding="utf-8"))
            run_config = json.loads((out / "deepface-retinaface" / "detector_run.json").read_text(encoding="utf-8"))
            self.assertEqual(report["max_reference_images"], 2)
            self.assertEqual(report["max_images"], 1)
            self.assertEqual(report["runs"][0]["reference_count"], 2)
            self.assertEqual(report["runs"][0]["image_count"], 1)
            self.assertEqual(run_config["max_reference_images"], "2")
            self.assertEqual(run_config["max_images"], "1")

    def test_distribute_vectorize_scores_only_assigned_local_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            generated = root / "generated"
            out = root / "worker_out"
            raw.mkdir()
            generated.mkdir()
            _write_image(raw / "a.png", (230, 210, 200))
            _write_image(raw / "b.png", (180, 160, 150))
            _write_image(generated / "selected.png", (231, 211, 201))
            _write_image(generated / "unselected.png", (80, 70, 60))

            model = root / "model"
            self.assertEqual(main(["build", "--images", str(raw), "--out", str(model)]), 0)
            scores = distribute_vectorize(
                [generated / "selected.png"],
                model,
                out,
                backend="deterministic",
                workers=[WorkerConfig(name="local-test", python="", project_dir="")],
            )

            self.assertEqual([score["image_id"] for score in scores], ["selected"])

    def test_cli_distributed_evaluate_writes_merged_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            generated = root / "generated"
            out = root / "distributed_eval"
            raw.mkdir()
            generated.mkdir()
            _write_image(raw / "a.png", (230, 210, 200))
            _write_image(raw / "b.png", (180, 160, 150))
            _write_image(generated / "selected.png", (231, 211, 201))

            model = root / "model"
            self.assertEqual(main(["build", "--images", str(raw), "--out", str(model)]), 0)
            self.assertEqual(
                main(
                    [
                        "distributed-evaluate",
                        "--model",
                        str(model),
                        "--images",
                        str(generated),
                        "--out",
                        str(out),
                        "--backend",
                        "deterministic",
                    ]
                ),
                0,
            )

            scores = (out / "scores.csv").read_text(encoding="utf-8-sig")
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            distributed = json.loads((out / "distributed_scores.json").read_text(encoding="utf-8"))
            self.assertIn("selected", scores)
            self.assertEqual(summary["image_count"], 1)
            self.assertEqual(summary["failed_count"], 0)
            self.assertEqual(distributed[0]["image_id"], "selected")

    def test_cli_distributed_evaluate_returns_nonzero_when_worker_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            generated = root / "generated"
            out = root / "distributed_eval"
            raw.mkdir()
            generated.mkdir()
            _write_image(raw / "a.png", (230, 210, 200))
            _write_image(generated / "selected.png", (231, 211, 201))

            model = root / "model"
            self.assertEqual(main(["build", "--images", str(raw), "--out", str(model)]), 0)
            result = main(
                [
                    "distributed-evaluate",
                    "--model",
                    str(model),
                    "--images",
                    str(generated),
                    "--out",
                    str(out),
                    "--backend",
                    "missing-backend",
                ]
            )

            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(result, 1)
            self.assertEqual(summary["image_count"], 0)
            self.assertEqual(summary["worker_failures"][0]["worker"], "ultra2025-4090")

    def test_cli_distributed_evaluate_writes_empty_outputs_for_empty_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            generated = root / "generated"
            out = root / "distributed_eval"
            raw.mkdir()
            generated.mkdir()
            _write_image(raw / "a.png", (230, 210, 200))

            model = root / "model"
            self.assertEqual(main(["build", "--images", str(raw), "--out", str(model)]), 0)
            result = main(
                [
                    "distributed-evaluate",
                    "--model",
                    str(model),
                    "--images",
                    str(generated),
                    "--out",
                    str(out),
                ]
            )

            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            distributed = json.loads((out / "distributed_scores.json").read_text(encoding="utf-8"))
            self.assertEqual(result, 0)
            self.assertEqual(summary["image_count"], 0)
            self.assertEqual(summary["failed_count"], 0)
            self.assertEqual(distributed, [])

    def test_distributed_evaluate_preserves_full_failed_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            out = root / "distributed_eval"
            raw.mkdir()
            _write_image(raw / "a.png", (230, 210, 200))
            bad_paths = []
            for index in range(21):
                bad = root / f"bad_{index}.png"
                bad.write_text("not an image", encoding="utf-8")
                bad_paths.append(bad)

            model = root / "model"
            self.assertEqual(main(["build", "--images", str(raw), "--out", str(model)]), 0)
            scores = distribute_vectorize(
                bad_paths,
                model,
                out,
                backend="deterministic",
                workers=[WorkerConfig(name="local-test", python="", project_dir="")],
            )

            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(scores, [])
            self.assertEqual(summary["failed_count"], 21)
            self.assertEqual(len(summary["failed_paths"]), 20)

    def test_worker_evaluate_reports_failed_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            _write_image(raw / "a.png", (230, 210, 200))
            bad = root / "bad.png"
            bad2 = root / "bad2.png"
            bad.write_text("not an image", encoding="utf-8")
            bad2.write_text("not an image either", encoding="utf-8")

            model = root / "model"
            out = root / "worker_out"
            self.assertEqual(main(["build", "--images", str(raw), "--out", str(model)]), 0)
            scores = run_local_evaluate([bad, bad2], model, out, backend="deterministic")
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(scores, [])
            self.assertEqual(summary["failed_count"], 2)
            self.assertEqual(summary["failed_paths"], [str(bad), str(bad2)])

    def test_worker_diagnostics_writes_local_probe_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "worker_diag"
            worker = WorkerConfig(name="local-test", python="python", project_dir=str(root))
            probe = {
                "hostname": "ultra2025",
                "seju_face_lab_importable": True,
                "torch_importable": True,
                "torch_cuda_available": True,
                "torch_cuda_device_count": 1,
                "torch_cuda_device_name": "NVIDIA GeForce RTX 4090",
                "configured_project_exists": True,
                "configured_python_exists": True,
            }

            with patch(
                "seju_face_lab.workers.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=["python"],
                    returncode=0,
                    stdout=json.dumps(probe) + "\n",
                    stderr="",
                ),
            ) as run:
                report = write_worker_diagnostics(out, workers=[worker])

            self.assertEqual(report["worker_count"], 1)
            self.assertTrue(report["workers"][0]["ok"])
            self.assertEqual(report["workers"][0]["probe"]["torch_cuda_device_name"], "NVIDIA GeForce RTX 4090")
            self.assertTrue((out / "worker_diagnostics.json").exists())
            self.assertIn("NVIDIA GeForce RTX 4090", (out / "worker_diagnostics.md").read_text(encoding="utf-8"))
            self.assertEqual(run.call_args.kwargs["cwd"], str(root))
            self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")

    def test_worker_diagnostics_can_probe_remote_over_ssh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "worker_diag"
            worker = WorkerConfig(
                name="remote-test",
                python=r"C:\repo\.venv\Scripts\python.exe",
                project_dir=r"C:\repo",
                remote_host="nicolas2025",
            )
            probe = {
                "hostname": "nicolas2025",
                "seju_face_lab_importable": True,
                "torch_importable": True,
                "torch_cuda_available": True,
                "torch_cuda_device_count": 1,
                "torch_cuda_device_name": "NVIDIA GeForce RTX 5060 Ti",
                "configured_project_exists": True,
                "configured_python_exists": True,
            }

            with patch(
                "seju_face_lab.workers.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=["ssh"],
                    returncode=0,
                    stdout=json.dumps(probe) + "\n",
                    stderr="",
                ),
            ) as run:
                report = write_worker_diagnostics(out, workers=[worker])

            self.assertTrue(report["workers"][0]["ok"])
            self.assertEqual(report["workers"][0]["remote_host"], "nicolas2025")
            self.assertEqual(run.call_args.args[0][0], "ssh")
            self.assertIn("nicolas2025", run.call_args.args[0])

    def test_worker_diagnostics_requires_cuda_for_ready_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "worker_diag"
            worker = WorkerConfig(name="cpu-only", python="python", project_dir=str(Path(tmp)))
            probe = {
                "hostname": "cpu-host",
                "seju_face_lab_importable": True,
                "torch_importable": True,
                "torch_cuda_available": False,
                "torch_cuda_device_count": 0,
                "torch_cuda_device_name": None,
            }

            with patch(
                "seju_face_lab.workers.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=["python"],
                    returncode=0,
                    stdout=json.dumps(probe) + "\n",
                    stderr="",
                ),
            ):
                report = write_worker_diagnostics(out, workers=[worker])

            self.assertFalse(report["workers"][0]["ok"])
            self.assertFalse(report["workers"][0]["probe"]["torch_cuda_available"])

    def test_worker_diagnostics_requires_configured_remote_paths_for_ready_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "worker_diag"
            worker = WorkerConfig(
                name="remote-stale-venv",
                python=r"C:\missing\.venv\Scripts\python.exe",
                project_dir=r"C:\missing\repo",
                remote_host="nicolas2025",
            )
            probe = {
                "hostname": "nicolas2025",
                "seju_face_lab_importable": True,
                "torch_importable": True,
                "torch_cuda_available": True,
                "torch_cuda_device_count": 1,
                "torch_cuda_device_name": "NVIDIA GeForce RTX 3070",
                "configured_project_exists": False,
                "configured_python_exists": False,
            }

            with patch(
                "seju_face_lab.workers.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=["ssh"],
                    returncode=0,
                    stdout=json.dumps(probe) + "\n",
                    stderr="",
                ),
            ):
                report = write_worker_diagnostics(out, workers=[worker])

            self.assertFalse(report["workers"][0]["ok"])
            self.assertFalse(report["workers"][0]["probe"]["configured_python_exists"])

    def test_cli_worker_diagnostics_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "worker_diag"
            probe = {
                "hostname": "ultra2025",
                "seju_face_lab_importable": True,
                "torch_importable": True,
                "torch_cuda_available": True,
                "torch_cuda_device_count": 1,
                "torch_cuda_device_name": "NVIDIA GeForce RTX 4090",
                "configured_project_exists": True,
                "configured_python_exists": True,
            }

            with patch(
                "seju_face_lab.workers.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=["python"],
                    returncode=0,
                    stdout=json.dumps(probe) + "\n",
                    stderr="",
                ),
            ):
                self.assertEqual(main(["worker-diagnostics", "--out", str(out)]), 0)

            report = json.loads((out / "worker_diagnostics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["worker_count"], 1)
            self.assertTrue(report["workers"][0]["ok"])

    def test_distribute_vectorize_rejects_remote_workers_until_sync_exists(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "remote worker subset evaluation"):
            distribute_vectorize(
                [Path("image.png")],
                Path("model"),
                Path("out"),
                workers=[
                    WorkerConfig(
                        name="remote-test",
                        python="python",
                        project_dir="repo",
                        remote_host="nicolas2025",
                    )
                ],
            )

    def test_organizer_parses_downloader_and_legacy_filenames(self) -> None:
        self.assertEqual(
            _talent_slug_from_filename(Path("airi-yamakawa_92064b501f.jpg")),
            "airi-yamakawa",
        )
        self.assertEqual(
            _talent_slug_from_filename(Path("0001_airi-yamakawa_92064b501f.jpg")),
            "airi-yamakawa",
        )
        self.assertIsNone(_talent_slug_from_filename(Path("unrecognized.jpg")))


def _engagement(
    platform: str,
    handle: str,
    followers: int,
    total_engagement: int,
) -> SnsEngagement:
    return SnsEngagement(
        platform=platform,
        handle=handle,
        profile_url=f"https://example.test/{handle}",
        followers=followers,
        following=None,
        posts=10,
        total_engagement=total_engagement,
        engagement_rate=round(total_engagement / followers / 10, 6),
        bio=None,
        display_name=handle,
        fetch_status="ok",
        fetch_error=None,
        retrieved_at="2026-06-15T00:00:00+00:00",
    )


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 32), color).save(path)


class _NoFaceApp:
    def get(self, _image: np.ndarray) -> list[object]:
        return []


class _FakeCV2:
    @staticmethod
    def imread(_path: str) -> np.ndarray:
        return np.zeros((32, 32, 3), dtype=np.uint8)


class _FakeReconfigurableStream:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def reconfigure(self, **kwargs: str) -> None:
        self.calls.append(kwargs)


class _FakeDeepFace:
    def __init__(self, representations: list[dict]) -> None:
        self.representations = representations

    def represent(self, **_kwargs: object) -> list[dict]:
        return self.representations


class _DetectorAwareDeepFace:
    def represent(self, **kwargs: object) -> list[dict]:
        detector = str(kwargs["detector_backend"])
        stem = Path(str(kwargs["img_path"])).stem
        if detector == "opencv" and stem.endswith("b"):
            return []
        if stem.endswith("a"):
            embedding = [1.0, 0.0, 0.0]
        else:
            embedding = [0.0, 1.0, 0.0]
        return [{"embedding": embedding, "facial_area": {"x": 4, "y": 4, "w": 16, "h": 16}}]


if __name__ == "__main__":
    unittest.main()
