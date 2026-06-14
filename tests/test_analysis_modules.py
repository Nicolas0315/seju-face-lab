from __future__ import annotations

import json
import os
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
from seju_face_lab.precision import write_precision_report
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
from seju_face_lab.workers import WorkerConfig, _split_paths, distribute_vectorize, run_local_evaluate


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
            out = root / "precision"
            model.mkdir()
            generation.mkdir()
            evaluation.mkdir()
            subjects.mkdir()
            backend_comparison.mkdir()
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

            report = write_precision_report(
                model_dir=model,
                out_dir=out,
                generation_review=generation,
                evaluation=evaluation,
                subject_review=subjects,
                backend_comparison=backend_comparison,
            )

            self.assertEqual(report["model"]["image_count"], 3)
            self.assertEqual(report["model"]["centroid_vectors"]["mean_embedding"]["shape"], [2])
            self.assertEqual(report["model"]["centroid_vectors"]["mean_embedding"]["l2_norm"], 1.0)
            self.assertEqual(len(report["model"]["centroid_vectors"]["mean_embedding"]["sha256"]), 64)
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
            self.assertTrue((out / "precision_report.json").exists())
            self.assertIn(
                "Backend Comparison",
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
            out = root / "precision"
            model.mkdir()
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

            self.assertEqual(
                main(["precision-report", "--model", str(model), "--out", str(out)]),
                0,
            )

            report = json.loads((out / "precision_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["model"]["embedding_dim"], 4)

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

            self.assertEqual(summary["best_combined_image_id"], "balanced")
            self.assertEqual(summary["best_combined_path"], "balanced.png")
            self.assertEqual(summary["best_combined_score"], 0.7)
            self.assertEqual(summary["runs"][0]["best_combined_image_id"], "balanced")
            self.assertEqual(summary["runs"][0]["best_combined_path"], "balanced.png")

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

            self.assertEqual(summary["best_combined_path"], "candidate.png")
            self.assertEqual(summary["best_combined_score"], 0.8)

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
