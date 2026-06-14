from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

import bootstrap  # noqa: F401
from seju_face_lab.cli import _sources_download, main
from seju_face_lab.model import load_model
from seju_face_lab.sources import DownloadResult


class PipelineTests(unittest.TestCase):
    def test_build_prompt_render_and_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            generated = root / "generated"
            subjects = root / "subjects"
            model_dir = root / "model space"
            eval_dir = root / "eval"
            generation_dir = root / "generated faces"
            review_dir = root / "review"
            raw.mkdir()
            generated.mkdir()
            (subjects / "near_subject").mkdir(parents=True)
            (subjects / "far_subject").mkdir(parents=True)

            _write_face_like_image(raw / "a.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(raw / "b.png", (225, 198, 184), eye_offset=2)
            _write_face_like_image(raw / "c.png", (240, 210, 196), eye_offset=-2)
            _write_face_like_image(generated / "candidate.png", (232, 202, 188), eye_offset=1)
            _write_face_like_image(subjects / "near_subject" / "near.png", (232, 202, 188), eye_offset=1)
            _write_face_like_image(subjects / "far_subject" / "far.png", (170, 145, 130), eye_offset=7)
            (subjects / "far_subject" / "broken.jpg").write_text("not an image", encoding="utf-8")

            self.assertEqual(main(["build", "--images", str(raw), "--out", str(model_dir)]), 0)
            self.assertEqual(
                main(["build", "--images", str(raw), "--out", str(root / "model_backend"), "--backend", "deterministic"]),
                0,
            )
            model = load_model(model_dir)
            self.assertEqual(len(model.image_ids), 3)
            self.assertGreater(model.embedding_dim, 100)
            self.assertTrue((model_dir / "mean_face.png").exists())
            self.assertTrue((model_dir / "median_face.png").exists())
            self.assertIn("aggregate traits", (model_dir / "prompt.txt").read_text(encoding="utf-8"))
            self.assertTrue((model_dir / "generation_manifest.json").exists())

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
            scores = (eval_dir / "scores.csv").read_text(encoding="utf-8-sig")
            self.assertIn("candidate", scores)
            self.assertIn("centroid_score", scores)
            self.assertTrue((eval_dir / "summary.json").exists())

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
            self.assertEqual(generation_run["config"]["prompt"], "")
            self.assertEqual(generation_run["config"]["variant"], "fp16")
            self.assertEqual(generation_run["config"]["negative_prompt"], "copied identity")
            self.assertIn("evaluate", generation_run["result"]["evaluation_command"])
            self.assertIn('"', generation_run["result"]["evaluation_command"])
            self.assertEqual(generation_run["result"]["evaluation_argv"][5], str(model_dir))

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

    def test_backends_command_lists_planned_backends(self) -> None:
        self.assertEqual(main(["backends"]), 0)

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


if __name__ == "__main__":
    unittest.main()
