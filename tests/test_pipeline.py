from __future__ import annotations

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
            model_dir = root / "model"
            eval_dir = root / "eval"
            raw.mkdir()
            generated.mkdir()

            _write_face_like_image(raw / "a.png", (235, 205, 190), eye_offset=0)
            _write_face_like_image(raw / "b.png", (225, 198, 184), eye_offset=2)
            _write_face_like_image(raw / "c.png", (240, 210, 196), eye_offset=-2)
            _write_face_like_image(generated / "candidate.png", (232, 202, 188), eye_offset=1)

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

    def test_backends_command_lists_planned_backends(self) -> None:
        self.assertEqual(main(["backends"]), 0)

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
