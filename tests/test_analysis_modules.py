from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bootstrap  # noqa: F401
import numpy as np
from PIL import Image
from scripts.organize_by_talent import _talent_slug_from_filename
from seju_face_lab.backends import InsightFaceBackend
from seju_face_lab.cli import main
from seju_face_lab.correlation import (
    build_correlation_dataset,
    compute_correlations,
    write_correlation_report,
)
from seju_face_lab.sns_metrics import (
    SnsEngagement,
    SnsHandleRecord,
    TalentEngagementRecord,
    extract_sns_handles_from_links,
    read_engagement_manifest,
    read_handles_manifest,
    write_engagement_manifest,
    write_handles_manifest,
)
from seju_face_lab.workers import WorkerConfig, _split_paths, distribute_vectorize


class AnalysisModuleTests(unittest.TestCase):
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

    def test_split_paths_chunks_by_worker_count(self) -> None:
        chunks = _split_paths([Path(f"{idx}.png") for idx in range(5)], 2)
        self.assertEqual([len(chunk) for chunk in chunks], [3, 2])

    def test_insightface_no_face_raises_instead_of_mixing_dimensions(self) -> None:
        backend = InsightFaceBackend()
        backend._app = _NoFaceApp()  # noqa: SLF001 - direct injection keeps this test offline.

        with patch("seju_face_lab.backends._import_cv2", return_value=_FakeCV2()):
            with self.assertRaisesRegex(ValueError, "No face detected"):
                backend.vectorize(Path("missing-face.jpg"))

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


if __name__ == "__main__":
    unittest.main()
