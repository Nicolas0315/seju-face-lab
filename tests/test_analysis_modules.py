from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import bootstrap  # noqa: F401
import numpy as np
from PIL import Image
from scripts.organize_by_talent import _talent_slug_from_filename
from seju_face_lab.backends import DeepFaceBackend, InsightFaceBackend
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

    def test_split_paths_chunks_by_worker_count(self) -> None:
        chunks = _split_paths([Path(f"{idx}.png") for idx in range(5)], 2)
        self.assertEqual([len(chunk) for chunk in chunks], [3, 2])

    def test_insightface_no_face_raises_instead_of_mixing_dimensions(self) -> None:
        backend = InsightFaceBackend()
        backend._app = _NoFaceApp()  # noqa: SLF001 - direct injection keeps this test offline.

        with patch("seju_face_lab.backends._import_cv2", return_value=_FakeCV2()):
            with self.assertRaisesRegex(ValueError, "No face detected"):
                backend.vectorize(Path("missing-face.jpg"))

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


class _FakeDeepFace:
    def __init__(self, representations: list[dict]) -> None:
        self.representations = representations

    def represent(self, **_kwargs: object) -> list[dict]:
        return self.representations


if __name__ == "__main__":
    unittest.main()
