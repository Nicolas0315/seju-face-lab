from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import bootstrap  # noqa: F401
from seju_face_lab.sources import (
    SourceCandidate,
    download_source_images,
    parse_profile,
    parse_talent_links,
    read_source_manifest,
    write_source_manifest,
    _quote_url,
)


class SourceParsingTests(unittest.TestCase):
    def test_parse_talent_links_keeps_only_profile_pages(self) -> None:
        html = """
        <a href="/talents/">talent</a>
        <a href="https://seju.tokyo/talents/rei-mukai/">Rei</a>
        <a href="/talents/rei-mukai/">Duplicate</a>
        <a href="/topics/topic-1/">Topic</a>
        <a href="https://example.com/talents/outside/">Outside</a>
        """
        self.assertEqual(
            parse_talent_links(html, "https://seju.tokyo/talents/"),
            ["https://seju.tokyo/talents/rei-mukai/"],
        )

    def test_parse_profile_finds_name_birthdate_and_profile_images(self) -> None:
        html = """
        <title>向井 怜衣 | seju</title>
        <meta name="description" content="向井 怜衣 Profile；2007年6月25日・広島県・159cm" />
        <meta property="og:image" content="https://seju.tokyo/wp-content/uploads/2023/10/rei_thumbnail.jpg" />
        <img src="/wp-content/uploads/2023/10/rei_01.jpg" alt="向井 怜衣の画像1枚目" />
        <img src="/wp-content/uploads/2025/05/cropped-favicon-32x32.png" alt="favicon" />
        """
        profile, images = parse_profile(html, "https://seju.tokyo/talents/rei-mukai/")
        self.assertEqual(profile["name"], "向井 怜衣")
        self.assertEqual(profile["birthdate"], "2007-06-25")
        self.assertEqual(len(images), 2)
        self.assertTrue(all("wp-content/uploads" in image[0] for image in images))

    def test_parse_profile_falls_back_to_body_text_birthdate(self) -> None:
        html = """
        <title>Example | seju</title>
        <meta name="description" content="Example Profile" />
        <main><p>Profile</p><p>2000年1月2日・東京都</p></main>
        <meta property="og:image" content="https://seju.tokyo/wp-content/uploads/example.jpg" />
        """
        profile, _images = parse_profile(html, "https://seju.tokyo/talents/example/")
        self.assertEqual(profile["birthdate"], "2000-01-02")

    def test_write_source_manifest_outputs_jsonl_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sources.jsonl"
            candidate = SourceCandidate(
                profile_url="https://seju.tokyo/talents/example/",
                talent_slug="example",
                name="Example",
                birthdate="2000-01-01",
                age_as_of=24,
                image_url="https://seju.tokyo/wp-content/uploads/example.jpg",
                image_kind="og:image",
                alt=None,
                eligible_for_analysis=True,
                exclusion_reason=None,
                retrieved_at=date(2026, 6, 14).isoformat(),
                source_policy="manifest_only_review_before_download",
            )
            write_source_manifest([candidate], out)
            self.assertIn("example.jpg", out.read_text(encoding="utf-8"))
            self.assertIn("Manifest-only", out.with_suffix(".audit.md").read_text(encoding="utf-8"))
            self.assertEqual(read_source_manifest(out)[0].talent_slug, "example")

    def test_download_source_images_dry_run_and_fixture_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = SourceCandidate(
                profile_url="https://seju.tokyo/talents/example/",
                talent_slug="example",
                name="Example",
                birthdate="2000-01-01",
                age_as_of=24,
                image_url="https://seju.tokyo/wp-content/uploads/example.jpg",
                image_kind="og:image",
                alt=None,
                eligible_for_analysis=True,
                exclusion_reason=None,
                retrieved_at=date(2026, 6, 14).isoformat(),
                source_policy="manifest_only_review_before_download",
            )
            dry_run = download_source_images([candidate], root / "dry", dry_run=True)
            self.assertEqual(dry_run[0].status, "planned")

            downloaded = download_source_images(
                [candidate],
                root / "raw",
                fetch_bytes=lambda _url: (b"fake-jpeg-bytes", "image/jpeg"),
            )
            self.assertEqual(downloaded[0].status, "downloaded")
            self.assertEqual(downloaded[0].bytes, len(b"fake-jpeg-bytes"))
            self.assertTrue(Path(downloaded[0].path or "").exists())
            self.assertTrue((root / "raw" / "download_manifest.jsonl").exists())

    def test_quote_url_preserves_ascii_and_encodes_japanese_path(self) -> None:
        quoted = _quote_url("https://seju.tokyo/wp-content/uploads/2023/07/秋葉聡さん撮影.jpg")
        self.assertIn("%E7%A7%8B", quoted)
        self.assertTrue(quoted.endswith(".jpg"))


if __name__ == "__main__":
    unittest.main()
