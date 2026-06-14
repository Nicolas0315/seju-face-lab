"""Organize downloaded images into per-talent subdirectories.

Usage:
    python scripts/organize_by_talent.py \\
        --src data/raw/seju_official \\
        --dst data/raw/seju_by_talent

Images in src are named: {talent_slug}_{digest}.{ext}
Legacy numbered files like {num}_{talent_slug}_{digest}.{ext} are also accepted.
Output: dst/{talent_slug}/{filename}
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize seju images by talent slug.")
    parser.add_argument("--src", type=Path, default=Path("data/raw/seju_official"))
    parser.add_argument("--dst", type=Path, default=Path("data/raw/seju_by_talent"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src: Path = args.src
    dst: Path = args.dst
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    images = [p for p in src.iterdir() if p.suffix.lower() in suffixes]
    print(f"source images: {len(images)}")

    by_talent: dict[str, list[Path]] = {}
    skipped = 0
    for img in sorted(images):
        slug = _talent_slug_from_filename(img)
        if slug is None:
            skipped += 1
            continue
        by_talent.setdefault(slug, []).append(img)

    print(f"talents found: {len(by_talent)}")
    print(f"skipped (unrecognized name format): {skipped}")

    for slug, paths in sorted(by_talent.items()):
        talent_dir = dst / slug
        if not args.dry_run:
            talent_dir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            dest = talent_dir / p.name
            if args.dry_run:
                print(f"  [dry] {p} -> {dest}")
            else:
                shutil.copy2(p, dest)
        print(f"  {slug}: {len(paths)} images -> {talent_dir}")

    if args.dry_run:
        print("\nDry run complete. Pass without --dry-run to copy files.")
    else:
        print(f"\nDone. Organized into: {dst}")
        print(
            "Next: seju-face-lab review-subjects --model outputs/seju_model_official "
            f"--subjects {dst} --out outputs/seju_subject_reviews"
        )


def _talent_slug_from_filename(path: Path) -> str | None:
    parts = path.stem.rsplit("_", 1)
    if len(parts) != 2 or len(parts[1]) != 10:
        return None
    slug = parts[0]
    legacy_parts = slug.split("_", 1)
    if len(legacy_parts) == 2 and legacy_parts[0].isdigit():
        return legacy_parts[1]
    return slug or None


if __name__ == "__main__":
    main()
