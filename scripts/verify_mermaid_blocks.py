from __future__ import annotations

import argparse
from pathlib import Path


def extract_mermaid_blocks(path: Path, out_dir: Path) -> list[Path]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[list[str]] = []
    current: list[str] | None = None

    for line in lines:
        if current is None:
            if line.strip() == "```mermaid":
                current = []
            continue
        if line.strip() == "```":
            if not any(item.strip() for item in current):
                raise ValueError("empty mermaid block")
            blocks.append(current)
            current = None
            continue
        current.append(line)

    if current is not None:
        raise ValueError("unclosed mermaid block")
    if not blocks:
        raise ValueError("no mermaid blocks found")

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, block in enumerate(blocks, start=1):
        output = out_dir / f"diagram-{index:02d}.mmd"
        output.write_text("\n".join(block) + "\n", encoding="utf-8")
        paths.append(output)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract Mermaid blocks for parser checks")
    parser.add_argument("markdown", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    outputs = extract_mermaid_blocks(args.markdown, args.out)
    print(f"mermaid_blocks={len(outputs)}")
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
