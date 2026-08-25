from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.verify_mermaid_blocks import extract_mermaid_blocks


class VectorModelV1Tests(unittest.TestCase):
    def test_extract_mermaid_blocks_rejects_unclosed_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "bad.md"
            source.write_text("```mermaid\nflowchart TD\nA-->B\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unclosed mermaid block"):
                extract_mermaid_blocks(source, root / "out")


if __name__ == "__main__":
    unittest.main()
