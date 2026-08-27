from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class PccPresentationContractTests(unittest.TestCase):
    def test_cli_starts_a_zero_observation_review_with_full_task_denominator(self) -> None:
        """Catches a bridge that cannot report an honest zero-image experiment start."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract = root / "contract.json"
            out = root / "out"
            contract.write_text(json.dumps(self._contract()), encoding="utf-8")
            environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "seju_face_lab",
                    "pcc-presentation-review",
                    "--contract",
                    str(contract),
                    "--out",
                    str(out),
                ],
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            summary = json.loads((out / "pcc_presentation_review.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["task_count"], 2)
            self.assertEqual(summary["mapped_count"], 0)
            self.assertEqual(summary["missing_task_count"], 2)
            self.assertFalse(summary["promotion_ready"])

    def test_cli_aggregates_pcc_observations_without_promoting_a_candidate(self) -> None:
        """Catches a missing contract bridge or one that promotes without human review."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract = root / "contract.json"
            observations = root / "observations.jsonl"
            out = root / "out"
            contract.write_text(json.dumps(self._contract()), encoding="utf-8")
            observations.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "task_id": "GEN_001",
                                "image_id": "P2",
                                "face_qa_pass": True,
                                "presentation_flags": [],
                            }
                        ),
                        json.dumps(
                            {
                                "task_id": "GEN_002",
                                "image_id": "R003",
                                "face_qa_pass": True,
                                "presentation_flags": ["low_global_contrast"],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "seju_face_lab",
                    "pcc-presentation-review",
                    "--contract",
                    str(contract),
                    "--observations",
                    str(observations),
                    "--out",
                    str(out),
                ],
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            summary = json.loads((out / "pcc_presentation_review.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["mapped_count"], 2)
            self.assertEqual(summary["face_qa_pass_count"], 2)
            self.assertEqual(summary["presentation_clear_count"], 1)
            self.assertFalse(summary["promotion_ready"])
            self.assertIn("blinded human review", summary["promotion_reason"])

    @staticmethod
    def _contract() -> dict[str, object]:
        return {
            "contract_version": "pcc-seju-presentation/v1",
            "study_id": "PCC-NH-TEST",
            "presentation_modes": ["full", "face_crop", "face_hidden"],
            "privacy_boundary": {
                "contains_raw_images": False,
                "contains_image_locations": False,
                "contains_face_embeddings": False,
                "contains_prompt_text": False,
                "contains_personal_attributes": False,
            },
            "tasks": [
                {"task_id": "GEN_001", "condition_id": "C_FULL_CORE"},
                {"task_id": "GEN_002", "condition_id": "C_SHORT_BASELINE"},
            ],
        }


if __name__ == "__main__":
    unittest.main()
