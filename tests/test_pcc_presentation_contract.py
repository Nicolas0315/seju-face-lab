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
    def test_cli_rejects_contract_task_fields_outside_the_public_schema(self) -> None:
        """Catches private prompt or attribute data hidden in task metadata."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract = self._contract()
            contract["tasks"][0]["prompt_text"] = "private prompt"  # type: ignore[index]
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            out = root / "out"

            result = self._run_review(contract_path, None, out)

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out.exists())

    def test_cli_rejects_extra_privacy_boundary_fields(self) -> None:
        """Catches a path or other private artifact hidden beside boolean declarations."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract = self._contract()
            contract["privacy_boundary"]["image_path"] = r"C:\private\portrait.png"  # type: ignore[index]
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            out = root / "out"

            result = self._run_review(contract_path, None, out)

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out.exists())

    def test_cli_rejects_path_like_image_ids_and_undeclared_observation_fields(self) -> None:
        """Catches image paths or vectors entering the anonymous observation channel."""
        cases = [
            {"task_id": "GEN_000000000001", "image_id": r"C:\private\portrait.png", "face_qa_pass": True, "presentation_flags": []},
            {"task_id": "GEN_000000000001", "image_id": "P2", "face_qa_pass": True, "presentation_flags": [], "face_vector": [0.1]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(self._contract()), encoding="utf-8")
            for index, observation in enumerate(cases):
                with self.subTest(observation=index):
                    observations = root / f"observations_{index}.jsonl"
                    observations.write_text(json.dumps(observation) + "\n", encoding="utf-8")
                    out = root / f"out_{index}"
                    result = self._run_review(contract_path, observations, out)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(out.exists())

    def test_cli_rejects_presentation_flags_outside_the_defined_vocabulary(self) -> None:
        """Catches free-text annotations being re-exported as candidate metadata."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(self._contract()), encoding="utf-8")
            observations = root / "observations.jsonl"
            observations.write_text(
                json.dumps(
                    {
                        "task_id": "GEN_000000000001",
                        "image_id": "P2",
                        "face_qa_pass": True,
                        "presentation_flags": ["looks like a specific person"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            out = root / "out"

            result = self._run_review(contract_path, observations, out)

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out.exists())

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
                                "task_id": "GEN_000000000001",
                                "image_id": "P2",
                                "face_qa_pass": True,
                                "presentation_flags": [],
                            }
                        ),
                        json.dumps(
                            {
                                "task_id": "GEN_000000000002",
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
            "study_id": "PCC-NH-01",
            "study_version": "1.0.0",
            "presentation_modes": ["full", "face_crop", "face_hidden"],
            "privacy_boundary": {
                "contains_raw_images": False,
                "contains_image_locations": False,
                "contains_face_embeddings": False,
                "contains_prompt_text": False,
                "contains_personal_attributes": False,
            },
            "tasks": [
                {
                    "task_id": "GEN_000000000001",
                    "stage": "A",
                    "model_family": "imagen_family",
                    "culture_cell": "japan",
                    "language_cell": "english",
                    "condition_id": "C_FULL_CORE",
                    "seed_requested": "7",
                    "condition_type": "control",
                    "factors": {},
                    "prompt_sha256": "a" * 64,
                },
                {
                    "task_id": "GEN_000000000002",
                    "stage": "A",
                    "model_family": "imagen_family",
                    "culture_cell": "japan",
                    "language_cell": "english",
                    "condition_id": "C_SHORT_BASELINE",
                    "seed_requested": "8",
                    "condition_type": "control",
                    "factors": {},
                    "prompt_sha256": "b" * 64,
                },
            ],
        }

    @staticmethod
    def _run_review(
        contract: Path, observations: Path | None, out: Path
    ) -> subprocess.CompletedProcess[str]:
        environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
        args = [
            sys.executable,
            "-m",
            "seju_face_lab",
            "pcc-presentation-review",
            "--contract",
            str(contract),
            "--out",
            str(out),
        ]
        if observations is not None:
            args.extend(["--observations", str(observations)])
        return subprocess.run(
            args,
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )


if __name__ == "__main__":
    unittest.main()
