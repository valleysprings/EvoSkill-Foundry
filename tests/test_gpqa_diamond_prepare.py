from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.bench.benchmark_support import public_question_payload


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "benchmark" / "science_verified" / "gpqa-diamond" / "prepare.py"
VERIFIER_PATH = ROOT / "benchmark" / "science_verified" / "gpqa-diamond" / "verifier.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_prepare_module():
    return _load_module(SCRIPT_PATH, "gpqa_diamond_prepare")


verifier_module = _load_module(VERIFIER_PATH, "gpqa_diamond_verifier")
evaluate_candidate = verifier_module.evaluate_candidate


def _write_candidate(path: Path, answer_literal: str) -> None:
    path.write_text(f"def solve(question):\n    return {answer_literal}\n")


class GPQADiamondPrepareTest(unittest.TestCase):
    def test_prepare_stably_shuffles_choices_and_hides_explanations(self) -> None:
        module = _load_prepare_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_root = Path(tmp_dir)
            data_dir = task_root / "data"
            module.ROOT = task_root
            module.DATA_DIR = data_dir
            module.MANIFEST_PATH = data_dir / "questions.json"

            rows = []
            for index in range(198):
                rows.append(
                    {
                        "Question": f"Question {index}?",
                        "Correct Answer": f"Correct {index}",
                        "Incorrect Answer 1": f"Wrong A {index}",
                        "Incorrect Answer 2": f"Wrong B {index}",
                        "Incorrect Answer 3": f"Wrong C {index}",
                        "Explanation": f"Explanation {index}",
                        "Subdomain": "Physics (general)" if index % 2 == 0 else "Chemistry (general)",
                        "High-level domain": "Physics" if index % 2 == 0 else "Chemistry",
                        "Record ID": f"rec-{index:03d}",
                    }
                )

            with (
                patch.object(sys, "argv", ["prepare.py"]),
                patch.object(module, "_load_gpqa_rows", return_value=rows),
            ):
                module.main()
            first_payload = json.loads((data_dir / "questions.json").read_text())

            with (
                patch.object(sys, "argv", ["prepare.py"]),
                patch.object(module, "_load_gpqa_rows", return_value=rows),
            ):
                module.main()
            second_payload = json.loads((data_dir / "questions.json").read_text())

            self.assertEqual(first_payload["prepared_count"], 198)
            self.assertEqual(first_payload["dataset_size"], 198)
            self.assertEqual(first_payload["items"], second_payload["items"])

            first_item = first_payload["items"][0]
            self.assertEqual(len(first_item["choices"]), 4)
            self.assertIn(first_item["expected_answer"], first_item["choices"])
            self.assertEqual(
                first_item["choices"][first_item["metadata"]["correct_choice_index"]],
                first_item["expected_answer"],
            )
            self.assertNotIn("Explanation", first_item)
            self.assertNotIn("explanation", first_item["metadata"])
            self.assertNotIn("Explanation", public_question_payload(first_item))
            self.assertNotIn("Explanation 0", json.dumps(public_question_payload(first_item)))

    def test_verifier_accepts_text_label_and_index(self) -> None:
        item = {
            "item_id": "gpqa-diamond-sample",
            "expected_answer": "Correct",
            "choices": ["Wrong A", "Correct", "Wrong B", "Wrong C"],
            "metadata": {"correct_choice_index": 1, "answer_aliases": ["Correct"]},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            task = {"entry_symbol": "solve", "question_item": item}
            candidate_root = Path(tmp_dir)

            candidate_path = candidate_root / "candidate_text.py"
            _write_candidate(candidate_path, repr("Correct"))
            result = evaluate_candidate(
                task=task,
                candidate_path=candidate_path,
                source_code="",
                baseline_metrics=None,
                memory_applied=False,
            )
            self.assertEqual(result["status"], "pass")

            candidate_path = candidate_root / "candidate_label.py"
            _write_candidate(candidate_path, repr("B"))
            result = evaluate_candidate(
                task=task,
                candidate_path=candidate_path,
                source_code="",
                baseline_metrics=None,
                memory_applied=False,
            )
            self.assertEqual(result["status"], "pass")

            candidate_path = candidate_root / "candidate_index.py"
            _write_candidate(candidate_path, repr("2"))
            result = evaluate_candidate(
                task=task,
                candidate_path=candidate_path,
                source_code="",
                baseline_metrics=None,
                memory_applied=False,
            )
            self.assertEqual(result["status"], "pass")

            candidate_path = candidate_root / "candidate_wrong.py"
            _write_candidate(candidate_path, repr("Wrong A"))
            result = evaluate_candidate(
                task=task,
                candidate_path=candidate_path,
                source_code="",
                baseline_metrics=None,
                memory_applied=False,
            )
            self.assertEqual(result["status"], "fail")


if __name__ == "__main__":
    unittest.main()
