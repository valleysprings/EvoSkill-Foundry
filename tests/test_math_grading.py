from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.bench.math_grading import evaluate_math_dataset_candidate


def _write_candidate(path: Path, answer_literal: str) -> None:
    path.write_text(f"def solve(question):\n    return {answer_literal}\n")


class MathGradingTest(unittest.TestCase):
    def test_choice_answers_include_compact_display_text(self) -> None:
        item = {
            "item_id": "math-choice-sample",
            "expected_answer": "5",
            "choices": ["3", "5", "7", "9"],
            "metadata": {
                "answer_format": "choice",
                "correct_choice_index": 1,
                "answer_aliases": ["5", "B"],
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            candidate_path = Path(tmp_dir) / "candidate.py"
            _write_candidate(candidate_path, repr("B"))
            result = evaluate_math_dataset_candidate(
                task={"entry_symbol": "solve", "question_item": item},
                candidate_path=candidate_path,
                source_code="",
                baseline_metrics=None,
                memory_applied=False,
            )

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["test_results"][0]["actual_display"], "B -> 5")


if __name__ == "__main__":
    unittest.main()
