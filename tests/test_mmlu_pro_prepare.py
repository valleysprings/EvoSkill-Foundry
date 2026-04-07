from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "benchmark" / "reasoning_verified" / "mmlu-pro" / "prepare.py"
VERIFIER_PATH = ROOT / "benchmark" / "reasoning_verified" / "mmlu-pro" / "verifier.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_prepare_module():
    return _load_module(SCRIPT_PATH, "mmlu_pro_prepare")


verifier_module = _load_module(VERIFIER_PATH, "mmlu_pro_verifier")
evaluate_candidate = verifier_module.evaluate_candidate


def _write_candidate(path: Path, answer_literal: str) -> None:
    path.write_text(f"def solve(question):\n    return {answer_literal}\n")


class MMLUProPrepareTest(unittest.TestCase):
    def test_prepare_preserves_variable_length_choices_and_answer_aliases(self) -> None:
        module = _load_prepare_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_root = Path(tmp_dir)
            data_dir = task_root / "data"
            module.ROOT = task_root
            module.DATA_DIR = data_dir
            module.MANIFEST_PATH = data_dir / "questions.json"

            samples = [
                {
                    "question_id": 10,
                    "question": "Pick the correct business answer.",
                    "options": ["A1", "B1", "C1", "D1", "E1"],
                    "answer": "E",
                    "answer_index": 4,
                    "category": "business",
                    "src": "sample-business",
                },
                {
                    "question_id": 11,
                    "question": "Pick the correct math answer.",
                    "options": ["A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2", "I2", "J2"],
                    "answer": "J",
                    "answer_index": 9,
                    "category": "math",
                    "src": "sample-math",
                },
            ]

            with (
                patch.object(sys, "argv", ["prepare.py", "--items", "2"]),
                patch.object(module, "load_dataset", return_value=samples),
            ):
                module.main()

            payload = json.loads((data_dir / "questions.json").read_text())
            self.assertEqual(payload["prepared_count"], 2)
            self.assertEqual(payload["dataset_size"], 12032)

            first_item = payload["items"][0]
            self.assertEqual(first_item["choices"], ["A1", "B1", "C1", "D1", "E1"])
            self.assertEqual(first_item["expected_answer"], "E1")
            self.assertEqual(first_item["metadata"]["correct_choice_index"], 4)
            self.assertEqual(first_item["metadata"]["answer_aliases"], ["E1", "E"])

            second_item = payload["items"][1]
            self.assertEqual(len(second_item["choices"]), 10)
            self.assertEqual(second_item["expected_answer"], "J2")
            self.assertEqual(second_item["metadata"]["correct_choice_index"], 9)
            self.assertEqual(second_item["metadata"]["answer_aliases"], ["J2", "J"])

    def test_verifier_accepts_text_label_and_index(self) -> None:
        item = {
            "item_id": "mmlu-pro-sample",
            "expected_answer": "J2",
            "choices": ["A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2", "I2", "J2"],
            "metadata": {"correct_choice_index": 9, "answer_aliases": ["J2", "J"]},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            task = {"entry_symbol": "solve", "question_item": item}
            candidate_root = Path(tmp_dir)

            candidate_path = candidate_root / "candidate_text.py"
            _write_candidate(candidate_path, repr("J2"))
            result = evaluate_candidate(
                task=task,
                candidate_path=candidate_path,
                source_code="",
                baseline_metrics=None,
                memory_applied=False,
            )
            self.assertEqual(result["status"], "pass")

            candidate_path = candidate_root / "candidate_label.py"
            _write_candidate(candidate_path, repr("J"))
            result = evaluate_candidate(
                task=task,
                candidate_path=candidate_path,
                source_code="",
                baseline_metrics=None,
                memory_applied=False,
            )
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["test_results"][0]["actual_display"], "J -> J2")

            candidate_path = candidate_root / "candidate_index.py"
            _write_candidate(candidate_path, repr("10"))
            result = evaluate_candidate(
                task=task,
                candidate_path=candidate_path,
                source_code="",
                baseline_metrics=None,
                memory_applied=False,
            )
            self.assertEqual(result["status"], "pass")

            candidate_path = candidate_root / "candidate_wrong.py"
            _write_candidate(candidate_path, repr("A2"))
            result = evaluate_candidate(
                task=task,
                candidate_path=candidate_path,
                source_code="",
                baseline_metrics=None,
                memory_applied=False,
            )
            self.assertEqual(result["status"], "fail")

    def test_verifier_truncates_verbose_unparsed_response_display(self) -> None:
        item = {
            "item_id": "mmlu-pro-sample",
            "expected_answer": "J2",
            "choices": ["A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2", "I2", "J2"],
            "metadata": {"correct_choice_index": 9, "answer_aliases": ["J2", "J"]},
        }
        verbose_response = (
            "Answer 当然可以！通过分析文章的语气和文字表现，我可以帮助您提炼作者的个人形象、叙事风格，以及主题核心。"
            "针对您的职业背景和写作习惯，我建议您附上文章具体段落中包含的重要对话或情节发展，这样我们可以更细腻地解析作者。"
            "此外，根据您热衷对角色塑造进行深层次研究的习惯，建议重点关注作者是否通过细腻的心理刻画或表征物件传递观点。"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            task = {"entry_symbol": "solve", "question_item": item}
            candidate_path = Path(tmp_dir) / "candidate_verbose.py"
            _write_candidate(candidate_path, repr(verbose_response))
            result = evaluate_candidate(
                task=task,
                candidate_path=candidate_path,
                source_code="",
                baseline_metrics=None,
                memory_applied=False,
            )

        self.assertEqual(result["status"], "fail")
        actual_display = result["test_results"][0]["actual_display"]
        self.assertTrue(actual_display.startswith("Answer"))
        self.assertLess(len(actual_display), len(verbose_response))
        self.assertTrue(actual_display.endswith("..."))


if __name__ == "__main__":
    unittest.main()
