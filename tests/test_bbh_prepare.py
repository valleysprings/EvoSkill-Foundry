from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.reasoning_verified.bbh.verifier import _match_answer


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "benchmark" / "reasoning_verified" / "bbh" / "prepare.py"


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location("bbh_prepare", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BBHPrepareTest(unittest.TestCase):
    def test_prepare_normalizes_choice_and_short_answer_items(self) -> None:
        module = _load_prepare_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_root = Path(tmp_dir)
            data_dir = task_root / "data"

            module.ROOT = task_root
            module.DATA_DIR = data_dir
            module.MANIFEST_PATH = data_dir / "questions.json"

            samples = {
                "movie_recommendation": [
                    {
                        "input": (
                            "Find a movie similar to Batman, The Mask, The Fugitive, Pretty Woman:\n"
                            "Options:\n"
                            "(A) The Front Page\n"
                            "(B) Maelstrom\n"
                            "(C) The Lion King\n"
                            "(D) Lamerica"
                        ),
                        "target": "(C)",
                    }
                ],
                "boolean_expressions": [
                    {
                        "input": "not ( True ) and ( True ) is",
                        "target": "False",
                    }
                ],
            }

            with (
                patch.object(
                    sys,
                    "argv",
                    ["prepare.py", "--subset", "movie_recommendation", "--subset", "boolean_expressions", "--items", "2"],
                ),
                patch.object(module, "_load_subset", side_effect=lambda config: samples[config]),
            ):
                module.main()

            payload = json.loads((data_dir / "questions.json").read_text())
            self.assertEqual(payload["prepared_count"], 2)
            self.assertEqual(payload["configs"], ["movie_recommendation", "boolean_expressions"])

            choice_item = payload["items"][0]
            self.assertEqual(choice_item["expected_answer"], "(C)")
            self.assertEqual(choice_item["choices"][2], "The Lion King")
            self.assertEqual(choice_item["metadata"]["correct_choice_index"], 2)
            self.assertEqual(choice_item["metadata"]["correct_choice_text"], "The Lion King")
            self.assertEqual(choice_item["metadata"]["answer_format"], "choice")
            self.assertEqual(choice_item["metadata"]["answer_aliases"], ["(C)"])

            short_answer_item = payload["items"][1]
            self.assertEqual(short_answer_item["expected_answer"], "False")
            self.assertEqual(short_answer_item["metadata"]["answer_format"], "short_text")
            self.assertNotIn("choices", short_answer_item)

    def test_verifier_requires_direct_official_answer(self) -> None:
        choice_item = {
            "expected_answer": "(C)",
            "choices": ["The Front Page", "Maelstrom", "The Lion King", "Lamerica"],
            "metadata": {"answer_aliases": ["(C)"], "correct_choice_index": 2, "correct_choice_text": "The Lion King"},
        }
        numeric_item = {
            "expected_answer": "24",
            "metadata": {"answer_aliases": ["24"]},
        }
        self.assertEqual(_match_answer(choice_item, "(C)"), (True, "(c)"))
        self.assertEqual(_match_answer(choice_item, "So the answer is (C)."), (False, "so the answer is (c)."))
        self.assertEqual(_match_answer(choice_item, "The Lion King"), (False, "the lion king"))
        self.assertEqual(_match_answer(numeric_item, "24"), (True, "24"))
        self.assertEqual(_match_answer(numeric_item, "Therefore, 24"), (False, "therefore, 24"))


if __name__ == "__main__":
    unittest.main()
