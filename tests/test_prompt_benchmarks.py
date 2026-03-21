from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "fetch_prompt_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("fetch_prompt_benchmarks", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PromptBenchmarkNormalizationTest(unittest.TestCase):
    def test_banking77_normalization(self) -> None:
        record = MODULE.normalize_record(
            "banking77_small",
            {
                "text": "I need to activate my card",
                "label": 4,
                "label_text": "activate_my_card",
            },
        )
        self.assertEqual(record["input"], "I need to activate my card")
        self.assertEqual(record["target"], "activate_my_card")
        self.assertEqual(record["metadata"]["label_id"], 4)

    def test_boolq_normalization(self) -> None:
        record = MODULE.normalize_record(
            "boolq_small",
            {
                "question": "Is the sky blue?",
                "passage": "Rayleigh scattering makes the sky appear blue.",
                "answer": True,
            },
        )
        self.assertEqual(record["input"]["question"], "Is the sky blue?")
        self.assertEqual(record["target"], "yes")

    def test_ag_news_normalization(self) -> None:
        record = MODULE.normalize_record(
            "ag_news_small",
            {
                "label": 2,
                "title": "Final Score",
                "description": "The home team won in overtime.",
            },
        )
        self.assertEqual(record["target"], "sports")
        self.assertEqual(record["input"]["title"], "Final Score")
        self.assertIn("overtime", record["input"]["article"])


if __name__ == "__main__":
    unittest.main()
