from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARE_PATH = ROOT / "benchmark" / "personalization_verified" / "socialbench" / "prepare.py"
VERIFIER_PATH = ROOT / "benchmark" / "personalization_verified" / "socialbench" / "verifier.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SocialBenchAlignmentTest(unittest.TestCase):
    def test_prepare_uses_upstream_choice_prompt_and_insertion_order(self) -> None:
        module = _load_module(PREPARE_PATH, "socialbench_prepare")
        prompt, choices = module._format_prompt(
            {
                "dialogue": [{"from": "User", "value": "Which reply fits you best?"}],
                "choices": {"B": "Pick the second response.", "A": "Pick the first response."},
                "meta": {
                    "lang": "en",
                    "category": "Individual-SA-RoleStyle",
                    "name": "Aiden Callaway",
                    "profile": {"Aiden Callaway": "Driven and direct."},
                },
            }
        )

        self.assertEqual(choices, ["Pick the second response.", "Pick the first response."])
        self.assertIn("please choose the best option (A, B, C, or D):", prompt)
        self.assertIn("You are playing the role of Aiden Callaway, you need to embody the knowledge and style of Aiden Callaway.", prompt)

    def test_prepare_normalizes_released_memory_categories_to_upstream_open_prompt(self) -> None:
        module = _load_module(PREPARE_PATH, "socialbench_prepare_memory")
        prompt, choices = module._format_prompt(
            {
                "dialogue": [{"from": "User", "value": "Can you remind me of your favorite football team?"}],
                "meta": {
                    "lang": "en",
                    "category": "Individual-MEM-Long",
                    "name": "Aiden Callaway",
                    "profile": {"Aiden Callaway": "Driven and direct."},
                },
            }
        )

        self.assertEqual(choices, [])
        self.assertIn(
            "you must produce a reply as the Assistant to response to the latest User's message (one term is enough):",
            prompt,
        )

    def test_verifier_scores_memory_aliases_with_upstream_keyword_matching(self) -> None:
        module = _load_module(VERIFIER_PATH, "socialbench_verifier")

        self.assertEqual(
            module._score_prediction(
                "I've always been a supporter of the New England Patriots.",
                ["New", "England", "Patriots"],
                "Individual-MEM-Long",
            ),
            1.0,
        )
        self.assertIsNone(module._score_prediction("", ["Patriots"], "Individual-MEM-Short"))


if __name__ == "__main__":
    unittest.main()
