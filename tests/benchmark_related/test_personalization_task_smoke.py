from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.helpers import runnable_personalization_task_ids


ROOT = Path(__file__).resolve().parents[2]
PREPARE_SPECS = {
    "incharacter": {"items": 448},
    "characterbench": {"items": 3250},
    "personafeedback": {"items": 8298},
    "alpsbench-extraction": {"items": 466},
    "alpsbench-update": {"items": 469},
    "alpsbench-retrieval": {"items": 2380},
    "alpsbench-utilization": {"items": 577},
    "alpbench": {
        "items": 800,
        "fake_rows": [
            {
                "inputs_1y": "- cuisine: ['Italian', 'Japanese']\n- budget: ['cheap', 'expensive']",
                "ground-truth": "{'cuisine': 'Italian', 'budget': 'cheap'}",
            },
            {
                "inputs_1y": "- cuisine: ['Thai', 'Mexican']\n- budget: ['low', 'high']",
                "ground-truth": "{'cuisine': 'Mexican', 'budget': 'high'}",
            },
        ],
    },
    "timechara": {
        "items": 10895,
        "fake_rows": [
            {
                "series": "harry_potter",
                "character": "Harry Potter",
                "character_period": "1st-year/on halloween",
                "question": "What do you think is happening in the castle tonight?",
                "data_type": "future",
                "temporal_label": "Future: Harry should not know the full future outcome yet.",
                "spatial_label": "-",
            },
            {
                "series": "the_lord_of_the_rings",
                "character": "Frodo Baggins",
                "character_period": "1/end of the scene",
                "question": "How are you feeling about the road ahead?",
                "data_type": "past-only",
                "temporal_label": "Past: Frodo should only reference events already lived through.",
                "spatial_label": "-",
            },
        ],
    },
}
DETERMINISTIC_VERIFIER_SPECS = {
    "personafeedback": {"illegal_output": "option z"},
    "alpsbench-extraction": {"illegal_output": "option z"},
    "alpsbench-update": {"illegal_output": "option z"},
    "alpsbench-retrieval": {"illegal_output": "option z"},
    "alpsbench-utilization": {"illegal_output": "option z"},
    "alpbench": {"illegal_output": "mystery_label"},
}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_file(root: Path, body: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "candidate.py"
    path.write_text(body)
    return path


def _configured_prepare_module(task_id: str, *, suffix: str):
    module = _load_module(ROOT / "benchmark" / "personalization_verified" / task_id / "prepare.py", f"{task_id}_{suffix}")
    spec = PREPARE_SPECS[task_id]
    fake_rows = spec.get("fake_rows")
    if fake_rows is not None:
        if hasattr(module, "_load_rows"):
            module._load_rows = lambda: list(fake_rows)
        else:
            raise AssertionError(f"{task_id} prepare module does not expose _load_rows for test patching.")
    return module


def _incharacter_choices_for_labels(item: dict[str, object], *, wrong: bool = False) -> dict[str, dict[str, float]]:
    raw_context = dict(item.get("raw_context") or {})
    metadata = dict(item.get("metadata") or {})
    questions = [question for question in list(raw_context.get("questions") or []) if isinstance(question, dict)]
    questionnaire_name = str(metadata.get("questionnaire") or "").strip()
    labels = {
        str(dimension).strip(): str((detail or {}).get("type") or "").strip().upper()
        for dimension, detail in dict(metadata.get("annotation_labels") or {}).items()
        if str(dimension).strip()
    }
    low, high = [float(value) for value in list(metadata.get("questionnaire_range") or [1, 5])[:2]]
    midpoint = (low + high) / 2.0
    choices: dict[str, dict[str, float]] = {}
    for question in questions:
        question_id = str(question.get("id") or "").strip()
        dimension = str(question.get("dimension") or "").strip()
        category = str(question.get("category") or "").strip()
        label = labels[dimension]
        if wrong:
            if label == "H":
                label = "L"
            elif label == "L":
                label = "H"
            else:
                label = "H"
        positive = category == dimension[0] if questionnaire_name == "16Personalities" else category != "negative"
        if label == "X":
            choice = midpoint
        elif label == "H":
            choice = high if positive else low
        else:
            choice = low if positive else high
        choices.setdefault(dimension, {})[question_id] = choice
    return choices


class PersonalizationTaskSmokeTest(unittest.TestCase):
    def test_default_smoke_targets_are_active_runnable_tasks(self) -> None:
        self.assertTrue(set(PREPARE_SPECS).issubset(runnable_personalization_task_ids()))
        self.assertTrue(set(DETERMINISTIC_VERIFIER_SPECS).issubset(runnable_personalization_task_ids()))

    def test_prepare_scripts_materialize_expected_manifest_prefix(self) -> None:
        for task_id, spec in PREPARE_SPECS.items():
            with self.subTest(task_id=task_id):
                module = _configured_prepare_module(task_id, suffix="prepare")
                with tempfile.TemporaryDirectory() as tmp_dir:
                    manifest_path = Path(tmp_dir) / "questions.json"
                    module.MANIFEST_PATH = manifest_path
                    module.main(["--items", "2"])
                    payload = json.loads(manifest_path.read_text())
                    self.assertEqual(payload["prepared_count"], 2)
                    self.assertEqual(payload["dataset_size"], spec["items"])
                    self.assertEqual(len(payload["items"]), 2)

    def test_deterministic_verifiers_cover_correct_wrong_missing_and_illegal_outputs(self) -> None:
        for task_id, spec in DETERMINISTIC_VERIFIER_SPECS.items():
            with self.subTest(task_id=task_id):
                prepare_module = _configured_prepare_module(task_id, suffix="prepare_verifier")
                verifier_module = _load_module(
                    ROOT / "benchmark" / "personalization_verified" / task_id / "verifier.py",
                    f"{task_id}_verifier",
                )
                item = prepare_module._build_items()[0]

                with tempfile.TemporaryDirectory() as tmp_dir:
                    workspace = Path(tmp_dir)
                    correct_path = _candidate_file(
                        workspace / "correct",
                        f"def solve(question):\n    return {item['expected_answer']!r}\n",
                    )
                    wrong_path = _candidate_file(
                        workspace / "wrong",
                        "def solve(question):\n    return 'definitely wrong'\n",
                    )
                    illegal_path = _candidate_file(
                        workspace / "illegal",
                        f"def solve(question):\n    return {spec['illegal_output']!r}\n",
                    )
                    task = {"entry_symbol": "solve", "question_item": item}

                    correct = verifier_module.evaluate_candidate(
                        task=task,
                        candidate_path=correct_path,
                        source_code="",
                        baseline_metrics=None,
                        memory_applied=False,
                    )
                    self.assertEqual(correct["status"], "pass")
                    self.assertEqual(correct["objective"], 1.0)

                    wrong = verifier_module.evaluate_candidate(
                        task=task,
                        candidate_path=wrong_path,
                        source_code="",
                        baseline_metrics=None,
                        memory_applied=False,
                    )
                    self.assertIn(wrong["status"], {"pass", "fail"})
                    self.assertLess(wrong["objective"], 1.0)

                    illegal = verifier_module.evaluate_candidate(
                        task=task,
                        candidate_path=illegal_path,
                        source_code="",
                        baseline_metrics=None,
                        memory_applied=False,
                    )
                    self.assertEqual(illegal["status"], "fail")
                    self.assertEqual(illegal["objective"], 0.0)

                    with self.assertRaises(ValueError):
                        verifier_module.evaluate_candidate(
                            task={"entry_symbol": "solve"},
                            candidate_path=correct_path,
                            source_code="",
                            baseline_metrics=None,
                            memory_applied=False,
                        )

    def test_incharacter_verifier_supports_mocked_conversion(self) -> None:
        prepare_module = _configured_prepare_module("incharacter", suffix="prepare_verifier")
        verifier_module = _load_module(
            ROOT / "benchmark" / "personalization_verified" / "incharacter" / "verifier.py",
            "incharacter_verifier",
        )
        item = prepare_module._build_items()[0]
        expected_labels = {
            str(dimension).strip(): str((detail or {}).get("type") or "").strip().upper()
            for dimension, detail in dict(item["metadata"]["annotation_labels"]).items()
        }
        correct_choices = _incharacter_choices_for_labels(item)
        wrong_choices = _incharacter_choices_for_labels(item, wrong=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            correct_path = _candidate_file(
                workspace / "correct",
                "def solve(question):\n    return {entry['id']: 'I answer in character.' for entry in question['raw_context']['questions']}\n",
            )
            wrong_path = _candidate_file(
                workspace / "wrong",
                "def solve(question):\n    return {entry['id']: 'I refuse to stay in role.' for entry in question['raw_context']['questions']}\n",
            )
            missing_path = _candidate_file(
                workspace / "missing",
                "def solve(question):\n    first = question['raw_context']['questions'][0]['id']\n    return {first: 'Only one answer'}\n",
            )
            illegal_path = _candidate_file(workspace / "illegal", "def solve(question):\n    return 'not json'\n")
            task = {"entry_symbol": "solve", "question_item": item, "eval_model": "gpt-4.1-mini"}

            def correct_eval(**kwargs):
                dimension = kwargs["purpose"].split(":")[-1]
                return correct_choices[dimension], {"model": "judge-a", "dimension": dimension}

            with patch.object(verifier_module, "run_eval_model_json", side_effect=correct_eval):
                correct = verifier_module.evaluate_candidate(
                    task=task,
                    candidate_path=correct_path,
                    source_code="",
                    baseline_metrics=None,
                    memory_applied=False,
                )
            self.assertEqual(correct["status"], "pass")
            self.assertEqual(correct["test_results"][0]["actual"], expected_labels)
            self.assertEqual(correct["objective"], 1.0)
            self.assertEqual(correct["test_results"][0]["full_profile_accuracy"], 1.0)

            def wrong_eval(**kwargs):
                dimension = kwargs["purpose"].split(":")[-1]
                return wrong_choices[dimension], {"model": "judge-b", "dimension": dimension}

            with patch.object(verifier_module, "run_eval_model_json", side_effect=wrong_eval):
                wrong = verifier_module.evaluate_candidate(
                    task=task,
                    candidate_path=wrong_path,
                    source_code="",
                    baseline_metrics=None,
                    memory_applied=False,
                )
            self.assertEqual(wrong["status"], "pass")
            self.assertNotEqual(wrong["test_results"][0]["actual"], expected_labels)
            self.assertLess(wrong["objective"], correct["objective"])

            with patch.object(verifier_module, "run_eval_model_json", side_effect=correct_eval):
                missing = verifier_module.evaluate_candidate(
                    task=task,
                    candidate_path=missing_path,
                    source_code="",
                    baseline_metrics=None,
                    memory_applied=False,
                )
            self.assertEqual(missing["status"], "fail")
            self.assertEqual(missing["objective"], 0.0)

            illegal = verifier_module.evaluate_candidate(
                task=task,
                candidate_path=illegal_path,
                source_code="",
                baseline_metrics=None,
                memory_applied=False,
            )
            self.assertEqual(illegal["status"], "fail")
            self.assertEqual(illegal["objective"], 0.0)

    def test_alpbench_label_display_uses_label_plus_combo_preview(self) -> None:
        prepare_module = _configured_prepare_module("alpbench", suffix="prepare_display")
        verifier_module = _load_module(
            ROOT / "benchmark" / "personalization_verified" / "alpbench" / "verifier.py",
            "alpbench_display_verifier",
        )
        item = prepare_module._build_items()[0]
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            candidate_path = _candidate_file(
                workspace / "candidate",
                f"def solve(question):\n    return {item['expected_answer']!r}\n",
            )
            task = {"entry_symbol": "solve", "question_item": item}
            result = verifier_module.evaluate_candidate(
                task=task,
                candidate_path=candidate_path,
                source_code="",
                baseline_metrics=None,
                memory_applied=False,
            )
        self.assertEqual(result["status"], "pass")
        actual_display = result["test_results"][0]["actual_display"]
        self.assertIn(str(item["expected_answer"]), actual_display)
        self.assertIn("->", actual_display)
        self.assertIn("cuisine=Italian", actual_display)
        self.assertIn("budget=cheap", actual_display)

    def test_alpbench_label_display_truncates_verbose_unparsed_output(self) -> None:
        prepare_module = _configured_prepare_module("alpbench", suffix="prepare_verbose_display")
        verifier_module = _load_module(
            ROOT / "benchmark" / "personalization_verified" / "alpbench" / "verifier.py",
            "alpbench_verbose_display_verifier",
        )
        item = prepare_module._build_items()[0]
        verbose_response = (
            "Answer 当然可以！通过分析文章的语气和文字表现，我可以帮助您提炼作者的个人形象、叙事风格，以及主题核心。"
            "针对您的职业背景和写作习惯，我建议您附上文章具体段落中包含的重要对话或情节发展，这样我们可以更细腻地解析作者。"
            "此外，根据您热衷对角色塑造进行深层次研究的习惯，建议重点关注作者是否通过细腻的心理刻画或表征物件传递观点。"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            candidate_path = _candidate_file(
                workspace / "candidate",
                f"def solve(question):\n    return {verbose_response!r}\n",
            )
            task = {"entry_symbol": "solve", "question_item": item}
            result = verifier_module.evaluate_candidate(
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

    def test_characterbench_verifier_supports_mocked_judging(self) -> None:
        prepare_module = _configured_prepare_module("characterbench", suffix="prepare_verifier")
        verifier_module = _load_module(
            ROOT / "benchmark" / "personalization_verified" / "characterbench" / "verifier.py",
            "characterbench_verifier",
        )
        item = prepare_module._build_items()[0]
        max_score = float(item["metadata"]["subset_max_annotation_score"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            candidate_path = _candidate_file(
                workspace / "candidate",
                "def solve(question):\n    return 'I will respond in character and keep the dialogue grounded.'\n",
            )
            task = {"entry_symbol": "solve", "question_item": item, "eval_model": "gpt-4.1-mini"}
            with patch.object(verifier_module, "run_eval_model_json", return_value=({"score": max_score}, {"model": "judge-a"})):
                result = verifier_module.evaluate_candidate(
                    task=task,
                    candidate_path=candidate_path,
                    source_code="",
                    baseline_metrics=None,
                    memory_applied=False,
                )
            self.assertEqual(result["status"], "pass")
            self.assertAlmostEqual(result["objective"], 1.0, places=6)

    def test_timechara_verifier_supports_mocked_spatiotemporal_judging(self) -> None:
        prepare_module = _configured_prepare_module("timechara", suffix="prepare_verifier")
        verifier_module = _load_module(
            ROOT / "benchmark" / "personalization_verified" / "timechara" / "verifier.py",
            "timechara_verifier",
        )
        item = prepare_module._build_items()[0]

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            candidate_path = _candidate_file(
                workspace / "candidate",
                "def solve(question):\n    return 'I answer with only timeline-appropriate knowledge.'\n",
            )
            task = {"entry_symbol": "solve", "question_item": item, "eval_model": "gpt-4.1-mini"}
            with patch.object(verifier_module, "run_eval_model_json", return_value=({"score": 1, "reasoning": "ok"}, {"model": "judge-a"})):
                result = verifier_module.evaluate_candidate(
                    task=task,
                    candidate_path=candidate_path,
                    source_code="",
                    baseline_metrics=None,
                    memory_applied=False,
                )
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["objective"], 1.0)

if __name__ == "__main__":
    unittest.main()
