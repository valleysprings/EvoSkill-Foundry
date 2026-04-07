from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.codegen import catalog
from app.codegen.catalog import list_codegen_task_summaries, load_codegen_tasks


class CodegenCatalogTest(unittest.TestCase):
    def test_active_registry_uses_one_main_benchmark_lane(self) -> None:
        tasks = load_codegen_tasks()
        comparable_tasks = [task for task in tasks if task["benchmark_tier"] == "comparable"]
        main_tasks = [task for task in tasks if task["included_in_main_comparison"]]
        off_main_tasks = [task for task in tasks if not task["included_in_main_comparison"]]

        self.assertTrue(comparable_tasks)
        self.assertEqual(
            {task["id"] for task in off_main_tasks},
            set(),
        )
        self.assertEqual(
            {task["track"] for task in comparable_tasks},
            {
                "math_verified",
                "reasoning_verified",
                "text2sql_verified",
                "longcontext_verified",
                "personalization_verified",
                "safety_verified",
                "science_verified",
                "coding_verified",
                "or_verified",
            },
        )
        self.assertEqual(
            {task["runtime_backend"] for task in tasks},
            {"dataset", "benchmark_adapter"},
        )
        self.assertTrue(all(task["included_in_main_comparison"] for task in main_tasks))
        self.assertEqual([task["id"] for task in main_tasks[:5]], ["olymmath", "math-500", "aime-2024", "aime-2025", "aime-2026"])

    def test_main_comparison_filter_returns_all_active_benchmark_tasks(self) -> None:
        all_tasks = load_codegen_tasks()
        main_tasks = load_codegen_tasks(included_in_main_comparison=True)
        self.assertTrue(main_tasks)
        self.assertEqual(len(main_tasks), len(all_tasks))
        self.assertTrue(all(task["included_in_main_comparison"] for task in main_tasks))
        self.assertTrue(all(task["benchmark_tier"] == "comparable" for task in main_tasks))
        self.assertEqual(
            {task["id"] for task in all_tasks if not task["included_in_main_comparison"]},
            set(),
        )

    def test_task_summaries_include_benchmark_metadata(self) -> None:
        summaries = list_codegen_task_summaries()
        olymmath = next(task for task in summaries if task["id"] == "olymmath")
        math_500 = next(task for task in summaries if task["id"] == "math-500")
        aime_2024 = next(task for task in summaries if task["id"] == "aime-2024")
        aime_2025 = next(task for task in summaries if task["id"] == "aime-2025")
        aime_2026 = next(task for task in summaries if task["id"] == "aime-2026")
        planbench = next(task for task in summaries if task["id"] == "planbench")
        arc_challenge = next(task for task in summaries if task["id"] == "arc-challenge")
        bbh = next(task for task in summaries if task["id"] == "bbh")
        mmlu_pro = next(task for task in summaries if task["id"] == "mmlu-pro")
        longbench = next(task for task in summaries if task["id"] == "longbench-v2")
        incharacter = next(task for task in summaries if task["id"] == "incharacter")
        characterbench = next(task for task in summaries if task["id"] == "characterbench")
        timechara = next(task for task in summaries if task["id"] == "timechara")
        personamem = next(task for task in summaries if task["id"] == "personamem-32k")
        socialbench = next(task for task in summaries if task["id"] == "socialbench")
        xstest = next(task for task in summaries if task["id"] == "xstest-refusal-calibration")
        harmbench = next(task for task in summaries if task["id"] == "harmbench-text-harmful")
        jailbreakbench = next(task for task in summaries if task["id"] == "jailbreakbench-harmful")
        or_bench_hard = next(task for task in summaries if task["id"] == "or-bench-hard-1k")
        or_bench_toxic = next(task for task in summaries if task["id"] == "or-bench-toxic")
        hallulens_precise = next(task for task in summaries if task["id"] == "hallulens-precisewikiqa")
        hallulens_mixed = next(task for task in summaries if task["id"] == "hallulens-mixedentities")
        hallulens_longwiki = next(task for task in summaries if task["id"] == "hallulens-longwiki")
        longsafety = next(task for task in summaries if task["id"] == "longsafety")
        sciq = next(task for task in summaries if task["id"] == "sciq")
        qasc = next(task for task in summaries if task["id"] == "qasc")
        scienceqa = next(task for task in summaries if task["id"] == "scienceqa")
        openbookqa = next(task for task in summaries if task["id"] == "openbookqa")
        gpqa_diamond = next(task for task in summaries if task["id"] == "gpqa-diamond")
        livecodebench = next(task for task in summaries if task["id"] == "livecodebench")
        co_bench = next(task for task in summaries if task["id"] == "co-bench")
        summary_ids = {task["id"] for task in summaries}
        self.assertNotIn("contains-duplicates", summary_ids)
        self.assertNotIn("planbench-lite", summary_ids)
        self.assertNotIn("terminal-bench", summary_ids)
        self.assertNotIn("bloom-self-preferential-bias", summary_ids)
        self.assertNotIn("bloom-trait-examples", summary_ids)
        self.assertNotIn("tom-gibbs-multiturn-jailbreak", summary_ids)
        self.assertNotIn("safemtdata-benign-utility", summary_ids)
        self.assertNotIn("tau-bench-retail", summary_ids)
        self.assertNotIn("tau-bench-airline", summary_ids)
        self.assertNotIn("alfworld", summary_ids)
        self.assertNotIn("assistantbench", summary_ids)
        self.assertNotIn("gaia", summary_ids)
        self.assertNotIn("gaia2", summary_ids)
        self.assertNotIn("osworld", summary_ids)

        self.assertTrue(olymmath["local_dataset_only"])
        self.assertEqual(olymmath["dataset_size"], 100)
        self.assertEqual(olymmath["split"], "en-hard:test")
        self.assertEqual(math_500["track"], "math_verified")
        self.assertEqual(math_500["split"], "test")
        self.assertEqual(aime_2024["dataset_size"], 30)
        self.assertEqual(aime_2024["split"], "train:2024-full")
        self.assertEqual(aime_2025["dataset_size"], 30)
        self.assertEqual(aime_2025["split"], "AIME2025-I:test + AIME2025-II:test")
        self.assertEqual(aime_2026["dataset_size"], 30)
        self.assertEqual(aime_2026["split"], "test")
        self.assertTrue(planbench["local_dataset_only"])
        self.assertEqual(planbench["dataset_size"], 2270)
        self.assertEqual(planbench["track"], "reasoning_verified")
        self.assertTrue(planbench["included_in_main_comparison"])
        self.assertEqual(planbench["split"], "task_1_plan_generation:train")
        self.assertEqual(planbench["runtime_backend"], "dataset")
        self.assertEqual(planbench["task_mode"], "answer")
        self.assertEqual(planbench["interaction_mode"], "single_turn")
        self.assertEqual(planbench["optimization_scope"], "wrapper")
        self.assertEqual(planbench["selection_spec"]["profile"], "objective_only")
        self.assertEqual(arc_challenge["dataset_size"], 299)
        self.assertEqual(arc_challenge["track"], "reasoning_verified")
        self.assertEqual(arc_challenge["split"], "validation:ARC-Challenge")
        self.assertEqual(bbh["dataset_size"], 6511)
        self.assertEqual(bbh["track"], "reasoning_verified")
        self.assertEqual(bbh["split"], "train:all_configs")
        self.assertTrue(bbh["included_in_main_comparison"])
        self.assertEqual(mmlu_pro["dataset_size"], 12032)
        self.assertEqual(mmlu_pro["track"], "reasoning_verified")
        self.assertEqual(mmlu_pro["split"], "default:test")
        self.assertEqual(mmlu_pro["interaction_mode"], "single_turn")
        self.assertTrue(mmlu_pro["included_in_main_comparison"])
        self.assertEqual(longbench["dataset_size"], 503)
        self.assertEqual(longbench["track"], "longcontext_verified")
        self.assertEqual(longbench["split"], "train")
        self.assertTrue(longbench["included_in_main_comparison"])
        self.assertEqual(incharacter["track"], "personalization_verified")
        self.assertEqual(incharacter["dataset_size"], 448)
        self.assertEqual(incharacter["interaction_mode"], "single_turn")
        self.assertEqual(incharacter["task_shape"], "dialogue_judgement")
        self.assertTrue(incharacter["supports_eval_model"])
        self.assertTrue(incharacter["requires_eval_model"])
        self.assertTrue(incharacter["included_in_main_comparison"])
        self.assertEqual(characterbench["track"], "personalization_verified")
        self.assertEqual(characterbench["dataset_size"], 3250)
        self.assertEqual(characterbench["interaction_mode"], "single_turn")
        self.assertEqual(characterbench["task_shape"], "dialogue_judgement")
        self.assertTrue(characterbench["supports_eval_model"])
        self.assertTrue(characterbench["requires_eval_model"])
        self.assertTrue(characterbench["included_in_main_comparison"])
        self.assertEqual(timechara["dataset_size"], 10895)
        self.assertEqual(timechara["interaction_mode"], "single_turn")
        self.assertEqual(timechara["task_shape"], "dialogue_judgement")
        self.assertTrue(timechara["supports_eval_model"])
        self.assertTrue(timechara["requires_eval_model"])
        self.assertTrue(timechara["included_in_main_comparison"])
        self.assertEqual(personamem["track"], "personalization_verified")
        self.assertEqual(personamem["dataset_size"], 589)
        self.assertEqual(personamem["split"], "benchmark:32k")
        self.assertEqual(personamem["research_line"], "personalization")
        self.assertEqual(personamem["personalization_category"], "user_persona")
        self.assertEqual(personamem["personalization_focus"], "preference_following")
        self.assertFalse(personamem["supports_eval_model"])
        self.assertTrue(personamem["included_in_main_comparison"])
        self.assertEqual(socialbench["track"], "personalization_verified")
        self.assertEqual(socialbench["dataset_size"], 7702)
        self.assertEqual(socialbench["split"], "official:all")
        self.assertEqual(socialbench["research_line"], "personalization")
        self.assertEqual(socialbench["personalization_category"], "role_play")
        self.assertEqual(socialbench["personalization_focus"], "sociality")
        self.assertFalse(socialbench["supports_eval_model"])
        self.assertTrue(socialbench["included_in_main_comparison"])
        self.assertEqual(xstest["track"], "safety_verified")
        self.assertEqual(xstest["title"], "XSTest Refusal Calibration")
        self.assertEqual(xstest["dataset_size"], 450)
        self.assertEqual(xstest["split"], "hf:train")
        self.assertEqual(xstest["research_line"], "safety")
        self.assertEqual(xstest["safety_category"], "over_refusal")
        self.assertEqual(xstest["safety_focus"], "over_refusal")
        self.assertTrue(xstest["included_in_main_comparison"])
        self.assertEqual(harmbench["track"], "safety_verified")
        self.assertEqual(harmbench["dataset_size"], 240)
        self.assertEqual(harmbench["runtime_backend"], "benchmark_adapter")
        self.assertEqual(harmbench["interaction_mode"], "single_turn")
        self.assertEqual(harmbench["safety_category"], "jailbreak_attack")
        self.assertEqual(harmbench["safety_focus"], "jailbreak_attack")
        self.assertTrue(harmbench["included_in_main_comparison"])
        self.assertTrue(harmbench["supports_max_items"])
        self.assertEqual(harmbench["default_max_items"], 240)
        self.assertFalse(harmbench["supports_max_episodes"])
        self.assertEqual(jailbreakbench["track"], "safety_verified")
        self.assertEqual(jailbreakbench["dataset_size"], 100)
        self.assertEqual(jailbreakbench["runtime_backend"], "benchmark_adapter")
        self.assertEqual(jailbreakbench["interaction_mode"], "single_turn")
        self.assertEqual(jailbreakbench["safety_category"], "jailbreak_attack")
        self.assertEqual(jailbreakbench["safety_focus"], "jailbreak_attack")
        self.assertTrue(jailbreakbench["included_in_main_comparison"])
        self.assertTrue(jailbreakbench["supports_max_items"])
        self.assertEqual(jailbreakbench["default_max_items"], 100)
        self.assertFalse(jailbreakbench["supports_max_episodes"])
        self.assertEqual(or_bench_hard["track"], "safety_verified")
        self.assertEqual(or_bench_hard["dataset_size"], 1319)
        self.assertEqual(or_bench_hard["runtime_backend"], "benchmark_adapter")
        self.assertEqual(or_bench_hard["interaction_mode"], "single_turn")
        self.assertEqual(or_bench_hard["safety_category"], "over_refusal")
        self.assertEqual(or_bench_hard["safety_focus"], "over_refusal")
        self.assertTrue(or_bench_hard["included_in_main_comparison"])
        self.assertTrue(or_bench_hard["supports_max_items"])
        self.assertEqual(or_bench_hard["default_max_items"], 1319)
        self.assertEqual(or_bench_toxic["track"], "safety_verified")
        self.assertEqual(or_bench_toxic["dataset_size"], 655)
        self.assertEqual(or_bench_toxic["runtime_backend"], "benchmark_adapter")
        self.assertEqual(or_bench_toxic["interaction_mode"], "single_turn")
        self.assertEqual(or_bench_toxic["safety_category"], "jailbreak_attack")
        self.assertEqual(or_bench_toxic["safety_focus"], "should_refuse")
        self.assertTrue(or_bench_toxic["included_in_main_comparison"])
        self.assertTrue(or_bench_toxic["supports_max_items"])
        self.assertEqual(or_bench_toxic["default_max_items"], 655)
        self.assertEqual(hallulens_precise["track"], "safety_verified")
        self.assertEqual(hallulens_precise["dataset_size"], 250)
        self.assertEqual(hallulens_precise["interaction_mode"], "single_turn")
        self.assertEqual(hallulens_precise["safety_category"], "factuality_hallucination")
        self.assertEqual(hallulens_precise["safety_focus"], "factuality_hallucination")
        self.assertTrue(hallulens_precise["supports_max_items"])
        self.assertEqual(hallulens_precise["default_max_items"], 250)
        self.assertEqual(hallulens_mixed["track"], "safety_verified")
        self.assertEqual(hallulens_mixed["dataset_size"], 400)
        self.assertEqual(hallulens_mixed["interaction_mode"], "single_turn")
        self.assertEqual(hallulens_mixed["safety_category"], "factuality_hallucination")
        self.assertEqual(hallulens_mixed["safety_focus"], "factuality_hallucination")
        self.assertTrue(hallulens_mixed["supports_max_items"])
        self.assertEqual(hallulens_mixed["default_max_items"], 400)
        self.assertEqual(hallulens_longwiki["track"], "safety_verified")
        self.assertEqual(hallulens_longwiki["dataset_size"], 250)
        self.assertEqual(hallulens_longwiki["interaction_mode"], "single_turn")
        self.assertEqual(hallulens_longwiki["safety_category"], "factuality_hallucination")
        self.assertEqual(hallulens_longwiki["safety_focus"], "factuality_hallucination")
        self.assertTrue(hallulens_longwiki["supports_max_items"])
        self.assertEqual(hallulens_longwiki["default_max_items"], 250)
        self.assertEqual(longsafety["track"], "safety_verified")
        self.assertEqual(longsafety["dataset_size"], 1543)
        self.assertEqual(longsafety["interaction_mode"], "single_turn")
        self.assertEqual(longsafety["safety_category"], "jailbreak_attack")
        self.assertEqual(longsafety["safety_focus"], "safety_degradation")
        self.assertTrue(longsafety["supports_max_items"])
        self.assertEqual(longsafety["default_max_items"], 1543)
        self.assertEqual(sciq["dataset_size"], 1000)
        self.assertEqual(sciq["track"], "science_verified")
        self.assertEqual(sciq["split"], "validation")
        self.assertEqual(qasc["dataset_size"], 926)
        self.assertEqual(qasc["split"], "validation")
        self.assertEqual(scienceqa["dataset_size"], 768)
        self.assertEqual(scienceqa["split"], "validation:natural-science:text-only:biology-chemistry-physics")
        self.assertEqual(openbookqa["dataset_size"], 500)
        self.assertEqual(openbookqa["track"], "science_verified")
        self.assertEqual(openbookqa["split"], "validation:additional")
        self.assertEqual(gpqa_diamond["dataset_size"], 198)
        self.assertEqual(gpqa_diamond["track"], "science_verified")
        self.assertEqual(gpqa_diamond["split"], "official:diamond")
        self.assertEqual(gpqa_diamond["interaction_mode"], "single_turn")
        self.assertTrue(gpqa_diamond["included_in_main_comparison"])
        self.assertEqual(livecodebench["dataset_size"], 1055)
        self.assertEqual(livecodebench["track"], "coding_verified")
        self.assertEqual(livecodebench["split"], "release_v6:test")
        self.assertTrue(livecodebench["included_in_main_comparison"])
        self.assertEqual(livecodebench["runtime_backend"], "dataset")
        self.assertEqual(livecodebench["task_mode"], "artifact")
        self.assertEqual(livecodebench["interaction_mode"], "single_turn")
        self.assertEqual(livecodebench["optimization_scope"], "implementation")
        self.assertFalse(livecodebench["supports_runtime_config"])
        self.assertTrue(livecodebench["supports_max_items"])
        self.assertEqual(livecodebench["default_max_items"], 1055)
        self.assertEqual(co_bench["track"], "or_verified")
        self.assertTrue(co_bench["included_in_main_comparison"])
        self.assertTrue(co_bench["local_dataset_only"])
        self.assertEqual(co_bench["dataset_size"], 36)
        self.assertEqual(co_bench["split"], "official:test")
        self.assertEqual(co_bench["runtime_backend"], "dataset")
        self.assertEqual(co_bench["task_mode"], "artifact")
        self.assertEqual(co_bench["optimization_scope"], "implementation")
        self.assertFalse(co_bench["supports_runtime_config"])
        self.assertTrue(co_bench["supports_max_items"])
        self.assertEqual(co_bench["default_max_items"], 36)
        self.assertFalse(co_bench["run_baseline_verifier"])
        self.assertFalse(hallulens_precise["run_baseline_verifier"])

    def test_missing_local_benchmark_assets_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {"id": "missing-task", "path": "missing-task", "enabled": True},
                        ]
                    }
                )
            )
            with (
                patch.object(catalog, "BENCHMARK_ROOT", root),
                patch.object(catalog, "REGISTRY_PATH", registry_path),
            ):
                self.assertEqual(load_codegen_tasks(), [])
                self.assertEqual(list_codegen_task_summaries(), [])

    def test_new_personalization_tasks_are_registered_with_expected_metadata(self) -> None:
        summaries = {task["id"]: task for task in list_codegen_task_summaries()}
        expected = {
            "incharacter": ("role_play", "single_turn", "dialogue_judgement", "hybrid", 448, True),
            "characterbench": ("role_play", "single_turn", "dialogue_judgement", "judge_model", 3250, True),
            "timechara": ("role_play", "single_turn", "dialogue_judgement", "judge_model", 10895, True),
            "socialbench": ("role_play", "single_turn", "dialogue_judgement", "hybrid", 7702, True),
            "personafeedback": ("user_persona", "single_turn", "mcq", "exact_match", 8298),
            "alpsbench": ("user_persona", "single_turn", "mcq", "exact_match", 577),
            "alpbench": ("user_persona", "single_turn", "classification", "label_match", 800),
        }

        for task_id, spec in expected.items():
            if len(spec) == 6:
                category, interaction_mode, task_shape, scoring_mode, dataset_size, included = spec
            else:
                category, interaction_mode, task_shape, scoring_mode, dataset_size = spec
                included = True
            task = summaries[task_id]
            self.assertEqual(task["track"], "personalization_verified")
            self.assertEqual(task["research_line"], "personalization")
            self.assertEqual(task["personalization_category"], category)
            self.assertEqual(task["interaction_mode"], interaction_mode)
            self.assertEqual(task["task_shape"], task_shape)
            self.assertEqual(task["scoring_mode"], scoring_mode)
            self.assertEqual(task["dataset_size"], dataset_size)
            self.assertEqual(task["included_in_main_comparison"], included)

    def test_lazy_manifest_keeps_declared_dataset_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_dir = root / "coding_verified" / "lazy-livecodebench"
            data_dir = task_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {"id": "livecodebench", "path": "coding_verified/lazy-livecodebench", "enabled": True},
                        ]
                    }
                )
            )
            (task_dir / "editable.py").write_text("def solve():\n    return None\n")
            (task_dir / "verifier.py").write_text("def evaluate_candidate(**_kwargs):\n    return {}\n")
            (task_dir / "task.json").write_text(
                json.dumps(
                    {
                        "id": "livecodebench",
                        "title": "Lazy LiveCodeBench",
                        "description": "Synthetic lazy dataset task.",
                        "benchmark_tier": "comparable",
                        "track": "coding_verified",
                        "dataset_id": "livecodebench_release_v6",
                        "dataset_size": 1055,
                        "local_dataset_only": True,
                        "lazy_item_manifest": True,
                        "item_manifest": "data/questions.json",
                        "split": "release_v6:test",
                        "allow_browsing": False,
                        "answer_metric": "test_pass_rate",
                        "family": "coding",
                        "task_signature": ["dataset-task"],
                        "runtime_backend": "dataset",
                        "task_mode": "artifact",
                        "interaction_mode": "single_turn",
                        "optimization_scope": "implementation",
                        "editable_file": "editable.py",
                        "verifier": "verifier.py",
                        "entry_symbol": "solve",
                        "generation_budget": 3,
                        "candidate_budget": 2,
                        "branching_factor": 3,
                        "item_workers": 6,
                        "epsilon": 0.01,
                        "objective_spec": {
                            "display_name": "Test pass rate",
                            "direction": "max",
                            "unit": "ratio",
                            "summary_template": "Higher is better.",
                            "formula": "test_pass_rate = passed_cases / total_cases"
                        }
                    }
                )
            )
            (data_dir / "questions.json").write_text(
                json.dumps(
                    {
                        "dataset_id": "livecodebench_release_v6",
                        "dataset_size": 1055,
                        "prepared_count": 2,
                        "items": [
                            {"item_id": "item-1", "prompt": "a", "expected_answer": "ok"},
                            {"item_id": "item-2", "prompt": "b", "expected_answer": "ok"},
                        ],
                    }
                )
            )

            with (
                patch.object(catalog, "BENCHMARK_ROOT", root),
                patch.object(catalog, "REGISTRY_PATH", registry_path),
            ):
                task = next(item for item in load_codegen_tasks() if item["id"] == "livecodebench")
                self.assertEqual(task["dataset_size"], 1055)
                self.assertEqual(task["prepared_item_count"], 2)


if __name__ == "__main__":
    unittest.main()
