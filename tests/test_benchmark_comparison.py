from __future__ import annotations

import unittest

from app.codegen.catalog import load_codegen_tasks


class BenchmarkComparisonTest(unittest.TestCase):
    def test_active_benchmark_tasks_cover_all_enabled_registry_entries(self) -> None:
        comparable_tasks = load_codegen_tasks(included_in_main_comparison=True)
        self.assertEqual(
            {task["id"] for task in comparable_tasks},
            {
                "olymmath",
                "math-500",
                "aime-2024",
                "aime-2025",
                "aime-2026",
                "planbench",
                "arc-challenge",
                "bbh",
                "mmlu-pro",
                "spider",
                "bird",
                "chase",
                "longbench-v2",
                "incharacter",
                "characterbench",
                "socialbench",
                "timechara",
                "personamem-32k",
                "personafeedback",
                "alpsbench",
                "alpbench",
                "xstest-refusal-calibration",
                "harmbench-text-harmful",
                "jailbreakbench-harmful",
                "or-bench-hard-1k",
                "or-bench-toxic",
                "hallulens-precisewikiqa",
                "hallulens-mixedentities",
                "hallulens-longwiki",
                "longsafety",
                "sciq",
                "qasc",
                "scienceqa",
                "openbookqa",
                "gpqa-diamond",
                "livecodebench",
                "co-bench",
            },
        )
        dataset_tasks = [task for task in comparable_tasks if task.get("local_dataset_only")]
        self.assertEqual(
            {task["id"] for task in dataset_tasks},
            {
                "olymmath",
                "math-500",
                "aime-2024",
                "aime-2025",
                "aime-2026",
                "planbench",
                "arc-challenge",
                "bbh",
                "mmlu-pro",
                "spider",
                "bird",
                "chase",
                "longbench-v2",
                "incharacter",
                "characterbench",
                "socialbench",
                "timechara",
                "personamem-32k",
                "personafeedback",
                "alpsbench",
                "alpbench",
                "xstest-refusal-calibration",
                "sciq",
                "qasc",
                "scienceqa",
                "openbookqa",
                "gpqa-diamond",
                "livecodebench",
                "co-bench",
            },
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
        self.assertEqual([task["id"] for task in comparable_tasks[:5]], ["olymmath", "math-500", "aime-2024", "aime-2025", "aime-2026"])


if __name__ == "__main__":
    unittest.main()
