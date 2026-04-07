from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.helpers import ROOT, make_runtime


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeAlfworldEnv:
    def reset(self):
        return ["Kitchen. Your task is to: cool the apple"], {"admissible_commands": [["cool apple"]], "won": [False], "goal_condition_success_rate": [0.0]}

    def step(self, commands):
        command = commands[0]
        if command == "cool apple":
            return ["Done."], [1.0], [True], {"admissible_commands": [[]], "won": [True], "goal_condition_success_rate": [1.0]}
        return ["Still working."], [0.0], [False], {"admissible_commands": [["cool apple"]], "won": [False], "goal_condition_success_rate": [0.0]}

    def close(self):
        return None


class AgentVerifiedOfficialEnvTest(unittest.TestCase):
    def test_alfworld_official_suite_uses_env_success(self) -> None:
        module = _load_module(
            "test_alfworld_verifier",
            ROOT / "benchmark" / "agent_verified" / "alfworld" / "verifier.py",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            candidate_path = Path(tmp_dir) / "editable.py"
            candidate_path.write_text(
                "def step(turn: dict, runtime) -> dict:\n"
                "    del runtime\n"
                "    return {\n"
                "        'message': 'act',\n"
                "        'tool_calls': [{'name': 'act', 'arguments': {'command': 'cool apple'}}],\n"
                "        'done': False,\n"
                "        'state': dict(turn.get('state') or {}),\n"
                "        'annotations': {},\n"
                "    }\n"
            )
            with (
                patch.object(module, "_episode_specs_for_run", return_value=[{"episode_id": "ep-1", "instruction": "cool the apple", "game_file": "/tmp/game.tw-pddl", "traj_file": "/tmp/traj.json", "split": "valid_seen"}]),
                patch.object(module, "_make_alfworld_env", return_value=FakeAlfworldEnv()),
            ):
                metrics = module._run_official_alfworld_suite(
                    task={"id": "alfworld", "runtime_backend": "benchmark_adapter"},
                    candidate_path=candidate_path,
                    proposal_runtime=make_runtime([]),
                    suite_name="alfworld-text",
                    domain="alfworld",
                    suite_config={"episode_split": "valid_seen", "max_turns": 5},
                    requested_limit=1,
                )
        self.assertEqual(metrics["objective"], 1.0)
        self.assertEqual(metrics["passed_tests"], 1)
        self.assertEqual(metrics["suite_summary"]["source"], "official_alfworld_text_env")
        self.assertEqual(metrics["suite_summary"]["split"], "valid_seen")
        self.assertEqual(metrics["item_runs"][0]["annotations"]["goal_condition_success_rate"], 1.0)

    def test_assistantbench_scaffold_returns_structured_runtime_unavailable_result(self) -> None:
        module = _load_module(
            "test_assistantbench_verifier",
            ROOT / "benchmark" / "agent_verified" / "assistantbench" / "verifier.py",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            candidate_path = Path(tmp_dir) / "editable.py"
            candidate_path.write_text(
                "def step(turn: dict, runtime) -> dict:\n"
                "    del runtime\n"
                "    return {\n"
                "        'message': 'noop',\n"
                "        'tool_calls': [],\n"
                "        'done': False,\n"
                "        'state': dict(turn.get('state') or {}),\n"
                "        'annotations': {},\n"
                "    }\n"
            )
            metrics = module.evaluate_candidate(
                task={"id": "assistantbench", "runtime_backend": "benchmark_adapter"},
                candidate_path=candidate_path,
                source_code=candidate_path.read_text(),
                baseline_metrics=None,
                memory_applied=False,
            )
            result = module.run_benchmark_adapter_task(
                task={
                    "id": "assistantbench",
                    "title": "AssistantBench",
                    "description": "AssistantBench scaffold",
                    "family": "agent-benchmark",
                    "function_name": "step",
                    "entry_symbol": "step",
                    "editable_file": "editable.py",
                    "answer_metric": "success_rate",
                    "objective_label": "Successful task rate",
                    "objective_direction": "max",
                    "objective_spec": {
                        "display_name": "Successful task rate",
                        "direction": "max",
                        "summary_template": "Higher is better.",
                        "formula": "success_rate = passed / total",
                    },
                    "selection_spec": {
                        "profile": "objective_only",
                        "display_name": "Objective only",
                        "summary_template": "Higher objective is better.",
                        "primary_formula": "primary_score = objective",
                        "gate_summary": "No extra gate.",
                        "tie_break_formula": "tie_break_score = 0",
                        "delta_template": "delta = current - previous",
                        "archive_summary": "archive",
                    },
                    "generation_budget": 2,
                    "candidate_budget": 1,
                    "branching_factor": 2,
                    "item_workers": 2,
                    "benchmark_tier": "experiment",
                    "track": "agent_verified",
                    "dataset_id": "assistantbench_public_validation",
                    "dataset_size": 214,
                    "local_dataset_only": False,
                    "split": None,
                    "included_in_main_comparison": False,
                    "runtime_backend": "benchmark_adapter",
                    "task_mode": "agent",
                    "interaction_mode": "multi_turn",
                    "optimization_scope": "wrapper",
                    "runtime_suite_config": {"task_limit": 214, "max_turns": 40},
                },
                candidate_path=candidate_path,
                source_code=candidate_path.read_text(),
                proposal_runtime=make_runtime([]),
                workspace_root=Path(tmp_dir),
                session_id="session-1",
                max_items=None,
                max_episodes=3,
                progress_callback=None,
                pace_ms=0,
            )
        self.assertEqual(metrics["status"], "error")
        self.assertEqual(metrics["suite_summary"]["status"], "runtime-missing")
        self.assertIn("AssistantBench", metrics["error"])
        self.assertEqual(result["winner"]["metrics"]["status"], "error")
        self.assertEqual(result["winner"]["metrics"]["suite_summary"]["status"], "runtime-missing")
        self.assertIn("AssistantBench", result["selection_reason"])


if __name__ == "__main__":
    unittest.main()
