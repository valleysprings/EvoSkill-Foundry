from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from app.entries import runner


SAMPLE_TASK = {
    "id": "livecodebench",
    "title": "LiveCodeBench v6",
    "description": "Synthetic coding benchmark summary for CLI tests.",
    "family": "coding",
    "function_name": "solve",
    "entry_symbol": "solve",
    "editable_file": "editable.py",
    "answer_metric": "test_pass_rate",
    "objective_label": "Test pass rate",
    "objective_direction": "max",
    "objective_spec": {
        "display_name": "Test pass rate",
        "direction": "max",
        "unit": "ratio",
        "formula": "test_pass_rate = passed_cases / total_cases",
    },
    "selection_spec": {
        "primary_formula": "primary_score = objective_score",
        "gate_summary": "gate: verifier_status == 'pass'",
        "tie_break_formula": "tie_break_score = 0.0 (no auxiliary tie-break metrics configured)",
        "archive_summary": "archive_features = none",
    },
    "generation_budget": 3,
    "candidate_budget": 2,
    "branching_factor": 3,
    "item_workers": 6,
    "benchmark_tier": "comparable",
    "track": "coding_verified",
    "dataset_id": "livecodebench_release_v6",
    "dataset_size": 1055,
    "local_dataset_only": True,
    "split": "release_v6:test",
    "runtime_backend": "dataset",
    "task_mode": "artifact",
    "optimization_scope": "implementation",
    "included_in_main_comparison": True,
    "supports_runtime_config": False,
    "suite_run_config": None,
    "supports_max_items": True,
    "default_max_items": 1055,
}


class RunnerCliTest(unittest.TestCase):
    def _run_cli(self, argv: list[str]) -> str:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            runner.main(argv)
        return buffer.getvalue()

    def test_tasks_command_returns_api_shaped_json(self) -> None:
        with patch.object(runner, "list_codegen_task_summaries", return_value=[dict(SAMPLE_TASK)]):
            output = self._run_cli(["tasks"])
        payload = json.loads(output)
        self.assertIn("tasks", payload)
        self.assertEqual(payload["tasks"][0]["id"], "livecodebench")
        self.assertEqual(payload["tasks"][0]["task_mode"], "artifact")

    def test_tasks_pretty_mode_renders_single_task_detail(self) -> None:
        with patch.object(runner, "list_codegen_task_summaries", return_value=[dict(SAMPLE_TASK)]):
            output = self._run_cli(["tasks", "--task-id", "livecodebench", "--pretty"])
        self.assertIn("task_mode_summary", output)
        self.assertIn("Artifact task", output)
        self.assertIn("objective_formula", output)
        self.assertIn("test_pass_rate = passed_cases / total_cases", output)

    def test_latest_run_command_renders_cached_summary(self) -> None:
        payload = {
            "summary": {
                "generated_at": "2026-03-25T12:00:00+08:00",
                "active_model": "deepseek-chat",
                "num_tasks": 1,
                "total_runs": 1,
                "total_generations": 3,
                "write_backs": 1,
                "experiment_runs": 0,
            },
            "audit": {
                "session_id": "20260325_120000",
                "workspace_root": "runs/workspace/20260325_120000",
                "max_items": 5,
            },
            "runs": [
                {
                    "task": {"id": "livecodebench"},
                    "winner": {"metrics": {"objective": 0.75, "primary_score": 0.75}},
                    "delta_primary_score": 0.12,
                }
            ],
        }
        with patch.object(runner, "load_cached_discrete_payload", return_value=payload) as load_cached:
            output = self._run_cli(["latest-run", "--task-id", "livecodebench", "--pretty"])
        load_cached.assert_called_once_with(task_id="livecodebench")
        self.assertIn("generated_at", output)
        self.assertIn("deepseek-chat", output)
        self.assertIn("livecodebench", output)
        self.assertIn("delta_primary_score=0.12", output)

    def test_run_task_returns_payload_json(self) -> None:
        runtime = object()
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "codegen-livecodebench.json"
            artifact_path.write_text(json.dumps({"summary": {"generated_at": "now"}, "runs": []}))
            with (
                patch.object(runner, "_runtime_for_cli", return_value=runtime) as runtime_for_cli,
                patch.object(runner, "write_discrete_artifacts", return_value=artifact_path) as write_discrete_artifacts,
            ):
                output = self._run_cli(["run-task", "--task-id", "livecodebench", "--max-items", "3"])
        runtime_for_cli.assert_called_once_with(None)
        write_discrete_artifacts.assert_called_once_with(
            task_id="livecodebench",
            proposal_runtime=runtime,
            generation_budget=None,
            candidate_budget=None,
            branching_factor=None,
            item_workers=None,
            max_items=3,
            suite_config=None,
        )
        payload = json.loads(output)
        self.assertEqual(payload["summary"]["generated_at"], "now")


if __name__ == "__main__":
    unittest.main()
