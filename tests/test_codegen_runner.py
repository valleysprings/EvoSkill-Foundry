from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.codegen.errors import ConfigError, LlmResponseError, LlmTransportError
from app.entries.discrete_demo import generate_discrete_payload, write_discrete_artifacts
from tests.helpers import chat_response, make_runtime


PROPOSAL_PAYLOAD = {
    "candidates": [
        {
            "name": "Set cardinality",
            "strategy": "Use a set cardinality check.",
            "rationale": "The runtime can detect duplicates by comparing lengths.",
            "imports": [],
            "function_body": "return len(values) != len(set(values))",
            "candidate_summary": "Single-pass set cardinality duplicate detection.",
        },
        {
            "name": "Seen set",
            "strategy": "Stream through the list with a seen set.",
            "rationale": "Early exit preserves correctness and still removes quadratic scans.",
            "imports": [],
            "function_body": "seen = set()\nfor value in values:\n    if value in seen:\n        return True\n    seen.add(value)\nreturn False",
            "candidate_summary": "Streaming duplicate detection with early exit.",
        },
        {
            "name": "Sorted scan",
            "strategy": "Sort then scan neighbors.",
            "rationale": "Sorting avoids quadratic pair comparisons.",
            "imports": [],
            "function_body": "ordered = sorted(values)\nfor index in range(1, len(ordered)):\n    if ordered[index] == ordered[index - 1]:\n        return True\nreturn False",
            "candidate_summary": "Sort-based duplicate detection.",
        },
    ]
}

FAILURE_PROPOSAL_PAYLOAD = {
    "candidates": [
        {
            "name": "Always false",
            "strategy": "Return a constant false value.",
            "rationale": "This is intentionally incorrect and should fail deterministic tests.",
            "imports": [],
            "function_body": "return False",
            "candidate_summary": "Incorrect constant-false duplicate detector.",
        },
        {
            "name": "Always true",
            "strategy": "Return a constant true value.",
            "rationale": "This is intentionally incorrect and should fail deterministic tests.",
            "imports": [],
            "function_body": "return True",
            "candidate_summary": "Incorrect constant-true duplicate detector.",
        },
        {
            "name": "Length gate",
            "strategy": "Use an obviously wrong shortcut.",
            "rationale": "This is intentionally incorrect and should fail deterministic tests.",
            "imports": [],
            "function_body": "return len(values) > 3",
            "candidate_summary": "Incorrect heuristic duplicate detector.",
        },
    ]
}

REFLECTION_PAYLOAD = {
    "failure_pattern": "Quadratic duplicate detection wasted time on repeated scans.",
    "strategy_hypothesis": "Hash-based membership checks will dominate the nested scan on this task family.",
    "successful_strategy": "Use a set cardinality check to detect duplicates.",
    "prompt_fragment": "Prefer a hash-based duplicate detector that preserves semantics and reduces the scan to near-linear work.",
    "tool_trace_summary": "The set-cardinality candidate passed all tests and outpaced the baseline benchmark.",
}

FAILURE_REFLECTION_PAYLOAD = {
    "failure_pattern": "The mutation preserved correctness but stalled because it did not improve on the incumbent benchmark.",
    "strategy_hypothesis": "Repeatedly proposing the same architecture wastes generations once the best set-based path is already in place.",
    "successful_strategy": "Shift to a materially different architecture only when it can beat the incumbent, otherwise avoid repeating the same plan.",
    "prompt_fragment": "Do not spend another generation on a semantic no-op that merely restates the incumbent strategy without a measurable gain.",
    "tool_trace_summary": "candidate passed tests, matched or trailed the incumbent benchmark, and was rejected as a non-improving repeat.",
}


class CodegenRunnerTest(unittest.TestCase):
    def test_missing_runtime_config_fails_before_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigError):
                generate_discrete_payload(task_id="contains-duplicates", runs_root=Path(tmp_dir), env_root=Path(tmp_dir))

    def test_invalid_llm_output_fails_immediately(self) -> None:
        runtime = make_runtime([chat_response({"candidates": [{"name": "bad"}]})])
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(LlmResponseError):
                generate_discrete_payload(
                    task_id="contains-duplicates",
                    proposal_runtime=runtime,
                    runs_root=Path(tmp_dir),
                )

    def test_success_path_writes_payload_memory_and_llm_trace(self) -> None:
        runtime = make_runtime(
            [
                chat_response(PROPOSAL_PAYLOAD),
                chat_response(REFLECTION_PAYLOAD),
                chat_response(FAILURE_PROPOSAL_PAYLOAD),
                chat_response(FAILURE_REFLECTION_PAYLOAD),
                chat_response(FAILURE_PROPOSAL_PAYLOAD),
                chat_response(FAILURE_REFLECTION_PAYLOAD),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = write_discrete_artifacts(
                task_id="contains-duplicates",
                proposal_runtime=runtime,
                runs_root=Path(tmp_dir),
            )
            payload = json.loads(artifact_path.read_text())
            run = payload["runs"][0]
            manifest_path = Path(tmp_dir) / run["handoff_bundle"]["manifest_path"]
            self.assertEqual(payload["run_mode"], "llm-required")
            self.assertEqual(payload["summary"]["active_model"], "deepseek-chat")
            self.assertTrue((Path(tmp_dir) / "codegen_working_memory.md").exists())
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["winner_candidate"], run["winner"]["agent"])
            self.assertIsNotNone(manifest["artifact_paths"]["llm_trace_jsonl"])
            self.assertTrue((Path(tmp_dir) / manifest["artifact_paths"]["llm_trace_jsonl"]).exists())
            self.assertTrue((Path(tmp_dir) / manifest["artifact_paths"]["report_svg"]).exists())
            self.assertTrue((Path(tmp_dir) / manifest["artifact_paths"]["improvement_table_json"]).exists())
            self.assertIn(manifest["session_id"], run["handoff_bundle"]["manifest_path"])
            self.assertEqual(run["session_id"], manifest["session_id"])
            self.assertEqual(payload["summary"]["write_backs"], 3)
            self.assertEqual(run["memory_before_count"], 2)
            self.assertEqual(run["memory_after_count"], 5)
            self.assertEqual(run["positive_experiences_added"], 1)
            self.assertEqual(run["negative_experiences_added"], 2)
            self.assertEqual(len(run["added_experiences"]), 3)
            self.assertEqual(run["added_experiences"][0]["generation"], 1)
            self.assertEqual(run["added_experiences"][0]["experience_outcome"], "success")
            memories = json.loads((Path(tmp_dir) / "codegen_working_memory.json").read_text())
            task_memories = [item for item in memories if item.get("source_task") == "contains-duplicates"]
            self.assertTrue(any(item.get("experience_outcome") == "success" for item in task_memories))
            self.assertTrue(any(item.get("experience_outcome") == "failure" for item in task_memories))
            self.assertIn("failure_memories:", (Path(tmp_dir) / "codegen_working_memory.md").read_text())

    def test_timeout_failure_aborts_run(self) -> None:
        runtime = make_runtime([TimeoutError("timed out")])
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(LlmTransportError):
                generate_discrete_payload(
                    task_id="contains-duplicates",
                    proposal_runtime=runtime,
                    runs_root=Path(tmp_dir),
                )

    def test_invalid_http_json_aborts_run(self) -> None:
        runtime = make_runtime(["not-json"])
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(LlmResponseError):
                generate_discrete_payload(
                    task_id="contains-duplicates",
                    proposal_runtime=runtime,
                    runs_root=Path(tmp_dir),
                )

    def test_reflection_failure_aborts_run(self) -> None:
        runtime = make_runtime(
            [
                chat_response(PROPOSAL_PAYLOAD),
                chat_response({"failure_pattern": "missing most fields"}),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(LlmResponseError):
                generate_discrete_payload(
                    task_id="contains-duplicates",
                    proposal_runtime=runtime,
                    runs_root=Path(tmp_dir),
                )


if __name__ == "__main__":
    unittest.main()
