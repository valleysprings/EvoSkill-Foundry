from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.bench.agent_benchmarks import run_scripted_multi_turn_suite
from app.bench.multi_turn_agent import (
    AgentRuntime,
    MULTI_TURN_AGENT_CONTRACT,
    normalize_step_result,
    validate_episode_payload,
    validate_turn_payload,
)
from tests.helpers import make_runtime


def _tool_call_response() -> str:
    return json.dumps(
        {
            "id": "resp-tool",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "done",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "complete",
                                    "arguments": json.dumps({"message": "done"}),
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 12},
        }
    )


class MultiTurnAgentTest(unittest.TestCase):
    def test_validate_episode_payload_normalizes_tool_schema(self) -> None:
        episode = validate_episode_payload(
            {
                "contract": MULTI_TURN_AGENT_CONTRACT,
                "suite": "fixture",
                "domain": "fixture",
                "episode_id": "ep-1",
                "instruction": "Complete the task.",
                "policy": {},
                "tools": [{"name": "complete", "description": "Finish the task."}],
                "limits": {"max_turns": 1},
                "metadata": {},
            }
        )
        self.assertEqual(episode["tools"][0]["function"]["name"], "complete")

    def test_validate_turn_payload_requires_contract(self) -> None:
        with self.assertRaises(ValueError):
            validate_turn_payload({"contract": "wrong"})

    def test_normalize_step_result_parses_json_tool_arguments(self) -> None:
        payload = normalize_step_result(
            {
                "message": "done",
                "tool_calls": [{"name": "complete", "arguments": json.dumps({"message": "done"})}],
                "done": True,
                "state": {},
            }
        )
        self.assertEqual(payload["tool_calls"][0]["arguments"]["message"], "done")

    def test_agent_runtime_chat_exposes_tool_calls(self) -> None:
        runtime = AgentRuntime(make_runtime([_tool_call_response()]))
        response = runtime.chat(
            [{"role": "user", "content": "finish"}],
            tools=[{"name": "complete", "description": "Finish"}],
        )
        self.assertEqual(response["tool_calls"][0]["name"], "complete")
        self.assertEqual(response["tool_calls"][0]["arguments"]["message"], "done")

    def test_scripted_suite_records_turn_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            candidate_path = Path(tmp_dir) / "editable.py"
            candidate_path.write_text(
                "def step(turn: dict, runtime) -> dict:\n"
                "    del runtime\n"
                "    return {\n"
                "        'message': 'done',\n"
                "        'tool_calls': [{'name': 'complete', 'arguments': {'message': 'done'}}],\n"
                "        'done': True,\n"
                "        'state': dict(turn.get('state') or {}),\n"
                "        'annotations': {},\n"
                "    }\n"
            )
            metrics = run_scripted_multi_turn_suite(
                task={"id": "fixture", "runtime_backend": "benchmark_adapter"},
                candidate_path=candidate_path,
                proposal_runtime=make_runtime([]),
                suite_name="fixture",
                domain="fixture",
                scripted_episodes=[
                    {
                        "episode_id": "episode-1",
                        "instruction": "Finish immediately.",
                        "turns": [
                            {
                                "observation": {"hint": "done"},
                                "expected_tool_name": "complete",
                                "stop_after_step": True,
                            }
                        ],
                    }
                ],
                suite_config={},
            )
        self.assertEqual(metrics["objective"], 1.0)
        self.assertEqual(metrics["item_runs"][0]["item_id"], "episode-1")
        self.assertEqual(len(metrics["item_runs"][0]["turns"]), 1)
        self.assertTrue(metrics["item_runs"][0]["success"])


if __name__ == "__main__":
    unittest.main()
