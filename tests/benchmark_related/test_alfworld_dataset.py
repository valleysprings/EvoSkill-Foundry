from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.agent_verified.alfworld import verifier as alfworld_verifier
from tests.helpers import make_runtime


class _FakeEnv:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def reset(self):
        return (
            ["Welcome to ALFWorld. Your task is to: put the apple in the fridge"],
            {"admissible_commands": [["open fridge", "take apple"]]},
        )

    def step(self, commands):
        command = str(commands[0])
        self.commands.append(command)
        return (
            ["The fridge is open and the apple is inside."],
            [1.0],
            [True],
            {
                "admissible_commands": [["complete"]],
                "won": [True],
                "goal_condition_success_rate": [1.0],
            },
        )

    def close(self) -> None:
        return None


class AlfworldDatasetVerifierTest(unittest.TestCase):
    def test_evaluate_candidate_runs_single_episode_under_dataset_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            episode_dir = root / "episode-1"
            episode_dir.mkdir(parents=True, exist_ok=True)
            game_path = episode_dir / "game.tw-pddl"
            traj_path = episode_dir / "traj_data.json"
            game_path.write_text('{"solvable": true}\n')
            traj_path.write_text("{}\n")

            candidate_path = root / "editable.py"
            candidate_path.write_text(
                "def init_episode(episode: dict) -> dict:\n"
                "    return {'steps': 0, 'episode_id': episode['episode_id']}\n"
                "\n"
                "def step(turn: dict, runtime) -> dict:\n"
                "    del runtime\n"
                "    state = dict(turn.get('state') or {})\n"
                "    state['steps'] = int(state.get('steps') or 0) + 1\n"
                "    return {\n"
                "        'message': 'opening the fridge',\n"
                "        'tool_calls': [{'name': 'act', 'arguments': {'command': 'open fridge'}}],\n"
                "        'done': False,\n"
                "        'state': state,\n"
                "        'annotations': {},\n"
                "    }\n"
            )

            fake_env = _FakeEnv()
            task = {
                "id": "alfworld",
                "entry_symbol": "step",
                "runtime_suite_config": {
                    "episode_split": "valid_seen",
                    "max_turns": 5,
                    "tools": [
                        {
                            "name": "act",
                            "description": "Submit one action.",
                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                        },
                        {
                            "name": "complete",
                            "description": "Stop the episode.",
                            "parameters": {"type": "object", "properties": {"message": {"type": "string"}}},
                        },
                    ],
                },
                "question_item": {
                    "item_id": "episode-1",
                    "prompt": "put the apple in the fridge",
                    "metadata": {
                        "episode_id": "episode-1",
                        "instruction": "put the apple in the fridge",
                        "split": "valid_seen",
                        "game_file": str(game_path.relative_to(root)),
                        "traj_file": str(traj_path.relative_to(root)),
                    },
                },
            }

            with (
                patch.object(alfworld_verifier, "TASK_ROOT", root),
                patch.object(alfworld_verifier, "_make_alfworld_env", return_value=fake_env),
                patch.object(alfworld_verifier, "runtime_for_task", return_value=make_runtime([])),
            ):
                result = alfworld_verifier.evaluate_candidate(
                    task=task,
                    candidate_path=candidate_path,
                    source_code=candidate_path.read_text(),
                    baseline_metrics=None,
                    memory_applied=False,
                )

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["verifier_status"], "pass")
        self.assertEqual(result["objective"], 1.0)
        self.assertEqual(result["passed_tests"], 1)
        self.assertEqual(result["suite_summary"]["split"], "valid_seen")
        self.assertEqual(fake_env.commands, ["open fridge"])
        self.assertEqual(len(result["item_runs"]), 1)
        self.assertEqual(result["item_runs"][0]["item_id"], "episode-1")
        self.assertTrue(result["item_runs"][0]["success"])
        self.assertEqual(len(result["item_runs"][0]["turns"]), 1)
        self.assertEqual(
            result["item_runs"][0]["turns"][0]["action"]["tool_calls"][0]["arguments"]["command"],
            "open fridge",
        )


if __name__ == "__main__":
    unittest.main()
