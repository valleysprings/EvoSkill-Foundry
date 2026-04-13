from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.bench.multi_turn_agent import (
    AgentRuntime,
    MULTI_TURN_AGENT_CONTRACT,
    load_agent_adapter,
    normalize_step_result,
    validate_episode_payload,
    validate_turn_payload,
)
from app.bench.runtime_support import effective_suite_run_config, runtime_for_task


ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = Path(__file__).resolve().parent
ALFWORLD_REPO_ROOT = ROOT / "external" / "alfworld"
ALFWORLD_DATA_ROOT = TASK_ROOT / "data"
DEFAULT_SPLIT = "valid_seen"
DEFAULT_MAX_TURNS = 50
VALID_SPLITS = {"train", "valid_seen", "valid_unseen"}


@contextmanager
def _repo_on_sys_path(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(repo_root))
        except ValueError:
            pass


def _require_question_item(task: dict[str, Any]) -> dict[str, Any]:
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("Dataset question task must provide question_item.")
    return item


def _history_message(role: str, content: str | None, *, tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": role, "content": content}
    if tool_calls:
        payload["tool_calls"] = list(tool_calls)
    return payload


def _tool_result_message(call_id: str, content: str, *, error: bool = False) -> dict[str, Any]:
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": call_id,
        "error": error,
    }


def _action_summary(action: dict[str, Any]) -> str:
    parts: list[str] = []
    for tool_call in list(action.get("tool_calls") or []):
        name = str(tool_call.get("name") or "").strip() or "tool"
        arguments = dict(tool_call.get("arguments") or {})
        detail = str(arguments.get("command") or arguments.get("message") or "").strip()
        parts.append(f"{name}({detail})" if detail else name)
    message = str(action.get("message") or "").strip()
    if message:
        parts.append(message)
    if bool(action.get("done")):
        parts.append("done")
    return " | ".join(part for part in parts if part)


def _batched_value(payload: Any) -> Any:
    if isinstance(payload, list) and payload:
        return payload[0]
    return payload


def _batched_info(payload: Any, key: str, default: Any) -> Any:
    if isinstance(payload, dict):
        value = payload.get(key, default)
        if isinstance(value, list) and value:
            return value[0]
        return value
    return default


def _configured_split(config: dict[str, Any], item: dict[str, Any]) -> str:
    metadata = dict(item.get("metadata") or {})
    split = str(metadata.get("split") or config.get("episode_split") or DEFAULT_SPLIT).strip() or DEFAULT_SPLIT
    if split not in VALID_SPLITS:
        raise ValueError(f"Unsupported ALFWorld split {split!r}. Expected one of {sorted(VALID_SPLITS)}.")
    return split


def _max_turns(config: dict[str, Any]) -> int:
    raw_value = config.get("max_turns")
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        parsed = DEFAULT_MAX_TURNS
    return parsed if parsed > 0 else DEFAULT_MAX_TURNS


def _extract_task_desc(observation_text: str, fallback: str) -> str:
    marker = "Your task is to: "
    if marker in observation_text:
        return observation_text.partition(marker)[-1].strip() or fallback
    return fallback


def _resolve_episode_spec(item: dict[str, Any], split: str) -> dict[str, str]:
    metadata = dict(item.get("metadata") or {})
    episode_id = str(metadata.get("episode_id") or item.get("item_id") or "").strip()
    instruction = str(metadata.get("instruction") or item.get("prompt") or "").strip()
    game_file = str(metadata.get("game_file") or "").strip()
    traj_file = str(metadata.get("traj_file") or "").strip()
    if not episode_id:
        raise ValueError("ALFWorld item is missing metadata.episode_id.")
    if not instruction:
        raise ValueError(f"ALFWorld item {episode_id!r} is missing instruction.")
    if not game_file:
        raise ValueError(f"ALFWorld item {episode_id!r} is missing metadata.game_file.")
    if not traj_file:
        raise ValueError(f"ALFWorld item {episode_id!r} is missing metadata.traj_file.")

    game_path = (TASK_ROOT / game_file).resolve()
    traj_path = (TASK_ROOT / traj_file).resolve()
    if not game_path.exists():
        raise FileNotFoundError(f"ALFWorld game file not found: {game_path}")
    if not traj_path.exists():
        raise FileNotFoundError(f"ALFWorld trajectory file not found: {traj_path}")
    return {
        "episode_id": episode_id,
        "instruction": instruction,
        "split": split,
        "game_file": str(game_path),
        "traj_file": str(traj_path),
    }


def _make_alfworld_env(game_file: str, max_turns: int):
    with _repo_on_sys_path(ALFWORLD_REPO_ROOT):
        try:
            import textworld  # type: ignore[import-not-found]
            import textworld.gym  # type: ignore[import-not-found]
            from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:  # pragma: no cover
            missing = getattr(exc, "name", None) or "official ALFWorld dependencies"
            raise RuntimeError(
                "ALFWorld official evaluation requires the external repo dependencies. "
                f"Missing module: {missing}. Install the ALFWorld text environment requirements first."
            ) from exc

        os.environ["ALFWORLD_DATA"] = str(ALFWORLD_DATA_ROOT)
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        env_id = textworld.gym.register_games(
            [game_file],
            request_infos,
            batch_size=1,
            asynchronous=False,
            max_episode_steps=max_turns,
            wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
        )
        return textworld.gym.make(env_id)


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    del source_code, baseline_metrics, memory_applied

    item = _require_question_item(task)
    suite_config = effective_suite_run_config(task, Path(candidate_path))
    split = _configured_split(suite_config, item)
    episode_spec = _resolve_episode_spec(item, split)
    max_turns = _max_turns(suite_config)
    init_episode, step = load_agent_adapter(Path(candidate_path))
    runtime = AgentRuntime(runtime_for_task(task))

    started = time.perf_counter()
    env = _make_alfworld_env(episode_spec["game_file"], max_turns)
    try:
        initial_obs, initial_infos = env.reset()
        observation_text = str(_batched_value(initial_obs) or "")
        instruction = _extract_task_desc(observation_text, episode_spec["instruction"])
        admissible_commands = list(_batched_info(initial_infos, "admissible_commands", []) or [])
        episode = validate_episode_payload(
            {
                "contract": MULTI_TURN_AGENT_CONTRACT,
                "suite": "alfworld-text",
                "domain": "alfworld",
                "episode_id": episode_spec["episode_id"],
                "instruction": instruction,
                "policy": {},
                "tools": list(suite_config.get("tools") or []),
                "limits": {"max_turns": max_turns},
                "metadata": {
                    "split": episode_spec["split"],
                    "game_file": episode_spec["game_file"],
                    "traj_file": episode_spec["traj_file"],
                    "official_env": True,
                },
            }
        )

        state_result = init_episode(episode)
        if state_result is None:
            state: dict[str, Any] = {}
        elif isinstance(state_result, dict):
            state = dict(state_result)
        else:
            raise ValueError("init_episode(...) must return a dict or None.")

        history: list[dict[str, Any]] = []
        turns: list[dict[str, Any]] = []
        success = False
        reward = 0.0
        goal_condition_success_rate = 0.0

        for turn_index in range(max_turns):
            observation = {
                "text": observation_text,
                "instruction": instruction,
                "valid_actions": admissible_commands,
                "won": success,
                "goal_condition_success_rate": goal_condition_success_rate,
            }
            turn_payload = validate_turn_payload(
                {
                    "contract": MULTI_TURN_AGENT_CONTRACT,
                    "episode": episode,
                    "turn_index": turn_index,
                    "history": history,
                    "observation": observation,
                    "state": state,
                    "metadata": {"official_env": True},
                }
            )
            action = normalize_step_result(step(turn_payload, runtime))
            tool_results: list[dict[str, Any]] = []
            episode_done = False

            if action["message"] is not None or action["tool_calls"]:
                history.append(_history_message("assistant", action["message"], tool_calls=action["tool_calls"]))

            for call_index, tool_call in enumerate(action["tool_calls"]):
                tool_name = str(tool_call.get("name") or "").strip()
                arguments = dict(tool_call.get("arguments") or {})
                call_id = str(tool_call.get("id") or f"{episode_spec['episode_id']}-tool-{turn_index}-{call_index}")
                error = False

                if tool_name == "act":
                    command = str(arguments.get("command") or "").strip()
                    next_obs, _scores, dones, infos = env.step([command])
                    observation_text = str(_batched_value(next_obs) or observation_text)
                    admissible_commands = list(_batched_info(infos, "admissible_commands", []) or [])
                    success = bool(_batched_info(infos, "won", False))
                    goal_condition_success_rate = float(_batched_info(infos, "goal_condition_success_rate", 0.0) or 0.0)
                    reward = 1.0 if success else 0.0
                    episode_done = bool(_batched_value(dones)) or episode_done
                    content = json.dumps(
                        {
                            "observation": observation_text,
                            "valid_actions": admissible_commands,
                            "won": success,
                            "goal_condition_success_rate": goal_condition_success_rate,
                            "done": episode_done,
                        },
                        ensure_ascii=True,
                    )
                elif tool_name == "complete":
                    episode_done = True
                    content = json.dumps(
                        {
                            "status": "completed",
                            "message": str(arguments.get("message") or ""),
                        },
                        ensure_ascii=True,
                    )
                else:
                    error = True
                    content = json.dumps({"error": f"Unknown tool {tool_name!r}"}, ensure_ascii=True)

                tool_call_with_id = {**tool_call, "id": call_id}
                tool_results.append(
                    {
                        "tool_call": tool_call_with_id,
                        "content": content,
                        "error": error,
                    }
                )
                history.append(_tool_result_message(call_id, content, error=error))

            turns.append(
                {
                    "turn_index": turn_index,
                    "observation": observation,
                    "action": action,
                    "tool_results": tool_results,
                }
            )
            state = dict(action["state"])
            if action["done"] or episode_done:
                break

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        item_run = {
            "item_id": episode_spec["episode_id"],
            "item_name": episode_spec["episode_id"],
            "payload": episode,
            "turns": turns,
            "success": success,
            "reward": reward,
            "annotations": {"goal_condition_success_rate": goal_condition_success_rate},
            "raw_artifact_path": episode_spec["game_file"],
        }
        test_result = {
            "name": episode_spec["episode_id"],
            "expected": 1.0,
            "actual": reward,
            "actual_raw": {
                "payload": episode,
                "turns": turns,
                "goal_condition_success_rate": goal_condition_success_rate,
                "action_summary": _action_summary(dict(turns[-1]["action"])) if turns else "",
            },
            "passed": success,
        }
        return {
            "status": "pass" if success else "fail",
            "verifier_status": "pass" if success else "fail",
            "correctness": reward,
            "passed_tests": 1 if success else 0,
            "total_tests": 1,
            "benchmark_ms": round(elapsed_ms, 3),
            "benchmark_samples_ms": [round(elapsed_ms, 3)],
            "objective": reward,
            "objective_score": reward,
            "objective_signal": reward,
            "error": None,
            "test_results": [test_result],
            "item_runs": [item_run],
            "suite_summary": {
                "suite": "alfworld-text",
                "domain": "alfworld",
                "source": "official_alfworld_text_env",
                "split": episode_spec["split"],
                "passed": 1 if success else 0,
                "total": 1,
                "average_goal_condition_success_rate": goal_condition_success_rate,
                "average_steps": float(len(turns)),
            },
        }
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
