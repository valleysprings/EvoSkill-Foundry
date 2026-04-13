from __future__ import annotations

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from app.bench.runtime_support import emit_progress
from app.codegen.llm import ProposalRuntime


MULTI_TURN_AGENT_CONTRACT = "multi_turn_agent_v1"
EpisodeHook = Callable[[dict[str, Any]], dict[str, Any]]
StepHook = Callable[[dict[str, Any], "AgentRuntime"], dict[str, Any]]


def _load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a dict.")
    return dict(value)


def _require_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _normalize_message(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise ValueError("history entries must be JSON objects.")
    role = _require_string(message.get("role"), label="history.role")
    normalized: dict[str, Any] = {"role": role}
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=True)
    normalized["content"] = content
    if isinstance(message.get("tool_calls"), list):
        normalized["tool_calls"] = [_normalize_tool_call(item) for item in message["tool_calls"]]
    if isinstance(message.get("metadata"), dict):
        normalized["metadata"] = dict(message["metadata"])
    return normalized


def normalize_tool_schema(tool: Any) -> dict[str, Any]:
    if not isinstance(tool, dict):
        raise ValueError("tools must contain JSON objects.")
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        function_payload = dict(tool["function"])
        name = _require_string(function_payload.get("name"), label="tool.function.name")
        description = function_payload.get("description")
        parameters = function_payload.get("parameters")
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": str(description or "").strip(),
                "parameters": dict(parameters) if isinstance(parameters, dict) else {"type": "object", "properties": {}},
            },
        }
    name = _require_string(tool.get("name"), label="tool.name")
    description = str(tool.get("description") or "").strip()
    parameters = tool.get("parameters")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": dict(parameters) if isinstance(parameters, dict) else {"type": "object", "properties": {}},
        },
    }


def _normalize_tool_call(tool_call: Any) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        raise ValueError("tool_calls must contain JSON objects.")
    name = _require_string(tool_call.get("name"), label="tool_call.name")
    arguments = tool_call.get("arguments")
    if arguments is None:
        normalized_arguments: dict[str, Any] = {}
    elif isinstance(arguments, dict):
        normalized_arguments = dict(arguments)
    elif isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("tool_call.arguments must be a dict or valid JSON object string.") from exc
        if not isinstance(parsed, dict):
            raise ValueError("tool_call.arguments JSON must decode to an object.")
        normalized_arguments = dict(parsed)
    else:
        raise ValueError("tool_call.arguments must be a dict or JSON string.")
    normalized = {
        "name": name,
        "arguments": normalized_arguments,
    }
    if isinstance(tool_call.get("id"), str) and tool_call["id"].strip():
        normalized["id"] = tool_call["id"].strip()
    return normalized


def validate_episode_payload(episode: dict[str, Any]) -> dict[str, Any]:
    payload = _require_mapping(episode, label="episode")
    contract = _require_string(payload.get("contract"), label="episode.contract")
    if contract != MULTI_TURN_AGENT_CONTRACT:
        raise ValueError(f"episode.contract must be {MULTI_TURN_AGENT_CONTRACT!r}.")
    normalized = {
        "contract": contract,
        "suite": _require_string(payload.get("suite"), label="episode.suite"),
        "domain": _require_string(payload.get("domain"), label="episode.domain"),
        "episode_id": _require_string(payload.get("episode_id"), label="episode.episode_id"),
        "instruction": _require_string(payload.get("instruction"), label="episode.instruction"),
        "policy": dict(payload.get("policy") or {}),
        "tools": [normalize_tool_schema(item) for item in list(payload.get("tools") or [])],
        "limits": dict(payload.get("limits") or {}),
        "metadata": dict(payload.get("metadata") or {}),
    }
    return normalized


def validate_turn_payload(turn: dict[str, Any]) -> dict[str, Any]:
    payload = _require_mapping(turn, label="turn")
    contract = _require_string(payload.get("contract"), label="turn.contract")
    if contract != MULTI_TURN_AGENT_CONTRACT:
        raise ValueError(f"turn.contract must be {MULTI_TURN_AGENT_CONTRACT!r}.")
    normalized = {
        "contract": contract,
        "episode": validate_episode_payload(_require_mapping(payload.get("episode"), label="turn.episode")),
        "turn_index": int(payload.get("turn_index") or 0),
        "history": [_normalize_message(item) for item in list(payload.get("history") or [])],
        "observation": dict(payload.get("observation") or {}),
        "state": dict(payload.get("state") or {}),
        "metadata": dict(payload.get("metadata") or {}),
    }
    return normalized


def normalize_step_result(result: dict[str, Any]) -> dict[str, Any]:
    payload = _require_mapping(result, label="step result")
    message = payload.get("message")
    if message is not None and not isinstance(message, str):
        raise ValueError("step result.message must be a string or null.")
    tool_calls = [_normalize_tool_call(item) for item in list(payload.get("tool_calls") or [])]
    state = payload.get("state")
    if state is None:
        normalized_state: dict[str, Any] = {}
    elif isinstance(state, dict):
        normalized_state = dict(state)
    else:
        raise ValueError("step result.state must be a dict.")
    annotations = payload.get("annotations")
    if annotations is None:
        normalized_annotations: dict[str, Any] = {}
    elif isinstance(annotations, dict):
        normalized_annotations = dict(annotations)
    else:
        raise ValueError("step result.annotations must be a dict.")
    return {
        "message": message.strip() if isinstance(message, str) else None,
        "tool_calls": tool_calls,
        "done": bool(payload.get("done", False)),
        "state": normalized_state,
        "annotations": normalized_annotations,
    }


def load_agent_adapter(candidate_path: Path) -> tuple[EpisodeHook, StepHook]:
    module_name = f"agent_adapter_{candidate_path.parent.name}_{candidate_path.stem}".replace("-", "_")
    module = _load_module_from_path(candidate_path, module_name)
    step = getattr(module, "step", None)
    if not callable(step):
        raise ValueError(f"{candidate_path} must export callable step(turn, runtime).")
    init_episode = getattr(module, "init_episode", None)
    if init_episode is None:
        def _default_init_episode(_episode: dict[str, Any]) -> dict[str, Any]:
            return {}

        return _default_init_episode, step
    if not callable(init_episode):
        raise ValueError(f"{candidate_path} exports init_episode but it is not callable.")
    return init_episode, step


class AgentRuntime:
    def __init__(self, runtime: ProposalRuntime) -> None:
        self._runtime = runtime

    @property
    def active_model(self) -> str:
        return self._runtime.active_model

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        purpose: str | None = None,
    ) -> dict[str, Any]:
        normalized_messages = [_normalize_message(message) for message in messages]
        normalized_tools = [normalize_tool_schema(tool) for tool in tools] if tools else None
        response, _trace = self._runtime.chat(
            purpose=purpose or "multi_turn_agent_step",
            messages=normalized_messages,
            tools=normalized_tools,
            tool_choice=tool_choice,
        )
        return response


def invoke_agent_init_episode(candidate_path: Path, episode: dict[str, Any]) -> dict[str, Any]:
    init_episode, _step = load_agent_adapter(candidate_path)
    result = init_episode(validate_episode_payload(episode))
    if result is None:
        return {}
    if not isinstance(result, dict):
        raise ValueError("init_episode(...) must return a dict or None.")
    return dict(result)


def invoke_agent_step(candidate_path: Path, turn: dict[str, Any], runtime: AgentRuntime) -> dict[str, Any]:
    _init_episode, step = load_agent_adapter(candidate_path)
    return normalize_step_result(step(validate_turn_payload(turn), runtime))


def _resolve_parallel_workers(
    *,
    task: dict[str, Any],
    total_items: int,
    suite_config: dict[str, Any] | None = None,
) -> int:
    if total_items <= 1:
        return 1
    raw_value = task.get("item_workers")
    if raw_value is None and isinstance(suite_config, dict):
        raw_value = suite_config.get("max_concurrency")
    try:
        configured = int(raw_value or 1)
    except (TypeError, ValueError):
        configured = 1
    configured = max(1, configured)
    return max(1, min(configured, total_items))


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


def _default_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "act",
            "description": "Submit one action to the scripted environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                    }
                },
                "required": ["command"],
            },
        },
        {
            "name": "complete",
            "description": "Declare the task complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                    }
                },
            },
        },
    ]


def _action_summary(action: dict[str, Any]) -> str:
    tool_calls = list(action.get("tool_calls") or [])
    parts: list[str] = []
    for tool_call in tool_calls:
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


def _episode_record(
    *,
    suite: str,
    domain: str,
    episode_id: str,
    instruction: str,
    tools: list[dict[str, Any]],
    limits: dict[str, Any],
    metadata: dict[str, Any],
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return validate_episode_payload(
        {
            "contract": MULTI_TURN_AGENT_CONTRACT,
            "suite": suite,
            "domain": domain,
            "episode_id": episode_id,
            "instruction": instruction,
            "policy": dict(policy or {}),
            "tools": tools,
            "limits": dict(limits),
            "metadata": dict(metadata),
        }
    )


def _turn_record(
    *,
    episode: dict[str, Any],
    turn_index: int,
    history: list[dict[str, Any]],
    observation: dict[str, Any],
    state: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return validate_turn_payload(
        {
            "contract": MULTI_TURN_AGENT_CONTRACT,
            "episode": episode,
            "turn_index": turn_index,
            "history": list(history),
            "observation": dict(observation),
            "state": dict(state),
            "metadata": dict(metadata or {}),
        }
    )


def _extract_turn_reward(turn: dict[str, Any], *, default_success: bool) -> tuple[float, bool]:
    reward_value = turn.get("reward")
    if reward_value is None:
        return (1.0 if default_success else 0.0), default_success
    reward = float(reward_value or 0.0)
    return reward, reward >= 1.0 - 1e-6


def _evaluate_scripted_expectations(turn: dict[str, Any], action: dict[str, Any]) -> tuple[float | None, bool | None]:
    checks: list[bool] = []
    expected_tool_name = str(turn.get("expected_tool_name") or "").strip()
    if expected_tool_name:
        checks.append(any(str(call.get("name") or "") == expected_tool_name for call in action["tool_calls"]))
    expected_tool_arguments = turn.get("expected_tool_arguments")
    if isinstance(expected_tool_arguments, dict):
        matched = False
        for call in action["tool_calls"]:
            arguments = dict(call.get("arguments") or {})
            if all(arguments.get(key) == value for key, value in expected_tool_arguments.items()):
                matched = True
                break
        checks.append(matched)
    expected_message_contains = str(turn.get("expected_message_contains") or "")
    if expected_message_contains:
        checks.append(expected_message_contains in str(action.get("message") or ""))
    if "expected_done" in turn:
        checks.append(bool(action.get("done")) is bool(turn.get("expected_done")))
    if "expected_tool_count" in turn:
        checks.append(len(list(action.get("tool_calls") or [])) == int(turn.get("expected_tool_count") or 0))
    if not checks:
        return None, None
    success = all(checks)
    return (1.0 if success else 0.0), success


def _run_scripted_episode(
    *,
    task: dict[str, Any],
    candidate_path: Path,
    proposal_runtime: ProposalRuntime,
    suite_name: str,
    domain: str,
    suite_config: dict[str, Any],
    scripted_episode: dict[str, Any],
    index: int,
    progress_callback=None,
) -> tuple[int, dict[str, Any], dict[str, Any], bool]:
    init_episode, step = load_agent_adapter(candidate_path)
    runtime = AgentRuntime(proposal_runtime)
    episode_id = str(scripted_episode.get("episode_id") or f"{task['id']}-episode-{index:04d}")
    instruction = str(scripted_episode.get("instruction") or "").strip()
    if not instruction:
        raise ValueError(f"Scripted episode {episode_id} is missing instruction.")
    episode = _episode_record(
        suite=suite_name,
        domain=domain,
        episode_id=episode_id,
        instruction=instruction,
        tools=list(scripted_episode.get("tools") or suite_config.get("tools") or _default_tools()),
        limits={"max_turns": int(scripted_episode.get("max_turns") or suite_config.get("max_turns") or 8)},
        metadata=dict(scripted_episode.get("metadata") or {}),
        policy=dict(scripted_episode.get("policy") or suite_config.get("policy") or {}),
    )
    state_result = init_episode(episode)
    state = dict(state_result or {})
    if state_result is not None and not isinstance(state_result, dict):
        raise ValueError("init_episode(...) must return a dict or None.")

    history: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    scripted_turns = list(scripted_episode.get("turns") or [{"observation": {"instruction": instruction}}])
    max_turns = int(episode.get("limits", {}).get("max_turns") or len(scripted_turns) or 1)
    reward = 0.0
    success = False
    raw_artifact_path = None
    emit_progress(
        progress_callback,
        task_id=str(task["id"]),
        phase="episode_started",
        item_id=episode_id,
        item_name=episode_id,
        item_brief=instruction,
        expected_answer="Solve episode",
        candidate_status="running",
        message=f"[{episode_id}] {instruction}",
    )

    for turn_index, scripted_turn in enumerate(scripted_turns[:max_turns]):
        observation = dict(scripted_turn.get("observation") or {})
        if turn_index == 0 and "instruction" not in observation:
            observation["instruction"] = instruction
        turn_payload = _turn_record(
            episode=episode,
            turn_index=turn_index,
            history=history,
            observation=observation,
            state=state,
            metadata={"scripted": True},
        )
        action = normalize_step_result(step(turn_payload, runtime))
        tool_results: list[dict[str, Any]] = []
        scripted_results = list(scripted_turn.get("tool_results") or [])
        for call_index, tool_call in enumerate(action["tool_calls"]):
            scripted_result = scripted_results[call_index] if call_index < len(scripted_results) else {}
            result_content = scripted_result.get("content", scripted_result.get("result", scripted_result))
            normalized_content = (
                result_content
                if isinstance(result_content, str)
                else json.dumps(result_content, ensure_ascii=True, sort_keys=False)
            )
            tool_results.append(
                {
                    "tool_call": dict(tool_call),
                    "content": normalized_content,
                    "error": bool(scripted_result.get("error", False)),
                }
            )

        if action["message"] is not None or action["tool_calls"]:
            history.append(_history_message("assistant", action["message"], tool_calls=action["tool_calls"]))
        for item in tool_results:
            call_id = str(item["tool_call"].get("id") or f"{episode_id}-tool-{turn_index}-{len(turns)}")
            history.append(_tool_result_message(call_id, str(item["content"]), error=bool(item["error"])))

        expectation_reward, expectation_success = _evaluate_scripted_expectations(scripted_turn, action)
        if expectation_reward is not None and expectation_success is not None:
            reward, success = expectation_reward, expectation_success
        else:
            reward, success = _extract_turn_reward(
                scripted_turn,
                default_success=bool(scripted_episode.get("expected_success")),
            )
        raw_artifact_path = scripted_turn.get("raw_artifact_path", raw_artifact_path)
        turns.append(
            {
                "turn_index": turn_index,
                "observation": observation,
                "action": action,
                "tool_results": tool_results,
            }
        )
        emit_progress(
            progress_callback,
            task_id=str(task["id"]),
            phase="episode_turn",
            item_id=episode_id,
            item_name=episode_id,
            item_brief=instruction,
            expected_answer="Solve episode",
            candidate_actual=_action_summary(action),
            candidate_status="running",
            message=f"[{episode_id}] t{turn_index}: {_action_summary(action) or 'no action'}",
        )
        state = dict(action["state"])
        if action["done"] or bool(scripted_turn.get("stop_after_step")):
            break

    item_run = {
        "item_id": episode_id,
        "item_name": episode_id,
        "payload": episode,
        "turns": turns,
        "success": success,
        "reward": reward,
        "raw_artifact_path": raw_artifact_path,
    }
    test_result = {
        "name": episode_id,
        "expected": 1.0,
        "actual": reward,
        "passed": success,
        "actual_raw": {"payload": episode, "turns": turns},
    }
    emit_progress(
        progress_callback,
        task_id=str(task["id"]),
        phase="episode_finished",
        item_id=episode_id,
        item_name=episode_id,
        item_brief=instruction,
        expected_answer="Solve episode",
        candidate_actual=_action_summary(dict(turns[-1].get("action") or {})) if turns else ("Episode solved" if success else "Episode not solved"),
        candidate_status="pass" if success else "fail",
        message=f"[{episode_id}] {'solved' if success else 'failed'} reward={reward}",
    )
    return index, item_run, test_result, success


def run_scripted_multi_turn_suite(
    *,
    task: dict[str, Any],
    candidate_path: Path,
    proposal_runtime: ProposalRuntime,
    suite_name: str,
    domain: str,
    scripted_episodes: list[dict[str, Any]],
    suite_config: dict[str, Any],
    progress_callback=None,
) -> dict[str, Any]:
    item_runs: list[dict[str, Any]] = []
    test_results: list[dict[str, Any]] = []
    passed = 0
    parallel_workers = _resolve_parallel_workers(
        task=task,
        total_items=len(scripted_episodes),
        suite_config=suite_config,
    )
    ordered_rows: list[tuple[int, dict[str, Any], dict[str, Any], bool]] = []
    if parallel_workers <= 1:
        for index, scripted_episode in enumerate(scripted_episodes, start=1):
            ordered_rows.append(
                _run_scripted_episode(
                    task=task,
                    candidate_path=candidate_path,
                    proposal_runtime=proposal_runtime,
                    suite_name=suite_name,
                    domain=domain,
                    suite_config=suite_config,
                    scripted_episode=scripted_episode,
                    index=index,
                    progress_callback=progress_callback,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [
                executor.submit(
                    _run_scripted_episode,
                    task=task,
                    candidate_path=candidate_path,
                    proposal_runtime=proposal_runtime,
                    suite_name=suite_name,
                    domain=domain,
                    suite_config=suite_config,
                    scripted_episode=scripted_episode,
                    index=index,
                    progress_callback=progress_callback,
                )
                for index, scripted_episode in enumerate(scripted_episodes, start=1)
            ]
            for future in as_completed(futures):
                ordered_rows.append(future.result())

    for _index, item_run, test_result, success in sorted(ordered_rows, key=lambda row: row[0]):
        if success:
            passed += 1
        item_runs.append(item_run)
        test_results.append(test_result)

    total = len(item_runs)
    objective = passed / total if total else 0.0
    return {
        "status": "pass",
        "verifier_status": "pass",
        "correctness": objective,
        "passed_tests": passed,
        "total_tests": total,
        "benchmark_ms": None,
        "benchmark_samples_ms": [],
        "objective": objective,
        "objective_score": objective,
        "objective_signal": objective,
        "test_results": test_results,
        "item_runs": item_runs,
        "suite_summary": {
            "suite": suite_name,
            "domain": domain,
            "passed": passed,
            "total": total,
            "source": "scripted",
        },
    }
