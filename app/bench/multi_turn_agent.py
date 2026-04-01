from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

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
