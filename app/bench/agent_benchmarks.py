from __future__ import annotations

import importlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.bench.benchmark_adapter_support import (
    build_benchmark_adapter_candidate,
    build_benchmark_adapter_result,
    emit_progress,
    effective_suite_run_config,
    ensure_repo_checkout,
    openai_compatible_env,
    require_command,
    run_command,
    runtime_for_benchmark_adapter_task,
    strip_socks_proxy_env,
)
from app.bench.multi_turn_agent import (
    MULTI_TURN_AGENT_CONTRACT,
    AgentRuntime,
    load_agent_adapter,
    normalize_step_result,
    validate_episode_payload,
    validate_turn_payload,
)
from app.codegen.llm import ProposalRuntime
from app.configs.paths import ROOT, RUNTIME_REPOS_ROOT


try:
    from harbor.agents.base import BaseAgent as HarborBaseAgent
    from harbor.environments.base import BaseEnvironment as HarborBaseEnvironment
    from harbor.models.agent.context import AgentContext as HarborAgentContext
except Exception:  # noqa: BLE001
    class HarborBaseAgent:  # type: ignore[too-many-ancestors]
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401, ANN003
            pass

    HarborBaseEnvironment = Any  # type: ignore[assignment]
    HarborAgentContext = Any  # type: ignore[assignment]


TERMINAL_BENCH_REPO_URL = "https://github.com/laude-institute/harbor.git"
TAU_BENCH_REPO_URL = "https://github.com/sierra-research/tau2-bench.git"


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "benchmark-adapter-job"


def runtime_repo_dir(repo_name: str) -> Path:
    return RUNTIME_REPOS_ROOT / repo_name


def _prefixed_model_name(model_name: str, provider: str) -> str:
    return model_name if "/" in model_name else f"{provider}/{model_name}"


def _coerce_tau_results_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = None
        for key in ("results", "rows", "trials", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                rows = value
                break
        if rows is None:
            raise RuntimeError("tau-bench results payload did not contain a result list.")
    else:
        raise RuntimeError("tau-bench results payload must be a JSON object or list.")
    return [dict(row) for row in rows if isinstance(row, dict)]


def _load_tau_results(result_path: Path, repo_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    candidates = [
        result_path,
        repo_dir / "data" / "simulations" / result_path.name,
        repo_dir / "data" / "tau2" / "simulations" / result_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return _coerce_tau_results_payload(json.loads(candidate.read_text())), candidate
    raise RuntimeError(f"No tau-bench result JSON file was found near {result_path}.")


def _legacy_tau_cli_is_likely(repo_dir: Path, error: RuntimeError) -> bool:
    lowered = str(error).lower()
    if not (repo_dir / "run.py").exists():
        return False
    if "tau2" not in lowered:
        return False
    missing_command_markers = (
        "failed to spawn",
        "command not found",
        "no such file or directory",
        "is not installed",
        "not found",
    )
    return any(marker in lowered for marker in missing_command_markers)


def _load_scripted_episodes(config: dict[str, Any]) -> list[dict[str, Any]] | None:
    inline_episodes = config.get("inline_episodes")
    if isinstance(inline_episodes, list):
        return [dict(item) for item in inline_episodes if isinstance(item, dict)]
    episodes_path = str(config.get("episodes_path") or "").strip()
    if not episodes_path:
        return None
    payload = json.loads(Path(episodes_path).read_text())
    if isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
        rows = payload["episodes"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("episodes_path must point to a JSON list or {'episodes': [...]} payload.")
    return [dict(item) for item in rows if isinstance(item, dict)]


def _default_terminal_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "exec",
            "description": "Run a shell command in the benchmark environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."},
                    "env": {
                        "type": "object",
                        "description": "Optional environment variables for this command.",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["command"],
            },
        },
        {
            "name": "complete",
            "description": "Declare the task complete and optionally summarize the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Short completion note."},
                },
            },
        },
    ]


def _tool_parameters_schema(tool: Any) -> dict[str, Any]:
    if isinstance(tool, dict):
        for key in ("parameters", "input_schema", "json_schema"):
            if isinstance(tool.get(key), dict):
                return dict(tool[key])
    for attribute in ("parameters", "input_schema", "json_schema"):
        value = getattr(tool, attribute, None)
        if isinstance(value, dict):
            return dict(value)
    model_json_schema = getattr(tool, "model_json_schema", None)
    if callable(model_json_schema):
        try:
            schema = model_json_schema()
        except Exception:  # noqa: BLE001
            return {"type": "object", "properties": {}}
        if isinstance(schema, dict):
            return schema
    return {"type": "object", "properties": {}}


def _normalize_external_tools(tools: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        name = ""
        description = ""
        if isinstance(tool, dict):
            name = str(tool.get("name") or "").strip()
            description = str(tool.get("description") or "").strip()
        else:
            name = str(getattr(tool, "name", "") or "").strip()
            description = str(getattr(tool, "description", "") or "").strip()
        if not name:
            continue
        normalized.append(
            {
                "name": name,
                "description": description,
                "parameters": _tool_parameters_schema(tool),
            }
        )
    return normalized


def _resolve_parallel_workers(
    *,
    task: dict[str, Any],
    proposal_runtime: ProposalRuntime,
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
    pace_ms: int = 0,
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
        tools=list(scripted_episode.get("tools") or suite_config.get("tools") or _default_terminal_tools()),
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
        pace_ms=pace_ms,
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
            pace_ms=pace_ms,
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
        pace_ms=pace_ms,
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
    pace_ms: int = 0,
) -> dict[str, Any]:
    item_runs: list[dict[str, Any]] = []
    test_results: list[dict[str, Any]] = []
    passed = 0
    parallel_workers = _resolve_parallel_workers(
        task=task,
        proposal_runtime=proposal_runtime,
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
                    pace_ms=pace_ms,
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
                    pace_ms=pace_ms,
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


class HarborBenchmarkAdapterAgent(HarborBaseAgent):
    CANDIDATE_PATH = ""
    SUITE_NAME = "terminal-bench"
    DOMAIN = "terminal"
    SUITE_CONFIG: dict[str, Any] = {}

    @staticmethod
    def name() -> str:
        return "autoresearch-terminal-adapter"

    def version(self) -> str:
        return "multi_turn_agent_v1"

    async def setup(self, environment: HarborBaseEnvironment) -> None:
        return None

    async def run(
        self,
        instruction: str,
        environment: HarborBaseEnvironment,
        context: HarborAgentContext,
    ) -> None:
        candidate_path = Path(self.CANDIDATE_PATH)
        init_episode, step = load_agent_adapter(candidate_path)
        runtime = AgentRuntime(ProposalRuntime.from_env().with_model(getattr(self, "model_name", None)))
        max_turns = int(self.SUITE_CONFIG.get("max_turns") or 20)
        episode = _episode_record(
            suite=self.SUITE_NAME,
            domain=self.DOMAIN,
            episode_id=_sanitize_name(f"{self.SUITE_NAME}-{int(time.time() * 1000)}"),
            instruction=instruction,
            tools=_default_terminal_tools(),
            limits={"max_turns": max_turns},
            metadata={"environment": type(environment).__name__},
            policy=dict(self.SUITE_CONFIG.get("policy") or {}),
        )
        state_result = init_episode(episode)
        state = dict(state_result or {})
        history: list[dict[str, Any]] = []
        turns: list[dict[str, Any]] = []
        last_tool_results: list[dict[str, Any]] = []

        for turn_index in range(max_turns):
            observation = {
                "instruction": instruction if turn_index == 0 else None,
                "last_tool_results": last_tool_results,
            }
            turn_payload = _turn_record(
                episode=episode,
                turn_index=turn_index,
                history=history,
                observation=observation,
                state=state,
            )
            action = normalize_step_result(step(turn_payload, runtime))
            tool_results: list[dict[str, Any]] = []
            completed = action["done"]
            completion_message = action["message"]

            if action["message"] is not None or action["tool_calls"]:
                history.append(_history_message("assistant", action["message"], tool_calls=action["tool_calls"]))

            for tool_call in action["tool_calls"]:
                tool_name = str(tool_call.get("name") or "")
                arguments = dict(tool_call.get("arguments") or {})
                call_id = str(tool_call.get("id") or f"terminal-{turn_index}-{len(tool_results)}")
                if tool_name == "exec":
                    exec_result = await environment.exec(
                        command=str(arguments.get("command") or ""),
                        env={
                            str(key): str(value)
                            for key, value in dict(arguments.get("env") or {}).items()
                        }
                        or None,
                    )
                    content = json.dumps(
                        {
                            "stdout": exec_result.stdout,
                            "stderr": exec_result.stderr,
                            "return_code": exec_result.return_code,
                        },
                        ensure_ascii=True,
                    )
                    tool_results.append(
                        {
                            "tool_call": {**tool_call, "id": call_id},
                            "content": content,
                            "error": exec_result.return_code != 0,
                        }
                    )
                    history.append(_tool_result_message(call_id, content, error=exec_result.return_code != 0))
                    continue
                if tool_name == "complete":
                    completed = True
                    completion_message = str(arguments.get("message") or completion_message or "").strip() or None
                    tool_results.append(
                        {
                            "tool_call": {**tool_call, "id": call_id},
                            "content": json.dumps({"status": "completed", "message": completion_message}, ensure_ascii=True),
                            "error": False,
                        }
                    )
                    history.append(_tool_result_message(call_id, json.dumps({"status": "completed"}, ensure_ascii=True)))
                    continue
                content = json.dumps({"error": f"Unknown tool {tool_name!r}"}, ensure_ascii=True)
                tool_results.append({"tool_call": {**tool_call, "id": call_id}, "content": content, "error": True})
                history.append(_tool_result_message(call_id, content, error=True))

            turns.append(
                {
                    "turn_index": turn_index,
                    "observation": observation,
                    "action": action,
                    "tool_results": tool_results,
                }
            )
            last_tool_results = [
                {
                    "name": str(item["tool_call"].get("name") or ""),
                    "content": item["content"],
                    "error": bool(item["error"]),
                }
                for item in tool_results
            ]
            state = dict(action["state"])
            if completed:
                break

        trace_path = Path(self.logs_dir) / "agent_adapter_trace.json"
        trace_path.write_text(json.dumps({"episode": episode, "turns": turns}, indent=2))
        metadata = {
            "episode": episode,
            "turns": turns,
            "trace_path": str(trace_path),
            "completion_message": completion_message,
        }
        if getattr(context, "metadata", None):
            metadata = {**dict(context.metadata), **metadata}
        context.metadata = metadata


def _write_harbor_agent_bridge(
    *,
    bridge_dir: Path,
    task_id: str,
    candidate_path: Path,
    suite_config: dict[str, Any],
) -> str:
    bridge_dir.mkdir(parents=True, exist_ok=True)
    module_stem = f"bridge_{_sanitize_name(task_id)}"
    bridge_path = bridge_dir / f"{module_stem}.py"
    bridge_path.write_text(
        "\n".join(
            [
                "from app.bench.agent_benchmarks import HarborBenchmarkAdapterAgent",
                "",
                "class CandidateAgent(HarborBenchmarkAdapterAgent):",
                f"    CANDIDATE_PATH = {candidate_path.as_posix()!r}",
                f"    SUITE_NAME = {task_id!r}",
                "    DOMAIN = 'terminal'",
                f"    SUITE_CONFIG = {json.dumps(suite_config, ensure_ascii=True, sort_keys=False)}",
                "",
            ]
        )
    )
    return f"{module_stem}:CandidateAgent"


def _latest_job_dir(jobs_dir: Path, preferred_name: str) -> Path:
    direct = jobs_dir / preferred_name
    if direct.exists():
        return direct
    candidates = [path for path in jobs_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No Harbor job directory found under {jobs_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_harbor_trial_results(job_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for result_path in sorted(job_dir.rglob("result.json")):
        if result_path.parent == job_dir:
            continue
        payload = json.loads(result_path.read_text())
        if "verifier_result" in payload:
            results.append(payload)
    if not results:
        raise RuntimeError(f"No Harbor trial result.json files found under {job_dir}")
    return results


def _ensure_docker_available() -> None:
    require_command("docker")
    try:
        run_command(["docker", "info"], cwd=ROOT, timeout_s=20)
    except RuntimeError as exc:
        raise RuntimeError(
            "Docker is installed but the daemon is not available. terminal-bench requires a running local Docker daemon."
        ) from exc


def _normalize_harbor_metrics(
    *,
    trial_results: list[dict[str, Any]],
    dataset_name: str,
    model_name: str,
    job_dir: Path,
    elapsed_ms: float,
) -> dict[str, Any]:
    passed = 0
    item_runs: list[dict[str, Any]] = []
    test_results: list[dict[str, Any]] = []
    for trial in trial_results:
        rewards = dict(trial.get("verifier_result", {}).get("rewards") or {})
        reward = float(rewards.get("reward") or 0.0)
        succeeded = reward >= 1.0 - 1e-6
        if succeeded:
            passed += 1
        agent_metadata = dict((trial.get("agent_result") or {}).get("metadata") or {})
        item_runs.append(
            {
                "item_id": str(trial.get("trial_name") or trial.get("task_name") or f"trial-{len(item_runs)+1}"),
                "item_name": str(trial.get("task_name") or trial.get("trial_name") or ""),
                "payload": dict(agent_metadata.get("episode") or {}),
                "turns": list(agent_metadata.get("turns") or []),
                "success": succeeded,
                "reward": reward,
                "raw_artifact_path": agent_metadata.get("trace_path"),
            }
        )
        test_results.append(
            {
                "name": trial.get("task_name") or trial.get("trial_name"),
                "expected": 1.0,
                "actual": reward,
                "passed": succeeded,
                "actual_raw": {
                    "trial_name": trial.get("trial_name"),
                    "trial_uri": trial.get("trial_uri"),
                    "rewards": rewards,
                },
            }
        )
    total = len(trial_results)
    objective = passed / total if total else 0.0
    return {
        "status": "pass",
        "verifier_status": "pass",
        "correctness": objective,
        "passed_tests": passed,
        "total_tests": total,
        "benchmark_ms": round(elapsed_ms, 3),
        "benchmark_samples_ms": [round(elapsed_ms, 3)],
        "objective": objective,
        "objective_score": objective,
        "objective_signal": objective,
        "test_results": test_results,
        "item_runs": item_runs,
        "suite_summary": {
            "dataset": dataset_name,
            "model": model_name,
            "passed": passed,
            "total": total,
            "job_dir": str(job_dir),
        },
    }


def evaluate_harbor_terminal_candidate(
    *,
    task: dict[str, Any],
    candidate_path: Path,
    source_code: str,
    workspace_root: Path,
    session_id: str,
    max_items: int | None,
    max_episodes: int | None,
    proposal_runtime: ProposalRuntime | None = None,
    progress_callback=None,
    pace_ms: int = 0,
) -> dict[str, Any]:
    del source_code
    config = effective_suite_run_config(task, candidate_path)
    scripted_episodes = _load_scripted_episodes(config)
    runtime = proposal_runtime or runtime_for_benchmark_adapter_task(task)
    if scripted_episodes is not None:
        return run_scripted_multi_turn_suite(
            task=task,
            candidate_path=candidate_path,
            proposal_runtime=runtime,
            suite_name="terminal-bench",
            domain="terminal",
            scripted_episodes=scripted_episodes[: max_episodes or len(scripted_episodes)],
            suite_config=config,
            progress_callback=progress_callback,
            pace_ms=pace_ms,
        )

    require_command("git")
    require_command("uv")
    _ensure_docker_available()
    repo_dir = ensure_repo_checkout(str(config.get("repo_url") or TERMINAL_BENCH_REPO_URL), runtime_repo_dir("harbor"))
    jobs_dir = workspace_root / "jobs"
    bridge_dir = workspace_root / "bridge_agents"
    dataset_name = str(config.get("dataset") or "terminal-bench@2.0")
    provider = str(config.get("model_provider") or "openai")
    model_name = _prefixed_model_name(str(config.get("model_name") or runtime.active_model), provider)
    task_limit = int(max_episodes if isinstance(max_episodes, int) and max_episodes > 0 else config.get("task_limit") or 5)
    max_concurrency = _resolve_parallel_workers(
        task=task,
        proposal_runtime=runtime,
        total_items=task_limit,
        suite_config=config,
    )
    job_name = _sanitize_name(f"{task['id']}-{session_id}")
    agent_import_path = _write_harbor_agent_bridge(
        bridge_dir=bridge_dir,
        task_id=str(task["id"]),
        candidate_path=candidate_path,
        suite_config=config,
    )
    env = strip_socks_proxy_env(dict(os.environ))
    env.update(openai_compatible_env(runtime))
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(bridge_dir), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)

    emit_progress(
        progress_callback,
        task_id=str(task["id"]),
        phase="benchmark_adapter_started",
        message=f"Running Harbor terminal-bench with custom adapter over {task_limit} tasks",
        pace_ms=pace_ms,
    )
    started = time.perf_counter()
    run_command(
        [
            "uv",
            "run",
            "harbor",
            "run",
            "--dataset",
            dataset_name,
            "--agent-import-path",
            agent_import_path,
            "--model",
            model_name,
            "--job-name",
            job_name,
            "--jobs-dir",
            str(jobs_dir),
            "--n-tasks",
            str(task_limit),
            "--n-concurrent",
            str(max_concurrency),
            "--quiet",
        ],
        cwd=repo_dir,
        env=env,
        timeout_s=int(config.get("timeout_s") or 7200),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    job_dir = _latest_job_dir(jobs_dir, job_name)
    return _normalize_harbor_metrics(
        trial_results=_load_harbor_trial_results(job_dir),
        dataset_name=dataset_name,
        model_name=model_name,
        job_dir=job_dir,
        elapsed_ms=elapsed_ms,
    )


def run_harbor_terminal_bench(
    *,
    task: dict[str, Any],
    candidate_path: Path,
    source_code: str,
    proposal_runtime: ProposalRuntime,
    workspace_root: Path,
    session_id: str,
    max_items: int | None,
    max_episodes: int | None,
    progress_callback,
    pace_ms: int,
) -> dict[str, Any]:
    raw_metrics = evaluate_harbor_terminal_candidate(
        task=task,
        candidate_path=candidate_path,
        source_code=source_code,
        proposal_runtime=proposal_runtime,
        workspace_root=workspace_root,
        session_id=session_id,
        max_items=max_items,
        max_episodes=max_episodes,
        progress_callback=progress_callback,
        pace_ms=pace_ms,
    )
    suite_summary = dict(raw_metrics.get("suite_summary") or {})
    passed = int(raw_metrics.get("passed_tests") or 0)
    total = int(raw_metrics.get("total_tests") or 0)
    baseline = build_benchmark_adapter_candidate(
        task=task,
        source_code=source_code,
        agent="checked-in-adapter",
        label="checked-in-adapter",
        strategy="Use the checked-in shared multi-turn terminal agent adapter.",
        rationale="The checked-in editable.py defines the benchmark adapter behavior that the terminal benchmark executes.",
        candidate_summary="Checked-in shared terminal benchmark adapter.",
        raw_metrics={"status": "not-run", "verifier_status": "not-run"},
        workspace_path=str(candidate_path),
    )
    winner = build_benchmark_adapter_candidate(
        task=task,
        source_code=source_code,
        agent="candidate-adapter",
        label="candidate-adapter",
        strategy="Evaluate the candidate's multi-turn terminal adapter against the official Harbor environment.",
        rationale="The benchmark loop owns the environment and scoring; the candidate owns turn-by-turn tool use and completion logic.",
        candidate_summary="Shared terminal benchmark adapter run over the selected Harbor slice.",
        raw_metrics=raw_metrics,
        workspace_path=str(suite_summary.get("job_dir") or workspace_root),
        proposal_model=str(suite_summary.get("model") or proposal_runtime.active_model),
    )
    return build_benchmark_adapter_result(
        task=task,
        proposal_runtime=proposal_runtime,
        baseline=baseline,
        winner=winner,
        selection_reason=f"terminal-bench finished with {passed}/{total} successful episodes.",
        extra_fields={
            "suite_summary": suite_summary,
            "item_runs": list(raw_metrics.get("item_runs") or []),
        },
    )


@contextmanager
def _sys_path(path: Path):
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(path))
        except ValueError:
            pass


def _tau_task_instruction(task_obj: Any) -> str:
    if hasattr(task_obj, "user_scenario") and getattr(task_obj.user_scenario, "instructions", None):
        return str(task_obj.user_scenario.instructions)
    for key in ("instruction", "prompt", "description"):
        value = getattr(task_obj, key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(task_obj, dict):
        for key in ("instruction", "prompt", "description"):
            value = task_obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise ValueError("Unable to extract tau-bench task instruction.")


def _tau_task_id(task_obj: Any) -> str:
    value = getattr(task_obj, "id", None)
    if value is None and isinstance(task_obj, dict):
        value = task_obj.get("id")
    return str(value or "tau-task")


def _tau_policy_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _normalize_tau_history(messages: list[Any]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for message in messages:
        role = getattr(message, "role", None)
        if hasattr(role, "value"):
            role = role.value
        role_text = str(role or "")
        content = getattr(message, "content", None)
        tool_calls = getattr(message, "tool_calls", None)
        normalized_tool_calls: list[dict[str, Any]] = []
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                normalized_tool_calls.append(
                    {
                        "id": str(getattr(tool_call, "id", "") or ""),
                        "name": str(getattr(tool_call, "name", "") or ""),
                        "arguments": dict(getattr(tool_call, "arguments", {}) or {}),
                    }
                )
        history.append(_history_message(role_text, content, tool_calls=normalized_tool_calls or None))
    return history


def _normalize_tau_result_messages(messages: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for message in messages:
        role = getattr(message, "role", None)
        if hasattr(role, "value"):
            role = role.value
        payload = {"role": str(role or ""), "content": getattr(message, "content", None)}
        if isinstance(getattr(message, "tool_calls", None), list):
            payload["tool_calls"] = [
                {
                    "id": str(getattr(tool_call, "id", "") or ""),
                    "name": str(getattr(tool_call, "name", "") or ""),
                    "arguments": dict(getattr(tool_call, "arguments", {}) or {}),
                }
                for tool_call in message.tool_calls
            ]
        rows.append(payload)
    return rows


def _tau_tool_calls(module: Any, tool_calls: list[dict[str, Any]]) -> list[Any]:
    ToolCall = getattr(module, "ToolCall")
    return [
        ToolCall(
            id=str(tool_call.get("id") or f"tool-{index}"),
            name=str(tool_call.get("name") or ""),
            arguments=dict(tool_call.get("arguments") or {}),
            requestor="assistant",
        )
        for index, tool_call in enumerate(tool_calls, start=1)
    ]


def _import_tau_modules(repo_dir: Path) -> dict[str, Any]:
    with _sys_path(repo_dir / "src"):
        modules = {
            "base_agent": importlib.import_module("tau2.agent.base_agent"),
            "message": importlib.import_module("tau2.data_model.message"),
            "simulation": importlib.import_module("tau2.data_model.simulation"),
            "runner": importlib.import_module("tau2.runner"),
            "registry": importlib.import_module("tau2.registry"),
            "orchestrator": importlib.import_module("tau2.orchestrator.orchestrator"),
            "evaluation": importlib.import_module("tau2.evaluator.evaluator"),
        }
    return modules


def _evaluate_tau_with_official_api(
    *,
    task: dict[str, Any],
    candidate_path: Path,
    proposal_runtime: ProposalRuntime,
    suite_config: dict[str, Any],
    env_name: str,
    max_items: int | None,
) -> dict[str, Any]:
    repo_dir = ensure_repo_checkout(str(suite_config.get("repo_url") or TAU_BENCH_REPO_URL), runtime_repo_dir("tau2-bench"))
    modules = _import_tau_modules(repo_dir)
    HalfDuplexAgent = getattr(modules["base_agent"], "HalfDuplexAgent")
    AssistantMessage = getattr(modules["message"], "AssistantMessage")
    TextRunConfig = getattr(modules["simulation"], "TextRunConfig")
    runner = modules["runner"]
    EvaluationType = getattr(modules["evaluation"], "EvaluationType")
    Orchestrator = getattr(modules["orchestrator"], "Orchestrator")

    external_runtime = AgentRuntime(proposal_runtime)
    init_episode_hook, step_hook = load_agent_adapter(candidate_path)

    class TauAdapterAgent(HalfDuplexAgent[list]):  # type: ignore[misc]
        def __init__(self, tools: list[Any], domain_policy: str, *, task_obj: Any, llm: str) -> None:
            super().__init__(tools=tools, domain_policy=domain_policy)
            self.task_obj = task_obj
            self.llm = llm
            self.turns: list[dict[str, Any]] = []
            self._episode = _episode_record(
                suite="tau-bench",
                domain=env_name,
                episode_id=_tau_task_id(task_obj),
                instruction=_tau_task_instruction(task_obj),
                tools=_normalize_external_tools(tools),
                limits={"max_turns": int(suite_config.get("max_turns") or 20)},
                metadata={"task_id": _tau_task_id(task_obj)},
                policy={"domain_policy": _tau_policy_text(domain_policy)},
            )

        def get_init_state(self, message_history: list[Any] | None = None) -> list[Any]:
            init_state = init_episode_hook(self._episode)
            if init_state is not None and not isinstance(init_state, dict):
                raise ValueError("init_episode(...) must return a dict or None.")
            state_payload = dict(init_state or {})
            history = list(message_history) if message_history else []
            return [history, state_payload]

        def generate_next_message(self, message: Any, state: list[Any]) -> tuple[Any, list[Any]]:
            history_messages = list(state[0])
            state_payload = dict(state[1] or {})
            history_messages.append(message)
            turn_payload = _turn_record(
                episode=self._episode,
                turn_index=len(self.turns),
                history=_normalize_tau_history(history_messages),
                observation={"incoming_message": getattr(message, "content", None)},
                state=state_payload,
            )
            action = normalize_step_result(step_hook(turn_payload, external_runtime))
            assistant_message = AssistantMessage.text(
                action["message"] or "Done.",
                tool_calls=_tau_tool_calls(modules["message"], action["tool_calls"]) or None,
                raw_data={"annotations": action["annotations"]} if action["annotations"] else None,
            )
            history_messages.append(assistant_message)
            self.turns.append(
                {
                    "turn_index": len(self.turns),
                    "observation": {"incoming_message": getattr(message, "content", None)},
                    "action": action,
                }
            )
            return assistant_message, [history_messages, dict(action["state"])]

    tasks = runner.get_tasks(
        env_name,
        task_ids=list(suite_config.get("task_ids") or []),
    )
    if max_items is not None and max_items > 0:
        tasks = list(tasks)[:max_items]
    elif int(suite_config.get("task_limit") or 0) > 0:
        tasks = list(tasks)[: int(suite_config["task_limit"])]
    else:
        tasks = list(tasks)

    user_model = str(suite_config.get("user_model") or proposal_runtime.active_model)
    agent_model = str(suite_config.get("agent_model") or proposal_runtime.active_model)
    item_runs: list[dict[str, Any]] = []
    test_results: list[dict[str, Any]] = []
    passed = 0
    started = time.perf_counter()
    for task_obj in tasks:
        environment = runner.build_environment(env_name)
        agent = TauAdapterAgent(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            task_obj=task_obj,
            llm=agent_model,
        )
        user = runner.build_user(
            str(suite_config.get("user") or "user_simulator"),
            environment,
            task_obj,
            llm=user_model,
        )
        orchestrator = Orchestrator(
            domain=env_name,
            agent=agent,
            user=user,
            environment=environment,
            task=task_obj,
            max_steps=int(suite_config.get("max_turns") or 20),
            max_errors=int(suite_config.get("max_errors") or 5),
            seed=int(suite_config.get("seed") or 42),
        )
        result = runner.run_simulation(orchestrator, evaluation_type=EvaluationType.ALL)
        reward = float(getattr(getattr(result, "reward_info", None), "reward", 0.0) or 0.0)
        succeeded = reward >= 1.0 - 1e-6
        if succeeded:
            passed += 1
        episode_id = _tau_task_id(task_obj)
        item_runs.append(
            {
                "item_id": episode_id,
                "item_name": episode_id,
                "payload": agent._episode,
                "turns": list(agent.turns),
                "success": succeeded,
                "reward": reward,
                "raw_artifact_path": None,
                "messages": _normalize_tau_result_messages(list(getattr(result, "messages", []) or [])),
            }
        )
        test_results.append(
            {
                "name": episode_id,
                "expected": 1.0,
                "actual": reward,
                "passed": succeeded,
                "actual_raw": {
                    "messages": _normalize_tau_result_messages(list(getattr(result, "messages", []) or [])),
                },
            }
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    total = len(item_runs)
    objective = passed / total if total else 0.0
    return {
        "status": "pass",
        "verifier_status": "pass",
        "correctness": objective,
        "passed_tests": passed,
        "total_tests": total,
        "benchmark_ms": round(elapsed_ms, 3),
        "benchmark_samples_ms": [round(elapsed_ms, 3)],
        "objective": objective,
        "objective_score": objective,
        "objective_signal": objective,
        "test_results": test_results,
        "item_runs": item_runs,
        "suite_summary": {
            "domain": env_name,
            "task_split": suite_config.get("task_split"),
            "passed": passed,
            "total": total,
            "agent_model": agent_model,
            "user_model": user_model,
        },
    }


def evaluate_tau_bench_candidate(
    *,
    task: dict[str, Any],
    candidate_path: Path,
    source_code: str,
    workspace_root: Path,
    session_id: str,
    max_items: int | None,
    max_episodes: int | None,
    env_name: str,
    proposal_runtime: ProposalRuntime | None = None,
    progress_callback=None,
    pace_ms: int = 0,
) -> dict[str, Any]:
    del source_code, workspace_root, session_id
    config = effective_suite_run_config(task, candidate_path)
    runtime = proposal_runtime or runtime_for_benchmark_adapter_task(task)
    scripted_episodes = _load_scripted_episodes(config)
    if scripted_episodes is not None:
        return run_scripted_multi_turn_suite(
            task=task,
            candidate_path=candidate_path,
            proposal_runtime=runtime,
            suite_name="tau-bench",
            domain=env_name,
            scripted_episodes=scripted_episodes[: max_episodes or len(scripted_episodes)],
            suite_config=config,
            progress_callback=progress_callback,
            pace_ms=pace_ms,
        )
    emit_progress(
        progress_callback,
        task_id=str(task["id"]),
        phase="benchmark_adapter_started",
        message=f"Running tau-bench {env_name} with shared multi-turn adapter",
        pace_ms=pace_ms,
    )
    return _evaluate_tau_with_official_api(
        task=task,
        candidate_path=candidate_path,
        proposal_runtime=runtime,
        suite_config=config,
        env_name=env_name,
        max_items=max_episodes if isinstance(max_episodes, int) and max_episodes > 0 else max_items,
    )


def run_tau_bench_suite(
    *,
    task: dict[str, Any],
    candidate_path: Path,
    source_code: str,
    proposal_runtime: ProposalRuntime,
    workspace_root: Path,
    session_id: str,
    max_items: int | None,
    max_episodes: int | None,
    env_name: str,
    progress_callback,
    pace_ms: int,
) -> dict[str, Any]:
    raw_metrics = evaluate_tau_bench_candidate(
        task=task,
        candidate_path=candidate_path,
        source_code=source_code,
        proposal_runtime=proposal_runtime,
        workspace_root=workspace_root,
        session_id=session_id,
        max_items=max_items,
        max_episodes=max_episodes,
        env_name=env_name,
        progress_callback=progress_callback,
        pace_ms=pace_ms,
    )
    suite_summary = dict(raw_metrics.get("suite_summary") or {})
    passed = int(raw_metrics.get("passed_tests") or 0)
    total = int(raw_metrics.get("total_tests") or 0)
    baseline = build_benchmark_adapter_candidate(
        task=task,
        source_code=source_code,
        agent="checked-in-adapter",
        label="checked-in-adapter",
        strategy="Use the checked-in shared tau-bench adapter.",
        rationale="The checked-in editable.py defines the turn-by-turn policy that tau-bench evaluates.",
        candidate_summary="Checked-in shared tau-bench adapter.",
        raw_metrics={"status": "not-run", "verifier_status": "not-run"},
        workspace_path=str(candidate_path),
    )
    winner = build_benchmark_adapter_candidate(
        task=task,
        source_code=source_code,
        agent="candidate-adapter",
        label="candidate-adapter",
        strategy="Evaluate the candidate's multi-turn conversation adapter against the official tau-bench loop.",
        rationale="The suite owns the environment, user simulator, and reward function; the candidate owns agent actions.",
        candidate_summary=f"tau-bench {env_name} run over the selected task slice.",
        raw_metrics=raw_metrics,
        workspace_path=str(workspace_root),
        proposal_model=proposal_runtime.active_model,
    )
    return build_benchmark_adapter_result(
        task=task,
        proposal_runtime=proposal_runtime,
        baseline=baseline,
        winner=winner,
        selection_reason=f"tau-bench {env_name} finished with {passed}/{total} successful episodes.",
        extra_fields={
            "suite_summary": suite_summary,
            "item_runs": list(raw_metrics.get("item_runs") or []),
        },
    )
