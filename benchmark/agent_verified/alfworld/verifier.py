from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.bench.agent_benchmarks import (
    _action_summary,
    _episode_record,
    _history_message,
    _resolve_parallel_workers,
    _tool_result_message,
    _turn_record,
)
from app.bench.benchmark_adapter_support import (
    build_benchmark_adapter_candidate,
    build_benchmark_adapter_result,
    effective_suite_run_config,
    emit_progress,
    runtime_for_benchmark_adapter_task,
)
from app.bench.multi_turn_agent import AgentRuntime, load_agent_adapter, normalize_step_result


ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = Path(__file__).resolve().parent
ALFWORLD_REPO_ROOT = ROOT / "external" / "alfworld"
ALFWORLD_DATA_ROOT = TASK_ROOT / "data"
ALFWORLD_JSON_ROOT = ALFWORLD_DATA_ROOT / "json_2.1.1"
DEFAULT_SPLIT = "valid_seen"
DEFAULT_TASK_LIMIT = 140
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


def _configured_split(config: dict[str, Any]) -> str:
    split = str(config.get("episode_split") or DEFAULT_SPLIT).strip() or DEFAULT_SPLIT
    if split not in VALID_SPLITS:
        raise ValueError(f"Unsupported ALFWorld split {split!r}. Expected one of {sorted(VALID_SPLITS)}.")
    return split


def _task_limit(config: dict[str, Any], requested_limit: int | None) -> int:
    if isinstance(requested_limit, int) and requested_limit > 0:
        return requested_limit
    raw_value = config.get("task_limit")
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        parsed = DEFAULT_TASK_LIMIT
    return parsed if parsed > 0 else DEFAULT_TASK_LIMIT


def _max_turns(config: dict[str, Any]) -> int:
    raw_value = config.get("max_turns")
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        parsed = DEFAULT_MAX_TURNS
    return parsed if parsed > 0 else DEFAULT_MAX_TURNS


def _instruction_from_traj(traj_payload: dict[str, Any]) -> str:
    anns = list((traj_payload.get("turk_annotations") or {}).get("anns") or [])
    for ann in anns:
        if not isinstance(ann, dict):
            continue
        task_desc = str(ann.get("task_desc") or "").strip()
        if task_desc:
            return task_desc
    task_type = str(traj_payload.get("task_type") or "alfworld task").replace("_", " ").strip()
    return task_type.capitalize()


def _episode_specs_for_run(config: dict[str, Any], requested_limit: int | None) -> list[dict[str, Any]]:
    split = _configured_split(config)
    split_root = ALFWORLD_JSON_ROOT / split
    if not split_root.exists():
        raise FileNotFoundError(f"ALFWorld split directory not found: {split_root}")
    rows: list[dict[str, Any]] = []
    for game_path in sorted(split_root.rglob("game.tw-pddl")):
        root_text = str(game_path.parent)
        if "movable" in root_text or "Sliced" in root_text:
            continue
        traj_path = game_path.with_name("traj_data.json")
        if not traj_path.exists():
            continue
        game_payload = json.loads(game_path.read_text())
        if not bool(game_payload.get("solvable")):
            continue
        traj_payload = json.loads(traj_path.read_text())
        episode_id = "/".join(game_path.relative_to(split_root).parts[:-1])
        rows.append(
            {
                "episode_id": episode_id,
                "instruction": _instruction_from_traj(traj_payload),
                "game_file": str(game_path),
                "traj_file": str(traj_path),
                "split": split,
            }
        )
    if not rows:
        raise ValueError(f"No solvable ALFWorld episodes found under {split_root}")
    return rows[: _task_limit(config, requested_limit)]


def _extract_task_desc(observation_text: str, fallback: str) -> str:
    marker = "Your task is to: "
    if marker in observation_text:
        return observation_text.partition(marker)[-1].strip() or fallback
    return fallback


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


def _make_alfworld_env(game_file: str, max_turns: int):
    with _repo_on_sys_path(ALFWORLD_REPO_ROOT):
        try:
            import textworld  # type: ignore[import-not-found]
            import textworld.gym  # type: ignore[import-not-found]
            from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised through runtime only
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


def _run_episode(
    *,
    task: dict[str, Any],
    candidate_path: Path,
    proposal_runtime,
    suite_name: str,
    domain: str,
    suite_config: dict[str, Any],
    episode_spec: dict[str, Any],
    index: int,
    progress_callback=None,
    pace_ms: int = 0,
) -> tuple[int, dict[str, Any], dict[str, Any], bool, float]:
    init_episode, step = load_agent_adapter(candidate_path)
    runtime = AgentRuntime(proposal_runtime)
    max_turns = _max_turns(suite_config)
    env = _make_alfworld_env(str(episode_spec["game_file"]), max_turns)
    try:
        initial_obs, initial_infos = env.reset()
        observation_text = str(_batched_value(initial_obs) or "")
        instruction = _extract_task_desc(observation_text, str(episode_spec["instruction"]))
        admissible_commands = list(_batched_info(initial_infos, "admissible_commands", []) or [])
        episode = _episode_record(
            suite=suite_name,
            domain=domain,
            episode_id=str(episode_spec["episode_id"]),
            instruction=instruction,
            tools=list(suite_config.get("tools") or []),
            limits={"max_turns": max_turns},
            metadata={
                "split": episode_spec["split"],
                "game_file": str(episode_spec["game_file"]),
                "traj_file": str(episode_spec["traj_file"]),
            },
            policy={},
        )
        state_result = init_episode(episode)
        state = dict(state_result or {})
        if state_result is not None and not isinstance(state_result, dict):
            raise ValueError("init_episode(...) must return a dict or None.")

        history: list[dict[str, Any]] = []
        turns: list[dict[str, Any]] = []
        success = False
        reward = 0.0
        goal_condition_success_rate = 0.0

        emit_progress(
            progress_callback,
            task_id=str(task["id"]),
            phase="episode_started",
            item_id=str(episode_spec["episode_id"]),
            item_name=str(episode_spec["episode_id"]),
            item_brief=instruction,
            expected_answer="Solve episode",
            candidate_status="running",
            message=f"[{episode_spec['episode_id']}] {instruction}",
            pace_ms=pace_ms,
        )

        for turn_index in range(max_turns):
            observation = {
                "text": observation_text,
                "instruction": instruction,
                "valid_actions": admissible_commands,
                "won": success,
                "goal_condition_success_rate": goal_condition_success_rate,
            }
            turn_payload = _turn_record(
                episode=episode,
                turn_index=turn_index,
                history=history,
                observation=observation,
                state=state,
                metadata={"official_env": True},
            )
            action = normalize_step_result(step(turn_payload, runtime))
            tool_results: list[dict[str, Any]] = []
            episode_done = False

            if action["message"] is not None or action["tool_calls"]:
                history.append(_history_message("assistant", action["message"], tool_calls=action["tool_calls"]))

            for call_index, tool_call in enumerate(action["tool_calls"]):
                tool_name = str(tool_call.get("name") or "")
                arguments = dict(tool_call.get("arguments") or {})
                call_id = str(tool_call.get("id") or f"{episode_spec['episode_id']}-tool-{turn_index}-{call_index}")
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
                    tool_results.append({"tool_call": {**tool_call, "id": call_id}, "content": content, "error": False})
                    history.append(_tool_result_message(call_id, content))
                    continue
                if tool_name == "complete":
                    episode_done = True
                    content = json.dumps({"status": "completed", "message": str(arguments.get("message") or "")}, ensure_ascii=True)
                    tool_results.append({"tool_call": {**tool_call, "id": call_id}, "content": content, "error": False})
                    history.append(_tool_result_message(call_id, content))
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
            emit_progress(
                progress_callback,
                task_id=str(task["id"]),
                phase="episode_turn",
                item_id=str(episode_spec["episode_id"]),
                item_name=str(episode_spec["episode_id"]),
                item_brief=instruction,
                expected_answer="Solve episode",
                candidate_actual=_action_summary(action),
                candidate_status="running",
                message=f"[{episode_spec['episode_id']}] t{turn_index}: {_action_summary(action) or 'no action'}",
                pace_ms=pace_ms,
            )
            state = dict(action["state"])
            if action["done"] or episode_done:
                break

        item_run = {
            "item_id": str(episode_spec["episode_id"]),
            "item_name": str(episode_spec["episode_id"]),
            "payload": episode,
            "turns": turns,
            "success": success,
            "reward": reward,
            "annotations": {"goal_condition_success_rate": goal_condition_success_rate},
            "raw_artifact_path": str(episode_spec["game_file"]),
        }
        test_result = {
            "name": str(episode_spec["episode_id"]),
            "expected": 1.0,
            "actual": reward,
            "passed": success,
            "actual_raw": {
                "payload": episode,
                "turns": turns,
                "goal_condition_success_rate": goal_condition_success_rate,
            },
        }
        emit_progress(
            progress_callback,
            task_id=str(task["id"]),
            phase="episode_finished",
            item_id=str(episode_spec["episode_id"]),
            item_name=str(episode_spec["episode_id"]),
            item_brief=instruction,
            expected_answer="Solve episode",
            candidate_actual=_action_summary(dict(turns[-1].get("action") or {})) if turns else ("Episode solved" if success else "Episode not solved"),
            candidate_status="pass" if success else "fail",
            message=f"[{episode_spec['episode_id']}] {'solved' if success else 'failed'} reward={reward}",
            pace_ms=pace_ms,
        )
        return index, item_run, test_result, success, goal_condition_success_rate
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _run_official_alfworld_suite(
    *,
    task: dict[str, Any],
    candidate_path: Path,
    proposal_runtime,
    suite_name: str,
    domain: str,
    suite_config: dict[str, Any],
    requested_limit: int | None,
    progress_callback=None,
    pace_ms: int = 0,
) -> dict[str, Any]:
    episode_specs = _episode_specs_for_run(suite_config, requested_limit)
    ordered_rows: list[tuple[int, dict[str, Any], dict[str, Any], bool, float]] = []
    parallel_workers = _resolve_parallel_workers(
        task=task,
        proposal_runtime=proposal_runtime,
        total_items=len(episode_specs),
        suite_config=suite_config,
    )
    if parallel_workers <= 1:
        for index, episode_spec in enumerate(episode_specs, start=1):
            ordered_rows.append(
                _run_episode(
                    task=task,
                    candidate_path=candidate_path,
                    proposal_runtime=proposal_runtime,
                    suite_name=suite_name,
                    domain=domain,
                    suite_config=suite_config,
                    episode_spec=episode_spec,
                    index=index,
                    progress_callback=progress_callback,
                    pace_ms=pace_ms,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [
                executor.submit(
                    _run_episode,
                    task=task,
                    candidate_path=candidate_path,
                    proposal_runtime=proposal_runtime,
                    suite_name=suite_name,
                    domain=domain,
                    suite_config=suite_config,
                    episode_spec=episode_spec,
                    index=index,
                    progress_callback=progress_callback,
                    pace_ms=pace_ms,
                )
                for index, episode_spec in enumerate(episode_specs, start=1)
            ]
            for future in as_completed(futures):
                ordered_rows.append(future.result())

    item_runs: list[dict[str, Any]] = []
    test_results: list[dict[str, Any]] = []
    passed = 0
    goal_condition_total = 0.0
    for _index, item_run, test_result, success, goal_condition_success_rate in sorted(ordered_rows, key=lambda row: row[0]):
        if success:
            passed += 1
        goal_condition_total += goal_condition_success_rate
        item_runs.append(item_run)
        test_results.append(test_result)

    total = len(item_runs)
    objective = passed / total if total else 0.0
    average_goal_condition = goal_condition_total / total if total else 0.0
    average_steps = (sum(len(item.get("turns") or []) for item in item_runs) / total) if total else 0.0
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
            "source": "official_alfworld_text_env",
            "split": _configured_split(suite_config),
            "passed": passed,
            "total": total,
            "average_goal_condition_success_rate": average_goal_condition,
            "average_steps": average_steps,
        },
    }


def evaluate_candidate(
    *,
    task,
    candidate_path,
    source_code,
    baseline_metrics,
    memory_applied,
):
    del source_code, baseline_metrics, memory_applied
    config = effective_suite_run_config(task, Path(candidate_path))
    runtime = runtime_for_benchmark_adapter_task(task)
    requested_limit = task.get("runtime_max_episodes")
    return _run_official_alfworld_suite(
        task=task,
        candidate_path=Path(candidate_path),
        proposal_runtime=runtime,
        suite_name="alfworld-text",
        domain="alfworld",
        suite_config=config,
        requested_limit=requested_limit,
    )


def run_benchmark_adapter_task(
    *,
    task,
    candidate_path,
    source_code,
    proposal_runtime,
    workspace_root,
    session_id,
    max_items,
    max_episodes,
    progress_callback,
    pace_ms,
):
    del workspace_root, session_id, max_items
    config = effective_suite_run_config(task, Path(candidate_path))
    raw_metrics = _run_official_alfworld_suite(
        task=task,
        candidate_path=Path(candidate_path),
        proposal_runtime=proposal_runtime,
        suite_name="alfworld-text",
        domain="alfworld",
        suite_config=config,
        requested_limit=max_episodes,
        progress_callback=progress_callback,
        pace_ms=pace_ms,
    )
    baseline = build_benchmark_adapter_candidate(
        task=task,
        source_code=source_code,
        agent="checked-in-adapter",
        label="checked-in-adapter",
        strategy="Use the checked-in ALFWorld official text-environment adapter.",
        rationale="The checked-in file is the baseline multi-turn adapter for the official ALFWorld TextWorld environment.",
        candidate_summary="Checked-in ALFWorld official text adapter.",
        raw_metrics={"status": "not-run", "verifier_status": "not-run"},
        workspace_path=str(candidate_path),
    )
    winner = build_benchmark_adapter_candidate(
        task=task,
        source_code=source_code,
        agent="candidate-adapter",
        label="candidate-adapter",
        strategy="Run the candidate adapter against the official ALFWorld text environment.",
        rationale="The official ALFWorld environment owns admissible commands, task dynamics, and success scoring; the candidate owns turn-by-turn action selection.",
        candidate_summary="ALFWorld official text benchmark run.",
        raw_metrics=raw_metrics,
        workspace_path=str(candidate_path),
        proposal_model=proposal_runtime.active_model,
    )
    passed = int(raw_metrics.get("passed_tests") or 0)
    total = int(raw_metrics.get("total_tests") or 0)
    return build_benchmark_adapter_result(
        task=task,
        proposal_runtime=proposal_runtime,
        baseline=baseline,
        winner=winner,
        selection_reason=f"ALFWorld official text environment finished with {passed}/{total} solved episodes.",
        extra_fields={
            "suite_summary": dict(raw_metrics.get("suite_summary") or {}),
            "item_runs": list(raw_metrics.get("item_runs") or []),
        },
    )
