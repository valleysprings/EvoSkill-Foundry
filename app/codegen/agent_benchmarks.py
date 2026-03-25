from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from app.codegen.external import (
    build_external_candidate,
    build_external_result,
    emit_progress,
    effective_external_run_config,
    ensure_repo_checkout,
    openai_compatible_env,
    require_command,
    render_external_run_config_source,
    runtime_for_external_task,
    run_command,
    strip_socks_proxy_env,
)
from app.codegen.llm import ProposalRuntime
from app.configs.paths import ROOT


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "external-job"


def _prefixed_model_name(model_name: str, provider: str) -> str:
    return model_name if "/" in model_name else f"{provider}/{model_name}"


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
            "Docker is installed but the daemon is not available. "
            "Harbor terminal-bench requires a running local Docker daemon."
        ) from exc


def _prefixed_tau_model_name(model_name: str, provider: str) -> str:
    provider = provider.strip()
    if not provider:
        return model_name
    return _prefixed_model_name(model_name, provider)


def _tau_result_candidates(result_path: Path, repo_dir: Path) -> list[Path]:
    base_name = result_path.name
    stem = result_path.stem
    candidates = [
        result_path,
        result_path.with_suffix(".json"),
        repo_dir / "data" / "simulations" / base_name,
        repo_dir / "data" / "simulations" / f"{stem}.json",
        repo_dir / "data" / "tau2" / "simulations" / base_name,
        repo_dir / "data" / "tau2" / "simulations" / f"{stem}.json",
    ]
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


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
    for candidate in _tau_result_candidates(result_path, repo_dir):
        if candidate.exists():
            return _coerce_tau_results_payload(json.loads(candidate.read_text())), candidate
    search_roots = [
        repo_dir / "data" / "simulations",
        repo_dir / "data" / "tau2" / "simulations",
    ]
    fallback_matches: list[Path] = []
    for search_root in search_roots:
        if not search_root.exists():
            continue
        fallback_matches.extend(sorted(search_root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True))
    if fallback_matches:
        latest = fallback_matches[0]
        return _coerce_tau_results_payload(json.loads(latest.read_text())), latest
    raise RuntimeError(
        "No tau-bench result JSON file was found in the expected output locations. "
        f"Checked around {result_path} and {repo_dir / 'data'}."
    )


def _legacy_tau_cli_is_likely(repo_dir: Path, error: RuntimeError) -> bool:
    lowered = str(error).lower()
    if not (repo_dir / "run.py").exists():
        return False
    legacy_markers = (
        "failed to spawn",
        "no such command",
        "unrecognized arguments",
        "can't open file",
        "no module named tau2",
    )
    return any(marker in lowered for marker in legacy_markers)


def _run_modern_tau_bench(
    *,
    repo_dir: Path,
    env_name: str,
    model_name: str,
    model_provider: str,
    user_model: str,
    user_model_provider: str,
    max_concurrency: int,
    task_limit: int,
    session_id: str,
    workspace_root: Path,
    env: dict[str, str],
    timeout_s: int,
) -> tuple[list[dict[str, Any]], Path]:
    result_path = workspace_root / f"{_sanitize_name(f'tau-{env_name}-{session_id}')}.json"
    agent_llm = _prefixed_tau_model_name(model_name, model_provider)
    user_llm = _prefixed_tau_model_name(user_model, user_model_provider)
    run_command(
        [
            "uv",
            "run",
            "tau2",
            "run",
            "--domain",
            env_name,
            "--agent-llm",
            agent_llm,
            "--user-llm",
            user_llm,
            "--num-trials",
            "1",
            "--num-tasks",
            str(task_limit),
            "--max-concurrency",
            str(max_concurrency),
            "--save-to",
            str(result_path),
        ],
        cwd=repo_dir,
        env=env,
        timeout_s=timeout_s,
    )
    results, resolved_result_path = _load_tau_results(result_path, repo_dir)
    return results, resolved_result_path


def _run_legacy_tau_bench(
    *,
    repo_dir: Path,
    env_name: str,
    model_name: str,
    model_provider: str,
    user_model: str,
    user_model_provider: str,
    agent_strategy: str,
    user_strategy: str,
    max_concurrency: int,
    task_limit: int,
    log_dir: Path,
    env: dict[str, str],
    timeout_s: int,
) -> tuple[list[dict[str, Any]], Path]:
    run_command(
        [
            "uv",
            "run",
            "python",
            "run.py",
            "--env",
            env_name,
            "--agent-strategy",
            agent_strategy,
            "--model",
            model_name,
            "--model-provider",
            model_provider,
            "--user-model",
            user_model,
            "--user-model-provider",
            user_model_provider,
            "--user-strategy",
            user_strategy,
            "--max-concurrency",
            str(max_concurrency),
            "--start-index",
            "0",
            "--end-index",
            str(task_limit),
            "--log-dir",
            str(log_dir),
        ],
        cwd=repo_dir,
        env=env,
        timeout_s=timeout_s,
    )
    result_files = sorted(log_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
    if not result_files:
        raise RuntimeError(f"No tau-bench result JSON files found under {log_dir}")
    results = _coerce_tau_results_payload(json.loads(result_files[-1].read_text()))
    return results, result_files[-1]


def evaluate_harbor_terminal_candidate(
    *,
    task: dict[str, Any],
    candidate_path: Path,
    source_code: str,
    workspace_root: Path,
    session_id: str,
    max_items: int | None,
    proposal_runtime: ProposalRuntime | None = None,
    progress_callback=None,
    pace_ms: int = 0,
) -> dict[str, Any]:
    config = effective_external_run_config(task, candidate_path)
    runtime = proposal_runtime or runtime_for_external_task(task)
    require_command("git")
    require_command("uv")
    _ensure_docker_available()
    repo_dir = ensure_repo_checkout(
        str(config.get("repo_url") or "https://github.com/laude-institute/harbor.git"),
        ROOT / "external" / "harbor",
    )
    jobs_dir = workspace_root / "jobs"
    job_name = _sanitize_name(f"{task['id']}-{session_id}")
    dataset_name = str(config.get("dataset") or "terminal-bench@2.0")
    agent_name = str(config.get("agent_name") or "terminus-2")
    provider = str(config.get("model_provider") or "openai")
    model_name = _prefixed_model_name(str(config.get("model_name") or runtime.active_model), provider)
    n_tasks = int(max_items if isinstance(max_items, int) and max_items > 0 else config.get("n_tasks") or 5)
    n_concurrent = int(config.get("n_concurrent") or 1)
    env = strip_socks_proxy_env(dict(os.environ))
    env.update(openai_compatible_env(runtime))

    emit_progress(
        progress_callback,
        task_id=str(task["id"]),
        phase="external_harness_started",
        message=f"Running Harbor {dataset_name} with {agent_name}",
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
            "--agent",
            agent_name,
            "--model",
            model_name,
            "--job-name",
            job_name,
            "--jobs-dir",
            str(jobs_dir),
            "--n-tasks",
            str(n_tasks),
            "--n-concurrent",
            str(n_concurrent),
            "--quiet",
        ],
        cwd=repo_dir,
        env=env,
        timeout_s=int(config.get("timeout_s") or 7200),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    job_dir = _latest_job_dir(jobs_dir, job_name)
    trial_results = _load_harbor_trial_results(job_dir)

    passed = 0
    test_results = []
    for trial in trial_results:
        rewards = dict(trial.get("verifier_result", {}).get("rewards") or {})
        reward = float(rewards.get("reward") or 0.0)
        succeeded = reward >= 1.0 - 1e-6
        if succeeded:
            passed += 1
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
        "external_summary": {
            "dataset": dataset_name,
            "agent": agent_name,
            "model": model_name,
            "passed": passed,
            "total": total,
            "job_dir": str(job_dir),
        },
    }


def run_harbor_terminal_bench(
    *,
    task: dict[str, Any],
    candidate_path: Path,
    source_code: str,
    proposal_runtime: ProposalRuntime,
    workspace_root: Path,
    session_id: str,
    max_items: int | None,
    progress_callback,
    pace_ms: int,
) -> dict[str, Any]:
    config = effective_external_run_config(task, candidate_path)
    rendered_source = render_external_run_config_source(config)
    raw_metrics = evaluate_harbor_terminal_candidate(
        task=task,
        candidate_path=candidate_path,
        source_code=source_code,
        proposal_runtime=proposal_runtime,
        workspace_root=workspace_root,
        session_id=session_id,
        max_items=max_items,
        progress_callback=progress_callback,
        pace_ms=pace_ms,
    )
    external_summary = dict(raw_metrics.get("external_summary") or {})
    passed = int(raw_metrics.get("passed_tests") or 0)
    total = int(raw_metrics.get("total_tests") or 0)
    baseline = build_external_candidate(
        task=task,
        source_code=rendered_source,
        agent="checked-in-config",
        label="checked-in-config",
        strategy="Use the checked-in Harbor benchmark configuration.",
        rationale="The checked-in file defines which Harbor dataset, agent, and concurrency settings to use.",
        candidate_summary="Checked-in Harbor benchmark configuration.",
        raw_metrics={"status": "not-run", "verifier_status": "not-run"},
        workspace_path=str(candidate_path),
    )
    winner = build_external_candidate(
        task=task,
        source_code=rendered_source,
        agent=str(external_summary.get("agent") or "candidate"),
        label=str(external_summary.get("agent") or "candidate"),
        strategy="Run Harbor's official terminal benchmark harness against the configured agent and model.",
        rationale="Harbor executes the real terminal-bench tasks inside containerized environments and returns task rewards.",
        candidate_summary="Harbor terminal benchmark run over the selected task subset.",
        raw_metrics=raw_metrics,
        workspace_path=str(external_summary.get("job_dir") or workspace_root),
        proposal_model=str(external_summary.get("model") or proposal_runtime.active_model),
    )
    return build_external_result(
        task=task,
        proposal_runtime=proposal_runtime,
        baseline=baseline,
        winner=winner,
        selection_reason=f"Harbor terminal-bench finished with {passed}/{total} successful trials.",
        extra_fields={"external_summary": external_summary},
    )


def evaluate_tau_bench_candidate(
    *,
    task: dict[str, Any],
    candidate_path: Path,
    source_code: str,
    workspace_root: Path,
    session_id: str,
    max_items: int | None,
    env_name: str,
    proposal_runtime: ProposalRuntime | None = None,
    progress_callback=None,
    pace_ms: int = 0,
) -> dict[str, Any]:
    config = effective_external_run_config(task, candidate_path)
    runtime = proposal_runtime or runtime_for_external_task(task)
    require_command("git")
    require_command("uv")
    repo_dir = ensure_repo_checkout(
        str(config.get("repo_url") or "https://github.com/sierra-research/tau2-bench.git"),
        ROOT / "external" / "tau2-bench",
    )
    log_dir = workspace_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_provider = str(config.get("model_provider") or "openai")
    model_name = str(config.get("model_name") or runtime.active_model)
    user_model = str(config.get("user_model") or model_name)
    user_model_provider = str(config.get("user_model_provider") or model_provider)
    agent_strategy = str(config.get("agent_strategy") or "tool-calling")
    user_strategy = str(config.get("user_strategy") or "llm")
    max_concurrency = int(config.get("max_concurrency") or 1)
    task_limit = int(max_items if isinstance(max_items, int) and max_items > 0 else config.get("n_tasks") or 10)
    env = strip_socks_proxy_env(dict(os.environ))
    env.update(openai_compatible_env(runtime))

    emit_progress(
        progress_callback,
        task_id=str(task["id"]),
        phase="external_harness_started",
        message=f"Running tau-bench {env_name} with {agent_strategy}",
        pace_ms=pace_ms,
    )
    started = time.perf_counter()
    timeout_s = int(config.get("timeout_s") or 7200)
    try:
        results, result_path = _run_modern_tau_bench(
            repo_dir=repo_dir,
            env_name=env_name,
            model_name=model_name,
            model_provider=model_provider,
            user_model=user_model,
            user_model_provider=user_model_provider,
            max_concurrency=max_concurrency,
            task_limit=task_limit,
            session_id=session_id,
            workspace_root=workspace_root,
            env=env,
            timeout_s=timeout_s,
        )
    except RuntimeError as exc:
        if not _legacy_tau_cli_is_likely(repo_dir, exc):
            raise
        results, result_path = _run_legacy_tau_bench(
            repo_dir=repo_dir,
            env_name=env_name,
            model_name=model_name,
            model_provider=model_provider,
            user_model=user_model,
            user_model_provider=user_model_provider,
            agent_strategy=agent_strategy,
            user_strategy=user_strategy,
            max_concurrency=max_concurrency,
            task_limit=task_limit,
            log_dir=log_dir,
            env=env,
            timeout_s=timeout_s,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    passed = 0
    test_results = []
    for row in results:
        reward_raw = row.get("reward")
        if reward_raw is None and isinstance(row.get("info"), dict):
            reward_raw = row["info"].get("reward")
        if reward_raw is None and row.get("success") is not None:
            reward_raw = 1.0 if bool(row.get("success")) else 0.0
        reward = float(reward_raw or 0.0)
        succeeded = reward >= 1.0 - 1e-6
        if succeeded:
            passed += 1
        test_results.append(
            {
                "name": str(row.get("task_name") or row.get("task_id") or row.get("id") or f"task-{len(test_results) + 1}"),
                "expected": 1.0,
                "actual": reward,
                "passed": succeeded,
                "actual_raw": dict(row),
            }
        )
    total = len(results)
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
        "external_summary": {
            "env": env_name,
            "agent_strategy": agent_strategy,
            "model": model_name,
            "passed": passed,
            "total": total,
            "result_path": str(result_path),
        },
    }


def run_tau_bench_suite(
    *,
    task: dict[str, Any],
    candidate_path: Path,
    source_code: str,
    proposal_runtime: ProposalRuntime,
    workspace_root: Path,
    session_id: str,
    max_items: int | None,
    env_name: str,
    progress_callback,
    pace_ms: int,
) -> dict[str, Any]:
    config = effective_external_run_config(task, candidate_path)
    rendered_source = render_external_run_config_source(config)
    raw_metrics = evaluate_tau_bench_candidate(
        task=task,
        candidate_path=candidate_path,
        source_code=source_code,
        proposal_runtime=proposal_runtime,
        workspace_root=workspace_root,
        session_id=session_id,
        max_items=max_items,
        env_name=env_name,
        progress_callback=progress_callback,
        pace_ms=pace_ms,
    )
    external_summary = dict(raw_metrics.get("external_summary") or {})
    passed = int(raw_metrics.get("passed_tests") or 0)
    total = int(raw_metrics.get("total_tests") or 0)
    baseline = build_external_candidate(
        task=task,
        source_code=rendered_source,
        agent="checked-in-config",
        label="checked-in-config",
        strategy="Use the checked-in tau-bench benchmark configuration.",
        rationale="The checked-in file defines the tau-bench environment and agent settings.",
        candidate_summary="Checked-in tau-bench configuration.",
        raw_metrics={"status": "not-run", "verifier_status": "not-run"},
        workspace_path=str(candidate_path),
    )
    winner = build_external_candidate(
        task=task,
        source_code=rendered_source,
        agent=str(external_summary.get("agent_strategy") or "candidate"),
        label=str(external_summary.get("agent_strategy") or "candidate"),
        strategy="Run the official tau-bench harness against the configured model and user simulator.",
        rationale="tau-bench measures full conversational tool-agent behavior, not just single-shot answers.",
        candidate_summary="tau-bench run over the selected task subset.",
        raw_metrics=raw_metrics,
        workspace_path=str(Path(str(external_summary.get("result_path") or workspace_root)).parent),
        proposal_model=str(external_summary.get("model") or proposal_runtime.active_model),
    )
    return build_external_result(
        task=task,
        proposal_runtime=proposal_runtime,
        baseline=baseline,
        winner=winner,
        selection_reason=f"tau-bench {env_name} finished with {passed}/{total} successful tasks.",
        extra_fields={"external_summary": external_summary},
    )
