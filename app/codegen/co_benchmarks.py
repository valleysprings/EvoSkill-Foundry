from __future__ import annotations

import contextlib
import importlib
import sys
import time
from pathlib import Path
from typing import Any

from app.codegen.external import (
    build_external_candidate,
    build_external_result,
    effective_external_run_config,
    emit_progress,
    ensure_repo_checkout,
    load_candidate_module,
    render_external_run_config_source,
    runtime_for_external_task,
)
from app.codegen.llm import ProposalRuntime
from app.configs.paths import ROOT


CO_BENCH_SYSTEM_PROMPT = (
    "You are an expert in combinatorial optimization and algorithm design. "
    "Return only a JSON object with keys algorithm_summary and python_code. "
    "python_code must define the exact function required by the prompt, without Markdown fences."
)


@contextlib.contextmanager
def _prepend_sys_path(path: Path):
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(path))
        except ValueError:
            pass


def _task_data_is_present(data_dir: Path, task_name: str) -> bool:
    task_dir = data_dir / task_name
    return task_dir.exists() and (task_dir / "config.py").exists() and any(task_dir.iterdir())


def _ensure_data(data_dir: Path, *, task_names: list[str] | None = None, max_workers: int = 2) -> Path:
    requested = [name for name in (task_names or []) if str(name).strip()]
    if data_dir.exists() and requested and all(_task_data_is_present(data_dir, task_name) for task_name in requested):
        return data_dir
    if data_dir.exists() and not requested and any(data_dir.iterdir()):
        return data_dir
    hub = importlib.import_module("huggingface_hub")
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    allow_patterns = None
    if requested:
        allow_patterns = [f"{task_name}/**" for task_name in requested]
    hub.snapshot_download(
        repo_id="CO-Bench/CO-Bench",
        repo_type="dataset",
        local_dir=str(data_dir),
        allow_patterns=allow_patterns,
        max_workers=max(1, int(max_workers)),
    )
    return data_dir


def _prompt_python_code(
    proposal_runtime: ProposalRuntime,
    *,
    candidate_path: Path,
    problem_description: str,
    purpose: str,
    queue_priority: int,
) -> tuple[str, str, dict[str, Any]]:
    module = load_candidate_module(candidate_path)
    system_prompt = str(getattr(module, "SYSTEM_PROMPT", CO_BENCH_SYSTEM_PROMPT) or CO_BENCH_SYSTEM_PROMPT).strip()
    build_user_prompt = getattr(module, "build_user_prompt", None)
    if callable(build_user_prompt):
        user_prompt = str(build_user_prompt(problem_description) or "").strip()
        if not user_prompt:
            raise ValueError("build_user_prompt(problem_description) must return a non-empty string.")
    else:
        user_prompt = (
            "Solve the following CO-Bench task by writing Python code for the required solve function.\n\n"
            f"{problem_description.strip()}\n"
        )
    payload, trace = proposal_runtime.complete_json(
        purpose=purpose,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        queue_priority=queue_priority,
    )
    code = str(payload.get("python_code") or "").strip()
    summary = str(payload.get("algorithm_summary") or "").strip()
    if not code:
        raise ValueError("Model response did not include python_code.")
    return code, summary, trace


def evaluate_co_bench_candidate(
    *,
    task: dict[str, Any],
    candidate_path: Path,
    source_code: str,
    workspace_root: Path,
    max_items: int | None,
    proposal_runtime: ProposalRuntime | None = None,
    progress_callback=None,
    pace_ms: int = 0,
) -> dict[str, Any]:
    config = effective_external_run_config(task, candidate_path)
    runtime = proposal_runtime or runtime_for_external_task(task)
    repo_dir = ensure_repo_checkout(
        str(config.get("repo_url") or "https://github.com/sunnweiwei/CO-Bench.git"),
        ROOT / "external" / "CO-Bench",
    )
    timeout_s = int(config.get("timeout_s") or 10)
    requested_tasks = list(config.get("problem_names") or [])
    download_max_workers = int(config.get("download_max_workers") or 2)

    with _prepend_sys_path(repo_dir):
        controller = importlib.import_module("evaluation.controller")
        evaluator_module = importlib.import_module("evaluation.evaluate")
        task_names = requested_tasks or list(getattr(controller, "TASK_LIST"))
        if isinstance(max_items, int) and max_items > 0:
            task_names = task_names[:max_items]
        data_dir = _ensure_data(
            ROOT / "external" / "co-bench-data",
            task_names=task_names,
            max_workers=download_max_workers,
        )

        emit_progress(
            progress_callback,
            task_id=str(task["id"]),
            phase="external_dataset_loading",
            message=f"Loading CO-Bench metadata for {len(task_names)} tasks",
            pace_ms=pace_ms,
        )

        test_results = []
        llm_traces: list[dict[str, Any]] = []
        score_total = 0.0
        started = time.perf_counter()
        for index, task_name in enumerate(task_names, start=1):
            item_id = f"co-bench-{index:03d}"
            data = controller.get_data(task_name, src_dir=str(data_dir))
            try:
                python_code, algorithm_summary, trace = _prompt_python_code(
                    runtime,
                    candidate_path=candidate_path,
                    problem_description=str(data.problem_description),
                    purpose=f"{task['id']}::{item_id}",
                    queue_priority=1000 + index,
                )
                llm_traces.append({**trace, "item_id": item_id, "problem_name": task_name})
                feedback = evaluator_module.Evaluator(data, timeout=timeout_s).evaluate(python_code)
                test_score = float(feedback.test_score)
                score_total += test_score
                test_results.append(
                    {
                        "name": task_name,
                        "expected": 1.0,
                        "actual": test_score,
                        "passed": test_score > 0.0,
                        "actual_raw": {
                            "algorithm_summary": algorithm_summary,
                            "dev_score": feedback.dev_score,
                            "test_score": feedback.test_score,
                            "test_feedback": feedback.test_feedback,
                        },
                    }
                )
            except Exception as exc:  # noqa: BLE001
                test_results.append(
                    {
                        "name": task_name,
                        "expected": 1.0,
                        "actual": 0.0,
                        "passed": False,
                        "actual_raw": {"error": str(exc)},
                    }
                )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    total = len(task_names)
    objective = score_total / total if total else 0.0
    passed = sum(1 for row in test_results if row["passed"])
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
            "tasks": task_names,
            "avg_test_score": objective,
            "passed_nonzero": passed,
            "total": total,
            "data_dir": str(data_dir),
        },
    }


def run_co_bench_suite(
    *,
    task: dict[str, Any],
    candidate_path: Path,
    source_code: str,
    proposal_runtime: ProposalRuntime,
    workspace_root: Path,
    max_items: int | None,
    progress_callback,
    pace_ms: int,
) -> dict[str, Any]:
    config = effective_external_run_config(task, candidate_path)
    rendered_source = render_external_run_config_source(config)
    raw_metrics = evaluate_co_bench_candidate(
        task=task,
        candidate_path=candidate_path,
        source_code=source_code,
        proposal_runtime=proposal_runtime,
        workspace_root=workspace_root,
        max_items=max_items,
        progress_callback=progress_callback,
        pace_ms=pace_ms,
    )
    external_summary = dict(raw_metrics.get("external_summary") or {})
    objective = float(raw_metrics.get("objective") or 0.0)
    passed = int(raw_metrics.get("passed_tests") or 0)
    total = int(raw_metrics.get("total_tests") or 0)
    baseline = build_external_candidate(
        task=task,
        source_code=rendered_source,
        agent="checked-in-config",
        label="checked-in-config",
        strategy="Use the checked-in CO-Bench configuration.",
        rationale="The checked-in file selects which CO-Bench tasks and evaluator timeout to use.",
        candidate_summary="Checked-in CO-Bench configuration.",
        raw_metrics={"status": "not-run", "verifier_status": "not-run"},
        workspace_path=str(candidate_path),
    )
    winner = build_external_candidate(
        task=task,
        source_code=rendered_source,
        agent=proposal_runtime.active_model,
        label=proposal_runtime.active_model,
        strategy="Generate task-specific solve(**kwargs) implementations and score them with the official CO-Bench evaluator.",
        rationale="CO-Bench measures optimization quality through official normalized test scores over benchmark tasks.",
        candidate_summary="CO-Bench evaluation using the configured model as the code-generating agent.",
        raw_metrics=raw_metrics,
        workspace_path=str(workspace_root),
        proposal_model=proposal_runtime.active_model,
    )
    return build_external_result(
        task=task,
        proposal_runtime=proposal_runtime,
        baseline=baseline,
        winner=winner,
        selection_reason=f"CO-Bench finished with average normalized test score {objective:.4f} over {total} tasks.",
        llm_traces=[],
        extra_fields={"external_summary": external_summary},
    )
