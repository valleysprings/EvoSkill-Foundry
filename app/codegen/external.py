from __future__ import annotations

import importlib.util
import pprint
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Callable

from app.codegen.verifier import finalize_candidate_metrics
from app.codegen.llm import ProposalRuntime
from app.codegen.task_contracts import infer_optimization_scope, infer_runtime_backend, infer_task_mode


ProgressCallback = Callable[[dict[str, Any]], None]


def is_external_task(task: dict[str, Any]) -> bool:
    return infer_runtime_backend(task) == "external"


def _load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_candidate_module(path: Path):
    module_name = f"external_candidate_{path.parent.name}_{path.stem}".replace("-", "_")
    return _load_module_from_path(path, module_name)


def _load_task_runner(task: dict[str, Any]):
    verifier_path = Path(str(task["verifier_path"]))
    module_name = f"task_external_runner_{task['id'].replace('-', '_')}"
    module = _load_module_from_path(verifier_path, module_name)
    runner = getattr(module, "run_external_task", None)
    if not callable(runner):
        raise ValueError(f"{verifier_path} must export callable run_external_task().")
    return runner


def load_value_from_candidate(path: Path, name: str, default: Any = None) -> Any:
    module = load_candidate_module(path)
    return getattr(module, name, default)


def effective_external_run_config(task: dict[str, Any], candidate_path: Path) -> dict[str, Any]:
    module = load_candidate_module(candidate_path)
    builder = getattr(module, "build_run_config", None)
    if callable(builder):
        base_config = builder()
    else:
        base_config = getattr(module, "RUN_CONFIG", {})
    if base_config is None:
        base = {}
    elif isinstance(base_config, dict):
        base = dict(base_config)
    else:
        raise ValueError("External candidate build_run_config()/RUN_CONFIG must produce a dict.")
    override = task.get("runtime_external_config")
    if isinstance(override, dict):
        return {**base, **override}
    return base


def render_external_run_config_source(config: dict[str, Any]) -> str:
    rendered = pprint.pformat(config, sort_dicts=False)
    return (
        "def build_run_config() -> dict:\n"
        + textwrap.indent(f"return {rendered}", "    ")
        + "\n\n"
        "RUN_CONFIG = build_run_config()\n"
    )


def runtime_for_external_task(task: dict[str, Any]) -> ProposalRuntime:
    requested_model = str(task.get("runtime_model_override") or "").strip() or None
    return ProposalRuntime.from_env().with_model(requested_model)


def require_command(command: str) -> str:
    resolved = shutil.which(command)
    if resolved is None:
        raise RuntimeError(f"Required command not found on PATH: {command}")
    return resolved


def ensure_repo_checkout(repo_url: str, target_dir: Path) -> Path:
    if (target_dir / ".git").exists():
        return target_dir
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
        cwd=target_dir.parent,
        timeout_s=300,
    )
    return target_dir


def run_command(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout_s: int | float | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        details = stderr or stdout or f"returncode={completed.returncode}"
        raise RuntimeError(f"Command failed: {' '.join(args)} :: {details}")
    return completed


def openai_compatible_env(runtime: ProposalRuntime) -> dict[str, str]:
    env = dict()
    if runtime.config.api_key:
        env["OPENAI_API_KEY"] = runtime.config.api_key
    if runtime.config.api_base:
        env["OPENAI_API_BASE"] = runtime.config.api_base
        env["OPENAI_BASE_URL"] = runtime.config.api_base
    return env


def strip_socks_proxy_env(env: dict[str, str]) -> dict[str, str]:
    cleaned = dict(env)
    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        value = cleaned.get(key)
        if isinstance(value, str) and value.lower().startswith("socks"):
            cleaned.pop(key, None)
    return cleaned


def extract_python_code(text: str) -> str:
    matches = re.findall(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    matches = re.findall(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()


def _default_raw_metrics(status: str = "not-run") -> dict[str, Any]:
    return {
        "status": status,
        "verifier_status": status,
        "correctness": 0.0,
        "passed_tests": 0,
        "total_tests": 0,
        "benchmark_ms": None,
        "benchmark_samples_ms": [],
        "objective": 0.0,
        "objective_score": 0.0,
        "objective_signal": 0.0,
        "error": None,
        "test_results": [],
    }


def build_external_candidate(
    *,
    task: dict[str, Any],
    source_code: str,
    agent: str,
    label: str,
    strategy: str,
    rationale: str,
    candidate_summary: str,
    raw_metrics: dict[str, Any] | None = None,
    workspace_path: str | None = None,
    proposal_model: str | None = None,
) -> dict[str, Any]:
    metrics = finalize_candidate_metrics(
        task=task,
        source_code=source_code,
        memory_applied=False,
        raw_metrics=dict(_default_raw_metrics(), **(raw_metrics or {})),
    )
    return {
        "candidate_id": None,
        "agent": agent,
        "label": label,
        "strategy": strategy,
        "rationale": rationale,
        "candidate_summary": candidate_summary,
        "proposal_model": proposal_model,
        "verifier_status": metrics["verifier_status"],
        "workspace_path": workspace_path,
        "source_code": source_code,
        "metrics": metrics,
    }


def _objective_point(generation: int, candidate: dict[str, Any], *, improved_global_best: bool) -> dict[str, Any]:
    metrics = candidate["metrics"]
    return {
        "generation": generation,
        "objective": metrics["objective"],
        "objective_score": metrics["objective_score"],
        "candidate_objective": metrics["objective"],
        "candidate_objective_score": metrics["objective_score"],
        "primary_score": metrics["primary_score"],
        "candidate_primary_score": metrics["primary_score"],
        "tie_break_score": metrics["tie_break_score"],
        "candidate_tie_break_score": metrics["tie_break_score"],
        "accepted": True,
        "accepted_count": 1,
        "improved_global_best": improved_global_best,
        "memory_delta": 0,
    }


def build_external_result(
    *,
    task: dict[str, Any],
    proposal_runtime: ProposalRuntime,
    baseline: dict[str, Any],
    winner: dict[str, Any],
    selection_reason: str,
    llm_traces: list[dict[str, Any]] | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline_primary = float(baseline["metrics"]["primary_score"])
    winner_primary = float(winner["metrics"]["primary_score"])
    objective_curve = [_objective_point(0, baseline, improved_global_best=False)]
    if winner_primary != baseline_primary or winner["metrics"]["objective"] != baseline["metrics"]["objective"]:
        objective_curve.append(_objective_point(1, winner, improved_global_best=True))

    result = {
        "run_mode": "external-benchmark",
        "active_model": proposal_runtime.active_model,
        "selection_spec": dict(task["selection_spec"]),
        "benchmark_tier": task["benchmark_tier"],
        "track": task["track"],
        "dataset_id": task["dataset_id"],
        "included_in_main_comparison": task["included_in_main_comparison"],
        "task": {
            "id": task["id"],
            "title": task["title"],
            "description": task["description"],
            "family": task["family"],
            "function_name": task["function_name"],
            "entry_symbol": task["entry_symbol"],
            "editable_file": task["editable_file"],
            "answer_metric": task["answer_metric"],
            "objective_label": task["objective_label"],
            "objective_direction": task["objective_direction"],
            "objective_spec": task["objective_spec"],
            "selection_spec": task["selection_spec"],
            "generation_budget": int(task.get("generation_budget") or 0),
            "candidate_budget": int(task.get("candidate_budget") or 0),
            "branching_factor": int(task.get("branching_factor") or 0),
            "item_workers": int(task.get("item_workers") or 0),
            "runtime_backend": infer_runtime_backend(task),
            "task_mode": infer_task_mode(task),
            "optimization_scope": infer_optimization_scope(task),
            "benchmark_tier": task["benchmark_tier"],
            "track": task["track"],
            "dataset_id": task["dataset_id"],
            "dataset_size": task.get("dataset_size") or 0,
            "local_dataset_only": bool(task.get("local_dataset_only")),
            "split": task.get("split"),
            "included_in_main_comparison": task["included_in_main_comparison"],
            "supports_runtime_config": isinstance(task.get("runtime_external_config"), dict) or is_external_task(task),
            "external_run_config": task.get("runtime_external_config"),
            "supports_max_items": bool(task.get("local_dataset_only") or is_external_task(task)),
            "default_max_items": None,
        },
        "baseline": baseline,
        "winner": winner,
        "dataset_summary": None,
        "item_runs": [],
        "generations": [],
        "objective_curve": objective_curve,
        "llm_traces": list(llm_traces or []),
        "memory_markdown": "",
        "memory_before_count": 0,
        "memory_after_count": 0,
        "positive_experiences_added": 0,
        "negative_experiences_added": 0,
        "added_experiences": [],
        "delta_primary_score": round(winner_primary - baseline_primary, 6),
        "run_delta_primary_score": round(winner_primary - baseline_primary, 6),
        "run_delta_objective": round(
            float(winner["metrics"]["objective"]) - float(baseline["metrics"]["objective"]),
            6,
        ),
        "selection_reason": selection_reason,
        "total_generations": 0,
    }
    if extra_fields:
        result.update(extra_fields)
    return result


def emit_progress(
    progress_callback: ProgressCallback | None,
    *,
    task_id: str,
    phase: str,
    message: str,
    pace_ms: int = 0,
    **extra: Any,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        {
            "phase": phase,
            "task_id": task_id,
            "message": message,
            **extra,
        }
    )


def run_external_task(
    task: dict[str, Any],
    *,
    proposal_runtime: ProposalRuntime,
    workspace_root: Path,
    session_id: str,
    max_items: int | None = None,
    progress_callback: ProgressCallback | None = None,
    pace_ms: int = 0,
) -> dict[str, Any]:
    runner = _load_task_runner(task)
    candidate_path = Path(str(task["editable_path"]))
    source_code = candidate_path.read_text()
    workspace_root.mkdir(parents=True, exist_ok=True)
    return runner(
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
