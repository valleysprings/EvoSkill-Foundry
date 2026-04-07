from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.bench.benchmark_adapter_support import (
    build_benchmark_adapter_candidate,
    build_benchmark_adapter_result,
)
from app.codegen.llm import ProposalRuntime


def ensure_task_data_dir(task_root: Path) -> Path:
    data_dir = task_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def benchmark_info_path(task_root: Path) -> Path:
    return ensure_task_data_dir(task_root) / "benchmark_info.json"


def write_benchmark_info(task_root: Path, payload: dict[str, Any]) -> Path:
    path = benchmark_info_path(task_root)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    return path


def load_benchmark_info(task_root: Path) -> dict[str, Any]:
    path = benchmark_info_path(task_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark_info.json under {task_root / 'data'}. Run prepare.py first.")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark_info.json must contain a JSON object: {path}")
    return dict(payload)


def missing_runtime_metrics(*, error: str) -> dict[str, Any]:
    return {
        "status": "error",
        "verifier_status": "error",
        "correctness": 0.0,
        "passed_tests": 0,
        "total_tests": 0,
        "benchmark_ms": None,
        "benchmark_samples_ms": [],
        "objective": 0.0,
        "objective_score": 0.0,
        "objective_signal": 0.0,
        "error": error,
        "test_results": [],
        "item_runs": [],
        "suite_summary": {"status": "runtime-missing", "error": error},
    }


def build_unavailable_result(
    *,
    task: dict[str, Any],
    candidate_path: Path,
    source_code: str,
    proposal_runtime: ProposalRuntime,
    benchmark_name: str,
    error: str,
) -> dict[str, Any]:
    raw_metrics = missing_runtime_metrics(error=error)
    baseline = build_benchmark_adapter_candidate(
        task=task,
        source_code=source_code,
        agent="checked-in-adapter",
        label="checked-in-adapter",
        strategy=f"Use the checked-in {benchmark_name} adapter scaffold.",
        rationale=f"The checked-in file is the baseline adapter scaffold for {benchmark_name}.",
        candidate_summary=f"Checked-in {benchmark_name} adapter scaffold.",
        raw_metrics={"status": "not-run", "verifier_status": "not-run"},
        workspace_path=str(candidate_path),
    )
    winner = build_benchmark_adapter_candidate(
        task=task,
        source_code=source_code,
        agent="candidate-adapter",
        label="candidate-adapter",
        strategy=f"Run the candidate adapter against the {benchmark_name} scaffold.",
        rationale=error,
        candidate_summary=f"{benchmark_name} adapter run unavailable.",
        raw_metrics=raw_metrics,
        workspace_path=str(candidate_path),
        proposal_model=proposal_runtime.active_model,
    )
    return build_benchmark_adapter_result(
        task=task,
        proposal_runtime=proposal_runtime,
        baseline=baseline,
        winner=winner,
        selection_reason=error,
        extra_fields={
            "suite_summary": dict(raw_metrics.get("suite_summary") or {}),
            "item_runs": [],
        },
    )
