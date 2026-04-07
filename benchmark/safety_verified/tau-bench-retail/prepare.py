from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
DATA_DIR = ROOT / "data"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.bench.agent_benchmarks import TAU_BENCH_REPO_URL, runtime_repo_dir
from app.bench.benchmark_adapter_support import ensure_repo_checkout
from app.bench.safety_support import write_json


ENV_NAME = "retail"
INFO_PATH = DATA_DIR / "benchmark_info.json"
FULL_DATASET_SIZE = 114


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare tau2-bench retail benchmark metadata.")
    return parser.parse_args()


def _coerce_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("tasks")
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
    raise ValueError("tau2-bench task file must be a list or {'tasks': [...]} payload.")


def main() -> None:
    _parse_args()
    repo_dir = ensure_repo_checkout(TAU_BENCH_REPO_URL, runtime_repo_dir("tau2-bench"))
    task_path = repo_dir / "data" / "tau2" / "domains" / ENV_NAME / "tasks.json"
    rows = _coerce_rows(json.loads(task_path.read_text()))
    info = {
        "benchmark": "tau2-bench",
        "domain": ENV_NAME,
        "repo_url": TAU_BENCH_REPO_URL,
        "task_file": str(task_path),
        "dataset_size": len(rows),
        "runtime_status": "Uses the official tau2-bench runtime loop through app.bench.agent_benchmarks.run_tau_bench_suite.",
        "sample_task_ids": [str(row.get("id") or "") for row in rows[:10]],
    }
    write_json(INFO_PATH, info)
    print(f"Wrote tau2-bench {ENV_NAME} metadata ({len(rows)} tasks) to {INFO_PATH}.")


if __name__ == "__main__":
    main()
