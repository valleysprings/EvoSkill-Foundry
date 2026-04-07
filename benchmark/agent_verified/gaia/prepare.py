from __future__ import annotations

import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TASK_ROOT.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.bench.agent_verified_support import ensure_task_data_dir, write_benchmark_info


def main() -> int:
    ensure_task_data_dir(TASK_ROOT)
    output_path = write_benchmark_info(
        TASK_ROOT,
        {
            "benchmark": "GAIA",
            "dataset_id": "gaia_validation",
            "dataset_size": 166,
            "default_split": "validation",
            "official_sources": [
                "https://huggingface.co/spaces/gaia-benchmark/leaderboard"
            ],
            "expected_runtime": "GAIA leaderboard / official evaluation harness",
            "expected_dependencies": [
                "official GAIA task runner"
            ],
            "runtime_status": "Official GAIA runtime bridge is not wired into this repo yet.",
            "notes": [
                "This prepare step records benchmark metadata only.",
                "Task-local runtime assets should live under this task's data/ directory."
            ],
        },
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
