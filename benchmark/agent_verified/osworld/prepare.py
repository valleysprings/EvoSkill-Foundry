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
            "benchmark": "OSWorld",
            "dataset_id": "osworld_tasks",
            "dataset_size": 369,
            "default_split": "default",
            "official_sources": [
                "https://github.com/xlang-ai/OSWorld"
            ],
            "expected_runtime": "OSWorld desktop environment",
            "expected_dependencies": [
                "desktop-env",
                "vmrun or alternative supported provider"
            ],
            "runtime_status": "Official OSWorld runtime bridge is not wired into this repo yet.",
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
