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
            "benchmark": "Gaia2",
            "dataset_id": "gaia2_public",
            "dataset_size": 800,
            "default_split": "validation",
            "official_sources": [
                "https://facebookresearch.github.io/meta-agents-research-environments/user_guide/gaia2_evaluation.html",
                "https://arxiv.org/abs/2602.11964"
            ],
            "expected_runtime": "Meta Agents Research Environments (ARE)",
            "expected_dependencies": [
                "meta-agents-research-environments"
            ],
            "runtime_status": "Official ARE / Gaia2 runtime bridge is not wired into this repo yet.",
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
