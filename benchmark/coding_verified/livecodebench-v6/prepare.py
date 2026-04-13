from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.bench.livecodebench_prepare import prepare_livecodebench_shard


ROOT = Path(__file__).resolve().parent


def main() -> None:
    prepare_livecodebench_shard(
        task_root=ROOT,
        task_id="livecodebench-v6",
        release_version="v6",
        full_dataset_size=175,
    )


if __name__ == "__main__":
    main()
