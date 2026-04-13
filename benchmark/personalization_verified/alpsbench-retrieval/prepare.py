from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRACK_ROOT = ROOT.parent
REPO_ROOT = ROOT.parents[2]
for candidate in (TRACK_ROOT, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from alpsbench_public import TASK3_DATASET_ID, TASK3_SIZE, add_items_argument, build_task3_items, requested_count, write_task_manifest


MANIFEST_PATH = ROOT / "data" / "questions.json"
SPLIT = "validation:task3:d100+d300+d500+d700+d1000"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize AlpsBench Task 3 validation items.")
    return add_items_argument(parser).parse_args(argv)


def _build_items(limit: int | None = None) -> list[dict]:
    return build_task3_items(benchmark="alpsbench-retrieval", limit=limit)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    requested = requested_count(TASK3_SIZE, args.items)
    items = _build_items(limit=requested)
    write_task_manifest(
        manifest_path=MANIFEST_PATH,
        dataset_id=TASK3_DATASET_ID,
        split=SPLIT,
        items=items,
        dataset_size=TASK3_SIZE,
    )
    print(f"Wrote {len(items)} AlpsBench task3 rows to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
