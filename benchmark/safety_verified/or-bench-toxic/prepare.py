from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
DATA_DIR = ROOT / "data"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.bench.safety_support import write_json


MANIFEST_PATH = DATA_DIR / "questions.json"
SOURCE_INFO_PATH = DATA_DIR / "source_info.json"
SUBSET = "or-bench-toxic"
DATASET_ID = "bench-llm/or-bench"
FULL_DATASET_SIZE = 655


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the OR-Bench toxic manifest.")
    parser.add_argument("--items", type=int, default=FULL_DATASET_SIZE, help="Optional prefix length.")
    return parser.parse_args()


def build_questions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        prompt = str(row.get("prompt") or "").strip()
        if not prompt:
            continue
        items.append(
            {
                "item_id": f"or-bench-toxic-{row_index:04d}",
                "name": f"OR-Bench Toxic #{row_index}",
                "prompt": prompt,
                "context": "",
                "expected_answer": "refuse",
                "metadata": {
                    "benchmark": "or_bench",
                    "subset": SUBSET,
                    "category": str(row.get("category") or "").strip(),
                    "expected_behavior": "refuse",
                    "source_index": row_index,
                },
            }
        )
    return items


def main() -> None:
    args = _parse_args()
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, SUBSET, split="train")
    rows = [dict(row) for row in dataset]
    items = build_questions(rows)
    requested_items = max(1, min(int(args.items or len(items)), len(items)))
    prepared_items = items[:requested_items]
    manifest = {
      "dataset_id": "or_bench_toxic",
      "split": f"{DATASET_ID}:{SUBSET}:train",
      "dataset_size": len(items),
      "prepared_count": len(prepared_items),
      "items": prepared_items
    }
    source_info = {
      "benchmark": "or_bench",
      "sources": {
        "dataset": f"https://huggingface.co/datasets/{DATASET_ID}"
      },
      "subset": SUBSET,
      "prepared_count": len(prepared_items),
      "dataset_size": len(items)
    }
    write_json(MANIFEST_PATH, manifest)
    write_json(SOURCE_INFO_PATH, source_info)
    print(f"Wrote {len(prepared_items)} OR-Bench toxic prompts to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
