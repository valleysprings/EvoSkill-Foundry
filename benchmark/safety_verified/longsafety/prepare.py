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
DATASET_ID = "thu-coai/LongSafety"
FULL_DATASET_SIZE = 1543


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the LongSafety manifest.")
    parser.add_argument("--items", type=int, default=FULL_DATASET_SIZE, help="Optional prefix length.")
    return parser.parse_args()


def build_questions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        prompt = str(row.get("instruction") or "").strip()
        context = str(row.get("context") or "").strip()
        if not prompt or not context:
            continue
        source_id = int(row.get("id") or len(items))
        items.append(
            {
                "item_id": f"longsafety-{source_id}",
                "name": f"LongSafety #{source_id}",
                "prompt": prompt,
                "context": context,
                "expected_answer": "refuse",
                "metadata": {
                    "benchmark": "longsafety",
                    "source_id": source_id,
                    "length": int(row.get("length") or 0),
                    "safety_type": str(row.get("safety_type") or "").strip(),
                    "task_type": str(row.get("task_type") or "").strip(),
                    "doc_num": int(row.get("doc_num") or 0),
                    "link": list(row.get("link") or []),
                    "key_words": list(row.get("key_words") or []),
                },
            }
        )
    return items


def main() -> None:
    args = _parse_args()
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, split="train")
    rows = [dict(row) for row in dataset]
    items = build_questions(rows)
    requested_items = max(1, min(int(args.items or len(items)), len(items)))
    prepared_items = items[:requested_items]
    manifest = {
        "dataset_id": "longsafety",
        "split": f"{DATASET_ID}:train",
        "dataset_size": len(items),
        "prepared_count": len(prepared_items),
        "items": prepared_items,
    }
    source_info = {
        "benchmark": "longsafety",
        "sources": {
            "dataset": f"https://huggingface.co/datasets/{DATASET_ID}",
        },
        "prepared_count": len(prepared_items),
        "dataset_size": len(items),
    }
    write_json(MANIFEST_PATH, manifest)
    write_json(SOURCE_INFO_PATH, source_info)
    print(f"Wrote {len(prepared_items)} LongSafety prompts to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
