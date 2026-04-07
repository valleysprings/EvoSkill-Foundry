from __future__ import annotations

import argparse
import re
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
DATASET_ID = "euirim/goodwiki"
FULL_DATASET_SIZE = 250
MAX_CONTEXT_CHARS = 2400


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the HalluLens PreciseWikiQA local slice.")
    parser.add_argument("--items", type=int, default=FULL_DATASET_SIZE, help="Optional prefix length.")
    return parser.parse_args()


def _normalize_markdown(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", str(text or "").strip())


def build_questions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        title = str(row.get("title") or "").strip()
        description = str(row.get("description") or "").strip()
        markdown = _normalize_markdown(str(row.get("markdown") or ""))
        if not title or not description or not markdown:
            continue
        pageid = int(row.get("pageid") or len(items))
        items.append(
            {
                "item_id": f"hallulens-precisewikiqa-{pageid}",
                "name": title,
                "prompt": f"According to the reference, what is {title}?",
                "context": markdown[:MAX_CONTEXT_CHARS],
                "expected_answer": description,
                "metadata": {
                    "benchmark": "hallulens_precisewikiqa",
                    "title": title,
                    "pageid": pageid,
                    "revid": row.get("revid"),
                    "question_style": "description_lookup",
                },
            }
        )
        if len(items) >= FULL_DATASET_SIZE:
            break
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
        "dataset_id": "hallulens_precisewikiqa",
        "split": f"{DATASET_ID}:train:local-precisewikiqa-slice",
        "dataset_size": len(items),
        "prepared_count": len(prepared_items),
        "items": prepared_items,
    }
    source_info = {
        "benchmark": "hallulens",
        "task": "PreciseWikiQA",
        "sources": {
            "official_repo": "https://github.com/facebookresearch/HalluLens",
            "goodwiki": f"https://huggingface.co/datasets/{DATASET_ID}",
        },
        "notes": {
            "fidelity": "HalluLens-style local slice",
            "why_local_slice": "The official HalluLens pipeline dynamically generates prompts from GoodWiki. This local task fixes a deterministic GoodWiki slice so the benchmark stays runnable without upstream generation services."
        },
        "prepared_count": len(prepared_items),
        "dataset_size": len(items),
    }
    write_json(MANIFEST_PATH, manifest)
    write_json(SOURCE_INFO_PATH, source_info)
    print(f"Wrote {len(prepared_items)} HalluLens PreciseWikiQA prompts to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
