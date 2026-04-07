from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
except ModuleNotFoundError:
    load_dataset = None


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.bench.personalization_support import benchmark_metadata, write_manifest


MANIFEST_PATH = ROOT / "data" / "questions.json"
DATASET_ID = "timechara_test"
SPLIT = "hf:test"
DATASET_REPO = "ahnpersie/timechara"
FULL_DATASET_SIZE = 10895
PROMPT = "Answer as the character at the stated timeline point. Do not mention facts that should be unknown at that point in the story."

BOOK_NO_TO_NAME = {
    "harry_potter": {
        "1": "Harry Potter and the Philosopher's Stone",
        "2": "Harry Potter and the Chamber of Secrets",
        "3": "Harry Potter and the Prisoner of Azkaban",
        "4": "Harry Potter and the Goblet of Fire",
        "5": "Harry Potter and the Order of the Phoenix",
        "6": "Harry Potter and the Half-Blood Prince",
        "7": "Harry Potter and the Deathly Hallows",
    },
    "the_lord_of_the_rings": {
        "1": "The Fellowship of the Ring",
        "2": "The Two Towers",
        "3": "The Return of the King",
    },
    "twilight": {
        "1": "Twilight",
        "2": "New Moon",
        "3": "Eclipse",
        "4": "Breaking Dawn",
    },
    "hunger_games": {
        "1": "The Hunger Games",
        "2": "Catching Fire",
        "3": "Mocking Jay",
    },
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the official TimeChara test split from Hugging Face.")
    parser.add_argument("--items", type=int, default=None, help="Optional prefix length.")
    return parser.parse_args(argv)


def _load_rows() -> list[dict[str, Any]]:
    if load_dataset is None:
        raise ModuleNotFoundError("datasets is required to prepare TimeChara.")
    dataset = load_dataset(DATASET_REPO, split="test")
    rows = [dict(row) for row in dataset]
    if len(rows) != FULL_DATASET_SIZE:
        raise ValueError(f"Expected {FULL_DATASET_SIZE} TimeChara rows, found {len(rows)}.")
    return rows


def _agent_name(series: str, character: str, character_period: str) -> str:
    book_no = character_period.split("/")[0].strip()
    book_name = BOOK_NO_TO_NAME.get(series, {}).get(book_no[0], book_no)
    if len(character_period.split("/")) == 1:
        day = f"at the end of the scene in {book_name}"
    else:
        day = character_period.split("/")[1].strip()
    if series == "harry_potter":
        if character_period.strip() == "7th-year":
            return f"{character} {day}"
        return f"{book_no} {character} {day}"
    return f"{character} {day}"


def _build_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, row in enumerate(_load_rows()):
        series = str(row.get("series") or "").strip()
        character = str(row.get("character") or "").strip()
        character_period = str(row.get("character_period") or "").strip()
        question = str(row.get("question") or "").strip()
        data_type = str(row.get("data_type") or "").strip()
        temporal_label = str(row.get("temporal_label") or "").strip()
        spatial_label = str(row.get("spatial_label") or "").strip()
        if not series or not character or not character_period or not question:
            continue
        agent_name = _agent_name(series, character, character_period)
        items.append(
            {
                "item_id": f"timechara-{index + 1}",
                "name": f"TimeChara {character} #{index + 1}",
                "prompt": PROMPT,
                "context": "\n\n".join(
                    [
                        f"Series: {series}",
                        f"Character: {character}",
                        f"Timeline point: {character_period}",
                        f"Interviewer question:\n{question}",
                    ]
                ),
                "raw_context": {
                    "series": series,
                    "character": character,
                    "character_period": character_period,
                    "question": question,
                    "data_type": data_type,
                    "temporal_label": temporal_label,
                    "spatial_label": spatial_label,
                    "agent_name": agent_name,
                },
                "expected_answer": "__judge_model__",
                "metadata": benchmark_metadata(
                    benchmark="timechara",
                    benchmark_category="explicit_character_persona",
                    interaction_mode="single_turn",
                    task_shape="dialogue_judgement",
                    scoring_mode="judge_model",
                    extra={
                        "series": series,
                        "character": character,
                        "character_period": character_period,
                        "data_type": data_type,
                    },
                ),
            }
        )
    return items


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    items = _build_items()
    requested = len(items) if args.items is None else max(1, min(int(args.items), len(items)))
    write_manifest(
        MANIFEST_PATH,
        dataset_id=DATASET_ID,
        split=SPLIT,
        items=items[:requested],
        dataset_size=FULL_DATASET_SIZE,
    )
    print(f"Wrote {requested} TimeChara rows to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
