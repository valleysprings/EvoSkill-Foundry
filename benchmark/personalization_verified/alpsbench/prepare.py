from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote

try:
    from huggingface_hub import hf_hub_download
except ModuleNotFoundError:
    hf_hub_download = None

ROOT = Path(__file__).resolve().parent
TRACK_ROOT = ROOT.parent
REPO_ROOT = ROOT.parents[2]
for candidate in (TRACK_ROOT, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from app.bench.personalization_support import (
    benchmark_metadata,
    make_choice_item,
    serialize_dialogue_history,
    write_manifest,
)


MANIFEST_PATH = ROOT / "data" / "questions.json"
CACHE_DIR = ROOT / "data" / "_downloads"
DATASET_ID = "alpsbench_task4_validation"
SPLIT = "validation:task4:ability1-5"
DATASET_REPO = "Cosineyx/Alpsbench"
FULL_DATASET_SIZE = 577
PROMPT = (
    "Choose the memory snippet that should be retrieved and used to answer the current user request. "
    "Return the option or the full matching memory snippet."
)
ABILITY_PATHS: tuple[str, ...] = ("ability1", "ability2", "ability3", "ability4", "ability5")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the full local AlpsBench manifest.")
    parser.add_argument("--items", type=int, default=None, help="Optional prefix length.")
    return parser.parse_args(argv)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _dataset_file(filename: str) -> Path:
    if hf_hub_download is not None:
        return Path(hf_hub_download(DATASET_REPO, filename=filename, repo_type="dataset"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    destination = CACHE_DIR / filename.replace("/", "__")
    if destination.exists():
        return destination
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    url = f"https://huggingface.co/datasets/{quote(DATASET_REPO, safe='/')}/resolve/main/{quote(filename, safe='/')}"
    with urllib.request.urlopen(url, timeout=120) as response:
        temp_path.write_bytes(response.read())
    temp_path.replace(destination)
    return destination


def _memory_text(memory: object) -> str:
    if isinstance(memory, dict):
        label = str(memory.get("label") or "").strip()
        value = str(memory.get("value") or "").strip()
        if label and value:
            return f"{label}: {value}"
        if value:
            return value
        memory_id = str(memory.get("memory_id") or "").strip()
        if memory_id:
            return memory_id
    return str(memory or "").strip()


def _load_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for ability in ABILITY_PATHS:
        input_path = _dataset_file(f"dataset/validation/task4_{ability}/model_input.jsonl")
        reference_path = _dataset_file(f"dataset/validation/task4_{ability}/reference_output.jsonl")
        input_rows = _load_jsonl(input_path)
        reference_rows = _load_jsonl(reference_path)
        if len(input_rows) != len(reference_rows):
            raise ValueError(f"AlpsBench {ability} input/output row counts differ.")
        for input_row, reference_row in zip(input_rows, reference_rows, strict=True):
            benchmark_id = str(input_row.get("benchmark_id") or input_row.get("canonical_id") or "").strip()
            input_payload = input_row.get("input")
            gold_payload = reference_row.get("gold")
            if not isinstance(input_payload, dict) or not isinstance(gold_payload, dict):
                raise ValueError(f"Malformed AlpsBench row for {benchmark_id or ability}.")
            audit_context = input_payload.get("audit_context")
            history = list(audit_context.get("conversation") or []) if isinstance(audit_context, dict) else []
            query = str(input_payload.get("query") or "").strip()
            correct_memory = _memory_text(gold_payload.get("selected_memory"))
            extracted_memories = [
                memory_text
                for memory_text in (_memory_text(item) for item in list(gold_payload.get("extracted_memory") or []))
                if memory_text
            ]
            if not benchmark_id or not query or not correct_memory:
                raise ValueError(f"Incomplete AlpsBench row for ability={ability!r}.")
            records.append(
                {
                    "item_id": benchmark_id,
                    "name": f"AlpsBench {ability} {benchmark_id}",
                    "ability": ability,
                    "query": query,
                    "history": history,
                    "correct_memory": correct_memory,
                    "extracted_memories": extracted_memories,
                }
            )
    if len(records) != FULL_DATASET_SIZE:
        raise ValueError(f"Expected {FULL_DATASET_SIZE} AlpsBench rows, found {len(records)}.")
    return records


def _pick_memory_distractors(records: list[dict[str, object]], index: int) -> list[str]:
    record = records[index]
    correct = str(record["correct_memory"])
    choices: list[str] = []
    seen = {correct}
    for memory in list(record["extracted_memories"]):
        candidate = str(memory).strip()
        if candidate and candidate not in seen:
            choices.append(candidate)
            seen.add(candidate)
        if len(choices) == 3:
            return choices
    total = len(records)
    for attempt in range(1, total + 1):
        candidate_record = records[(index + (attempt * 29)) % total]
        candidate_memory = str(candidate_record["correct_memory"]).strip()
        if candidate_memory and candidate_memory not in seen:
            choices.append(candidate_memory)
            seen.add(candidate_memory)
        if len(choices) == 3:
            return choices
    raise ValueError(f"Unable to find enough AlpsBench distractors for row {index}.")


def _build_items() -> list[dict[str, object]]:
    records = _load_records()
    items: list[dict[str, object]] = []
    for index, record in enumerate(records):
        correct = str(record["correct_memory"])
        distractors = _pick_memory_distractors(records, index)
        correct_position = index % 4
        choices = list(distractors)
        choices.insert(correct_position, correct)
        history_text = serialize_dialogue_history(record["history"])
        context = "\n\n".join(
            block
            for block in (
                f"Conversation history:\n{history_text}" if history_text else "",
                f"Current user request:\n{record['query']}",
            )
            if block
        )
        items.append(
            make_choice_item(
                item_id=str(record["item_id"]),
                name=str(record["name"]),
                prompt=PROMPT,
                context=context,
                choices=choices,
                correct_choice_index=correct_position,
                metadata=benchmark_metadata(
                    benchmark="alpsbench",
                    benchmark_category="user_persona_personalization",
                    interaction_mode="single_turn",
                    task_shape="mcq",
                    scoring_mode="exact_match",
                    extra={"ability": record["ability"]},
                ),
            )
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
    print(f"Wrote {requested} AlpsBench rows to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
