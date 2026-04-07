from __future__ import annotations

import argparse
import ast
import csv
import json
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote

try:
    from huggingface_hub import hf_hub_download
except ModuleNotFoundError:
    hf_hub_download = None


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "_downloads"
MANIFEST_PATH = DATA_DIR / "questions.json"
DATASET_ID = "bowen-upenn/PersonaMem"
DATASET_SPLIT = "32k"
FULL_DATASET_SIZE = 589


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the local PersonaMem 32k manifest.")
    parser.add_argument("--items", type=int, default=FULL_DATASET_SIZE, help="Optional prefix length.")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    temp_path.replace(path)


def _dataset_file(filename: str) -> Path:
    if hf_hub_download is not None:
        return Path(hf_hub_download(DATASET_ID, filename=filename, repo_type="dataset"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    destination = CACHE_DIR / filename.replace("/", "__")
    if destination.exists():
        return destination
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    url = f"https://huggingface.co/datasets/{quote(DATASET_ID, safe='/')}/resolve/main/{quote(filename, safe='/')}"
    with urllib.request.urlopen(url, timeout=120) as response:
        temp_path.write_bytes(response.read())
    temp_path.replace(destination)
    return destination


def _format_context(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "unknown").strip().title()
        content = str(message.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def _parse_options(raw_value: str) -> list[str]:
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(raw_value)
    if not isinstance(parsed, list):
        raise ValueError(f"PersonaMem all_options must parse to a list, got {type(parsed).__name__}.")
    return [str(option).strip() for option in parsed]


def _load_shared_contexts() -> dict[str, list[dict[str, Any]]]:
    path = _dataset_file("shared_contexts_32k.jsonl")
    mapping: dict[str, list[dict[str, Any]]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            context_id = next(iter(row))
            mapping[context_id] = list(row[context_id])
    return mapping


def _choice_index(correct_answer: str) -> int:
    normalized = correct_answer.strip().lower()
    if normalized.startswith("(") and normalized.endswith(")") and len(normalized) == 3:
        return ord(normalized[1]) - ord("a")
    raise ValueError(f"Unexpected PersonaMem correct answer marker: {correct_answer!r}")


def main() -> None:
    args = _parse_args()
    requested_items = max(1, min(int(args.items or FULL_DATASET_SIZE), FULL_DATASET_SIZE))
    questions_path = _dataset_file("questions_32k.csv")
    shared_contexts = _load_shared_contexts()

    items: list[dict[str, Any]] = []
    with open(questions_path, encoding="utf-8") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            if row_index >= requested_items:
                break
            options = _parse_options(str(row["all_options"]))
            correct_marker = str(row["correct_answer"]).strip()
            correct_choice_index = _choice_index(correct_marker)
            if correct_choice_index < 0 or correct_choice_index >= len(options):
                raise ValueError(
                    f"PersonaMem row {row.get('question_id')} points at option {correct_choice_index} outside {len(options)} choices."
                )
            shared_context_id = str(row["shared_context_id"]).strip()
            shared_context = shared_contexts.get(shared_context_id)
            if shared_context is None:
                raise KeyError(f"Missing shared context {shared_context_id} for PersonaMem row {row.get('question_id')}.")
            end_index = int(row["end_index_in_shared_context"])
            context_text = _format_context(shared_context[:end_index])
            expected_answer = str(options[correct_choice_index]).strip()
            items.append(
                {
                    "item_id": str(row["question_id"]).strip(),
                    "name": f"PersonaMem 32k #{row_index + 1}",
                    "prompt": str(row["user_question_or_message"]).strip(),
                    "context": context_text,
                    "choices": [str(option).strip() for option in options],
                    "expected_answer": expected_answer,
                    "metadata": {
                        "dataset": "personamem",
                        "split": DATASET_SPLIT,
                        "persona_id": str(row["persona_id"]).strip(),
                        "topic": str(row["topic"]).strip(),
                        "question_type": str(row["question_type"]).strip(),
                        "shared_context_id": shared_context_id,
                        "context_length_in_tokens": int(row["context_length_in_tokens"]),
                        "context_length_in_letters": int(row["context_length_in_letters"]),
                        "distance_to_ref_in_blocks": int(row["distance_to_ref_in_blocks"]),
                        "distance_to_ref_in_tokens": int(row["distance_to_ref_in_tokens"]),
                        "distance_to_ref_proportion_in_context": str(row["distance_to_ref_proportion_in_context"]).strip(),
                        "num_irrelevant_tokens": int(row["num_irrelevant_tokens"]),
                        "correct_choice_index": correct_choice_index,
                        "answer_aliases": [correct_marker, expected_answer],
                    },
                }
            )

    manifest = {
        "dataset_id": "personamem_32k",
        "split": "benchmark:32k",
        "dataset_size": FULL_DATASET_SIZE,
        "prepared_count": len(items),
        "items": items,
    }
    _write_json(MANIFEST_PATH, manifest)
    print(f"Wrote {len(items)} PersonaMem rows to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
