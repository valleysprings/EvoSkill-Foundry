from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
TRACK_ROOT = ROOT.parent
REPO_ROOT = ROOT.parents[2]
for candidate in (TRACK_ROOT, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from sync_external import ensure_repo_checkout
from app.bench.personalization_support import benchmark_metadata, serialize_dialogue_history, write_manifest


MANIFEST_PATH = ROOT / "data" / "questions.json"
DATASET_ID = "characterbench_characterjudge"
SPLIT = "official:raw_data:all-tests"
FULL_DATASET_SIZE = 3250
PROMPT = "Write the next character response only. Stay faithful to the character card, dialogue, and current user message."


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize CharacterBench items for CharacterJudge-style evaluation.")
    parser.add_argument("--items", type=int, default=None, help="Optional prefix length.")
    return parser.parse_args(argv)


def _english_row(row: dict[str, Any]) -> dict[str, Any]:
    translated = row.get("translation_en")
    return translated if isinstance(translated, dict) else row


def _recent_dialogue(dialogue: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    return dialogue[-limit:] if len(dialogue) > limit else dialogue


def _extract_reference_fields(row: dict[str, Any], translated: dict[str, Any], subset: str) -> dict[str, Any]:
    messages = translated.get("messages")
    translated_messages = messages if isinstance(messages, dict) else {}
    output = translated_messages.get("output")
    output_dict = output if isinstance(output, dict) else {}
    response_messages = translated.get("response_messages")
    response_dict = response_messages if isinstance(response_messages, dict) else {}
    original_messages = row.get("messages")
    original_output = original_messages.get("output") if isinstance(original_messages, dict) else {}
    original_output_dict = original_output if isinstance(original_output, dict) else {}

    reference_payload: dict[str, Any] = {}
    if subset == "memory_consistency":
        reference_payload["reference_dialogue"] = list(output_dict.get("dialogue_segments") or [])
    elif subset == "attribute_consistency_bot":
        reference_payload["reference_answer"] = str(output_dict.get("answer") or "").strip()
    elif subset == "attribute_consistency_human":
        reference_payload["reference_answer"] = str(translated_messages.get("reference") or "").strip()
    elif subset == "behavior_consistency_bot":
        reference_payload["reference_behavior"] = str(output_dict.get("reference") or original_output_dict.get("reference") or "").strip()
    elif subset == "behavior_consistency_human":
        reference_payload["reference_behavior"] = str(response_dict.get("character_style") or "").strip()
    elif subset == "boundary_consistency":
        reference_payload["reference_response"] = str(response_dict.get("reference_response") or "").strip()
    elif subset == "fact_accuracy":
        reference_payload["reference_answer"] = str(output_dict.get("answer") or "").strip()
    elif subset in {"emotion_self_regulation", "empathetic_responsiveness"}:
        reference_payload["reference_emotion"] = str(output_dict.get("emotion") or "").strip()
    return reference_payload


def _build_items() -> list[dict[str, Any]]:
    source_root = ensure_repo_checkout("characterbench")
    raw_dir = source_root / "eval_data" / "raw_data"
    rows_by_subset: dict[str, list[dict[str, Any]]] = {}
    subset_max_score: dict[str, float] = {}
    for path in sorted(raw_dir.glob("*_test.json")):
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError(f"CharacterBench raw file must contain a list: {path}")
        subset = path.stem.removesuffix("_test")
        typed_rows = [row for row in rows if isinstance(row, dict)]
        rows_by_subset[subset] = typed_rows
        scores = [float(row.get("annotation_score") or 0.0) for row in typed_rows]
        subset_max_score[subset] = max(scores) if scores else 5.0

    items: list[dict[str, Any]] = []
    for subset in sorted(rows_by_subset):
        for row_index, row in enumerate(rows_by_subset[subset], start=1):
            translated = _english_row(row)
            character_name = str(translated.get("character_name") or row.get("character_name") or "").strip()
            dialogue = [turn for turn in list(translated.get("dialogue") or []) if isinstance(turn, dict)]
            if not character_name or not dialogue:
                continue
            user_message = str(dialogue[-1].get("utterance") or "").strip()
            prior_dialogue = dialogue[:-1]
            if not user_message:
                continue
            character_profile = str(translated.get("character_profile") or row.get("character_profile") or "").strip()
            greeting = str(translated.get("greeting") or row.get("greeting") or "").strip()
            reference_fields = _extract_reference_fields(row, translated, subset)
            context_blocks = [
                f"Character: {character_name}",
                f"Evaluation subset: {subset}",
                f"Character profile:\n{character_profile}" if character_profile else "",
                f"Greeting:\n{greeting}" if greeting else "",
                f"Recent dialogue:\n{serialize_dialogue_history(_recent_dialogue(prior_dialogue))}" if prior_dialogue else "",
                f"Current user message:\n{user_message}",
            ]
            items.append(
                {
                    "item_id": f"characterbench-{subset}-{row.get('id')}",
                    "name": f"CharacterBench {character_name} {subset} #{row_index}",
                    "prompt": PROMPT,
                    "context": "\n\n".join(block for block in context_blocks if block),
                    "raw_context": {
                        "character_name": character_name,
                        "subset": subset,
                        "character_profile": character_profile,
                        "dialogue": prior_dialogue,
                        "user_message": user_message,
                        "greeting": greeting,
                        **reference_fields,
                    },
                    "expected_answer": "__judge_model__",
                    "metadata": benchmark_metadata(
                        benchmark="characterbench",
                        benchmark_category="explicit_character_persona",
                        interaction_mode="single_turn",
                        task_shape="dialogue_judgement",
                        scoring_mode="judge_model",
                        extra={
                            "subset": subset,
                            "character_name": character_name,
                            "source_id": row.get("id"),
                            "annotation_score": row.get("annotation_score"),
                            "subset_max_annotation_score": subset_max_score.get(subset, 5.0),
                        },
                    ),
                }
            )
    if len(items) != FULL_DATASET_SIZE:
        raise ValueError(f"Expected {FULL_DATASET_SIZE} CharacterBench rows, found {len(items)}.")
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
    print(f"Wrote {requested} CharacterBench rows to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
