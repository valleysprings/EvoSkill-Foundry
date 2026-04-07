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
from app.bench.personalization_support import benchmark_metadata, write_manifest


MANIFEST_PATH = ROOT / "data" / "questions.json"
DATASET_ID = "incharacter_questionnaire_alignment"
SPLIT = "official:questionnaire-bundles"
FULL_DATASET_SIZE = 448
PROMPT = (
    "Answer every questionnaire prompt in character. Return strict JSON only, where each key is a question id and "
    "each value is your in-character answer text for that question."
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the official-protocol InCharacter questionnaire bundles.")
    parser.add_argument("--items", type=int, default=None, help="Optional prefix length.")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return payload


def _dimension_description(questionnaire: dict[str, Any], dimension: str) -> str:
    prompts = questionnaire.get("prompts")
    if not isinstance(prompts, dict):
        return ""
    dim_desc = prompts.get("dim_desc")
    if isinstance(dim_desc, dict):
        value = dim_desc.get(dimension)
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            for nested in value.values():
                text = str(nested or "").strip()
                if text:
                    return text
    return ""


def _questionnaire_bundle(questionnaire: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    question_map = questionnaire.get("questions")
    if not isinstance(question_map, dict):
        raise ValueError("InCharacter questionnaire metadata must expose a questions object.")
    categories = questionnaire.get("categories")
    if not isinstance(categories, list):
        raise ValueError("InCharacter questionnaire metadata must expose a categories list.")
    reverse_ids = {str(value).strip() for value in list(questionnaire.get("reverse") or []) if str(value).strip()}
    rows: list[dict[str, Any]] = []
    dimension_questions: dict[str, list[str]] = {}
    seen_question_ids: set[str] = set()
    for category in categories:
        if not isinstance(category, dict):
            continue
        dimension = str(category.get("cat_name") or "").strip()
        if not dimension:
            continue
        question_ids: list[str] = []
        for question_id in list(category.get("cat_questions") or []):
            normalized = str(question_id).strip()
            if not normalized:
                continue
            question_ids.append(normalized)
            if normalized in seen_question_ids:
                continue
            raw_question = question_map.get(normalized)
            if not isinstance(raw_question, dict):
                raise ValueError(f"Missing InCharacter question metadata for id={normalized!r}.")
            rows.append(
                {
                    "id": normalized,
                    "statement": str(raw_question.get("origin_en") or "").strip(),
                    "question": str(raw_question.get("rewritten_en") or raw_question.get("origin_en") or "").strip(),
                    "category": str(raw_question.get("category") or "").strip(),
                    "dimension": dimension,
                    "reverse_scored": normalized in reverse_ids,
                }
            )
            seen_question_ids.add(normalized)
        if question_ids:
            dimension_questions[dimension] = question_ids

    if not rows or not dimension_questions:
        raise ValueError("Unable to materialize the InCharacter questionnaire bundle.")
    return rows, dimension_questions


def _labels_by_dimension(raw_labels: dict[str, Any]) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for dimension, detail in sorted(raw_labels.items()):
        if not isinstance(detail, dict):
            continue
        labels[str(dimension)] = {
            "score": detail.get("score"),
            "type": str(detail.get("type") or "").strip().upper(),
        }
    return labels


def _build_items() -> list[dict[str, Any]]:
    source_root = ensure_repo_checkout("incharacter")
    characters = _load_json(source_root / "data" / "characters.json")
    labels_root = _load_json(source_root / "data" / "characters_labels.json")
    annotations = labels_root.get("annotation")
    pdb_labels_root = labels_root.get("pdb")
    if not isinstance(annotations, dict):
        raise ValueError("InCharacter annotations must provide an 'annotation' object.")
    if not isinstance(pdb_labels_root, dict):
        raise ValueError("InCharacter annotations must provide a 'pdb' object.")

    questionnaire_dir = source_root / "data" / "questionnaires"
    questionnaires = {path.stem: _load_json(path) for path in sorted(questionnaire_dir.glob("*.json"))}

    items: list[dict[str, Any]] = []
    for character_key in sorted(annotations):
        character_entry = characters.get(character_key)
        alias_list: list[str] = []
        if isinstance(character_entry, dict):
            aliases = character_entry.get("alias")
            if isinstance(aliases, list):
                alias_list = [str(alias).strip() for alias in aliases if str(alias).strip()]
        display_name = alias_list[0] if alias_list else str(character_key)

        tests = annotations.get(character_key)
        if not isinstance(tests, dict):
            continue
        pdb_tests = pdb_labels_root.get(character_key)
        if not isinstance(pdb_tests, dict):
            raise ValueError(f"Missing InCharacter pdb labels for {character_key!r}.")
        experimenter = ""
        if isinstance(character_entry, dict):
            experimenter = str(character_entry.get("experimenter") or "").strip()
        for questionnaire_name in sorted(tests):
            questionnaire = questionnaires.get(questionnaire_name)
            if questionnaire is None:
                raise KeyError(f"Missing questionnaire metadata for {questionnaire_name!r}.")
            full_name = str(questionnaire.get("full_name") or questionnaire_name).strip()
            dimensions = tests.get(questionnaire_name)
            pdb_dimensions = pdb_tests.get(questionnaire_name)
            if not isinstance(dimensions, dict):
                continue
            if not isinstance(pdb_dimensions, dict):
                pdb_dimensions = dimensions
            question_rows, dimension_questions = _questionnaire_bundle(questionnaire)
            dimension_descriptions = {
                dimension: _dimension_description(questionnaire, dimension)
                for dimension in sorted(dimension_questions)
            }
            question_blocks: list[str] = []
            for dimension in sorted(dimension_questions):
                description = dimension_descriptions.get(dimension) or ""
                question_lines = [
                    f"- {row['id']}: {row['question']}"
                    for row in question_rows
                    if str(row.get("dimension") or "").strip() == dimension
                ]
                block_parts = [f"{dimension}"]
                if description:
                    block_parts.append(description)
                block_parts.append("\n".join(question_lines))
                question_blocks.append("\n".join(part for part in block_parts if part))
            context_blocks = [
                f"Character: {display_name}",
                f"Aliases: {', '.join(alias_list)}" if alias_list else "",
                f"Questionnaire: {full_name}",
                "Dimensions:",
                "\n\n".join(question_blocks),
                'Return JSON only. Example: {"1": "answer one", "6": "answer two"}',
            ]
            item = {
                "item_id": f"incharacter-{character_key}-{questionnaire_name}",
                "name": f"InCharacter {display_name} {questionnaire_name}",
                "prompt": PROMPT,
                "context": "\n\n".join(block for block in context_blocks if block),
                "raw_context": {
                    "character": display_name,
                    "aliases": alias_list,
                    "experimenter": experimenter,
                    "questionnaire": questionnaire_name,
                    "questionnaire_full_name": full_name,
                    "dimension_descriptions": dimension_descriptions,
                    "dimension_questions": dimension_questions,
                    "questions": question_rows,
                    "response_format": {"question_id": "answer"},
                },
                "expected_answer": "__judge_model__",
                "metadata": benchmark_metadata(
                    benchmark="incharacter",
                    benchmark_category="explicit_character_persona",
                    interaction_mode="single_turn",
                    task_shape="dialogue_judgement",
                    scoring_mode="hybrid",
                    extra={
                        "character_key": character_key,
                        "character_aliases": alias_list,
                        "questionnaire": questionnaire_name,
                        "annotation_labels": _labels_by_dimension(dimensions),
                        "pdb_labels": _labels_by_dimension(pdb_dimensions),
                        "questionnaire_range": list(questionnaire.get("range") or []),
                        "questionnaire_compute_mode": questionnaire.get("compute_mode"),
                    },
                ),
            }
            items.append(item)
    if len(items) != FULL_DATASET_SIZE:
        raise ValueError(f"Expected {FULL_DATASET_SIZE} InCharacter items, found {len(items)}.")
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
    print(f"Wrote {requested} InCharacter rows to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
