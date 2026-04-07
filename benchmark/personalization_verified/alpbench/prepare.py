from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRACK_ROOT = ROOT.parent
REPO_ROOT = ROOT.parents[2]
for candidate in (TRACK_ROOT, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from app.bench.personalization_support import benchmark_metadata, make_label_item, write_manifest


MANIFEST_PATH = ROOT / "data" / "questions.json"
DATASET_ID = "alpbench_train_inputs_1y"
SPLIT = "train:inputs_1y"
FULL_DATASET_SIZE = 800
PROMPT = (
    "Infer the user's latent purchase-preference combination from the behavior history. "
    "Return exactly one option label or the full matching combination."
)
ROWS_API_TEMPLATE = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=OpenOneRec%2FALPBench&config=default&split=train&offset={offset}&length={length}"
)
PAGE_SIZE = 100
LABEL_NAMES = ("option_a", "option_b", "option_c", "option_d")
LABEL_LETTERS = ("A", "B", "C", "D")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the full local ALPBench manifest.")
    parser.add_argument("--items", type=int, default=None, help="Optional prefix length.")
    return parser.parse_args(argv)


def _fetch_page(offset: int, length: int) -> dict[str, object]:
    url = ROWS_API_TEMPLATE.format(offset=offset, length=length)
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ALPBench rows API returned a non-object payload.")
    return payload


def _load_rows() -> list[dict[str, object]]:
    first_page = _fetch_page(0, PAGE_SIZE)
    rows = list(first_page.get("rows") or [])
    total = int(first_page.get("num_rows_total") or len(rows))
    offset = len(rows)
    while offset < total:
        page = _fetch_page(offset, min(PAGE_SIZE, total - offset))
        rows.extend(list(page.get("rows") or []))
        offset = len(rows)
    normalized = [row.get("row") for row in rows if isinstance(row, dict) and isinstance(row.get("row"), dict)]
    if len(normalized) != FULL_DATASET_SIZE:
        raise ValueError(f"Expected {FULL_DATASET_SIZE} ALPBench rows, found {len(normalized)}.")
    return normalized


def _parse_ground_truth(raw_value: str) -> list[dict[str, str]]:
    matches = re.findall(r"\{[^{}]+\}", raw_value, flags=re.DOTALL)
    combos: list[dict[str, str]] = []
    for match in matches:
        payload = ast.literal_eval(match)
        if isinstance(payload, dict):
            combos.append({str(key): str(value) for key, value in payload.items()})
    if not combos:
        raise ValueError(f"Unable to parse ALPBench ground truth: {raw_value[:200]!r}")
    return combos


def _parse_candidate_values(text: str) -> dict[str, list[str]]:
    matches = re.findall(r"^\s*-\s*([^:：]+):\s*(\[[^\n]+\])", text, flags=re.MULTILINE)
    values: dict[str, list[str]] = {}
    for raw_name, raw_options in matches:
        parsed = ast.literal_eval(raw_options)
        if isinstance(parsed, list):
            values[str(raw_name).strip()] = [str(option).strip() for option in parsed if str(option).strip()]
    if values:
        return values

    dict_matches = re.findall(r"\{[^{}]*\[[^\]]+\][^{}]*\}", text, flags=re.DOTALL)
    candidate_dicts: list[dict[str, list[str]]] = []
    for raw_dict in dict_matches:
        try:
            parsed_dict = ast.literal_eval(raw_dict)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(parsed_dict, dict):
            continue
        normalized: dict[str, list[str]] = {}
        for key, raw_value in parsed_dict.items():
            if isinstance(raw_value, list):
                normalized[str(key).strip()] = [str(option).strip() for option in raw_value if str(option).strip()]
        if normalized:
            candidate_dicts.append(normalized)
    if candidate_dicts:
        candidate_dicts.sort(key=lambda payload: sum(len(options) for options in payload.values()), reverse=True)
        return candidate_dicts[0]

    raise ValueError("Unable to parse ALPBench candidate attribute values.")


def _combo_key(combo: dict[str, str], attribute_order: list[str]) -> tuple[tuple[str, str], ...]:
    return tuple((attribute, str(combo.get(attribute) or "")) for attribute in attribute_order)


def _format_combo(combo: dict[str, str], attribute_order: list[str]) -> str:
    return " | ".join(f"{attribute}={combo.get(attribute)}" for attribute in attribute_order)


def _build_distractors(
    *,
    primary_gold: dict[str, str],
    candidate_values: dict[str, list[str]],
    gold_set: set[tuple[tuple[str, str], ...]],
    attribute_order: list[str],
) -> list[dict[str, str]]:
    distractors: list[dict[str, str]] = []
    seen = set(gold_set)
    for attribute in attribute_order:
        current_value = primary_gold.get(attribute)
        for candidate_value in candidate_values.get(attribute, []):
            if candidate_value == current_value:
                continue
            combo = dict(primary_gold)
            combo[attribute] = candidate_value
            key = _combo_key(combo, attribute_order)
            if key in seen:
                continue
            distractors.append(combo)
            seen.add(key)
            if len(distractors) == 3:
                return distractors
    for left_index, left_attribute in enumerate(attribute_order):
        for right_attribute in attribute_order[left_index + 1 :]:
            for left_value in candidate_values.get(left_attribute, []):
                if left_value == primary_gold.get(left_attribute):
                    continue
                for right_value in candidate_values.get(right_attribute, []):
                    if right_value == primary_gold.get(right_attribute):
                        continue
                    combo = dict(primary_gold)
                    combo[left_attribute] = left_value
                    combo[right_attribute] = right_value
                    key = _combo_key(combo, attribute_order)
                    if key in seen:
                        continue
                    distractors.append(combo)
                    seen.add(key)
                    if len(distractors) == 3:
                        return distractors
    raise ValueError(f"Unable to build enough ALPBench distractors for gold combo {primary_gold!r}.")


def _build_items() -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    base_metadata = benchmark_metadata(
        benchmark="alpbench",
        benchmark_category="user_persona_personalization",
        interaction_mode="single_turn",
        task_shape="classification",
        scoring_mode="label_match",
    )
    for row_index, row in enumerate(_load_rows()):
        history = str(row.get("inputs_1y") or "").strip()
        if not history:
            raise ValueError(f"Missing ALPBench inputs_1y at row {row_index}.")
        gold_combos = _parse_ground_truth(str(row.get("ground-truth") or ""))
        primary_gold = gold_combos[0]
        attribute_order = list(primary_gold)
        candidate_values = _parse_candidate_values(history)
        gold_set = {_combo_key(combo, attribute_order) for combo in gold_combos}
        distractors = _build_distractors(
            primary_gold=primary_gold,
            candidate_values=candidate_values,
            gold_set=gold_set,
            attribute_order=attribute_order,
        )
        option_payloads = [_format_combo(primary_gold, attribute_order)] + [
            _format_combo(combo, attribute_order) for combo in distractors
        ]
        correct_position = row_index % 4
        ordered_payloads = list(option_payloads[1:])
        ordered_payloads.insert(correct_position, option_payloads[0])
        expected_label = LABEL_NAMES[correct_position]
        label_aliases = {
            label_name: [letter, ordered_payloads[position]]
            for position, (label_name, letter) in enumerate(zip(LABEL_NAMES, LABEL_LETTERS, strict=True))
        }
        label_aliases[expected_label].extend(_format_combo(combo, attribute_order) for combo in gold_combos[1:])
        option_lines = [f"{letter}. {ordered_payloads[position]}" for position, letter in enumerate(LABEL_LETTERS)]
        context = f"{history}\n\nCandidate option labels:\n" + "\n".join(option_lines)
        items.append(
            make_label_item(
                item_id=f"alpbench-{row_index + 1}",
                name=f"ALPBench #{row_index + 1}",
                prompt=PROMPT,
                expected_label=expected_label,
                allowed_labels=list(LABEL_NAMES),
                label_aliases=label_aliases,
                context=context,
                metadata={
                    **base_metadata,
                    "attribute_order": attribute_order,
                    "gold_combo_count": len(gold_combos),
                },
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
    print(f"Wrote {requested} ALPBench rows to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
