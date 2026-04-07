from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRACK_ROOT = ROOT.parent
REPO_ROOT = ROOT.parents[2]
for candidate in (TRACK_ROOT, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from sync_external import ensure_repo_checkout
from app.bench.personalization_support import benchmark_metadata, make_dialogue_next_turn_item, write_manifest


MANIFEST_PATH = ROOT / "data" / "questions.json"
DATASET_ID = "coser_test_set_fanout"
SPLIT = "official:test-set:fanout"
FULL_DATASET_SIZE = 1991
PROMPT = "Write the next in-character literary dialogue turn only. Keep the response grounded in the book context and current scene."


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the public CoSER literary dialogue fan-out.")
    parser.add_argument("--items", type=int, default=None, help="Optional prefix length.")
    return parser.parse_args(argv)


def _render_profile_block(character_profiles: dict[str, object], speaker: str, other_characters: list[str]) -> str:
    blocks: list[str] = []
    speaker_profile = character_profiles.get(speaker)
    if speaker_profile is not None:
        blocks.append(f"{speaker} profile:\n{str(speaker_profile).strip()}")
    for other in other_characters:
        profile = character_profiles.get(other)
        if profile is None:
            continue
        blocks.append(f"{other} profile:\n{str(profile).strip()}")
    return "\n\n".join(block for block in blocks if block)


def _build_items() -> list[dict[str, object]]:
    source_root = ensure_repo_checkout("coser")
    conversations = json.loads((source_root / "data" / "test" / "test_set.json").read_text())
    if not isinstance(conversations, list):
        raise ValueError("CoSER test_set.json must contain a list.")

    items: list[dict[str, object]] = []
    for case_index, conversation in enumerate(conversations):
        if not isinstance(conversation, dict):
            continue
        dialogues = list(conversation.get("dialogues") or [])
        major_characters = [str(value).strip() for value in list(conversation.get("major_characters") or []) if str(value).strip()]
        profile_map = conversation.get("character_profiles")
        character_profiles = profile_map if isinstance(profile_map, dict) else {}
        scenario = str(conversation.get("scenario") or "").strip()
        topic = str(conversation.get("topic") or "").strip()
        plot = conversation.get("plot")
        plot_summary = str(plot.get("summary") or "").strip() if isinstance(plot, dict) else ""
        book = str(conversation.get("book") or "").strip()
        tag = str(conversation.get("tag") or "").strip()

        for turn_index, turn in enumerate(dialogues):
            if not isinstance(turn, dict) or turn_index == 0:
                continue
            speaker = str(turn.get("character") or "").strip()
            message = str(turn.get("message") or "").strip()
            if not speaker or not message or speaker not in set(major_characters):
                continue
            previous_turn = dialogues[turn_index - 1]
            if isinstance(previous_turn, dict) and str(previous_turn.get("character") or "").strip() == "Environment":
                continue
            history = [
                (str(entry.get("character") or "").strip(), str(entry.get("message") or "").strip())
                for entry in dialogues[:turn_index]
                if isinstance(entry, dict) and str(entry.get("character") or "").strip() and str(entry.get("message") or "").strip()
            ]
            if not history:
                continue
            other_characters = [name for name in major_characters if name != speaker]
            context_blocks = [
                f"Book: {book}" if book else "",
                f"Topic: {topic}" if topic else "",
                f"Scenario:\n{scenario}" if scenario else "",
                f"Plot summary:\n{plot_summary}" if plot_summary else "",
                _render_profile_block(character_profiles, speaker, other_characters[:1]),
            ]
            items.append(
                make_dialogue_next_turn_item(
                    item_id=f"coser-{case_index + 1}-{turn_index + 1}-{speaker}",
                    name=f"CoSER {speaker} #{case_index + 1}/{turn_index + 1}",
                    prompt=PROMPT,
                    dialogue_history=history,
                    expected_reply=message,
                    metadata=benchmark_metadata(
                        benchmark="coser",
                        benchmark_category="explicit_character_persona",
                        interaction_mode="multi_turn",
                        task_shape="dialogue_next_turn",
                        scoring_mode="hybrid",
                        extra={
                            "speaker": speaker,
                            "book": book,
                            "topic": topic,
                            "tag": tag,
                            "case_index": case_index,
                            "turn_index": turn_index,
                            "profile_context": "\n\n".join(block for block in context_blocks if block),
                        },
                    ),
                    response_aliases=[message],
                )
            )
            items[-1]["context"] = "\n\n".join(
                block
                for block in (
                    "\n\n".join(block for block in context_blocks if block),
                    items[-1]["context"],
                )
                if block
            )
    if len(items) != FULL_DATASET_SIZE:
        raise ValueError(f"Expected {FULL_DATASET_SIZE} CoSER fan-out items, found {len(items)}.")
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
    print(f"Wrote {requested} CoSER rows to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
