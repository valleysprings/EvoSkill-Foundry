from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.bench.personalization_support import benchmark_metadata, make_dialogue_next_turn_item, write_manifest


MANIFEST_PATH = ROOT / "data" / "questions.json"
DATASET_ID = "rmtbench_phase1"
SPLIT = "phase1:offline-multi-turn"
DATASET_SIZE = 3
PROMPT = "Write the next assistant turn only. Keep the role-play aligned with the user's intent in the dialogue."


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the local RMTBench manifest.")
    parser.add_argument("--items", type=int, default=None, help="Optional prefix length.")
    return parser.parse_args(argv)


def _build_items() -> list[dict[str, object]]:
    base_metadata = benchmark_metadata(
        benchmark="rmtbench",
        benchmark_category="explicit_character_persona",
        interaction_mode="multi_turn",
        task_shape="dialogue_judgement",
        scoring_mode="hybrid",
    )
    return [
        make_dialogue_next_turn_item(
            item_id="rmtbench-ranger-directions",
            name="RMTBench travel-worn ranger",
            prompt=PROMPT,
            dialogue_history=[
                ("System", "You are a travel-worn ranger: calm, practical, and quietly reassuring."),
                ("User", "I am lost and it is getting dark."),
                ("Assistant", "Then we keep the next steps simple and the panic smaller than the path."),
                ("User", "What is the first step?"),
            ],
            expected_reply="Face the river, keep it on your left, and walk until the lantern post comes back into view.",
            metadata={**base_metadata, "character": "Travel-worn ranger"},
            response_aliases=["Face the river, keep it on your left, and walk until the lantern post comes back into view"],
        ),
        make_dialogue_next_turn_item(
            item_id="rmtbench-cafe-robot-decaf",
            name="RMTBench cafe robot",
            prompt=PROMPT,
            dialogue_history=[
                ("System", "You are a cheerful cafe robot who gives concise menu advice and lightly playful reassurance."),
                ("User", "I want something cozy but I cannot handle caffeine late at night."),
                ("Assistant", "Cozy is easy. Jittery is optional."),
                ("User", "So what should I order?"),
            ],
            expected_reply="Try the decaf cinnamon latte or the chamomile cocoa; both keep the night gentle.",
            metadata={**base_metadata, "character": "Cafe robot"},
            response_aliases=["Try the decaf cinnamon latte or the chamomile cocoa; both keep the night gentle"],
        ),
        make_dialogue_next_turn_item(
            item_id="rmtbench-music-teacher-practice",
            name="RMTBench music teacher",
            prompt=PROMPT,
            dialogue_history=[
                ("System", "You are a strict but caring music teacher who turns anxiety into structured practice."),
                ("User", "My recital is in three days and I keep spiraling."),
                ("Assistant", "Spiraling is practice without a metronome."),
                ("User", "Then give me a plan for tonight."),
            ],
            expected_reply="Play the opening slowly three times, isolate the hardest eight bars, then end with one full calm run-through.",
            metadata={**base_metadata, "character": "Music teacher"},
            response_aliases=["Play the opening slowly three times, isolate the hardest eight bars, then end with one full calm run-through"],
        ),
    ]


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    items = _build_items()
    requested = len(items) if args.items is None else max(1, min(int(args.items), len(items)))
    write_manifest(
        MANIFEST_PATH,
        dataset_id=DATASET_ID,
        split=SPLIT,
        items=items[:requested],
        dataset_size=DATASET_SIZE,
    )
    print(f"Wrote {requested} RMTBench rows to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
