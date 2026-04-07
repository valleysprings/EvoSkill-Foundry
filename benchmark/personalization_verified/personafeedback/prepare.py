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

from app.bench.personalization_support import benchmark_metadata, make_choice_item, write_manifest


MANIFEST_PATH = ROOT / "data" / "questions.json"
CACHE_DIR = ROOT / "data" / "_downloads"
DATASET_ID = "personafeedback_public"
SPLIT = "general+spedific:easy+medium+hard"
DATASET_REPO = "PersonalAILab/PersonaFeedback"
FULL_DATASET_SIZE = 8298
PROMPT = (
    "Choose the response that best fits the user's explicit persona and request. "
    "Return the option or the full matching reply."
)
DATA_FILES: tuple[tuple[str, str, str], ...] = (
    ("general", "easy", "data/general/easy.jsonl"),
    ("general", "medium", "data/general/medium.jsonl"),
    ("general", "hard", "data/general/hard.jsonl"),
    ("spedific", "easy", "data/spedific/easy.jsonl"),
    ("spedific", "medium", "data/spedific/medium.jsonl"),
    ("spedific", "hard", "data/spedific/hard.jsonl"),
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the full local PersonaFeedback manifest.")
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


def _flatten_dict(prefix: str, value: Any, lines: list[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_dict(next_prefix, nested, lines)
        return
    if isinstance(value, list):
        text = ", ".join(str(item).strip() for item in value if str(item).strip())
    else:
        text = str(value or "").strip()
    if text:
        lines.append(f"- {prefix}: {text}")


def _persona_context(persona: dict[str, Any]) -> str:
    lines = ["User persona:"]
    name = str(persona.get("Name") or "").strip()
    if name:
        lines.append(f"- Name: {name}")
    for section_name in ("Demographics", "Personality", "Preference", "Bias"):
        section = persona.get(section_name)
        if isinstance(section, dict):
            for key, value in section.items():
                _flatten_dict(f"{section_name}.{key}", value, lines)
        elif section is not None:
            text = str(section).strip()
            if text:
                lines.append(f"- {section_name}: {text}")
    return "\n".join(lines)


def _load_personas() -> dict[str, dict[str, Any]]:
    path = _dataset_file("persona/all_personas.jsonl")
    personas: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            payload = json.loads(line)
            if isinstance(payload, dict):
                personas[str(index)] = payload
    return personas


def _build_items() -> list[dict[str, object]]:
    personas = _load_personas()
    items: list[dict[str, object]] = []
    row_index = 0
    for track_name, difficulty, relative_path in DATA_FILES:
        path = _dataset_file(relative_path)
        for row in _load_jsonl(path):
            user_id = str(row.get("user_id") or "").strip()
            persona = personas.get(user_id)
            if persona is None:
                raise KeyError(f"Missing PersonaFeedback persona for user_id={user_id!r}.")
            chosen = str(row.get("chosen") or "").strip()
            rejected = str(row.get("reject") or "").strip()
            question = str(row.get("question") or "").strip()
            if not question or not chosen or not rejected:
                raise ValueError(f"Incomplete PersonaFeedback row in {relative_path}: {row}")
            correct_position = row_index % 2
            choices = [rejected, chosen]
            if correct_position == 0:
                choices = [chosen, rejected]
            items.append(
                make_choice_item(
                    item_id=f"personafeedback-{track_name}-{difficulty}-{row_index + 1}",
                    name=f"PersonaFeedback {track_name} {difficulty} #{row_index + 1}",
                    prompt=PROMPT,
                    context=f"{_persona_context(persona)}\n\nUser request:\n{question}",
                    choices=choices,
                    correct_choice_index=correct_position,
                    metadata=benchmark_metadata(
                        benchmark="personafeedback",
                        benchmark_category="user_persona_personalization",
                        interaction_mode="single_turn",
                        task_shape="mcq",
                        scoring_mode="exact_match",
                        extra={
                            "track_name": track_name,
                            "difficulty": difficulty,
                            "user_id": user_id,
                        },
                    ),
                )
            )
            row_index += 1
    if len(items) != FULL_DATASET_SIZE:
        raise ValueError(f"Expected {FULL_DATASET_SIZE} PersonaFeedback rows, found {len(items)}.")
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
    print(f"Wrote {requested} PersonaFeedback rows to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
