from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
TRACK_ROOT = ROOT.parent
if str(TRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACK_ROOT))

from sync_external import ensure_repo_checkout

DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "questions.json"
SOURCE_FILES: tuple[str, ...] = (
    "conversation_memory.json",
    "emotional_perception.json",
    "self_awareness.json",
    "social_preference.json",
)
FULL_DATASET_SIZE = 7702

PROMPT_EN = """
==Profile==
{role_profile}

==Conversations==
{conversations}

You are playing the role of {role_name}, you need to embody the knowledge and style of {role_name}.
Based on the provided role Profile and Conversations, please choose the best option (A, B, C, or D):
{options}

Your selection:
"""

PROMPT_ZH = """
==角色描述==
{role_profile}

==对话历史==
{conversations}

你要扮演{role_name}角色，你在聊天中要具备该角色对应的知识背景，语气风格等特征。
请根据所给的{role_name}角色描述和对话历史，从下面四个选项（A. B. C.和D.）中选择符合{role_name}的选项：
{options}

你的选择：
"""

PROMPT_DIALOGUE_EMOTION_EN = """
==Conversations==
{conversations}

Select the option () that best matches the mood in utterance "{utterance}". Single Choice
{options}

Your selection:
"""

PROMPT_DIALOGUE_EMOTION_ZH = """
==对话历史==
{conversations}

单选选择题，选择最符合"{utterance}"说话者当时心情的选项()
{options}

你的选择:
"""

PROMPT_OPEN_ZH = """
==角色描述==
{role_profile}

==对话历史==
{conversations}

你要扮演{role_name}角色，你在聊天中要具备该角色对应的知识背景，语气风格等特征。
请根据所给的{role_name}角色描述和对话历史，根据最后一个User的对话再补充一轮你作为Assistant的回复（一轮就好）：
Assistant: 
"""

PROMPT_OPEN_EN = """
==Profile==
{role_profile}

==Conversations==
{conversations}

You are playing the role of {role_name}, you need to embody the knowledge and style of {role_name}.
Based on the provided role Profile and Conversations, you must produce a reply as the Assistant to response to the latest User's message (one term is enough):
Assistant: 
"""

PROMPT_GROUP_EN = """
==Profiles==
{role_profiles}

==Conversations==
{conversations}

You are playing the role of {role_name}, you need to embody the social preference of {role_name}.
Based on the provided role profile and conversations, please choose the best option (A, B, C, or D) as your response:
{options}

Your selection (You can only output A, B, C or D, and no other characters.):
"""


PROMPT_GROUP_ZH = """
==角色描述==
{role_profiles}

==对话历史==
{conversations}

你要扮演{role_name}角色，你在聊天中要具备该角色对应的社交偏好。
请根据所给的{role_name}角色描述和对话历史，从下面四个选项（A. B. C.和D.）中选择最优的选项作为你的回复：
{options}

你的选择（你只能输出A，B，C或D，不要输出其他单词。）：
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the local SocialBench manifest.")
    parser.add_argument("--items", type=int, default=FULL_DATASET_SIZE, help="Optional prefix length.")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    temp_path.replace(path)


def _canonical_category(category: str) -> str:
    # The released data uses Individual-MEM-Short/Long while upstream dataset.py
    # keys the open-ended prompt/scoring path off the shared Individual-MEM base.
    if category.startswith("Individual-MEM"):
        return "Individual-MEM"
    return category


def _format_name(name: str) -> str:
    return name.replace(" ", "_").replace(".txt", "").replace(".json", "")


def _format_question(dialogue: list[dict[str, Any]], choices: dict[str, Any] | None) -> tuple[str, str]:
    conversations = ""
    for turn in dialogue:
        conversations += f"{turn['from']}: {turn['value']}\n"

    options = ""
    if choices is not None:
        for choice, text in choices.items():
            options += f"{choice}. {text}\n"
    return conversations, options


def _group_profiles(profiles: dict[str, Any], *, skip_role_name: str | None = None, shorten: bool = True, limit: int = 10) -> str:
    rendered: list[str] = []
    for role_name, role_profile in profiles.items():
        if skip_role_name is not None and _format_name(str(role_name)) == _format_name(skip_role_name):
            continue
        normalized = re.sub(r"\n+", "\n", str(role_profile))
        rendered.append("\n".join(normalized.split("\n")[:limit]) if shorten else normalized)
    return "\n\n\n".join(rendered)


def _choice_values(choices: dict[str, Any] | None) -> list[str]:
    if not isinstance(choices, dict):
        return []
    return [str(text).strip() for text in choices.values()]


def _format_prompt(row: dict[str, Any]) -> tuple[str, list[str]]:
    dialogue = list(row.get("dialogue") or [])
    meta = dict(row.get("meta") or {})
    lang = str(meta.get("lang") or "en").strip().lower()
    category = _canonical_category(str(meta.get("category") or "").strip())
    role_name = str(meta.get("name") or "").strip()
    profile = dict(meta.get("profile") or {})
    choices = row.get("choices") if isinstance(row.get("choices"), dict) else None
    conversations, options = _format_question(dialogue, choices)
    choice_values = _choice_values(choices)
    if category == "Individual-MEM":
        template = PROMPT_OPEN_EN if lang == "en" else PROMPT_OPEN_ZH
        return (
            template.format_map(
                {
                    "role_profile": str(profile.get(role_name) or ""),
                    "conversations": conversations,
                    "role_name": role_name,
                }
            ),
            choice_values,
        )
    if category == "Individual-EP-DialogueEmotionDetect":
        template = PROMPT_DIALOGUE_EMOTION_EN if lang == "en" else PROMPT_DIALOGUE_EMOTION_ZH
        utterance = str(dialogue[-1].get("value") or "").strip() if dialogue else ""
        return (
            template.format_map(
                {
                    "conversations": conversations,
                    "options": options,
                    "utterance": utterance,
                }
            ),
            choice_values,
        )
    if category in {"Individual-EP-HumorSarcasmDetect", "Individual-EP-SituationUnderstanding"}:
        return (f"{conversations}\n{options}", choice_values)
    if category in {"Group-SAP-Positive", "Group-SAP-Negative", "Group-SAP-Neutral"}:
        template = PROMPT_GROUP_EN if lang == "en" else PROMPT_GROUP_ZH
        return (
            template.format_map(
                {
                    "role_profiles": _group_profiles(profile),
                    "conversations": conversations,
                    "role_name": role_name,
                    "options": options,
                }
            ),
            choice_values,
        )
    if category in {"Individual-SA-RoleStyle", "Individual-SA-RoleKnowledge"}:
        template = PROMPT_EN if lang == "en" else PROMPT_ZH
        return (
            template.format_map(
                {
                    "role_profile": str(profile.get(role_name) or ""),
                    "conversations": conversations,
                    "role_name": role_name,
                    "options": options,
                }
            ),
            choice_values,
        )
    raise ValueError(category)


def main() -> None:
    args = _parse_args()
    requested_items = max(1, min(int(args.items or FULL_DATASET_SIZE), FULL_DATASET_SIZE))
    source_root = ensure_repo_checkout("socialbench")

    items: list[dict[str, Any]] = []
    for source_name in SOURCE_FILES:
        path = source_root / "data" / source_name
        if not path.exists():
            raise FileNotFoundError(f"Missing SocialBench source file: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row_index, row in enumerate(rows):
            if len(items) >= requested_items:
                break
            meta = dict(row.get("meta") or {})
            prompt, choices = _format_prompt(row)
            labels = [str(value).strip() for value in list(row.get("label") or []) if str(value).strip()]
            item_id = f"{path.stem}-{row_index}"
            items.append(
                {
                    "item_id": item_id,
                    "name": f"SocialBench {path.stem} #{row_index + 1}",
                    "prompt": prompt,
                    "choices": choices,
                    "expected_answer": labels,
                    "metadata": {
                        "dataset": "socialbench",
                        "source_file": path.stem,
                        "category": str(meta.get("category") or "").strip(),
                        "lang": str(meta.get("lang") or "").strip(),
                        "role_name": str(meta.get("name") or "").strip(),
                    },
                }
            )
        if len(items) >= requested_items:
            break

    manifest = {
        "dataset_id": "socialbench",
        "split": "official:all",
        "dataset_size": FULL_DATASET_SIZE,
        "prepared_count": len(items),
        "items": items,
    }
    _write_json(MANIFEST_PATH, manifest)
    print(f"Wrote {len(items)} SocialBench rows to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
