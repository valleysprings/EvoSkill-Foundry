from __future__ import annotations


def solve(question: dict) -> dict[str, str]:
    raw_context = question.get("raw_context")
    if not isinstance(raw_context, dict):
        return {}
    answers: dict[str, str] = {}
    for item in list(raw_context.get("questions") or []):
        if not isinstance(item, dict):
            continue
        question_id = str(item.get("id") or "").strip()
        prompt = str(item.get("question") or item.get("statement") or "").strip()
        if question_id and prompt:
            answers[question_id] = f"I would answer this carefully in character: {prompt}"
    return answers
