from __future__ import annotations


def solve(question: dict) -> str:
    raw_context = question.get("raw_context")
    if not isinstance(raw_context, dict):
        return ""
    character = str(raw_context.get("character") or "").strip()
    question_text = str(raw_context.get("question") or "").strip()
    if not character or not question_text:
        return ""
    return f"As {character}, I would answer this timeline-specific question carefully: {question_text}"
