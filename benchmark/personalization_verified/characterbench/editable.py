from __future__ import annotations


def solve(question: dict) -> str:
    raw_context = question.get("raw_context")
    if not isinstance(raw_context, dict):
        return ""
    user_message = str(raw_context.get("user_message") or "").strip()
    if not user_message:
        return ""
    return f"I will stay in character while answering: {user_message}"
