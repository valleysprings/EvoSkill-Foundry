from __future__ import annotations


def solve(question: dict) -> dict:
    context = dict(question.get("raw_context") or question.get("context") or {})
    selected_memory = dict(context.get("selected_memory") or {})
    return {
        "answer": "Stub personalized response.",
        "used_memory_fact": str(selected_memory.get("value") or ""),
    }
