from __future__ import annotations


def solve(question: dict) -> dict:
    context = dict(question.get("raw_context") or question.get("context") or {})
    candidate_memories = [entry for entry in list(context.get("candidate_memories") or []) if isinstance(entry, dict)]
    first = candidate_memories[0] if candidate_memories else {}
    return {
        "answer": str(first.get("value") or ""),
        "reason": "Selected the first candidate memory.",
        "selected_memory_id": str(first.get("memory_id") or ""),
    }
