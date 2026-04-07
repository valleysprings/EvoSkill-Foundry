from __future__ import annotations


def solve(question: dict) -> str:
    context = str(question.get("context") or "").strip()
    if not context:
        return ""
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    return lines[-1] if lines else ""
