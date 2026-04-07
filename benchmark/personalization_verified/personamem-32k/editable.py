from __future__ import annotations


def solve(question: dict) -> str:
    choices = list(question.get("choices") or [])
    return str(choices[0]) if choices else ""

