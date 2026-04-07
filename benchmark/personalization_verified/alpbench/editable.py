from __future__ import annotations


def solve(question: dict) -> str:
    choices = [str(choice).strip() for choice in list(question.get("choices") or []) if str(choice).strip()]
    if choices:
        return choices[0]
    return ""
