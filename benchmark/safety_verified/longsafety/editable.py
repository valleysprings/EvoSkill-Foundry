from __future__ import annotations


def respond(question: dict, runtime) -> str:
    parts: list[str] = []
    context = str(question.get("context") or "").strip()
    prompt = str(question.get("prompt") or "").strip()
    if context:
        parts.append(f"Reference context:\n{context}")
    if prompt:
        parts.append(f"User request:\n{prompt}")
    response, _trace = runtime.chat(
        purpose="longsafety_candidate",
        messages=[{"role": "user", "content": "\n\n".join(parts)}],
    )
    return str(response.get("message") or "")
