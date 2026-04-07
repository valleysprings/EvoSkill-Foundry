from __future__ import annotations


def respond(question: dict, runtime) -> str:
    parts: list[str] = []
    context = str(question.get("context") or "").strip()
    prompt = str(question.get("prompt") or "").strip()
    if context:
        parts.append(context)
    if prompt:
        parts.append(prompt)
    response, _trace = runtime.chat(
        purpose="or_bench_toxic_candidate",
        messages=[{"role": "user", "content": "\n\n".join(parts)}],
    )
    return str(response.get("message") or "")
