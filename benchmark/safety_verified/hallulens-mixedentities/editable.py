from __future__ import annotations


def respond(question: dict, runtime) -> str:
    prompt = str(question.get("prompt") or "").strip()
    response, _trace = runtime.chat(
        purpose="hallulens_mixedentities_candidate",
        messages=[{"role": "user", "content": prompt}],
    )
    return str(response.get("message") or "")
