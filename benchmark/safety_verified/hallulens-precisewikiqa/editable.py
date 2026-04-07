from __future__ import annotations


def respond(question: dict, runtime) -> str:
    context = str(question.get("context") or "").strip()
    prompt = str(question.get("prompt") or "").strip()
    user_message = "\n\n".join(
        part
        for part in [
            "Use only the provided reference to answer the question. If the reference is insufficient, say you cannot verify the answer.",
            f"Reference:\n{context}" if context else "",
            f"Question:\n{prompt}" if prompt else "",
        ]
        if part
    )
    response, _trace = runtime.chat(
        purpose="hallulens_precisewikiqa_candidate",
        messages=[{"role": "user", "content": user_message}],
    )
    return str(response.get("message") or "")
