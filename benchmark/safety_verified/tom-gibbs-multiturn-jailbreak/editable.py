from __future__ import annotations


def init_episode(episode: dict) -> dict:
    del episode
    return {}


def step(turn: dict, runtime) -> dict:
    messages: list[dict[str, str]] = []
    instruction = str((turn.get("episode") or {}).get("instruction") or "").strip()
    if instruction:
        messages.append({"role": "system", "content": instruction})
    for message in list(turn.get("history") or []):
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role in {"user", "assistant", "system"} and content:
            messages.append({"role": role, "content": content})
    response, _trace = runtime.chat(
        purpose="tom_gibbs_multiturn_jailbreak_step",
        messages=messages,
    )
    return {
        "message": str(response.get("message") or ""),
        "tool_calls": [],
        "done": False,
        "state": dict(turn.get("state") or {}),
        "annotations": {},
    }
