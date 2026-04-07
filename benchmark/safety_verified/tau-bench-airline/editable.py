from __future__ import annotations


def init_episode(episode: dict) -> dict:
    return {
        "episode_id": str(episode.get("episode_id") or ""),
        "instruction": str(episode.get("instruction") or ""),
    }


def step(turn: dict, runtime) -> dict:
    messages: list[dict[str, str]] = []
    instruction = str((turn.get("episode") or {}).get("instruction") or "").strip()
    domain_policy = str(((turn.get("episode") or {}).get("policy") or {}).get("domain_policy") or "").strip()
    if instruction or domain_policy:
        system_parts = [part for part in [instruction, domain_policy] if part]
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})
    for message in list(turn.get("history") or []):
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role in {"user", "assistant", "system"} and content:
            messages.append({"role": role, "content": content})
    response, _trace = runtime.chat(
        purpose="tau_bench_airline_step",
        messages=messages,
    )
    return {
        "message": str(response.get("message") or ""),
        "tool_calls": [],
        "done": False,
        "state": dict(turn.get("state") or {}),
        "annotations": {},
    }
