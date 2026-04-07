def init_episode(episode: dict) -> dict:
    return {
        "episode_id": str(episode.get("episode_id") or ""),
        "instruction": str(episode.get("instruction") or ""),
        "steps": 0,
    }


def step(turn: dict, runtime) -> dict:
    del runtime
    state = dict(turn.get("state") or {})
    state["steps"] = int(state.get("steps") or 0) + 1
    return {
        "message": "Need more observation before acting.",
        "tool_calls": [],
        "done": False,
        "state": state,
        "annotations": {},
    }
