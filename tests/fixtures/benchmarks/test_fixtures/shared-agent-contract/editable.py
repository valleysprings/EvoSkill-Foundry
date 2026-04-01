from __future__ import annotations


def init_episode(episode: dict) -> dict:
    return {"episode_id": episode["episode_id"]}


def step(turn: dict, runtime) -> dict:
    del runtime
    return {
        "message": "still investigating",
        "tool_calls": [],
        "done": False,
        "state": dict(turn.get("state") or {}),
        "annotations": {},
    }
