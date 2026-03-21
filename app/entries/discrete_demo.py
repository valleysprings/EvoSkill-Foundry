from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from app.discrete.catalog import list_discrete_task_summaries, load_discrete_tasks
from app.discrete.trainer import run_discrete_task
from app.memory.store import MemoryStore

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
WORKING_MEMORY = RUNS / "discrete_working_memory.json"
WORKING_MEMORY_MD = RUNS / "discrete_working_memory.md"
ProgressCallback = Callable[[dict[str, Any]], None]

J_FORMULA = (
    "J = 1.25 * feasible + 1.20 * quality_score + 0.25 * memory_bonus "
    "+ 0.15 * stability - 0.12 * latency_norm - 0.10 * complexity"
)
QUALITY_FORMULA = "quality_score = min(relative_gain / 0.35, 1.0)"
DELTA_FORMULA = "delta_J = J(best_after_run) - J(baseline)"


def generate_discrete_payload(
    task_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
    pace_ms: int = 0,
) -> dict[str, Any]:
    tasks = load_discrete_tasks()
    if task_id is not None:
        tasks = [task for task in tasks if task["id"] == task_id]
        if not tasks:
            raise ValueError(f"Unknown task id: {task_id}")

    RUNS.mkdir(exist_ok=True)
    store = MemoryStore(
        WORKING_MEMORY,
        markdown_path=WORKING_MEMORY_MD,
        title="Discrete Autoresearch Memory",
    )
    seed_memories = store.seed_from(DATA / "discrete_experiences.json")

    runs = []
    write_backs = 0
    total_generations = 0
    for task in tasks:
        before_count = store.count()
        result = run_discrete_task(
            task,
            store,
            progress_callback=progress_callback,
            pace_ms=pace_ms,
        )
        after_count = store.count()
        write_backs += after_count - before_count
        total_generations += len(result["generations"])
        result["memory_before_count"] = before_count
        result["memory_after_count"] = after_count
        result["memory_markdown"] = store.load_markdown()
        runs.append(result)

    winners = Counter(run["winner"]["agent"] for run in runs)
    return {
        "summary": {
            "project": "autoresearch-with-experience-replay",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "num_tasks": len(runs),
            "total_generations": total_generations,
            "initial_memory_count": len(seed_memories),
            "memory_size_after_run": store.count(),
            "write_backs": write_backs,
            "winner_agents": dict(winners),
            "scope": {
                "in_scope": [
                    "outer-loop search over discrete artifacts",
                    "operator, code, prompt, recipe, and schedule selection",
                    "memory write-back only after measured gains",
                ],
                "hybrid": [
                    "optimizer or schedule search around a training loop",
                    "architecture or loss recipe search under fixed budgets",
                ],
                "out_of_scope": [
                    "raw weight training by itself",
                    "open-ended generation without a verifier",
                ],
            },
            "karpathy_alignment": {
                "matches": [
                    "artifact-first outer loop instead of chat-first orchestration",
                    "deterministic verifier and explicit keep/discard policy",
                    "markdown memory ledger plus machine-readable replay store",
                    "visible improvement trajectory across generations",
                ],
                "gaps": [
                    "still synthetic local instances, not full research code or cluster jobs",
                    "proposal policy is hand-authored operators, not yet an LLM mutator",
                    "continuous inner-loop training is only defined as a future hybrid tier",
                ],
            },
        },
        "formulas": {
            "J": J_FORMULA,
            "quality_score": QUALITY_FORMULA,
            "delta_J": DELTA_FORMULA,
        },
        "task_catalog": list_discrete_task_summaries(),
        "memory_markdown": store.load_markdown(),
        "runs": runs,
    }


def write_discrete_artifacts(
    task_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
    pace_ms: int = 0,
) -> Path:
    payload = generate_discrete_payload(task_id=task_id, progress_callback=progress_callback, pace_ms=pace_ms)
    out_name = f"discrete-{task_id}.json" if task_id else "discrete-latest.json"
    out = RUNS / out_name
    out.write_text(json.dumps(payload, indent=2))
    (RUNS / "discrete-latest.json").write_text(json.dumps(payload, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the discrete autoresearch demo.")
    parser.add_argument("--task", help="Run only one task id from the discrete catalog.")
    parser.add_argument("--list-tasks", action="store_true", help="List available discrete task ids.")
    args = parser.parse_args()

    if args.list_tasks:
        for task in list_discrete_task_summaries():
            print(f"{task['id']}: {task['title']}")
        return

    out = write_discrete_artifacts(task_id=args.task)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
