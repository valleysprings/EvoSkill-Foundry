# autoresearch-with-experience-replay

A local-first autoresearch flywheel that borrows the fixed-budget experiment loop from `karpathy/autoresearch`, the Apple Silicon execution posture from `autoresearch-macos`, and the candidate-selection pressure of `openevolve`.

The repo is now split into two layers:

- a legacy small-task runner for simple code mutations
- a new discrete-optimization runner for auditable, visual NP-hard demos

The current main direction is the second one:

`baseline heuristic -> operator proposals -> verifier/objective -> generation selection -> selective memory write-back`

That keeps the project aligned with outer-loop autoresearch instead of raw model training.

## What this repo shows

- A macOS-first local runner that executes real Python optimization tasks.
- Experience replay as the central organizing object.
- Proposal competition across multiple agent lanes.
- Deterministic test-first evaluation before benchmark-based selection.
- A frontend that can trigger tasks and inspect winners on localhost.

## Run the task runner

List available tasks:

```bash
python3 -m app.legacy.demo_run --list-tasks
```

Run `task1` directly:

```bash
python3 -m app.legacy.demo_run --task contains-duplicates
```

Run the full sequence:

```bash
python3 -m app.legacy.demo_run
```

## Run the discrete demo

List the main discrete tasks:

```bash
python3 -m app.entries.discrete_demo --list-tasks
```

Run the Euclidean TSP demo:

```bash
python3 -m app.entries.discrete_demo --task euclidean-tsp
```

Run the Weighted Max-Cut demo:

```bash
python3 -m app.entries.discrete_demo --task weighted-max-cut
```

Artifacts are written under `runs/`:

- `runs/discrete-euclidean-tsp.json`
- `runs/discrete-weighted-max-cut.json`
- `runs/discrete_working_memory.json`
- `runs/discrete_working_memory.md`

Start the local server:

```bash
python3 -m app.entries.server
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Direct API calls:

```bash
curl http://127.0.0.1:8000/api/tasks
curl http://127.0.0.1:8000/api/latest-run?task_id=contains-duplicates
curl -X POST http://127.0.0.1:8000/api/run-task?task_id=contains-duplicates
curl -X POST http://127.0.0.1:8000/api/run-sequence
```

## Fetch prompt benchmarks

The repo now includes small local benchmark slices for prompt optimization under
`benchmarks/prompt/`.

Fetch all benchmark slices:

```bash
bash scripts/fetch_prompt_benchmarks.sh
```

Fetch one benchmark only:

```bash
bash scripts/fetch_prompt_benchmarks.sh --benchmark boolq_small
```

Current benchmark set:

- `banking77_small`: intent routing, 80 train + 60 eval
- `boolq_small`: yes/no QA, 80 train + 60 eval
- `ag_news_small`: topic classification, 80 train + 60 eval

Fetched data lands in `benchmarks/prompt/data/<benchmark_id>/` with:

- `train.jsonl`
- `eval.jsonl`
- `manifest.json`

These are designed to be the next input layer for a local prompt-evolution runner.

## Repo map

- `app/engine.py`: runner orchestration, candidate selection, and write-back policy.
- `app/evaluator.py`: actual code execution, correctness tests, and benchmark scoring.
- `app/memory_store.py`: file-backed retrieval and memory append logic.
- `app/demo_run.py`: end-to-end demo artifact generation.
- `app/entries/server.py`: tiny local backend that serves the UI and latest run JSON.
- `app/legacy/`: legacy small-task runner kept for comparison.
- `app/discrete/`: discrete optimization catalog, operators, evals, and trainer.
- `app/entries/discrete_demo.py`: dedicated entrypoint for the discrete demo.
- `app/memory/`: JSON + markdown memory store.
- `data/tasks.json`: the local task catalog.
- `data/experiences.json`: seed experience memory.
- `data/discrete_experiences.json`: seed memory for the discrete optimization tier.
- `examples/evolve/*/initial_program.py`: baseline functions for real tasks.
- `docs/plan.md`: implementation plan and scope lock.
- `docs/framework.md`: core mechanism and system view.
- `docs/demo.md`: narration for the frontend demo.
- `docs/scope.md`: in-scope vs out-of-scope definition.
- `docs/benchmarks.md`: benchmark tiers and metrics.
- `docs/todo.md`: inference-first TODO closure before trainer work.
- `docs/post-grpo-notes.md`: 2025-2026 reference notes.
- `paper/outline.md`: short paper structure for the concept.
- `benchmarks/prompt/configs/*.json`: prompt benchmark definitions.
- `benchmarks/prompt/data/*`: local prompt benchmark slices and manifests.
- `scripts/*.sh`: shell wrappers only; Python entries live under `app/`.

## Design choices

- **Validated experience over chat transcripts.** The reusable unit is a scored experience, not a conversation.
- **Deterministic evaluation first.** Keep/discard decisions stay stable and inspectable.
- **Local-first constraints.** The first loop must make sense on a Mac without GPU assumptions.
- **Inference first, training later.** Continuous weight training is a hybrid outer-loop tier, not the first-class story.
- **Dual memory.** JSON remains machine-readable, markdown remains human-auditable and LLM-readable.

## Reference repos cloned locally

These were cloned into `external/` for planning and comparison and are intentionally ignored by git:

- `external/autoresearch`
- `external/autoresearch-macos`
- `external/openevolve`
