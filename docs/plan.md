# Implementation plan

## Success criteria

Build a local-first autoresearch demo that feels structurally close to `karpathy/autoresearch`, runs on macOS without GPU requirements, exposes a deterministic selection loop, and can later plug into the stronger `../www-demo/autonomous-research-agent` workflow as a small but auditable flywheel core.

## Reference mapping

- `karpathy/autoresearch`: fixed-budget experiment loop, program-centric mutation, results logging, and explicit keep/discard decisions.
- `miolini/autoresearch-macos`: MPS-safe execution path, smaller local budgets, and Apple Silicon compatibility.
- `openevolve`: diversity across candidate lanes, mutation pressure, and evaluator-driven selection.
- `../www-demo/autonomous-research-agent`: literature-aware multi-agent front-end, process UI, LLM logs, and the longer-term "complete form" of the research agent.

## Chosen mechanism

The prototype organizes **validated experience units**, not free-form chat. Each task:

1. Retrieves prior experience by task signature and target device.
2. Generates competing research proposals from specialist lanes.
3. Scores proposals with a deterministic evaluator.
4. Selects the highest-scoring candidate against a baseline.
5. Writes back a new experience only if `delta_J > epsilon`.

## Scope lock

The first implementation deliberately avoids full training, cluster orchestration, and heavy live LLM coupling. Instead it builds a small, deterministic flywheel over auditable tasks so the core loop is visible and stable on any Mac first.

This repo is therefore the `small flywheel` layer:

- verifier
- operator search
- memory replay
- write-back policy
- auditable run artifacts

The `../www-demo/autonomous-research-agent` project is treated as the future `complete form`:

- literature retrieval
- multi-agent ideation
- richer UI and process visualization
- broader LLM traces and human-facing orchestration

## Phase plan

### Phase 1

Implement the local macOS loop:

- file-backed memory retrieval
- deterministic evaluator
- operator competition across auditable task families
- markdown memory ledger plus JSON replay store

### Phase 2

Show experience replay:

- first discrete task creates a reusable memory rule
- second discrete task or later generation retrieves it
- replay-guided operator wins with measured objective gains

### Phase 3

Add the integration bridge:

- package local flywheel outputs for `../www-demo/autonomous-research-agent`
- reuse that project's process UI, session logs, and LLM trace conventions
- keep evaluator and memory schemas stable enough to later hand off to an H200 training lane

## Integration note

The intended merge direction is:

1. this repo supplies the small, local, verifier-heavy optimization flywheel
2. `../www-demo/autonomous-research-agent` supplies the broader autonomous research workflow
3. the combined system becomes:
   - literature- and idea-aware at the front
   - memory- and evaluator-heavy at the optimization core
   - trainer-capable only after the inference-first loop is stable

## Demo acceptance

- `python3 -m app.legacy.demo_run` produces `runs/latest_run.json`
- `python3 -m app.entries.discrete_demo --task euclidean-tsp` produces a discrete optimization artifact
- `python3 -m app.entries.discrete_demo --task weighted-max-cut` produces a second auditable artifact
- `python3 -m app.entries.server` serves the dashboard locally
- the UI shows memory retrieval, candidate scores, winner selection, and write-back
- the roadmap clearly separates local Mac execution, the `www-demo` integration layer, and later H200 expansion
