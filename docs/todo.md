# TODO

This plan is ordered to keep the repo inference-first, auditable, and locally runnable before any trainer integration.

## Phase 1: Close The Discrete Demo Loop

- Finish the dedicated discrete entrypoint and wire it into the local server.
- Replace the old toy homepage with the two main demo tasks:
  - `euclidean-tsp`
  - `weighted-max-cut`
- Render generation-by-generation objective curves.
- Render memory retrieval and write-back events.
- Render final route / graph state directly in the UI.
- Add tests that assert both tasks improve over baseline.

## Phase 2: Auditability Layer

- Add a token calculator utility for future LLM-driven proposals.
- Define an `llm_trace.jsonl` schema for every proposal/eval step.
- Log prompt, model id, input tokens, output tokens, latency, and selected tool/operator.
- Add a markdown ledger view that mirrors the machine-readable memory store.
- Add a run manifest with:
  - git commit
  - task id
  - seed
  - objective curve
  - written memories
  - upstream session id when integrated with `../www-demo/autonomous-research-agent`

## Phase 3: Inference-First LLM Driver

- Add an LLM proposal interface for operator selection and mutation suggestion.
- Keep the verifier deterministic and outside the model.
- Make all proposal calls optional behind a local config switch.
- Track proposal quality against the hand-authored operator baseline.
- Align the trace fields with `../www-demo/autonomous-research-agent/logs/llm/*.jsonl`.
- Align the run/session structure with that project's process UI.

## Phase 4: Hybrid Training Tier

- Add a trainer adapter only after the discrete and inference layers are stable.
- Limit the first training tier to tiny, cheap outer-loop search problems.
- Search:
  - optimizer family
  - schedule
  - loss shaping
  - augmentation or curriculum policy
- Keep continuous weight updates inside the trainer adapter, not inside the core autoresearch abstraction.

## Integration Track

- Treat `../www-demo/autonomous-research-agent` as the future full-stack orchestrator.
- Keep this repo focused on the small optimization flywheel:
  - verifier
  - replay memory
  - operator search
  - auditable selection
- Define a clean handoff:
  - topic / task arrives from `www-demo`
  - local flywheel optimizes an auditable subproblem
  - results, markdown memory, and traces flow back to `www-demo`

## Bug-Fix Gate Before Expansion

Before any trainer work, the following must be true:

- the discrete demo runs end-to-end locally
- tests are green
- `scripts/` contains shell wrappers only
- entries live under `app/entries/`
- memory is dual-tracked as JSON plus markdown
- run artifacts are reproducible enough to diff and inspect
