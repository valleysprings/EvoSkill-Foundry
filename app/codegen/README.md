# Codegen Engine

This directory contains the active benchmark-driven codegen loop.

## Design

### Proposal runtime

- loads strict runtime config from repo-root `.env` or shell env
- sends one-model OpenAI-compatible chat requests
- expects strict JSON candidates with `file_body`

### Trainer

- retrieves prompt-ready memory fragments
- asks the model for candidate file rewrites
- materializes and verifies each candidate
- writes back reusable success or failure experiences

### Verifier

- materializes a single editable file from `file_body`
- imports the declared entry symbol
- runs deterministic correctness checks first
- benchmarks only verified candidates
- computes task-facing `objective` plus internal selection score `J`

### Reporting and handoff

- emits payload JSON, traces, `llm_trace.jsonl`, markdown memory, and report artifacts
- exposes those artifacts to the UI and downstream handoff consumers

## Objective and Selection

The runner uses two related scores:

- `objective`
  the task-facing metric declared by the benchmark verifier
- `J`
  the always-max internal selection score used to rank verified candidates across tasks

Current `J` formula:

`J = 1.20 * correctness + 0.95 * objective_signal + 0.20 * memory_bonus + 0.15 * stability - 0.18 * complexity - 0.05 * (line_count / 10)`

Selection rules:

- correctness is gated first
- failing or erroring candidates do not enter the winner lane
- generations mutate a selected frontier parent, not only the global incumbent
- a generation is accepted when it beats its selected parent by `epsilon`
- the global best updates only when a frontier winner also beats the current best by `epsilon`
- passing but stagnant candidates do not get written back as failure memory
