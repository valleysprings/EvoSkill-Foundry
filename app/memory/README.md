# Memory

This directory contains the strategy-memory layer used by the benchmark loop.

## Design

Memory is prompt-ready, not a generic event log.

Each experience stores fields such as:

- `failure_pattern`
- `strategy_hypothesis`
- `successful_strategy`
- `prompt_fragment`
- `tool_trace_summary`
- `process_failure_mode`
- `process_repair_hint`
- `process_trace_summary`
- `knowledge_scope`
- `distilled_skill`
- `applicability_notes`
- `source_dataset_ids`
- `delta_primary_score`
- `proposal_model`
- `candidate_summary`
- `experience_outcome`
- `verifier_status`

Suggested scopes:

- `knowledge_scope=episode_strategy`
  the current default: per-attempt strategy memory distilled from one candidate run
- `knowledge_scope=dataset_prior`
  reserved for future stronger-model distillation over a larger dataset slice or an entire benchmark

The dataset-prior shape is intentionally reserved now so an external distillation skill can write large reusable priors into the same retrieval layer without changing the main loop contract.

Important properties:

- memory persists across runs and is not reset to seeds each time
- retrieval prefers success experiences and caps failure fragments
- failure memory is reserved for informative verifier failures or execution errors
- duplicate memory fragments are suppressed before write-back
- markdown output is an auditable ledger, while the UI also surfaces run-local fragments directly
