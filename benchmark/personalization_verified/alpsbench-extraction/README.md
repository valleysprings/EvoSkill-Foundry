# AlpsBench Task 1

- Upstream paper: [arXiv: 2603.26680](https://arxiv.org/abs/2603.26680)
- Local task id: `alpsbench-extraction`
- Current fidelity: `proxy_local`

## Official Protocol

1. Read the released dialogue-only input.
2. Extract stable long-term user memories as structured `memory_items`.
3. Score with memory matching (`precision` / `recall` / `f1`) plus an optional structured judge.

## Local Proxy

1. Uses the released public validation split.
2. Preserves the public JSON prediction contract for `task1`.
3. Scores with the official public local scorer from the AlpsBench release.

## Required Models / Assets

- `policy_model`
- released AlpsBench validation data

## Known Deviations

- The benchmark-side structured `LLM-as-Judge` schema is described publicly but not fully released.
- This local task therefore follows the released public scorer, not the hidden benchmark-side judge stack.
