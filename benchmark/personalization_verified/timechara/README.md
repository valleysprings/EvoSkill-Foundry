# TimeChara

- Upstream paper/repo: [ahnjaewoo/timechara](https://github.com/ahnjaewoo/timechara)
- Local task id: `timechara`
- Current fidelity: `official`

## Official Protocol

1. Evaluate role-play outputs against a character's temporal state and plot stage.
2. Measure whether the reply is aligned with the correct timeline point rather than a generic character summary.
3. Aggregate timeline-consistency results over the released benchmark split.

## Required Models / Assets

- `policy_model`
- `judge_model`
- gated Hugging Face dataset access for `ahnpersie/timechara`
- released TimeChara prompts / timeline annotations

## Known Deviations

- The local runnable path now uses the released test split together with the published spatiotemporal evaluation prompt and item-level score extraction.
- The shared `eval_model` slot stands in for the benchmark's evaluator model while preserving the official future / past-absence / past-presence / past-only scoring logic.
