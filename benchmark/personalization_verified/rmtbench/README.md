# RMTBench

- Upstream paper: [arXiv: 2507.20352](https://arxiv.org/abs/2507.20352)
- Local task id: `rmtbench`
- Current fidelity: `reference_only`

## Official Protocol

1. Run the model in the benchmark's original user-oriented multi-turn role-play setting using the released user-side dialogue histories.
2. Generate the assistant side of the full conversation rather than a single deterministic next turn.
3. Score the completed dialogue with the benchmark's released multi-turn judge dimensions and aggregate benchmark-level results.

## Required Models / Assets

- `policy_model`
- `judge_model`
- gated official dataset access
- final paper-aligned evaluation assets

## Known Deviations

- The current hidden local wrapper is still only a tiny exploratory proxy and does not yet implement the full-dialogue release contract.
- The local proxy now uses judged next-turn scoring over the placeholder slice instead of pure exact-match continuation checks, but it remains far from the official benchmark.
- Runnable recovery is blocked on gated dataset inspection plus final paper-level scoring alignment.
- Not official; hidden from runnable picker until rewritten.
