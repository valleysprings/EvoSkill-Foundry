# ALPBench

- Upstream paper: [arXiv: 2602.03056](https://arxiv.org/abs/2602.03056)
- Local task id: `alpbench`
- Current fidelity: `official`

## Official Protocol

1. Read the released behavioral history for the user.
2. Predict the target latent attribute label defined by the benchmark.
3. Compute label-match accuracy over the released benchmark split.

## Required Models / Assets

- `policy_model`
- released ALPBench split and label schema

## Known Deviations

- No concrete protocol mismatch has been identified during this cleanup.
- The task remains runnable in the main personalization lane.
