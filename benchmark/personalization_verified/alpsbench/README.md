# AlpsBench

- Upstream paper: [arXiv: 2603.26680](https://arxiv.org/abs/2603.26680)
- Local task id: `alpsbench`
- Current fidelity: `official`

## Official Protocol

1. Read the released long-horizon user history and the current benchmark question.
2. Select the benchmark-approved personalized answer.
3. Compute exact-match accuracy on the released validation split.

## Required Models / Assets

- `policy_model`
- released AlpsBench validation data

## Known Deviations

- No concrete protocol mismatch has been identified during this cleanup.
- The task remains runnable while the rest of the track is reset.
