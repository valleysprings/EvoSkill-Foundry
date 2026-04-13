# AlpsBench Task 4

- Upstream paper: [arXiv: 2603.26680](https://arxiv.org/abs/2603.26680)
- Local task id: `alpsbench-utilization`
- Current fidelity: `proxy_local`

## Official Protocol

1. Use the relevant memory plus current query to generate a personalized response.
2. Return both the response and the memory fact used.
3. Benchmark-side evaluation uses ability-specific structured judges, scored independently per ability.

## Local Proxy

1. Uses the released public validation splits for `ability1..5`.
2. Preserves the public JSON prediction contract for `task4`.
3. Scores with the official public local grounding proxy from the AlpsBench release.

## Required Models / Assets

- `policy_model`
- released AlpsBench validation data

## Known Deviations

- The full benchmark-side Task 4 judge prompts and exact decision rules are intentionally not released.
- This local task therefore implements the public grounding proxy, not the hidden ability-specific judges.
