# AlpsBench Task 3

- Upstream paper: [arXiv: 2603.26680](https://arxiv.org/abs/2603.26680)
- Local task id: `alpsbench-retrieval`
- Current fidelity: `proxy_local`

## Official Protocol

1. Read the query and candidate memory pool.
2. Return a response, a rationale, and the selected supporting memory id.
3. Benchmark-side evaluation checks memory usage; public local scoring uses selected-memory matching.

## Local Proxy

1. Uses the released public validation splits for `d100`, `d300`, `d500`, `d700`, and `d1000`.
2. Preserves the public JSON prediction contract for `task3`.
3. Scores with the official public local scorer from the AlpsBench release.

## Required Models / Assets

- `policy_model`
- released AlpsBench validation data

## Known Deviations

- The benchmark-side memory-usage judge is documented but not fully released.
- This local task uses the released selected-memory public proxy.
