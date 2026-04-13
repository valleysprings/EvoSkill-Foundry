# AlpsBench Task 2

- Upstream paper: [arXiv: 2603.26680](https://arxiv.org/abs/2603.26680)
- Local task id: `alpsbench-update`
- Current fidelity: `proxy_local`

## Official Protocol

1. Read existing memories plus old and new dialogue.
2. Update the user's long-term memory state.
3. Score with memory matching (`precision` / `recall` / `f1`) plus an optional structured judge.

## Local Proxy

1. Uses the released public validation split.
2. Preserves the public JSON prediction contract for `task2`.
3. Scores with the official public local scorer from the AlpsBench release.

## Required Models / Assets

- `policy_model`
- released AlpsBench validation data

## Known Deviations

- The benchmark-side structured judge and blended scoring details are described but not fully released.
- This local task therefore follows the released public scorer only.
