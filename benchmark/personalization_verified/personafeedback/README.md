# PersonaFeedback

- Upstream paper: [arXiv: 2506.12915](https://arxiv.org/abs/2506.12915)
- Local task id: `personafeedback`
- Current fidelity: `official`

## Official Protocol

1. Read the released explicit persona profile and the candidate responses for the item.
2. Select the response that best matches the benchmark's persona-conditioned target.
3. Compute exact-match accuracy over the released benchmark split.

## Required Models / Assets

- `policy_model`
- released PersonaFeedback benchmark split

## Known Deviations

- No concrete protocol mismatch has been identified during this cleanup.
- This task remains runnable while broader personalization audits continue.
