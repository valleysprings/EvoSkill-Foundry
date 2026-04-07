# PersonaMem 32k

- Upstream source: [bowen-upenn/PersonaMem](https://huggingface.co/datasets/bowen-upenn/PersonaMem)
- Local task id: `personamem-32k`
- Current fidelity: `official`

## Official Protocol

1. Read the long user-history context and the candidate response set.
2. Select the response that best matches the stored persona and preference evidence.
3. Compute exact-match accuracy on the released 32k benchmark split.

## Required Models / Assets

- `policy_model`
- released PersonaMem 32k data

## Known Deviations

- No concrete protocol mismatch has been identified during this cleanup.
- The task remains in the runnable lane.
