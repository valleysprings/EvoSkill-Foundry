# SocialBench

- Upstream paper/repo: [X-PLUG/SocialBench](https://github.com/X-PLUG/SocialBench)
- Local task id: `socialbench`
- Current fidelity: `official`

## Official Protocol

1. Format prompts using the released category-specific SocialBench templates.
2. Run the policy model on individual and group-level sociality items.
3. Score outputs with the official deterministic parsing rules across SA, EP, CM, and SP dimensions.

## Required Models / Assets

- `policy_model`
- released SocialBench prompt and scoring code

## Known Deviations

- The current local task uses the released SocialBench prompt templates and deterministic scoring path.
- No protocol deviation is currently tracked for the runnable lane.
