# CoSER

- Upstream paper/repo: [Neph0s/CoSER](https://github.com/Neph0s/CoSER)
- Local task id: `coser`
- Current fidelity: `reference_only`

## Official Protocol

1. Run Given-Circumstance Acting with character agents, an environment agent, and a next-speaker predictor.
2. Produce a multi-turn simulation transcript for each literary scenario.
3. Judge the simulation with LLM critiques plus BLEU and ROUGE-L over the released evaluation dimensions.

## Required Models / Assets

- `policy_model`
- `env_model`
- `nsp_model`
- `judge_model`
- released GCA evaluation prompts and data

## Known Deviations

- The benchmark is now in the protocol-audit lane; the GCA runtime and extra model roles are documented but not yet supported locally.
- The current hidden local wrapper is still a fan-out simplification and does not implement the benchmark-native GCA simulation runtime.
- The hidden proxy now supports optional eval-model judging for character fidelity, scene grounding, and dialogue quality so the fan-out slice is less brittle than pure exact-match continuation checks.
- Not official; hidden from runnable picker until rewritten.
