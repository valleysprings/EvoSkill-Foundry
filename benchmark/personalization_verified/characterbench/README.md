# CharacterBench

- Upstream paper/repo: [thu-coai/CharacterBench](https://github.com/thu-coai/CharacterBench)
- Local task id: `characterbench`
- Current fidelity: `official`

## Official Protocol

1. Generate character responses on the released evaluation set.
2. Convert generated outputs into CharacterJudge prompt inputs.
3. Run CharacterJudge and aggregate the released evaluation dimensions.

## Required Models / Assets

- `policy_model`
- `evaluator_model` (`CharacterJudge`)
- released raw evaluation data and prompt-construction scripts

## Known Deviations

- The local verifier now uses the released English prompt-construction templates from `construct_prompts`, then normalizes the CharacterJudge score onto `[0, 1]` for the shared comparison UI.
- The underlying prompt flow and per-subset score normalization remain aligned with the released CharacterBench evaluation path.
