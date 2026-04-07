# InCharacter

- Upstream paper/repo: [Neph0s/InCharacter](https://github.com/Neph0s/InCharacter-Role-Playing-Agents-Effectively-Capture-Characters-Personalities-Traits)
- Local task id: `incharacter`
- Current fidelity: `official`

## Official Protocol

1. Run the role-playing model on a full released questionnaire bundle for a character.
2. Use an evaluator LLM to convert questionnaire answers back into the benchmark's published choice space.
3. Compute per-dimension labels, then report single-dimension accuracy and full-profile accuracy against the released labels.

## Required Models / Assets

- `policy_model`
- `judge_model` / evaluator LLM
- released questionnaire files and character labels

## Known Deviations

- The local task now evaluates full questionnaire bundles rather than a dimension slice, so the verifier can report both per-dimension accuracy and full-profile correctness.
- The local objective is the official single-dimension accuracy normalized to `[0, 1]`; full-profile accuracy is surfaced in the verifier trace for each item.
