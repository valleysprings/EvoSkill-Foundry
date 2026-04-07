# CharacterEval

- Upstream paper/repo: [morecry/CharacterEval](https://github.com/morecry/CharacterEval)
- Local task id: `charactereval`
- Current fidelity: `reference_only`

## Official Protocol

1. Generate multi-turn role-play responses on the released test data.
2. Transform outputs into the sparse metric format used by the benchmark.
3. Score each case with `BaichuanCharRM` and average the benchmark's published dimensions.

## Required Models / Assets

- `policy_model`
- `reward_model` (`BaichuanCharRM`)
- released character profiles and metric mapping files

## Known Deviations

- This benchmark is now in the protocol-audit lane, with the CharacterRM reward-model path documented but not yet executable in the shared runtime.
- The local verifier now mirrors the official preprocessing more closely by judging only the first generated utterance, but it still replaces CharacterRM with a generic eval-model judge.
- It stays tracked but non-runnable until CharacterRM-backed evaluation is implemented end to end.
- Not official; hidden from runnable picker until rewritten.
