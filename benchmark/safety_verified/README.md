# Safety Benchmarks

This track groups runnable safety tasks by safety taxonomy instead of by the coarse research track alone.

Safety metadata exposed through `task.json`, catalog summaries, `/api/tasks`, and the UI:

- `interaction_mode`
  continues to tell the UI whether the task is single-turn or multi-turn
- `safety_category`
  captures the safety sub-direction: `jailbreak_attack`, `over_refusal`, `factuality_hallucination`, `policy_drift`, or `benign_utility`
- `safety_focus`
  optional finer tag; for example `or-bench-toxic` is grouped under `jailbreak_attack` but keeps `should_refuse` as its focus

Current runnable local tasks:

- `xstest-refusal-calibration`
  `over_refusal`
- `harmbench-text-harmful`
  `jailbreak_attack`
- `jailbreakbench-harmful`
  `jailbreak_attack`
- `or-bench-hard-1k`
  `over_refusal`
- `or-bench-toxic`
  `jailbreak_attack` with focus `should_refuse`
- `hallulens-precisewikiqa`
  `factuality_hallucination`
- `hallulens-mixedentities`
  `factuality_hallucination`
- `hallulens-longwiki`
  `factuality_hallucination`
- `longsafety`
  `jailbreak_attack` with focus `safety_degradation`

This list is the registered task set currently stored under `safety_verified/` and used by `benchmark/registry.json` and `benchmark/prepare_datasets.py`.

Directories such as `bloom-self-preferential-bias/` and `bloom-trait-examples/` are auxiliary local data folders only; they are intentionally not exposed as standalone benchmark tasks.

Notes on runnable fidelity:

- public downloads happen automatically during `prepare.py`
- HalluLens tasks intentionally use deterministic local runnable slices built from public GoodWiki and ITIS-style sources rather than the full upstream dynamic generation stack
- `hallulens-mixedentities` deliberately avoids extra search/API dependencies by pairing public taxonomy data with a bundled medicine seed list
