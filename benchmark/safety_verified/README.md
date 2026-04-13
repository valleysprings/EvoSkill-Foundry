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

Disabled pending official recovery:

- `harmbench-text-harmful`
- `jailbreakbench-harmful`
- `or-bench-hard-1k`
- `or-bench-toxic`
- `longsafety`

Disabled under the stricter official-only policy because they are local slices or substitutes:

- `hallulens-precisewikiqa`
- `hallulens-mixedentities`
- `hallulens-longwiki`

This track still stores both enabled and disabled wrappers under `safety_verified/`, but only the enabled set from `benchmark/registry.json` is used by `benchmark/prepare_datasets.py`.

Directories such as `bloom-self-preferential-bias/` and `bloom-trait-examples/` are auxiliary local data folders only; they are intentionally not exposed as standalone benchmark tasks.

Notes on runnable fidelity:

- public downloads happen automatically during `prepare.py`
- shared generic safety judges are no longer treated as official benchmark implementations by default
