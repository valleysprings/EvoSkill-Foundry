# planbench

Comparable reasoning benchmark backed by the public `tasksource/planbench` dataset.

This local task tracks the official direct `task_1_plan_generation` setting as closely as practical while keeping benchmark-side logic self-contained under the task directory:

- input: the original natural-language planning prompt
- output: a directly generated plan in a format accepted by upstream `text_to_plan`
- scoring: upstream-compatible task1 parsing plus semantic validation with `VAL`
- unified repo contract: `task_mode=answer`, `interaction_mode=single_turn`
- verifier style: adapter/semantic validation, not exact string match or permissive local fallback parsing

Current local normalization target:

- config: `task_1_plan_generation`
- split: `train`

Prepare locally with:

```bash
python3 prepare.py
```

Task-local official evaluation support lives under:

- `benchmark/reasoning_verified/planbench/official/official_adapter.py`
- `benchmark/reasoning_verified/planbench/official/plan-bench`
- `benchmark/reasoning_verified/planbench/official/VAL/build/bin/{validate,Validate}`

- `official/` for static benchmark-side validator assets

This task does not use task-local runtime directories under `benchmark/` outside its own task folder. `External/` or `external/` clones elsewhere in the repo are reference-only and are not consulted by the verifier.

You can override those paths with:

- `PLANBENCH_OFFICIAL_ROOT`
- `PLANBENCH_VAL_BINARY`
