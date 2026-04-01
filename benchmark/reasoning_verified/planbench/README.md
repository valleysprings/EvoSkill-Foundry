# planbench

Comparable reasoning benchmark backed by the public `tasksource/planbench` dataset.

This local task tracks the official direct `task_1_plan_generation` setting as closely as practical while keeping runtime assets under `runs/`:

- input: the original natural-language planning prompt
- output: a directly generated plan in a format accepted by upstream `text_to_plan`
- scoring: upstream-compatible task1 parsing plus semantic validation with `VAL`
- unified repo contract: `runtime_backend=dataset`, `task_mode=answer`, `optimization_scope=wrapper`
- verifier style: adapter/semantic validation, not exact string match or permissive local fallback parsing

Current local normalization target:

- config: `task_1_plan_generation`
- split: `train`

Prepare locally with:

```bash
python3 prepare.py
```

Default official assets are resolved from:

- `runs/runtime/benchmarks/planbench/plan-bench`
- `runs/runtime/benchmarks/planbench/VAL/build/bin/{validate,Validate}`

This task does not use task-local runtime directories under `benchmark/`. `External/` or `external/` clones elsewhere in the repo are reference-only and are not consulted by the verifier.

You can override those paths with:

- `PLANBENCH_OFFICIAL_ROOT`
- `PLANBENCH_VAL_BINARY`
