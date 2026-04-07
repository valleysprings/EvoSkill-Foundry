# JailbreakBench Harmful

Single-turn safety benchmark built from the harmful split of JailbreakBench plus published attack artifacts.

- runtime backend: `benchmark_adapter`
- task contract: `respond(question: dict, runtime) -> str`
- primary metric: `attack_success_rate` (`direction=min`)
- behavior aggregation: a behavior is counted as jailbroken if any fixed artifact prompt succeeds

Local assets are materialized into:

- `benchmark/safety_verified/jailbreakbench-harmful/data/source/`
- `benchmark/safety_verified/jailbreakbench-harmful/data/questions.json`
- `benchmark/safety_verified/jailbreakbench-harmful/data/source_info.json`

Prepare locally with:

```bash
python benchmark/prepare_datasets.py --task-id jailbreakbench-harmful
```
