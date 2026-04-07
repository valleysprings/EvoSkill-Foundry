# HarmBench Text Harmful

Single-turn safety benchmark built from HarmBench harmful text behaviors.

- runtime backend: `benchmark_adapter`
- task contract: `respond(question: dict, runtime) -> str`
- primary metric: `attack_success_rate` (`direction=min`)
- behavior aggregation: a behavior is counted as jailbroken if any fixed prompt succeeds

Local assets are materialized into:

- `benchmark/safety_verified/harmbench-text-harmful/data/source/`
- `benchmark/safety_verified/harmbench-text-harmful/data/questions.json`
- `benchmark/safety_verified/harmbench-text-harmful/data/source_info.json`

Prepare locally with:

```bash
python benchmark/prepare_datasets.py --task-id harmbench-text-harmful
```
