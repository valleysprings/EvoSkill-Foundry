# HarmBench Text Harmful

Single-turn safety benchmark built from HarmBench harmful text behaviors.

- runtime backend: `dataset`
- task contract: `respond(question: dict, runtime) -> str`
- primary metric: `attack_success_rate` (`direction=min`)
- behavior aggregation: a behavior is counted as jailbroken if any fixed prompt succeeds
- verifier semantics: official HarmBench behavior-classifier prompt (`yes` / `no`), not the shared local harmfulness judge

Local assets are materialized into:

- `benchmark/safety_verified/harmbench-text-harmful/data/source/`
- `benchmark/safety_verified/harmbench-text-harmful/data/questions.json`
- `benchmark/safety_verified/harmbench-text-harmful/data/source_info.json`

Prepare locally with:

```bash
python benchmark/prepare_datasets.py --task-id harmbench-text-harmful
```

Preparation now requires the official HarmBench results archive. Fallback prompts are no longer accepted.
