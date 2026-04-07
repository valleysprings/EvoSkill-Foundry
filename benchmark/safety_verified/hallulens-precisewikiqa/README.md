# HalluLens PreciseWikiQA

Single-turn grounded factuality benchmark built as a deterministic HalluLens-style local GoodWiki slice.

- runtime backend: `benchmark_adapter`
- task contract: `respond(question: dict, runtime) -> str`
- primary metric: `hallucination_rate` (`direction=min`)
- item aggregation: a prompt fails when the assistant abstains unnecessarily or gives an incorrect or unsupported answer relative to the provided reference

Notes:

- The official HalluLens pipeline dynamically generates these prompts. This local task fixes a deterministic slice from `euirim/goodwiki` so it remains fully runnable offline after prepare.

Prepare locally with:

```bash
python benchmark/prepare_datasets.py --task-id hallulens-precisewikiqa
```
