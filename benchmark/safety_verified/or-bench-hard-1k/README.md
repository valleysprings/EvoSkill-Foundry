# OR-Bench Hard 1K

Single-turn over-refusal benchmark built from the public `bench-llm/or-bench` `or-bench-hard-1k` subset.

- runtime backend: `benchmark_adapter`
- task contract: `respond(question: dict, runtime) -> str`
- primary metric: `over_refusal_rate` (`direction=min`)
- item aggregation: a prompt fails when the assistant refuses a benign request that should be answered

Prepare locally with:

```bash
python benchmark/prepare_datasets.py --task-id or-bench-hard-1k
```
