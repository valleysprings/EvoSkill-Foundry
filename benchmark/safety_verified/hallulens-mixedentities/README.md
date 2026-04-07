# HalluLens MixedEntities

Single-turn hallucination benchmark built as a runnable HalluLens-style local slice for nonexistent mixed entities.

- runtime backend: `benchmark_adapter`
- task contract: `respond(question: dict, runtime) -> str`
- primary metric: `hallucination_rate` (`direction=min`)
- item aggregation: a prompt fails when the assistant invents facts instead of abstaining on a nonexistent entity

Notes:

- This task intentionally avoids HalluLens' external medicine CSV dependency so the local benchmark stays runnable without extra credentials.
- The current local slice uses public ITIS taxonomy data plus a bundled generic-medicine seed list and preserves the same failure mode: fabricated answers on nonexistent entities.

Prepare locally with:

```bash
python benchmark/prepare_datasets.py --task-id hallulens-mixedentities
```
