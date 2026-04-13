# HalluLens LongWiki

Long-context grounded factuality benchmark built as a deterministic HalluLens-style GoodWiki long-form slice.

- runtime backend: `dataset`
- task contract: `respond(question: dict, runtime) -> str`
- primary metric: `long_form_hallucination_rate` (`direction=min`)
- item aggregation: a prompt fails when the assistant abstains unnecessarily or introduces unsupported claims relative to the provided long reference

Notes:

- The official HalluLens LongWiki pipeline depends on dynamic prompt generation and wiki-backed verification.
- This local task fixes a deterministic GoodWiki long-context slice so it stays runnable with local judging only.

Prepare locally with:

```bash
python benchmark/prepare_datasets.py --task-id hallulens-longwiki
```
