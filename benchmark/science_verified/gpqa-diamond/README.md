# GPQA Diamond

Comparable science benchmark backed by the official [GPQA GitHub repository](https://github.com/idavidrein/gpqa).

This local task treats GPQA Diamond as a single science dataset under the unified dataset runtime:

- benchmark task unit: the full dataset
- evolution unit: one question per independent question-run
- current normalization target: `dataset/gpqa_diamond.csv`, `198` items total
- prompt/eval style: direct-answer only
- prepare reads the official password-protected `dataset.zip` release rather than relying on the gated Hugging Face mirror
- choice order is shuffled once at prepare time using a stable `Record ID`-based seed so prepared manifests stay reproducible across runs

Prepare locally with:

```bash
python3 prepare.py
python3 prepare.py --items 50
```
