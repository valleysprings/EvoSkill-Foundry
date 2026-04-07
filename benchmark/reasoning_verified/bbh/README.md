# BIG-Bench Hard

Comparable reasoning benchmark backed by the public [Hugging Face `maveriq/bigbenchhard`](https://huggingface.co/datasets/maveriq/bigbenchhard) dataset card and the upstream [BIG-Bench-Hard GitHub raw JSON files](https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/bbh).

This local task treats BBH as a single reasoning dataset under the unified dataset runtime:

- benchmark task unit: the full dataset
- evolution unit: one question per independent question-run
- current normalization target: all `27` official BBH configs, `6511` items total
- prompt/eval style: direct-answer only
- verifier compares the solver return value directly against the raw dataset target
- multiple-choice subsets preserve parsed `choices` plus `correct_choice_index` for solver context, but scoring still follows the raw official target such as `(C)`

The Hugging Face dataset currently ships as a legacy dataset script, so `prepare.py` pulls the official upstream raw JSON files directly instead of relying on `datasets.load_dataset(...)`.

Prepare locally with:

```bash
python3 prepare.py
python3 prepare.py --items 250
python3 prepare.py --subset boolean_expressions --items 50
```
