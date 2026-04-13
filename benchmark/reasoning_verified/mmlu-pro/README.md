# MMLU-Pro

Comparable reasoning benchmark backed by the public [Hugging Face `TIGER-Lab/MMLU-Pro`](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) dataset.

This local task treats MMLU-Pro as a single reasoning dataset under the unified dataset runtime:

- benchmark task unit: the full dataset
- evolution unit: one question per independent question-run
- current normalization target: config `default`, split `test`, `12032` items total
- prompt/eval style: direct-answer only
- verifier now follows the official option-letter judgment path and scores only the extracted final option label
- variable-length `choices` are preserved exactly as shipped by the dataset

Prepare locally with:

```bash
python3 prepare.py
python3 prepare.py --items 250
```
