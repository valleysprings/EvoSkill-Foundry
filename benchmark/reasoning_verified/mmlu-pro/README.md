# MMLU-Pro

Comparable reasoning benchmark backed by the public [Hugging Face `TIGER-Lab/MMLU-Pro`](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) dataset.

This local task treats MMLU-Pro as a single reasoning dataset under the unified dataset runtime:

- benchmark task unit: the full dataset
- evolution unit: one question per independent question-run
- current normalization target: config `default`, split `test`, `12032` items total
- prompt/eval style: direct-answer only
- verifier accepts either the correct answer text or the official option label for the prepared choice order
- variable-length `choices` are preserved exactly as shipped by the dataset

Prepare locally with:

```bash
python3 prepare.py
python3 prepare.py --items 250
```
