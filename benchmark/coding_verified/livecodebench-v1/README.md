# LiveCodeBench v1

This task mirrors the non-overlapping `livecodebench/code_generation_lite` `v1` shard as a dataset task under the current dataset runtime.

It aligns to the official LiveCodeBench code-generation evaluation path:

- `prepare.py` streams the official Hugging Face JSONL file for `v1`
- `verifier.py` evaluates candidate code with the official `lcb_runner/evaluation/testing_util.py` execution semantics adapted locally
- functional and stdin problems therefore follow the official checker behavior rather than the older substitute harness

This shard contains `400` problems.
