# nl4opt

OR benchmark task for `CardinalOperations/NL4OPT` using the custom benchmark-adapter runtime backend.

It prompts the configured model for `coptpy` code, executes the generated program locally, and scores exact optimal-value matches against the benchmark answers.

Here `runtime_backend=benchmark_adapter` means this task runs through a benchmark-owned adapter instead of the generic dataset fan-out path. It does not mean the benchmark is remote-only.

`python benchmark/prepare_datasets.py --task-id nl4opt` is the expected prerequisite after clone.

You can also run `python prepare.py` directly to materialize the benchmark question manifest into `data/questions.json`.
