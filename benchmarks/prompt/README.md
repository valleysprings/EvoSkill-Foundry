# Prompt Optimization Benchmarks

This directory defines small, local-first benchmark slices for prompt optimization.

The goal is not to clone massive benchmark corpora. The goal is to keep a reproducible
subset on disk that is cheap to run on a Mac while still being representative enough to
drive an evolutionary flywheel.

## Chosen task families

### 1. Intent routing

- Dataset: `mteb/banking77`
- Why: deterministic label space, realistic support-style user utterances, and clear
  task utility for prompt evolution.
- Task form: classify each utterance into one of the Banking77 intents.
- Primary metric: exact-match accuracy

### 2. Yes/No reading comprehension

- Dataset: `google/boolq`
- Why: simple binary answer space, passage-conditioned reasoning, low evaluator cost.
- Task form: answer `yes` or `no` given a question and passage.
- Primary metric: exact-match accuracy

### 3. News topic classification

- Dataset: `sh0416/ag_news`
- Why: compact four-way label space and a clean classification benchmark that is widely
  reused in NLP evaluation.
- Task form: classify a news article into `world`, `sports`, `business`, or `sci_tech`.
- Primary metric: exact-match accuracy

## Why these are good for the current repo

- They are deterministic, so `J` stays easy to interpret.
- They are cheap enough to run many prompt candidates locally.
- They are diverse enough to exercise retrieval, operator selection, and prompt reuse.

## Fetching

Use the fetch script to download small benchmark slices into `benchmarks/prompt/data/`:

```bash
bash scripts/fetch_prompt_benchmarks.sh
```

To refresh one benchmark only:

```bash
bash scripts/fetch_prompt_benchmarks.sh --benchmark banking77_small
```

## Data source notes

The fetch script now prefers `datasets.load_dataset()` so the benchmark slices can be
downloaded directly without depending on the dataset viewer service. It falls back to the
public Hugging Face dataset viewer API only if the `datasets` package is unavailable.

Relevant references:

- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- [Dataset Viewer Quickstart](https://huggingface.co/docs/dataset-viewer/quick_start)

The current benchmark choices are based on these dataset cards:

- [mteb/banking77](https://huggingface.co/datasets/mteb/banking77)
- [google/boolq](https://huggingface.co/datasets/google/boolq)
- [sh0416/ag_news](https://huggingface.co/datasets/sh0416/ag_news)

## Next integration step

These datasets are meant to feed a prompt-evolution runner with:

- prompt variants
- operator lineage
- semantic memory retrieval
- `J` / `delta_J` scoring

They are not yet wired into the live UI runner in this commit.
