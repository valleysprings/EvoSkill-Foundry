# Benchmark Registry

This repo treats `benchmark/` as the source of truth for local research datasets and their task wrappers.

The active tasks live in `benchmark/registry.json`. Each task directory includes:

- `task.json`
- `editable.py`
- `verifier.py`
- `prepare.py` for local dataset normalization when the benchmark ships as raw local assets
- `data/`
- `README.md`

Each task is expected to declare a clear contract:

- `runtime_backend`
  use `dataset` or `benchmark_adapter`; runtime-managed assets belong under `runs/`, not `External/`
- `task_mode`
  what the editable file represents: `answer` or `artifact`
- `interaction_mode`
  whether the task is `single_turn` or `multi_turn`
- `optimization_scope`
  what is allowed to change: `prompt`, `wrapper`, or `implementation`
- `answer_metric`
  the task-native headline metric name
- `objective_spec`
  display name, direction, unit, and formula for that metric
- `safety_category`
  required for `safety_verified` tasks; it captures the safety sub-direction while `interaction_mode` continues to capture single-turn vs multi-turn execution shape

For `personalization_verified`, the browse hierarchy is also fixed and catalog-driven:

- `Browse Mode`
  `Personalization`
- `Turn Mode`
  `single_turn` or `multi_turn`
- `Category`
  driven by `primary_category` from `benchmark/personalization_verified/reference_benchmarks.json`

That personalization catalog keeps three different layers on purpose:

- `benchmark_category`
  coarse research-family compatibility field
- `primary_category`
  the UI-facing coarse literature-backed category
- `secondary_categories`
  paper-level nuance tags that stay visible without exploding the dropdown

See [benchmark/personalization_verified/README.md](./personalization_verified/README.md) for the current full tracked benchmark slate, runnable subset, and category mapping.

For `task_mode=answer`, also make the verifier style explicit in task-local docs and prompt context:

- `exact_match`
  the returned answer is compared directly against a reference answer or label set
- `adapter` / `semantic`
  the returned answer is first parsed, normalized, or adapted by the verifier and then judged by a semantic checker or task-native evaluator

Read [references/evaluation-contract.md](../references/evaluation-contract.md) for the metric and selection semantics behind these fields.

Only `benchmark/registry.json` is intended to sync by default. Dataset directories under `benchmark/` stay local.

## Prepare Local Datasets

Preparing local datasets is a prerequisite after clone.

Use the shared entrypoint to materialize every benchmark-local dataset setup that exposes a `prepare.py`:

```bash
python benchmark/prepare_datasets.py
```

Useful variants:

```bash
python benchmark/prepare_datasets.py --list
python benchmark/prepare_datasets.py --task-id co-bench
```

The shared script just dispatches each task's own `prepare.py`, so task-specific customization still lives beside the task itself. When a task is missing local assets, runtime errors should point back to this README for the matching source location.

## Prepare Coverage

`benchmark/prepare_datasets.py` dispatches only enabled tasks from `benchmark/registry.json`.

- current registry size: `49` tasks
- current prepare coverage: every registered task ships a task-local `prepare.py`
- current implementation-track split:
  - `11` `personalization_verified` tasks
  - `13` `safety_verified` tasks
  - `25` tasks across `math_verified`, `reasoning_verified`, `science_verified`, `text2sql_verified`, `longcontext_verified`, `coding_verified`, `or_verified`, and `agent_verified`
- current README grouping:
  - `11` `Personalization` tasks
  - `11` `Safety` tasks
  - `27` `General Intelligence` tasks

Two directories under `benchmark/safety_verified/` are intentionally excluded from the registry and shared prepare flow:

- `bloom-self-preferential-bias`
- `bloom-trait-examples`

They are support-data directories, not standalone benchmark tasks, so `benchmark/prepare_datasets.py` does not treat them as datasets to prepare.

## Raw Data Sources

If `python benchmark/prepare_datasets.py` reports missing assets, use the matching source below and then rerun prepare.

### Enabled Registry Tasks By Group

| Group | Tasks In This README Grouping | Notes |
| --- | --- | --- |
| `Personalization` | `11` | Matches `personalization_verified`. |
| `Safety` | `9` | Safety-focused tasks from `safety_verified`. |
| `General Intelligence` | `20` | All tasks from `math_verified`, `reasoning_verified`, `science_verified`, `text2sql_verified`, `longcontext_verified`, `coding_verified`, and `or_verified`. |

### Personalization

| Task | Track | Source | Notes |
| --- | --- | --- | --- |
| `incharacter` | `personalization_verified` | [GitHub: `Neph0s/InCharacter`](https://github.com/Neph0s/InCharacter-Role-Playing-Agents-Effectively-Capture-Characters-Personalities-Traits) | Prepare from the mirrored `external/personalization/incharacter/` checkout. The local task now materializes the official questionnaire-bundle evaluation path and requires an `eval_model` for answer-to-choice conversion. |
| `characterbench` | `personalization_verified` | [GitHub: `thu-coai/CharacterBench`](https://github.com/thu-coai/CharacterBench) | Prepare from the mirrored `external/personalization/characterbench/` checkout. The local task now uses the released CharacterJudge prompt construction over the full evaluation subsets and requires an `eval_model`. |
| `personamem-32k` | `personalization_verified` | [Hugging Face: `bowen-upenn/PersonaMem`](https://huggingface.co/datasets/bowen-upenn/PersonaMem) | `prepare.py` downloads `questions_32k.csv` and `shared_contexts_32k.jsonl` from the public dataset repo and materializes a local question manifest with sliced conversation history. |
| `socialbench` | `personalization_verified` | [GitHub: `X-PLUG/SocialBench`](https://github.com/X-PLUG/SocialBench) | Prepare from the mirrored `external/personalization/socialbench/` checkout. The local task reuses the released JSON benchmark with deterministic local scoring for choice and keyword-memory items. |
| `timechara` | `personalization_verified` | [GitHub: `ahnjaewoo/timechara`](https://github.com/ahnjaewoo/timechara) | `prepare.py` pulls the full public test split from Hugging Face (`10895` rows). The local task follows the released spatiotemporal judge protocol and requires an `eval_model`. |
| `rmtbench` | `personalization_verified` | [arXiv: `RMTBench`](https://arxiv.org/abs/2507.20352) | Hidden planned task only. The current local assets are a phase-1 proxy scaffold kept out of the runnable lane until the full official multi-turn release contract is aligned. |
| `charactereval` | `personalization_verified` | [GitHub: `morecry/CharacterEval`](https://github.com/morecry/CharacterEval) | Hidden planned task only. The local scaffold prepares the public contexts (`4564` items) while the benchmark remains blocked on faithful `BaichuanCharRM` reward-model execution. |
| `coser` | `personalization_verified` | [GitHub: `Neph0s/CoSER`](https://github.com/Neph0s/CoSER) | Hidden planned task only. The current local scaffold fans out the public literary dialogue set, but official runnable recovery still needs the full multi-role GCA runtime. |
| `personafeedback` | `personalization_verified` | [arXiv: `PersonaFeedback`](https://arxiv.org/abs/2506.12915) | The local task wraps the published persona-conditioned preference benchmark as deterministic MCQ selection. |
| `alpsbench` | `personalization_verified` | [arXiv: `AlpsBench`](https://arxiv.org/abs/2603.26680) | The local task wraps the long-horizon personalization benchmark as deterministic MCQ selection over pre-materialized memory contexts. |
| `alpbench` | `personalization_verified` | [Hugging Face: `OpenOneRec/ALPBench`](https://huggingface.co/datasets/OpenOneRec/ALPBench) | `prepare.py` pulls the public train rows through the dataset server and converts them into latent-trait classification items (`800` rows). |

### Safety

| Task | Track | Source | Notes |
| --- | --- | --- | --- |
| `xstest-refusal-calibration` | `safety_verified` | [Hugging Face: `Paul/XSTest`](https://huggingface.co/datasets/Paul/XSTest) | `interaction_mode=single_turn`, `safety_category=over_refusal`. `prepare.py` downloads the public XSTest prompt set and turns it into a local refusal-boundary benchmark for answer-vs-refuse calibration. |
| `harmbench-text-harmful` | `safety_verified` | [GitHub: `centerforaisafety/HarmBench`](https://github.com/centerforaisafety/HarmBench) and [Zenodo: HarmBench Results 1.0](https://zenodo.org/records/10714577) | `interaction_mode=single_turn`, `safety_category=jailbreak_attack`. `prepare.py` mirrors the public harmful text behaviors into task-local data, then selects fixed attack prompts from the public results archive when available and falls back to deterministic local prompt templates otherwise. |
| `jailbreakbench-harmful` | `safety_verified` | [Hugging Face: `JailbreakBench/JBB-Behaviors`](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) and [GitHub: `JailbreakBench/artifacts`](https://github.com/JailbreakBench/artifacts) | `interaction_mode=single_turn`, `safety_category=jailbreak_attack`. `prepare.py` materializes the harmful split into task-local data and groups successful published attack artifacts by behavior, with deterministic fallback prompts when no public artifact is available. |
| `or-bench-hard-1k` | `safety_verified` | [Hugging Face: `bench-llm/or-bench`](https://huggingface.co/datasets/bench-llm/or-bench) | `interaction_mode=single_turn`, `safety_category=over_refusal`. Uses subset `or-bench-hard-1k` and scores whether the model answers benign but refusal-sensitive prompts. |
| `or-bench-toxic` | `safety_verified` | [Hugging Face: `bench-llm/or-bench`](https://huggingface.co/datasets/bench-llm/or-bench) | `interaction_mode=single_turn`, `safety_category=jailbreak_attack`, `safety_focus=should_refuse`. Uses subset `or-bench-toxic` and scores whether the model refuses prompts that should be blocked. |
| `hallulens-precisewikiqa` | `safety_verified` | [GitHub: `facebookresearch/HalluLens`](https://github.com/facebookresearch/HalluLens) and [Hugging Face: `euirim/goodwiki`](https://huggingface.co/datasets/euirim/goodwiki) | `interaction_mode=single_turn`, `safety_category=factuality_hallucination`. This is a deterministic local GoodWiki slice aligned to the HalluLens `PreciseWikiQA` task so it stays runnable without dynamic upstream generation services. |
| `hallulens-mixedentities` | `safety_verified` | [GitHub: `facebookresearch/HalluLens`](https://github.com/facebookresearch/HalluLens), [ITIS public taxonomy dump](https://www.itis.gov/downloads/index.html), and bundled medicine seeds | `interaction_mode=single_turn`, `safety_category=factuality_hallucination`. This local runnable slice mirrors HalluLens `MixedEntities` by generating nonexistent public-data entities that calibrated assistants should decline to fabricate. |
| `hallulens-longwiki` | `safety_verified` | [GitHub: `facebookresearch/HalluLens`](https://github.com/facebookresearch/HalluLens) and [Hugging Face: `euirim/goodwiki`](https://huggingface.co/datasets/euirim/goodwiki) | `interaction_mode=single_turn`, `safety_category=factuality_hallucination`. This is a deterministic long-context GoodWiki slice aligned to HalluLens `LongWiki`, evaluated with local groundedness judging only. |
| `longsafety` | `safety_verified` | [GitHub: `thu-coai/longsafety`](https://github.com/thu-coai/longsafety) and [Hugging Face: `thu-coai/LongSafety`](https://huggingface.co/datasets/thu-coai/LongSafety) | `interaction_mode=single_turn`, `safety_category=jailbreak_attack`, `safety_focus=safety_degradation`. `prepare.py` downloads the public long-context harmful prompts and merges instruction/context metadata into a local manifest for refusal-style judging. |

### General Intelligence

| Task | Track | Source | Notes |
| --- | --- | --- | --- |
| `olymmath` | `math_verified` | [Hugging Face: `RUC-AIBOX/OlymMATH`](https://huggingface.co/datasets/RUC-AIBOX/OlymMATH) | Uses config `en-hard`, split `test`. |
| `math-500` | `math_verified` | [Hugging Face: `HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | Uses split `test`. |
| `aime-2024` | `math_verified` | [Hugging Face: `HuggingFaceH4/aime_2024`](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) | Uses split `train`. |
| `aime-2025` | `math_verified` | [Hugging Face: `opencompass/AIME2025`](https://huggingface.co/datasets/opencompass/AIME2025) | Uses configs `AIME2025-I` and `AIME2025-II`, split `test`. |
| `aime-2026` | `math_verified` | [Hugging Face: `math-ai/aime26`](https://huggingface.co/datasets/math-ai/aime26) | Uses split `test`. |
| `planbench` | `reasoning_verified` | [Hugging Face: `tasksource/planbench`](https://huggingface.co/datasets/tasksource/planbench) | Uses config `task_1_plan_generation`, split `train`. |
| `arc-challenge` | `reasoning_verified` | [Hugging Face: `allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc) | Uses config `ARC-Challenge`, split `validation`. |
| `bbh` | `reasoning_verified` | [Hugging Face: `maveriq/bigbenchhard`](https://huggingface.co/datasets/maveriq/bigbenchhard) and [upstream raw BBH JSON](https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/bbh) | Aggregates all `27` official BBH configs (`6511` items total). `prepare.py` downloads the upstream raw JSON files directly because the HF dataset currently uses a legacy dataset script. |
| `mmlu-pro` | `reasoning_verified` | [Hugging Face: `TIGER-Lab/MMLU-Pro`](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | Uses config `default`, split `test`. The local task preserves variable-length choice lists and accepts either the final answer text or the official option label. |
| `spider` | `text2sql_verified` | [Google Drive archive](https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view) | Extract locally under `benchmark/text2sql_verified/spider/data/`. Expected files include `dev.json`, `tables.json`, and `database/`. |
| `bird` | `text2sql_verified` | [Official `dev.zip`](https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip) | `prepare.py` prefers local `data/dev.json`, `data/dev_tables.json`, and `data/dev_databases/` when present. |
| `chase` | `text2sql_verified` | [CHASE questions zip](https://raw.githubusercontent.com/xjtu-intsoft/chase/page/data/Chase.zip) and [CHASE database zip](https://raw.githubusercontent.com/xjtu-intsoft/chase/page/data/database.zip) | Extracted into `benchmark/text2sql_verified/chase/data/` by `prepare.py` if missing. |
| `longbench-v2` | `longcontext_verified` | [Hugging Face: `zai-org/LongBench-v2`](https://huggingface.co/datasets/zai-org/LongBench-v2) | Raw manifest URL used by the script: `https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/main/data.json`. |
| `sciq` | `science_verified` | [Hugging Face: `allenai/sciq`](https://huggingface.co/datasets/allenai/sciq) | Uses split `validation`. |
| `qasc` | `science_verified` | [Hugging Face: `allenai/qasc`](https://huggingface.co/datasets/allenai/qasc) | Uses split `validation`. |
| `scienceqa` | `science_verified` | [Hugging Face: `derek-thomas/ScienceQA`](https://huggingface.co/datasets/derek-thomas/ScienceQA) | Uses split `validation`, then filters to text-only natural-science rows. |
| `openbookqa` | `science_verified` | [Hugging Face: `allenai/openbookqa`](https://huggingface.co/datasets/allenai/openbookqa) | Uses config `additional`, split `validation`. |
| `gpqa-diamond` | `science_verified` | [GitHub: `idavidrein/gpqa`](https://github.com/idavidrein/gpqa) | Reads `dataset/gpqa_diamond.csv` from the official password-protected `dataset.zip` release and applies a stable per-item choice shuffle based on `Record ID`. |
| `livecodebench` | `coding_verified` | [Hugging Face: `livecodebench/code_generation_lite`](https://huggingface.co/datasets/livecodebench/code_generation_lite) | Script reads `test.jsonl` through `test6.jsonl` from the dataset's raw file URLs. |
| `co-bench` | `or_verified` | [Hugging Face: `CO-Bench/CO-Bench`](https://huggingface.co/datasets/CO-Bench/CO-Bench) | Pulled via `huggingface_hub.snapshot_download`. |
## Add A New Benchmark Dataset

1. add a task directory under the appropriate research track such as `math_verified/`, `science_verified/`, `reasoning_verified/`, `text2sql_verified/`, `longcontext_verified/`, `personalization_verified/`, or `safety_verified/`
2. register the dataset task in `benchmark/registry.json`
3. provide:
   - `task.json`
   - `editable.py`
   - `verifier.py`
   - `prepare.py` when raw local data must be normalized or lazily materialized into the shared manifest shape
   - `data/questions.json` or another local question manifest
   - `README.md`

Required `task.json` fields for dataset tasks:

- `track`
- `dataset_id`
- `dataset_size`
- `local_dataset_only`
- `item_manifest`
- `runtime_backend`
- `task_mode`
- `interaction_mode`
- `optimization_scope`
- `answer_metric`
- `objective_spec`
- `editable_file`
- `entry_symbol`

Recommended conventions:

- use the real dataset name for `id`, directory name, and `dataset_id`
- declare `interaction_mode` explicitly for every task; do not rely on implicit defaults
- put subset details like `validation`, `test`, `numeric_verified`, or `local_eval` in `split`
- keep one dataset-native headline metric; do not make `tie_break_score` or code-shape preferences the headline claim
- for answer tasks, state whether the verifier is `exact_match` or `adapter/semantic`; this belongs in the task README and prompt context even if it is not a dedicated schema field
- normalize raw dataset rows into the shared question schema:
  - `item_id`
  - `prompt`
  - optional `context`
  - optional `choices`
  - `expected_answer`
  - `metadata`
- for very large per-item payloads, a manifest row can also declare `item_file`, pointing to a local JSON object that is merged into that row at load time
- math datasets should declare `metadata.answer_format` as `symbolic`, `numeric`, or `choice`
- choice-form math datasets should also declare `metadata.correct_choice_index`
- safety datasets should declare `safety_category`; `safety_focus` is currently kept as a compatibility alias and mirrors the same sub-direction value
- treat the dataset as the benchmark task and each question as an independent question-run
