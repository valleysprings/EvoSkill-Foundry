# Personalization Benchmarks

This track now treats [reference_benchmarks.json](/Users/david/coding/2026/autoresearch-foundry/benchmark/personalization_verified/reference_benchmarks.json) as the single source of truth for taxonomy, benchmark status, and runnable eligibility.

## Environment Setup

Team default is the shared Conda env `autoresearch`, with Python dependencies managed by the root [pyproject.toml](/Users/david/coding/2026/autoresearch-foundry/pyproject.toml) and [uv.lock](/Users/david/coding/2026/autoresearch-foundry/uv.lock).

```bash
conda activate autoresearch
uv sync --active
```

If you change dependency ranges or add a new benchmark-side package, regenerate the lockfile from the repo root:

```bash
uv lock
uv sync --active
```

The current personalization stack depends on project-managed Hugging Face dataset clients plus the official RoleMRC metric stack, so packages such as `datasets`, `huggingface-hub`, `evaluate`, `rouge-score`, `sacrebleu`, `nltk`, `bert-score`, `transformers`, `pandas`, `numpy`, `tqdm`, and `openai` now live in the root lockfile instead of being installed ad hoc.

## Browse Hierarchy

- `Browse Mode`
  `Personalization`
- `Turn Mode`
  `single_turn` or `multi_turn`
- `Category`
  `character_portrayal`
  `consistency_robustness`
  `user_personalization`

`benchmark_category` remains the coarse research-family field:

- `explicit_character_persona`
- `user_persona_personalization`
- `trait_behavior`

`primary_category` is the UI dropdown key.
`secondary_categories` and `official_dimensions` preserve paper-level distinctions without turning every benchmark into its own category.

## Current Catalog State

The reference catalog currently tracks `11` running local tasks, all single-turn:

- `InCharacter`
- `CharacterBench`
- `SocialBench`
- `TimeChara`
- `PersonaFeedback`
- `PersonaMem`
- `AlpsBench Task 1 (Extraction)`
- `AlpsBench Task 2 (Update)`
- `AlpsBench Task 3 (Retrieval)`
- `AlpsBench Task 4 (Utilization)`
- `ALPBench`

The registry-enabled preparation set is narrower and currently includes:

- `incharacter`
- `characterbench`
- `socialbench`
- `personafeedback`
- `personamem-32k`
- `alpsbench-extraction`
- `alpsbench-update`
- `alpsbench-retrieval`
- `alpsbench-utilization`
- `alpbench`

## Official-Fidelity Policy

Benchmarks can remain `local_task + running` in two cases:

- `metric_fidelity=official`
  the local runnable path matches the published protocol closely enough to benchmark directly
- `metric_fidelity=proxy_local`
  the local runnable path follows the released public contract and scorer, but the benchmark-side judge or hidden aggregation is not fully public

If the public release is incomplete and no honest runnable proxy exists, the benchmark is downgraded to `planned_task` or `external_reference`.

`metric_fidelity` is interpreted strictly:

- `official`
  the local runnable path matches the published protocol closely enough to benchmark directly
- `adapted_local`
  kept for schema compatibility, but not used by the current tracked catalog after this reset
- `proxy_local`
  the local runnable path matches the public release but not the hidden benchmark-side judge stack
- `reference_only`
  tracked in catalog metadata only; not considered an official runnable implementation

## Schema Fields

Every tracked benchmark now declares:

- `status`
  `local_task`, `planned_task`, or `external_reference`
- `interaction_mode`
  `single_turn` or `multi_turn`
- `benchmark_category`
  coarse research family
- `primary_category`
  coarse UI category
- `secondary_categories`
  paper-level nuance
- `official_dimensions`
  the benchmark's published metric or evaluation dimensions
- `protocol_summary`
  a short description of the official evaluation flow
- `implementation_note`
  the concrete rewrite or audit target for the current start wave
- `required_runtime_roles`
  benchmark-native model roles such as `policy_model`, `judge_model`, `reward_model`, `interrogator_model`, `env_model`, or `nsp_model`
- `metric_fidelity`
  whether the local system currently implements the benchmark faithfully

## Runnable vs Hidden Local Tasks

Some hidden local wrappers still exist in this directory for future rewrite work.
They are not treated as official benchmark implementations and are not exposed in the runnable picker.

That means:

- tracked benchmark metadata can show a benchmark as `planned_task`
- a hidden local wrapper may still exist for development
- only `included_in_main_comparison=true` tasks appear in the picker

## Task READMEs

Each task README should state:

- upstream paper / repo
- the official protocol in 1-2-3 steps
- current local fidelity
- required models / evaluator assets
- known deviations

If a local wrapper is only a public-release proxy, the README must say so explicitly.

If a local wrapper is not runnable and not protocol-faithful, the README must say:

- `Not official; hidden from runnable picker until rewritten.`

## Preparing Data

Only the current official runnable set should be treated as active benchmark tasks:

```bash
python benchmark/prepare_datasets.py \
  --task-id incharacter \
  --task-id characterbench \
  --task-id socialbench \
  --task-id personafeedback \
  --task-id personamem-32k \
  --task-id alpsbench-extraction \
  --task-id alpsbench-update \
  --task-id alpsbench-retrieval \
  --task-id alpsbench-utilization \
  --task-id alpbench
```

## Mirrors

Mirrorable upstream repos are catalog-driven:

```bash
python benchmark/personalization_verified/sync_external.py
python benchmark/personalization_verified/sync_external.py --list
```

Mirrors live under `external/personalization/`.
