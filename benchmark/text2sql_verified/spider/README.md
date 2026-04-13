# spider

Comparable text-to-SQL benchmark backed by the local official Spider `dev` assets under `data/`.

This task keeps the repo-local benchmark contract:

- input: one Spider question plus its schema summary and local SQLite path
- output: one SQL query string
- scoring: Spider official exact-match evaluation via the released parser/evaluator modules
- unified repo contract: `task_mode=answer`, `interaction_mode=single_turn`
- verifier style: adapter, because the verifier extracts SQL text before matching

Prepare locally with:

```bash
python3 prepare.py
```

Required local source files:

- `data/dev.json`
- `data/tables.json`
- `data/database/`

`prepare.py` reads the official Spider `dev` JSON and schema assets directly from `data/` and writes `data/questions.json`.

Source provenance:

- Official Spider data archive, extracted locally under `data/`
- Optional reference clone: `external/spider`
