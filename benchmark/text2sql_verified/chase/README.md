# chase

Comparable text-to-SQL benchmark backed by the official GitHub dataset `xjtu-intsoft/chase` on the `page` branch.

This task keeps the repo-local benchmark contract:

- input: one CHASE dev turn, its database schema text, and prior utterance history
- output: one SQL query string
- scoring: Spider official-style execution match on the local sqlite database
- unified repo contract: `runtime_backend=dataset`, `task_mode=answer`, `optimization_scope=implementation`
- verifier style: adapter, because the verifier extracts SQL text before Spider execution evaluation

Prepare locally with:

```bash
python3 prepare.py
```

The benchmark-local source mirror lives under `data/source/`:

- `data/source/dev.json`
- `data/source/train.json`
- `data/source/test.json`
- `data/source/tables.json`
- `data/database/<db_id>.sqlite`

`prepare.py` refreshes these files from the official `page` branch archive and extracts the sqlite databases needed by the requested manifest prefix into `data/database/`.

Source provenance:

- GitHub dataset: `xjtu-intsoft/chase` (`page` branch)
- Evaluation contract: Spider official `evaluation.py`
