# bird

Comparable text-to-SQL benchmark backed by the official BIRD dev package from `AlibabaResearch/DAMO-ConvAI`.

This task keeps the repo-local benchmark contract:

- input: one BIRD dev question plus its schema summary, evidence, and local database assets
- output: one SQL query string
- scoring: official-style execution accuracy against the local SQLite database
- unified repo contract: `runtime_backend=dataset`, `task_mode=answer`, `optimization_scope=implementation`
- verifier style: adapter, because the verifier extracts SQL text before executing predicted and gold SQL

Prepare locally with:

```bash
python3 prepare.py
```

This downloads the official `dev.zip` package as needed, materializes `data/dev.json`, `data/dev_tables.json`, and the required `data/dev_databases/...` SQLite assets, then builds `data/questions.json`.

Source provenance:

- Official repo: `AlibabaResearch/DAMO-ConvAI`
- Official dev package: `https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip`
