def build_run_config() -> dict:
    return {
        "repo_url": "https://github.com/sierra-research/tau2-bench.git",
        "model_provider": "openai",
        "user_model_provider": "openai",
        "agent_strategy": "tool-calling",
        "user_strategy": "llm",
        "max_concurrency": 1,
        "n_tasks": 10,
        "timeout_s": 7200,
    }


RUN_CONFIG = build_run_config()
