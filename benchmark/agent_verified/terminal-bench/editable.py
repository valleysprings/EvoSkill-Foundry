def build_run_config() -> dict:
    return {
        "repo_url": "https://github.com/laude-institute/harbor.git",
        "dataset": "terminal-bench@2.0",
        "agent_name": "terminus-2",
        "model_provider": "openai",
        "n_tasks": 5,
        "n_concurrent": 1,
        "timeout_s": 7200,
    }


RUN_CONFIG = build_run_config()
