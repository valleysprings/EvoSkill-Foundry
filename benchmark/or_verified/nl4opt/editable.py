def build_run_config() -> dict:
    return {
        "dataset_name": "CardinalOperations/NL4OPT",
        "dataset_split": "test",
        "execution_timeout_s": 600,
        "numerical_err_tolerance": 1e-4,
    }


RUN_CONFIG = build_run_config()

SYSTEM_PROMPT = (
    "You are an expert in operations research modeling. "
    "Return only a JSON object with keys modeling_summary and python_code. "
    "python_code must be directly executable Python using coptpy, without Markdown fences. "
    "It must define the complete optimization model needed to solve the question."
)


def build_user_prompt(question: str) -> str:
    return (
        "Below is an operations research question. Build a mathematical model and corresponding "
        "python code using `coptpy` that appropriately addresses the question.\n\n"
        f"# Question:\n{question.strip()}\n\n"
        "# Response:\n"
    )
