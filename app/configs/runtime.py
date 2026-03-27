from __future__ import annotations

from app.configs.paths import ROOT


REQUIRED_ENV_KEYS = (
    "AUTORESEARCH_API_KEY",
    "AUTORESEARCH_API_BASE",
)
AVAILABLE_MODELS_ENV_KEY = "AUTORESEARCH_AVAILABLE_MODELS"
PRIMARY_MODEL_ENV_KEY = "AUTORESEARCH_PRIMARY_MODEL"
LLM_CONCURRENCY_ENV_KEY = "AUTORESEARCH_LLM_CONCURRENCY"
DEFAULT_PRIMARY_MODEL = "deepseek-chat"
DEFAULT_AVAILABLE_MODELS = (
    "deepseek-chat",
    "deepseek-reasoner",
    "gemini-3-flash-preview",
    "gpt-5.4",
    "claude-sonnet-4-6",
    "kimi-k2.5",
    "glm-5",
)
DEFAULT_RUNTIME_TEMPERATURE = 0.2
DEFAULT_RUNTIME_MAX_TOKENS = 16384
DEFAULT_RUNTIME_TIMEOUT_S = 1200
DEFAULT_NON_REASONING_RUNTIME_MAX_TOKENS = max(1, DEFAULT_RUNTIME_MAX_TOKENS // 4)
DEFAULT_NON_REASONING_RUNTIME_TIMEOUT_S = max(1, DEFAULT_RUNTIME_TIMEOUT_S // 4)
REASONING_MODELS = frozenset({"deepseek-reasoner", "gpt-5.4"})
DEFAULT_LLM_CONCURRENCY = 20


def is_reasoning_model(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized in REASONING_MODELS or "reasoner" in normalized


def default_max_tokens_for_model(model: str) -> int:
    if is_reasoning_model(model):
        return DEFAULT_RUNTIME_MAX_TOKENS
    return DEFAULT_NON_REASONING_RUNTIME_MAX_TOKENS


def default_timeout_for_model(model: str) -> int:
    if is_reasoning_model(model):
        return DEFAULT_RUNTIME_TIMEOUT_S
    return DEFAULT_NON_REASONING_RUNTIME_TIMEOUT_S

__all__ = [
    "AVAILABLE_MODELS_ENV_KEY",
    "DEFAULT_AVAILABLE_MODELS",
    "DEFAULT_LLM_CONCURRENCY",
    "DEFAULT_NON_REASONING_RUNTIME_MAX_TOKENS",
    "DEFAULT_NON_REASONING_RUNTIME_TIMEOUT_S",
    "DEFAULT_PRIMARY_MODEL",
    "DEFAULT_RUNTIME_MAX_TOKENS",
    "DEFAULT_RUNTIME_TEMPERATURE",
    "DEFAULT_RUNTIME_TIMEOUT_S",
    "LLM_CONCURRENCY_ENV_KEY",
    "PRIMARY_MODEL_ENV_KEY",
    "REASONING_MODELS",
    "REQUIRED_ENV_KEYS",
    "ROOT",
    "default_max_tokens_for_model",
    "default_timeout_for_model",
    "is_reasoning_model",
]
