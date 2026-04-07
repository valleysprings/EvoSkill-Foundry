from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlparse

from app.codegen.errors import ConfigError
from app.configs.runtime import (
    ACTIVE_PROFILE_ENV_KEY,
    DEFAULT_LLM_CONCURRENCY,
    DEFAULT_RUNTIME_TEMPERATURE,
    ROOT,
    default_max_tokens_for_model,
    default_timeout_for_model,
)


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigError(f"Malformed .env line {line_number}: expected KEY=VALUE.")
        key, value = line.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key:
            raise ConfigError(f"Malformed .env line {line_number}: empty key.")
        values[normalized_key] = _strip_quotes(value.strip())
    return values


def load_repo_env(root: Path | None = None) -> Path:
    resolved_root = root or ROOT
    env_path = resolved_root / ".env"
    for key, value in parse_dotenv(env_path).items():
        os.environ.setdefault(key, value)
    return env_path


def _require_string(mapping: dict[str, object], field_name: str) -> str:
    value = mapping.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Profile field {field_name} must be a non-empty string.")
    return value.strip()


def _optional_string(mapping: dict[str, object], field_name: str) -> str | None:
    value = mapping.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"Profile field {field_name} must be a string.")
    normalized = value.strip()
    return normalized or None


def _parse_base_url(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Profile field {field_name} must be a non-empty string.")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError(f"Profile field {field_name} must be a valid http(s) URL.")
    return value.rstrip("/")


def _optional_bool(mapping: dict[str, object], field_name: str, *, default: bool) -> bool:
    value = mapping.get(field_name)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ConfigError(f"Profile field {field_name} must be a boolean.")
    return value


def _optional_positive_int(mapping: dict[str, object], field_name: str, *, default: int) -> int:
    value = mapping.get(field_name)
    if value is None:
        return default
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"Profile field {field_name} must be a positive integer.")
    return value


def _optional_float(mapping: dict[str, object], field_name: str, *, default: float) -> float:
    value = mapping.get(field_name)
    if value is None:
        return default
    if not isinstance(value, (int, float)):
        raise ConfigError(f"Profile field {field_name} must be a float.")
    parsed = float(value)
    if not 0.0 <= parsed <= 2.0:
        raise ConfigError(f"Profile field {field_name} must be between 0.0 and 2.0.")
    return parsed


def _parse_available_models(value: object, *, default_model: str) -> tuple[str, ...]:
    models: list[str] = []
    source = [default_model] if value is None else value
    if not isinstance(source, list):
        raise ConfigError("Profile field available_models must be a list of strings.")
    for item in source:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError("Profile field available_models must contain non-empty strings.")
        normalized = item.strip()
        if normalized not in models:
            models.append(normalized)
    if default_model not in models:
        models.insert(0, default_model)
    return tuple(models)


def _parse_api_key(profile: dict[str, object]) -> str | None:
    inline_api_key = _optional_string(profile, "api_key")
    api_key_env = _optional_string(profile, "api_key_env")
    if inline_api_key and api_key_env:
        raise ConfigError("Profile fields api_key and api_key_env are mutually exclusive.")
    if api_key_env:
        value = os.getenv(api_key_env)
        if value is None:
            raise ConfigError(f"Profile api_key_env={api_key_env} is set but the environment variable is missing.")
        return value.strip() or None
    return inline_api_key


def _load_profiles_document(root: Path) -> dict[str, object]:
    config_path = root / "llm_profiles.toml"
    if not config_path.exists():
        raise ConfigError(f"Missing required runtime profile config: {config_path}.")
    try:
        with config_path.open("rb") as handle:
            parsed = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Malformed {config_path.name}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfigError(f"Malformed {config_path.name}: expected a TOML table.")
    return parsed


def _active_profile_name(document: dict[str, object]) -> str:
    override = os.getenv(ACTIVE_PROFILE_ENV_KEY)
    if isinstance(override, str) and override.strip():
        return override.strip()
    configured = document.get("active_profile")
    if not isinstance(configured, str) or not configured.strip():
        raise ConfigError("Runtime config must define a non-empty active_profile.")
    return configured.strip()


def _profile_section(document: dict[str, object], profile_name: str) -> dict[str, object]:
    profiles = document.get("profiles")
    if not isinstance(profiles, dict):
        raise ConfigError("Runtime config must define a [profiles] table.")
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        raise ConfigError(f"Runtime config does not define profile {profile_name!r}.")
    return profile


@dataclass(slots=True)
class RuntimeConfig:
    profile: str
    provider: str
    transport: str
    api_key: str | None
    base_url: str
    default_model: str
    available_models: tuple[str, ...]
    temperature: float
    max_tokens: int
    timeout_s: int
    llm_concurrency: int
    supports_tools: bool = True
    supports_json_mode: bool = True
    selected_model: str | None = None
    max_tokens_is_default: bool = False
    timeout_s_is_default: bool = False

    @property
    def active_model(self) -> str:
        return self.selected_model or self.default_model

    def with_model(self, model: str | None) -> "RuntimeConfig":
        if model is None or model == "":
            updated = replace(self, selected_model=None)
        else:
            if model not in self.available_models:
                raise ConfigError(
                    f"Model {model} is not enabled. Choose one of: {', '.join(self.available_models)}."
                )
            updated = replace(self, selected_model=model)
        overrides: dict[str, int] = {}
        if updated.max_tokens_is_default:
            overrides["max_tokens"] = default_max_tokens_for_model(updated.active_model)
        if updated.timeout_s_is_default:
            overrides["timeout_s"] = default_timeout_for_model(updated.active_model)
        if overrides:
            updated = replace(updated, **overrides)
        return updated

    def with_llm_concurrency(self, llm_concurrency: int | None) -> "RuntimeConfig":
        if llm_concurrency is None:
            return self
        if llm_concurrency <= 0:
            raise ConfigError("llm_concurrency must be a positive integer.")
        return replace(self, llm_concurrency=int(llm_concurrency))

    def describe(self) -> dict[str, object]:
        return {
            "mode": "llm-required",
            "profile": self.profile,
            "provider": self.provider,
            "transport": self.transport,
            "default_model": self.default_model,
            "active_model": self.active_model,
            "available_models": list(self.available_models),
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_s": self.timeout_s,
            "llm_concurrency": self.llm_concurrency,
            "supports_tools": self.supports_tools,
            "supports_json_mode": self.supports_json_mode,
        }


def load_runtime_config(root: Path | None = None) -> RuntimeConfig:
    resolved_root = root or ROOT
    load_repo_env(resolved_root)
    document = _load_profiles_document(resolved_root)
    profile_name = _active_profile_name(document)
    profile = _profile_section(document, profile_name)
    transport = _require_string(profile, "transport")
    if transport != "openai-compatible":
        raise ConfigError(f"Unsupported transport {transport!r}.")
    default_model = _require_string(profile, "default_model")
    max_tokens_is_default = "max_tokens" not in profile
    timeout_s_is_default = "timeout_s" not in profile
    return RuntimeConfig(
        profile=profile_name,
        provider=_require_string(profile, "provider"),
        transport=transport,
        api_key=_parse_api_key(profile),
        base_url=_parse_base_url(profile.get("base_url"), field_name="base_url"),
        default_model=default_model,
        available_models=_parse_available_models(profile.get("available_models"), default_model=default_model),
        temperature=_optional_float(profile, "temperature", default=DEFAULT_RUNTIME_TEMPERATURE),
        max_tokens=_optional_positive_int(profile, "max_tokens", default=default_max_tokens_for_model(default_model)),
        timeout_s=_optional_positive_int(profile, "timeout_s", default=default_timeout_for_model(default_model)),
        llm_concurrency=_optional_positive_int(profile, "llm_concurrency", default=DEFAULT_LLM_CONCURRENCY),
        supports_tools=_optional_bool(profile, "supports_tools", default=True),
        supports_json_mode=_optional_bool(profile, "supports_json_mode", default=True),
        max_tokens_is_default=max_tokens_is_default,
        timeout_s_is_default=timeout_s_is_default,
    )
