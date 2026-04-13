from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from app.codegen.config import load_runtime_config
from app.codegen.errors import ConfigError
from app.configs.runtime import (
    ACTIVE_PROFILE_ENV_KEY,
    DEFAULT_AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_NON_REASONING_RUNTIME_MAX_TOKENS,
    DEFAULT_NON_REASONING_RUNTIME_TIMEOUT_S,
    DEFAULT_RUNTIME_MAX_TOKENS,
    DEFAULT_RUNTIME_TEMPERATURE,
    DEFAULT_RUNTIME_TIMEOUT_S,
)


def _write_profiles(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).strip() + "\n")


class CodegenConfigTest(unittest.TestCase):
    def test_profile_config_loads_env_backed_secret(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp_dir)
            (root / ".env").write_text("OPENAI_API_KEY=test-key\n")
            _write_profiles(
                root / "llm_profiles.toml",
                """
                active_profile = "openai-main"

                [profiles.openai-main]
                provider = "openai"
                transport = "openai-compatible"
                base_url = "https://api.example.com/v1"
                api_key_env = "OPENAI_API_KEY"
                default_model = "deepseek-chat"
                available_models = [
                  "deepseek-chat",
                  "deepseek-reasoner",
                  "gemini-3-flash-preview",
                  "gpt-5.4",
                  "claude-sonnet-4-6",
                  "kimi-k2.5",
                  "glm-5",
                ]
                """,
            )
            config = load_runtime_config(root)
            self.assertEqual(config.profile, "openai-main")
            self.assertEqual(config.provider, "openai")
            self.assertEqual(config.transport, "openai-compatible")
            self.assertEqual(config.api_key, "test-key")
            self.assertEqual(config.base_url, "https://api.example.com/v1")
            self.assertEqual(config.default_model, DEFAULT_MODEL)
            self.assertEqual(config.available_models, DEFAULT_AVAILABLE_MODELS)
            self.assertEqual(config.temperature, DEFAULT_RUNTIME_TEMPERATURE)
            self.assertEqual(config.max_tokens, DEFAULT_NON_REASONING_RUNTIME_MAX_TOKENS)
            self.assertEqual(config.timeout_s, DEFAULT_NON_REASONING_RUNTIME_TIMEOUT_S)
            self.assertEqual(config.llm_concurrency, 20)

    def test_active_profile_can_be_overridden_by_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
            os.environ,
            {ACTIVE_PROFILE_ENV_KEY: "ollama-local"},
            clear=True,
        ):
            root = Path(tmp_dir)
            _write_profiles(
                root / "llm_profiles.toml",
                """
                active_profile = "openai-main"

                [profiles.openai-main]
                provider = "openai"
                transport = "openai-compatible"
                base_url = "https://api.example.com/v1"
                api_key = "test-key"
                default_model = "deepseek-chat"

                [profiles.ollama-local]
                provider = "ollama"
                transport = "openai-compatible"
                base_url = "http://127.0.0.1:11434/v1"
                api_key = ""
                default_model = "qwen3:8b"
                available_models = ["qwen3:8b", "qwen3:14b"]
                """,
            )
            config = load_runtime_config(root)
            self.assertEqual(config.profile, "ollama-local")
            self.assertEqual(config.provider, "ollama")
            self.assertIsNone(config.api_key)
            self.assertEqual(config.default_model, "qwen3:8b")
            self.assertEqual(config.available_models, ("qwen3:8b", "qwen3:14b"))

    def test_reasoning_model_defaults_expand_when_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp_dir)
            _write_profiles(
                root / "llm_profiles.toml",
                """
                active_profile = "openai-main"

                [profiles.openai-main]
                provider = "openai"
                transport = "openai-compatible"
                base_url = "https://api.example.com/v1"
                api_key = "test-key"
                default_model = "deepseek-chat"
                available_models = ["deepseek-chat", "deepseek-reasoner"]
                """,
            )
            config = load_runtime_config(root).with_model("deepseek-reasoner")
            self.assertEqual(config.max_tokens, DEFAULT_RUNTIME_MAX_TOKENS)
            self.assertEqual(config.timeout_s, DEFAULT_RUNTIME_TIMEOUT_S)

    def test_non_reasoning_models_use_reduced_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp_dir)
            _write_profiles(
                root / "llm_profiles.toml",
                """
                active_profile = "openai-main"

                [profiles.openai-main]
                provider = "openai"
                transport = "openai-compatible"
                base_url = "https://api.example.com/v1"
                api_key = "test-key"
                default_model = "deepseek-chat"
                available_models = ["deepseek-chat", "claude-sonnet-4-6"]
                """,
            )
            config = load_runtime_config(root).with_model("claude-sonnet-4-6")
            self.assertEqual(config.max_tokens, DEFAULT_NON_REASONING_RUNTIME_MAX_TOKENS)
            self.assertEqual(config.timeout_s, DEFAULT_NON_REASONING_RUNTIME_TIMEOUT_S)

    def test_available_models_profile_is_normalized_and_keeps_default_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp_dir)
            _write_profiles(
                root / "llm_profiles.toml",
                """
                active_profile = "openai-main"

                [profiles.openai-main]
                provider = "openai"
                transport = "openai-compatible"
                base_url = "https://api.example.com/v1"
                api_key = "test-key"
                default_model = "deepseek-chat"
                available_models = ["kimi-k2.5", "glm-5"]
                """,
            )
            config = load_runtime_config(root)
            self.assertEqual(config.available_models, ("deepseek-chat", "kimi-k2.5", "glm-5"))
            self.assertEqual(config.with_model("glm-5").active_model, "glm-5")

    def test_invalid_base_url_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp_dir)
            _write_profiles(
                root / "llm_profiles.toml",
                """
                active_profile = "bad-profile"

                [profiles.bad-profile]
                provider = "openai"
                transport = "openai-compatible"
                base_url = "not-a-url"
                api_key = "test-key"
                default_model = "deepseek-chat"
                """,
            )
            with self.assertRaises(ConfigError):
                load_runtime_config(root)

    def test_llm_concurrency_profile_is_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp_dir)
            _write_profiles(
                root / "llm_profiles.toml",
                """
                active_profile = "openai-main"

                [profiles.openai-main]
                provider = "openai"
                transport = "openai-compatible"
                base_url = "https://api.example.com/v1"
                api_key = "test-key"
                default_model = "deepseek-chat"
                llm_concurrency = 7
                """,
            )
            config = load_runtime_config(root)
            self.assertEqual(config.llm_concurrency, 7)

    def test_runtime_knobs_can_be_set_per_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp_dir)
            _write_profiles(
                root / "llm_profiles.toml",
                """
                active_profile = "openai-main"

                [profiles.openai-main]
                provider = "openai"
                transport = "openai-compatible"
                base_url = "https://api.example.com/v1"
                api_key = "test-key"
                default_model = "deepseek-chat"
                temperature = 0.4
                max_tokens = 2048
                timeout_s = 90
                supports_tools = false
                supports_json_mode = true
                """,
            )
            config = load_runtime_config(root)
            self.assertEqual(config.temperature, 0.4)
            self.assertEqual(config.max_tokens, 2048)
            self.assertEqual(config.timeout_s, 90)
            self.assertFalse(config.supports_tools)
            self.assertTrue(config.supports_json_mode)
            self.assertEqual(config.with_model("deepseek-chat").max_tokens, 2048)
            self.assertEqual(config.with_model("deepseek-chat").timeout_s, 90)

    def test_missing_api_key_env_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp_dir)
            _write_profiles(
                root / "llm_profiles.toml",
                """
                active_profile = "openai-main"

                [profiles.openai-main]
                provider = "openai"
                transport = "openai-compatible"
                base_url = "https://api.example.com/v1"
                api_key_env = "MISSING_API_KEY"
                default_model = "deepseek-chat"
                """,
            )
            with self.assertRaises(ConfigError):
                load_runtime_config(root)


if __name__ == "__main__":
    unittest.main()
