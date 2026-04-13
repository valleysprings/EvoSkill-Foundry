from __future__ import annotations

import asyncio
from contextlib import contextmanager
import json
import threading
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

from app.codegen import catalog
from app.codegen.config import RuntimeConfig
from app.codegen.llm import ProposalRuntime

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_BENCHMARK_ROOT = ROOT / "tests" / "fixtures" / "benchmarks"
FIXTURE_REGISTRY_PATH = FIXTURE_BENCHMARK_ROOT / "registry.json"
BENCHMARK_REGISTRY_PATH = ROOT / "benchmark" / "registry.json"
PERSONALIZATION_REFERENCE_CATALOG_PATH = ROOT / "benchmark" / "personalization_verified" / "reference_benchmarks.json"


def _read_json(path: Path) -> object:
    return json.loads(path.read_text())


def enabled_registry_entries() -> list[dict[str, object]]:
    payload = _read_json(BENCHMARK_REGISTRY_PATH)
    entries = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"Expected a top-level tasks list in {BENCHMARK_REGISTRY_PATH}")
    return [dict(entry) for entry in entries if isinstance(entry, dict) and bool(entry.get("enabled", True))]


def enabled_registry_task_ids() -> list[str]:
    return [str(entry["id"]) for entry in enabled_registry_entries()]


def disabled_registry_task_ids() -> set[str]:
    payload = _read_json(BENCHMARK_REGISTRY_PATH)
    entries = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"Expected a top-level tasks list in {BENCHMARK_REGISTRY_PATH}")
    return {
        str(entry["id"])
        for entry in entries
        if isinstance(entry, dict) and not bool(entry.get("enabled", True))
    }


def runnable_personalization_reference_entries() -> list[dict[str, object]]:
    payload = _read_json(PERSONALIZATION_REFERENCE_CATALOG_PATH)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a top-level list in {PERSONALIZATION_REFERENCE_CATALOG_PATH}")
    entries: list[dict[str, object]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("status") or "").strip() != "local_task":
            continue
        if str(entry.get("implementation_status") or "").strip() != "running":
            continue
        entries.append(dict(entry))
    return entries


def runnable_personalization_reference_ids() -> set[str]:
    return {str(entry["id"]) for entry in runnable_personalization_reference_entries()}


def runnable_personalization_task_ids() -> set[str]:
    task_ids: set[str] = set()
    for entry in runnable_personalization_reference_entries():
        task_id = str(entry.get("task_id") or entry.get("id") or "").strip()
        if task_id:
            task_ids.add(task_id)
    return task_ids


def chat_response(payload: dict, *, model: str = "deepseek-chat") -> str:
    return json.dumps(
        {
            "id": "resp-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(payload),
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
            },
            "model": model,
        }
    )


class QueueTransport:
    def __init__(self, responses: Sequence[object]):
        self.responses = list(responses)
        self._lock = threading.Lock()

    async def __call__(self, _request_body, _config) -> str:
        await asyncio.sleep(0)
        with self._lock:
            if not self.responses:
                raise AssertionError("No mocked LLM responses remain.")
            response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return str(response)


def make_runtime(responses: Sequence[object], *, model: str = "deepseek-chat") -> ProposalRuntime:
    return ProposalRuntime(
        RuntimeConfig(
            profile="test-profile",
            provider="openai",
            transport="openai-compatible",
            api_key="test-key",
            base_url="https://api.test/v1",
            default_model=model,
            available_models=(
                model,
                "deepseek-reasoner",
                "gemini-3-flash-preview",
                "gpt-5.4",
                "claude-sonnet-4-6",
                "kimi-k2.5",
                "glm-5",
            ),
            temperature=0.2,
            max_tokens=4096,
            timeout_s=45,
            llm_concurrency=20,
            supports_tools=True,
            supports_json_mode=True,
        ),
        transport=QueueTransport(responses),
    )


@contextmanager
def patched_fixture_catalog():
    with (
        patch.object(catalog, "BENCHMARK_ROOT", FIXTURE_BENCHMARK_ROOT),
        patch.object(catalog, "REGISTRY_PATH", FIXTURE_REGISTRY_PATH),
    ):
        yield


def load_fixture_codegen_tasks(
    task_id: str | None = None,
    *,
    included_in_main_comparison: bool | None = None,
) -> list[dict[str, object]]:
    with patched_fixture_catalog():
        return catalog.load_codegen_tasks(
            task_id,
            included_in_main_comparison=included_in_main_comparison,
        )


def list_fixture_codegen_task_summaries() -> list[dict[str, object]]:
    with patched_fixture_catalog():
        return catalog.list_codegen_task_summaries()


@contextmanager
def patch_runner_fixture_catalog():
    with (
        patch("app.entries.runner.load_codegen_tasks", side_effect=load_fixture_codegen_tasks),
        patch("app.entries.runner.list_codegen_task_summaries", side_effect=list_fixture_codegen_task_summaries),
    ):
        yield
