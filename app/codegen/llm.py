from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import textwrap
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
from app.configs.prompts import (
    CANDIDATE_LABEL_LIMIT,
    PROPOSAL_CANDIDATE_COUNT_TEMPLATE,
    PROPOSAL_CONCISE_FIELDS_INSTRUCTION,
    CANDIDATE_RESPONSE_REQUIRED_FIELDS,
    FAILURE_REFLECTION_OUTCOME_INSTRUCTIONS,
    FAILURE_REFLECTION_SYSTEM_PROMPT,
    MODEL_COMPLETION_MAX_ATTEMPTS,
    PROPOSAL_JSON_ONLY_INSTRUCTION,
    PROPOSAL_RESULT_INSTRUCTION,
    PROPOSAL_SYSTEM_PROMPT,
    RAW_PREVIEW_LIMIT,
    REFLECTION_FIELD_LIMIT,
    REFLECTION_FRAGMENT_INSTRUCTION,
    REFLECTION_OPTIONAL_FIELDS,
    REFLECTION_REQUIRED_FIELDS,
    REQUEST_PREVIEW_LIMIT,
    SUCCESS_REFLECTION_OUTCOME_INSTRUCTIONS,
    SUCCESS_REFLECTION_SYSTEM_PROMPT,
    TRIM_DEFAULT_LIMIT,
)
from app.codegen.selection import prompt_summary
from app.codegen.task_contracts import (
    infer_interaction_mode,
    infer_task_mode,
    interaction_mode_summary,
    task_mode_summary,
)
from app.configs.codegen import PROPOSAL_SELECTION_GUIDANCE
from app.codegen.config import ROOT, RuntimeConfig, load_runtime_config
from app.codegen.errors import LlmResponseError, LlmTransportError


Transport = Callable[[dict[str, Any], RuntimeConfig], Awaitable[str]]
RetryProgressCallback = Callable[[dict[str, Any]], None]
_TRANSPORT_GATE_LOCK = threading.Lock()
_TRANSPORT_DISPATCHERS: dict[tuple[str, int], "_TransportDispatcher"] = {}


class _TransportDispatcher:
    def __init__(self, *, base_url: str, max_workers: int) -> None:
        self._base_url = base_url
        self._max_workers = max_workers
        self._lock = threading.Lock()
        self._sequence = 0
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.PriorityQueue[tuple[int, int, Transport | None, dict[str, Any], RuntimeConfig, Future[str]]] | None = None
        self._client: httpx.AsyncClient | None = None
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run_event_loop,
            name=f"autoresearch-llm-dispatcher-{max_workers}",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()

    def submit(
        self,
        *,
        priority: int,
        sender: Transport | None,
        request_body: dict[str, Any],
        config: RuntimeConfig,
    ) -> Future[str]:
        future: Future[str] = Future()
        if self._startup_error is not None:
            raise RuntimeError(f"Failed to initialize async LLM transport dispatcher: {self._startup_error}") from self._startup_error
        with self._lock:
            sequence = self._sequence
            self._sequence += 1
        assert self._loop is not None
        assert self._queue is not None
        self._loop.call_soon_threadsafe(
            self._queue.put_nowait,
            (priority, sequence, sender, request_body, config, future),
        )
        return future

    def _run_event_loop(self) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._queue = asyncio.PriorityQueue()
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                limits=httpx.Limits(
                    max_connections=self._max_workers,
                    max_keepalive_connections=self._max_workers,
                ),
                timeout=None,
                trust_env=False,
            )
            for index in range(self._max_workers):
                loop.create_task(self._worker(index))
        except BaseException as exc:  # noqa: BLE001
            self._startup_error = exc
        finally:
            self._ready.set()
        if self._startup_error is None:
            assert self._loop is not None
            self._loop.run_forever()

    async def _worker(self, _index: int) -> None:
        assert self._queue is not None
        while True:
            _priority, _sequence, sender, request_body, config, future = await self._queue.get()
            try:
                if future.set_running_or_notify_cancel():
                    try:
                        if sender is None:
                            future.set_result(await self._default_transport(request_body, config))
                        else:
                            future.set_result(await sender(request_body, config))
                    except BaseException as exc:  # noqa: BLE001
                        future.set_exception(exc)
            finally:
                self._queue.task_done()

    async def _default_transport(self, request_body: dict[str, Any], config: RuntimeConfig) -> str:
        assert self._client is not None
        headers = {
            "Content-Type": "application/json",
        }
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        try:
            response = await self._client.post(
                "/chat/completions",
                json=request_body,
                headers=headers,
                timeout=config.timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise LlmTransportError("Model request timed out.", model=config.active_model) from exc
        except httpx.TransportError as exc:
            raise LlmTransportError(f"Model request failed: {exc}", model=config.active_model) from exc
        body = response.text
        if response.is_error:
            raise LlmTransportError(
                f"Model request failed with HTTP {response.status_code}: {body[:240]}",
                model=config.active_model,
            )
        return body


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return json.dumps(content)


def _normalize_tool_choice(tool_choice: str | dict[str, Any]) -> str | dict[str, Any]:
    if isinstance(tool_choice, str):
        normalized = tool_choice.strip() or "auto"
        if normalized not in {"none", "auto", "required"}:
            raise ValueError(f"Unsupported tool_choice={normalized!r}.")
        return normalized
    if isinstance(tool_choice, dict):
        return dict(tool_choice)
    raise ValueError("tool_choice must be a string or dict.")


def _normalize_tool_call_payloads(raw_tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_tool_calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw_tool_calls:
        if not isinstance(item, dict):
            continue
        function_payload = item.get("function")
        if not isinstance(function_payload, dict):
            continue
        name = str(function_payload.get("name") or "").strip()
        if not name:
            continue
        raw_arguments = function_payload.get("arguments")
        if isinstance(raw_arguments, dict):
            arguments = dict(raw_arguments)
        elif isinstance(raw_arguments, str):
            try:
                parsed_arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {"_raw": raw_arguments}
            else:
                arguments = dict(parsed_arguments) if isinstance(parsed_arguments, dict) else {"_raw": raw_arguments}
        else:
            arguments = {}
        normalized_item = {
            "name": name,
            "arguments": arguments,
        }
        identifier = item.get("id")
        if isinstance(identifier, str) and identifier.strip():
            normalized_item["id"] = identifier.strip()
        normalized.append(normalized_item)
    return normalized


def _transport_dispatcher(config: RuntimeConfig) -> _TransportDispatcher:
    key = (config.base_url, config.llm_concurrency)
    with _TRANSPORT_GATE_LOCK:
        dispatcher = _TRANSPORT_DISPATCHERS.get(key)
        if dispatcher is None:
            dispatcher = _TransportDispatcher(base_url=config.base_url, max_workers=config.llm_concurrency)
            _TRANSPORT_DISPATCHERS[key] = dispatcher
        return dispatcher


def _proposal_queue_priority(generation: int) -> int:
    return max(1, generation) * 10


def _reflection_queue_priority(generation: int) -> int:
    return _proposal_queue_priority(generation) + 5


def _extract_json_object(text: str) -> dict[str, Any]:
    normalized = text.strip()
    if not normalized:
        raise LlmResponseError("Model response was empty.")
    try:
        parsed = json.loads(normalized)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", normalized, flags=re.DOTALL)
    for candidate in fenced:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    for candidate in _balanced_json_objects(normalized):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise LlmResponseError("Model response did not contain a valid JSON object.")


def _balanced_json_objects(text: str) -> list[str]:
    candidates: list[str] = []
    start_index: int | None = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if start_index is None:
            if char == "{":
                start_index = index
                depth = 1
                in_string = False
                escaped = False
            continue

        if escaped:
            escaped = False
            continue
        if in_string:
            if char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                candidates.append(text[start_index : index + 1])
                start_index = None
    return candidates


def _trim(value: Any, *, limit: int = TRIM_DEFAULT_LIMIT) -> str:
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _proposal_system_prompt() -> str:
    return " ".join(
        (
            PROPOSAL_SYSTEM_PROMPT,
            PROPOSAL_JSON_ONLY_INSTRUCTION,
            PROPOSAL_CANDIDATE_COUNT_TEMPLATE,
            PROPOSAL_CONCISE_FIELDS_INSTRUCTION,
            "file_body must contain the full contents of the editable file and must preserve the declared entry symbol and task contract.",
        )
    )


def _proposal_contract_lines(task: dict[str, Any]) -> tuple[str, str]:
    task_mode = infer_task_mode(task)
    interaction_mode = infer_interaction_mode(task)
    return (
        f"Task mode: {task_mode} ({task_mode_summary(task_mode)})",
        f"Interaction mode: {interaction_mode} ({interaction_mode_summary(interaction_mode)})",
    )


def _reflection_interaction_guidance(task: dict[str, Any]) -> str:
    if infer_interaction_mode(task) != "multi_turn":
        return ""
    return (
        "For this multi-turn agent task, focus the reflection on reusable interaction failures or wins such as repeated loops, "
        "invalid actions, lost state updates, premature completion, weak search/navigation choices, or poor use of observations."
    )


def _looks_like_truncated_response(text: str, completion_tokens: Any, max_tokens: int) -> bool:
    completion_hit_limit = isinstance(completion_tokens, int) and completion_tokens >= max_tokens
    normalized = text.rstrip()
    if not normalized:
        return completion_hit_limit
    incomplete_tail = (
        normalized.startswith("```")
        and not normalized.endswith("```")
        or normalized.count("{") > normalized.count("}")
        or normalized.count("[") > normalized.count("]")
        or normalized.endswith("\\")
    )
    return completion_hit_limit or incomplete_tail


def _parse_failure_error(
    *,
    purpose: str,
    runtime: "ProposalRuntime",
    messages: list[dict[str, str]],
    usage: dict[str, Any],
    text: str,
    attempt: int,
    fallback_message: str,
) -> LlmResponseError:
    completion_tokens = usage.get("completion_tokens")
    response_truncated = _looks_like_truncated_response(text, completion_tokens, runtime.config.max_tokens)
    details = {
        "purpose": purpose,
        "selected_model": runtime.active_model,
        "parse_status": "truncated" if response_truncated else "invalid_json",
        "base_url": runtime.config.base_url,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "max_tokens": runtime.config.max_tokens,
        "request_preview": _request_preview(messages),
        "raw_preview": _trim(text, limit=RAW_PREVIEW_LIMIT),
        "attempt": attempt,
        "response_truncated": response_truncated,
    }
    if response_truncated:
        message = (
            "Model response appears truncated before completing a valid JSON object "
            f"(completion_tokens={completion_tokens}, max_tokens={runtime.config.max_tokens})."
        )
    else:
        message = fallback_message
    return LlmResponseError(message, model=runtime.active_model, details=details)


def _transport_failure_error(
    *,
    purpose: str,
    runtime: "ProposalRuntime",
    messages: list[dict[str, str]],
    attempt: int,
    exc: Exception,
) -> LlmTransportError:
    details = {
        "purpose": purpose,
        "selected_model": runtime.active_model,
        "parse_status": "transport_error",
        "base_url": runtime.config.base_url,
        "request_preview": _request_preview(messages),
        "attempt": attempt,
        "max_attempts": MODEL_COMPLETION_MAX_ATTEMPTS,
    }
    if isinstance(exc, LlmTransportError) and exc.details is not None:
        details.update(exc.details)
    return LlmTransportError(str(exc), model=runtime.active_model, details=details)


def _response_envelope_error(
    *,
    purpose: str,
    runtime: "ProposalRuntime",
    messages: list[dict[str, str]],
    raw_response: str,
    attempt: int,
    message: str,
) -> LlmResponseError:
    return LlmResponseError(
        message,
        model=runtime.active_model,
        details={
            "purpose": purpose,
            "selected_model": runtime.active_model,
            "parse_status": "invalid_http_json",
            "base_url": runtime.config.base_url,
            "request_preview": _request_preview(messages),
            "raw_preview": _trim(raw_response, limit=RAW_PREVIEW_LIMIT),
            "attempt": attempt,
        },
    )


def _emit_retry_progress(
    *,
    progress_callback: RetryProgressCallback | None,
    purpose: str,
    runtime: "ProposalRuntime",
    attempt: int,
    parse_status: str,
) -> None:
    if progress_callback is None or attempt >= MODEL_COMPLETION_MAX_ATTEMPTS:
        return
    next_attempt = attempt + 1
    parse_status_label = "connection error" if parse_status == "transport_error" else parse_status.replace("_", " ")
    progress_callback(
        {
            "phase": "llm_retry",
            "purpose": purpose,
            "selected_model": runtime.active_model,
            "parse_status": parse_status,
            "retry_attempt": next_attempt,
            "max_attempts": MODEL_COMPLETION_MAX_ATTEMPTS,
            "message": (
                f"Retrying {purpose} with {runtime.active_model} "
                f"(attempt {next_attempt}/{MODEL_COMPLETION_MAX_ATTEMPTS}) after {parse_status_label}."
            ),
        }
    )


def _normalize_imports(raw_imports: Any) -> list[str]:
    if raw_imports is None:
        return []
    if not isinstance(raw_imports, list):
        raise LlmResponseError("Candidate imports must be a list of strings.")
    imports: list[str] = []
    for item in raw_imports:
        if not isinstance(item, str):
            raise LlmResponseError("Candidate imports must contain only strings.")
        stripped = item.strip()
        if stripped:
            imports.append(stripped)
    return list(dict.fromkeys(imports))


def _normalize_candidate_payload(payload: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    missing = [
        key for key in CANDIDATE_RESPONSE_REQUIRED_FIELDS if not isinstance(payload.get(key), str) or not payload.get(key).strip()
    ]
    if missing:
        raise LlmResponseError(
            f"Candidate response is missing required string fields: {', '.join(missing)}.",
            model=trace.get("selected_model"),
        )
    file_body = str(payload["file_body"]).strip("\n")
    if not file_body.strip():
        raise LlmResponseError("Candidate must return a non-empty editable file.", model=trace.get("selected_model"))
    return {
        "agent": "candidate",
        "label": _trim(payload["name"], limit=CANDIDATE_LABEL_LIMIT),
        "strategy": _trim(payload["strategy"]),
        "rationale": _trim(payload["rationale"]),
        "imports": _normalize_imports(payload.get("imports")),
        "file_body": file_body,
        "candidate_summary": _trim(payload["candidate_summary"]),
        "run_mode": "llm-required",
        "proposal_model": trace.get("selected_model"),
    }


def _normalize_reflection_payload(payload: dict[str, Any], trace: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for field in REFLECTION_REQUIRED_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise LlmResponseError(f"Reflection response is missing required field {field}.", model=trace.get("selected_model"))
        normalized[field] = _trim(value, limit=REFLECTION_FIELD_LIMIT)
    for field in REFLECTION_OPTIONAL_FIELDS:
        value = payload.get(field)
        normalized[field] = _trim(value, limit=REFLECTION_FIELD_LIMIT) if isinstance(value, str) and value.strip() else ""
    return normalized


def _tool_call_digest(tool_call: dict[str, Any]) -> str:
    name = str(tool_call.get("name") or "").strip() or "unknown"
    arguments = dict(tool_call.get("arguments") or {})
    command = str(arguments.get("command") or arguments.get("message") or "").strip()
    return f"{name}({command})" if command else name


def _multi_turn_process_feedback(metrics: dict[str, Any]) -> str:
    item_runs = list(metrics.get("item_runs") or [])
    if not item_runs:
        return ""
    item_lines: list[str] = []
    for item_run in item_runs[:2]:
        turns = list(item_run.get("turns") or [])
        turn_lines: list[str] = []
        for turn in turns[:4]:
            action = dict(turn.get("action") or {})
            tool_calls = list(action.get("tool_calls") or [])
            tool_desc = ", ".join(_tool_call_digest(call) for call in tool_calls[:2]) or "no-tool-call"
            tool_errors = sum(1 for result in list(turn.get("tool_results") or []) if bool(result.get("error")))
            turn_lines.append(
                f"t{int(turn.get('turn_index') or 0)}:{tool_desc};done={bool(action.get('done'))};tool_errors={tool_errors}"
            )
        item_lines.append(
            " | ".join(
                [
                    f"item={str(item_run.get('item_id') or 'unknown')}",
                    f"success={bool(item_run.get('success'))}",
                    f"reward={item_run.get('reward')}",
                    f"turns={len(turns)}",
                    *turn_lines,
                ]
            )
        )
    return _trim(" || ".join(item_lines), limit=1200)


def _request_preview(messages: list[dict[str, str]]) -> str:
    user_messages = [message["content"] for message in messages if message["role"] == "user"]
    if not user_messages:
        return ""
    return _trim(user_messages[-1], limit=REQUEST_PREVIEW_LIMIT)


@dataclass(slots=True)
class ProposalRuntime:
    config: RuntimeConfig
    transport: Transport | None = None

    @classmethod
    def from_env(cls, root: Path | None = None) -> "ProposalRuntime":
        return cls(load_runtime_config(root or ROOT))

    @property
    def active_model(self) -> str:
        return self.config.active_model

    def with_model(self, model: str | None) -> "ProposalRuntime":
        return ProposalRuntime(config=self.config.with_model(model), transport=self.transport)

    def with_llm_concurrency(self, llm_concurrency: int | None) -> "ProposalRuntime":
        return ProposalRuntime(config=self.config.with_llm_concurrency(llm_concurrency), transport=self.transport)

    def describe(self) -> dict[str, object]:
        return self.config.describe()

    def chat(
        self,
        *,
        purpose: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        queue_priority: int = 1000,
        progress_callback: RetryProgressCallback | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        normalized_messages = [{"role": str(message.get("role") or ""), "content": message.get("content")} for message in messages]
        request_body: dict[str, Any] = {
            "model": self.active_model,
            "messages": normalized_messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            request_body["tools"] = list(tools)
            request_body["tool_choice"] = _normalize_tool_choice(tool_choice)
        sender = self.transport
        last_parse_error: LlmResponseError | None = None
        last_transport_error: LlmTransportError | None = None
        max_response_tokens = int(request_body["max_tokens"])
        for attempt in range(1, MODEL_COMPLETION_MAX_ATTEMPTS + 1):
            try:
                raw_response = _transport_dispatcher(self.config).submit(
                    priority=queue_priority,
                    sender=sender,
                    request_body=request_body,
                    config=self.config,
                ).result()
            except LlmTransportError as exc:
                last_transport_error = _transport_failure_error(
                    purpose=purpose,
                    runtime=self,
                    messages=normalized_messages,
                    attempt=attempt,
                    exc=exc,
                )
                if attempt >= MODEL_COMPLETION_MAX_ATTEMPTS:
                    raise last_transport_error from exc
                _emit_retry_progress(
                    progress_callback=progress_callback,
                    purpose=purpose,
                    runtime=self,
                    attempt=attempt,
                    parse_status="transport_error",
                )
                time.sleep(min(2**(attempt - 1), 5))
                continue
            except TimeoutError as exc:
                last_transport_error = _transport_failure_error(
                    purpose=purpose,
                    runtime=self,
                    messages=normalized_messages,
                    attempt=attempt,
                    exc=LlmTransportError("Model request timed out.", model=self.active_model),
                )
                if attempt >= MODEL_COMPLETION_MAX_ATTEMPTS:
                    raise last_transport_error from exc
                _emit_retry_progress(
                    progress_callback=progress_callback,
                    purpose=purpose,
                    runtime=self,
                    attempt=attempt,
                    parse_status="transport_error",
                )
                time.sleep(min(2**(attempt - 1), 5))
                continue
            try:
                parsed_response = json.loads(raw_response)
            except json.JSONDecodeError as exc:
                last_parse_error = _response_envelope_error(
                    purpose=purpose,
                    runtime=self,
                    messages=normalized_messages,
                    raw_response=raw_response,
                    attempt=attempt,
                    message="Model HTTP response was not valid JSON.",
                )
                if attempt >= MODEL_COMPLETION_MAX_ATTEMPTS:
                    raise last_parse_error from exc
                _emit_retry_progress(
                    progress_callback=progress_callback,
                    purpose=purpose,
                    runtime=self,
                    attempt=attempt,
                    parse_status="invalid_http_json",
                )
                continue

            try:
                response_message = parsed_response["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as exc:
                last_parse_error = _response_envelope_error(
                    purpose=purpose,
                    runtime=self,
                    messages=normalized_messages,
                    raw_response=raw_response,
                    attempt=attempt,
                    message="Model response did not contain choices[0].message.",
                )
                if attempt >= MODEL_COMPLETION_MAX_ATTEMPTS:
                    raise last_parse_error from exc
                _emit_retry_progress(
                    progress_callback=progress_callback,
                    purpose=purpose,
                    runtime=self,
                    attempt=attempt,
                    parse_status="invalid_http_json",
                )
                continue

            text = _message_text(response_message.get("content"))
            usage = parsed_response.get("usage", {})
            trace = {
                "purpose": purpose,
                "selected_model": self.active_model,
                "parse_status": "ok",
                "base_url": self.config.base_url,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "max_tokens": max_response_tokens,
                "request_preview": _request_preview(
                    [
                        {"role": str(item.get("role") or ""), "content": _message_text(item.get("content"))}
                        for item in normalized_messages
                    ]
                ),
                "raw_preview": _trim(text, limit=RAW_PREVIEW_LIMIT),
                "attempt": attempt,
            }
            return (
                {
                    "message": text.strip() or None,
                    "tool_calls": _normalize_tool_call_payloads(response_message.get("tool_calls")),
                    "raw": parsed_response,
                },
                trace,
            )
        if last_transport_error is not None:
            raise last_transport_error
        if last_parse_error is not None:
            raise last_parse_error
        raise LlmResponseError("Model response did not contain a valid chat message.", model=self.active_model)

    def complete_json(
        self,
        *,
        purpose: str,
        system_prompt: str,
        user_prompt: str,
        queue_priority: int = 1000,
        progress_callback: RetryProgressCallback | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_parse_error: LlmResponseError | None = None
        for attempt in range(1, MODEL_COMPLETION_MAX_ATTEMPTS + 1):
            response, trace = self.chat(
                purpose=purpose,
                messages=messages,
                queue_priority=queue_priority,
                progress_callback=progress_callback,
            )
            text = str(response.get("message") or "")
            usage = dict((response.get("raw") or {}).get("usage") or {})
            try:
                payload = _extract_json_object(text)
            except LlmResponseError as exc:
                last_parse_error = _parse_failure_error(
                    purpose=purpose,
                    runtime=self,
                    messages=messages,
                    usage=usage,
                    text=text,
                    attempt=attempt,
                    fallback_message=str(exc),
                )
                if attempt >= MODEL_COMPLETION_MAX_ATTEMPTS:
                    break
                _emit_retry_progress(
                    progress_callback=progress_callback,
                    purpose=purpose,
                    runtime=self,
                    attempt=attempt,
                    parse_status=str(last_parse_error.details.get("parse_status") or "invalid_json"),
                )
                continue
            trace["raw_preview"] = _trim(text, limit=RAW_PREVIEW_LIMIT)
            trace["attempt"] = max(int(trace.get("attempt") or 0), attempt)
            return payload, trace
        if last_parse_error is not None:
            raise last_parse_error
        raise LlmResponseError("Model response did not contain a valid JSON object.", model=self.active_model)


def _proposal_prompt(
    *,
    task: dict[str, Any],
    generation: int,
    parent_candidate: dict[str, Any],
    current_best: dict[str, Any],
    candidate_history: list[dict[str, Any]],
    memories: list[dict[str, Any]],
) -> tuple[str, str]:
    objective_spec = task.get("objective_spec") or {}
    selection_spec = task.get("selection_spec") or {}
    objective_name = objective_spec.get("display_name") or task.get("objective_label") or "objective"
    objective_direction = objective_spec.get("direction") or task.get("objective_direction") or "max"
    objective_formula = objective_spec.get("formula") or task.get("objective_label") or "objective"
    objective_summary = objective_spec.get("summary_template") or ""
    selection_summary = prompt_summary(selection_spec)
    memory_lines = [
        "- "
        + json.dumps(
            {
                "experience_id": memory.get("experience_id"),
                "experience_outcome": memory.get("experience_outcome", "success"),
                "verifier_status": memory.get("verifier_status"),
                "rejection_reason": memory.get("rejection_reason"),
                "failure_pattern": memory.get("failure_pattern"),
                "strategy_hypothesis": memory.get("strategy_hypothesis"),
                "successful_strategy": memory.get("successful_strategy"),
                "prompt_fragment": memory.get("prompt_fragment"),
                "candidate_summary": memory.get("candidate_summary"),
                "delta_primary_score": memory.get("delta_primary_score"),
                "knowledge_scope": memory.get("knowledge_scope", "episode_strategy"),
                "distilled_skill": memory.get("distilled_skill", ""),
                "applicability_notes": memory.get("applicability_notes", ""),
                "source_dataset_ids": memory.get("source_dataset_ids", []),
                "process_failure_mode": memory.get("process_failure_mode", ""),
                "process_repair_hint": memory.get("process_repair_hint", ""),
                "process_trace_summary": memory.get("process_trace_summary", ""),
            }
        )
        for memory in memories
    ]
    task_mode_line, interaction_mode_line = _proposal_contract_lines(task)
    system_prompt = _proposal_system_prompt()
    leakage_free = bool(task.get("leakage_free"))
    if leakage_free:
        history_lines = [
            "- "
            + json.dumps(
                {
                    "generation": item.get("generation"),
                    "candidate": item.get("agent"),
                    "self_critique_score": item.get("metrics", {}).get("self_critique_score"),
                    "primary_score": item.get("metrics", {}).get("primary_score"),
                    "tie_break_score": item.get("metrics", {}).get("tie_break_score"),
                    "gate_passed": item.get("metrics", {}).get("gate_passed"),
                    "selection_status": item.get("metrics", {}).get("selection_verifier_status"),
                    "candidate_summary": item.get("candidate_summary"),
                    "strategy": item.get("strategy"),
                }
            )
            for item in candidate_history[-6:]
        ]
    else:
        history_lines = [
            "- "
            + json.dumps(
                {
                    "generation": item.get("generation"),
                    "candidate": item.get("agent"),
                    "objective": item.get("metrics", {}).get("objective"),
                    "primary_score": item.get("metrics", {}).get("primary_score"),
                    "tie_break_score": item.get("metrics", {}).get("tie_break_score"),
                    "gate_passed": item.get("metrics", {}).get("gate_passed"),
                    "status": item.get("metrics", {}).get("status"),
                    "candidate_summary": item.get("candidate_summary"),
                    "strategy": item.get("strategy"),
                }
            )
            for item in candidate_history[-6:]
        ]
    if leakage_free:
        # In leakage-free mode expose only the hidden selection signal, never the ground-truth objective.
        parent_score_lines = (
            f"Selected parent self_critique_score: {parent_candidate['metrics'].get('self_critique_score')}\n"
            f"Selected parent primary_score: {parent_candidate['metrics'].get('primary_score')}\n"
            f"Selected parent gate_passed: {parent_candidate['metrics'].get('gate_passed')}\n"
        )
        best_score_lines = (
            f"Global best self_critique_score: {current_best['metrics'].get('self_critique_score')}\n"
            f"Global best primary_score: {current_best['metrics'].get('primary_score')}\n"
            f"Global best gate_passed: {current_best['metrics'].get('gate_passed')}\n"
        )
    else:
        parent_score_lines = (
            f"Selected parent objective: {parent_candidate['metrics']['objective']}\n"
            f"Selected parent objective_score: {parent_candidate['metrics'].get('objective_score')}\n"
            f"Selected parent primary_score: {parent_candidate['metrics'].get('primary_score')}\n"
            f"Selected parent tie_break_score: {parent_candidate['metrics'].get('tie_break_score')}\n"
            f"Selected parent gate_passed: {parent_candidate['metrics'].get('gate_passed')}\n"
        )
        best_score_lines = (
            f"Global best objective: {current_best['metrics']['objective']}\n"
            f"Global best objective_score: {current_best['metrics'].get('objective_score')}\n"
            f"Global best primary_score: {current_best['metrics'].get('primary_score')}\n"
            f"Global best tie_break_score: {current_best['metrics'].get('tie_break_score')}\n"
            f"Global best gate_passed: {current_best['metrics'].get('gate_passed')}\n"
        )
    user_prompt = (
        f"Task id: {task['id']}\n"
        f"Title: {task['title']}\n"
        f"Description: {task['description']}\n"
        f"Benchmark tier: {task['benchmark_tier']}\n"
        f"Track: {task['track']}\n"
        f"Dataset id: {task['dataset_id']}\n"
        f"Editable file: {task['editable_file']}\n"
        f"Entry symbol: {task['entry_symbol']}\n"
        f"{task_mode_line}\n"
        f"{interaction_mode_line}\n"
        f"Objective: {objective_name}\n"
        f"Objective direction: {objective_direction}\n"
        f"Objective formula: {objective_formula}\n"
        f"Objective summary: {objective_summary}\n"
        f"{PROPOSAL_SELECTION_GUIDANCE}\n"
        f"{selection_summary}\n"
        f"Prompt context: {task.get('prompt_context') or 'n/a'}\n"
        f"Generation: {generation}\n"
        f"Selected parent summary: {parent_candidate['candidate_summary']}\n"
        f"{parent_score_lines}"
        f"Global best summary: {current_best['candidate_summary']}\n"
        f"{best_score_lines}"
        "Baseline source:\n"
        f"{current_best['baseline_source']}\n"
        "Selected parent editable file:\n"
        f"{parent_candidate['source_code']}\n"
        "Global best editable file:\n"
        f"{current_best['source_code']}\n"
        "Retrieved strategy experiences (successful wins and failed attempts to avoid):\n"
        + ("\n".join(memory_lines) if memory_lines else "- none")
        + "\nPrevious candidate summaries:\n"
        + ("\n".join(history_lines) if history_lines else "- none")
        + f"\n{PROPOSAL_RESULT_INSTRUCTION}"
    )
    return system_prompt, user_prompt


def propose_code_candidate(
    runtime: ProposalRuntime,
    *,
    task: dict[str, Any],
    generation: int,
    parent_candidate: dict[str, Any],
    current_best: dict[str, Any],
    candidate_history: list[dict[str, Any]],
    memories: list[dict[str, Any]],
    progress_callback: RetryProgressCallback | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    system_prompt, user_prompt = _proposal_prompt(
        task=task,
        generation=generation,
        parent_candidate=parent_candidate,
        current_best=current_best,
        candidate_history=candidate_history,
        memories=memories,
    )
    payload, trace = runtime.complete_json(
        purpose="generation_proposals",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        queue_priority=_proposal_queue_priority(generation),
        progress_callback=progress_callback,
    )
    candidate = _normalize_candidate_payload(payload, trace)
    trace["candidate_count"] = 1
    return candidate, trace


def reflect_strategy_experience(
    runtime: ProposalRuntime,
    *,
    task: dict[str, Any],
    generation: int,
    previous_best: dict[str, Any],
    winner: dict[str, Any],
    delta_primary_score: float,
    outcome: str,
    rejection_reason: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    if outcome == "success":
        system_prompt = SUCCESS_REFLECTION_SYSTEM_PROMPT
        outcome_instructions = SUCCESS_REFLECTION_OUTCOME_INSTRUCTIONS
    else:
        system_prompt = FAILURE_REFLECTION_SYSTEM_PROMPT
        outcome_instructions = FAILURE_REFLECTION_OUTCOME_INSTRUCTIONS
    failed_tests = [result["name"] for result in winner["metrics"].get("test_results", []) if not result.get("passed")]
    process_trace_summary = _multi_turn_process_feedback(winner["metrics"])
    leakage_free = bool(task.get("leakage_free"))
    if leakage_free:
        # No ground truth signals: omit objective scores, failed test names, delta
        user_prompt = (
            f"Task id: {task['id']}\n"
            f"Generation: {generation}\n"
            f"Outcome: {outcome}\n"
            f"Interaction mode: {infer_interaction_mode(task)}\n"
            f"Previous best summary: {previous_best['candidate_summary']}\n"
            f"Winner summary: {winner['candidate_summary']}\n"
            f"Winner strategy: {winner['strategy']}\n"
            f"Winner rationale: {winner['rationale']}\n"
            f"Winner self_critique_score: {winner['metrics'].get('self_critique_score')}\n"
            f"Winner primary_score: {winner['metrics'].get('primary_score')}\n"
            f"Winner error: {winner['metrics'].get('error')}\n"
            f"Rejection reason: {rejection_reason or 'n/a'}\n"
            f"Winner process trace summary: {process_trace_summary or 'n/a'}\n"
            f"{_reflection_interaction_guidance(task)}\n"
            f"{outcome_instructions}\n"
            f"{REFLECTION_FRAGMENT_INSTRUCTION}"
        )
    else:
        user_prompt = (
            f"Task id: {task['id']}\n"
            f"Generation: {generation}\n"
            f"Outcome: {outcome}\n"
            f"Interaction mode: {infer_interaction_mode(task)}\n"
            f"Previous best summary: {previous_best['candidate_summary']}\n"
            f"Previous best objective: {previous_best['metrics']['objective']}\n"
            f"Winner summary: {winner['candidate_summary']}\n"
            f"Winner strategy: {winner['strategy']}\n"
            f"Winner rationale: {winner['rationale']}\n"
            f"Winner verifier_status: {winner['metrics']['verifier_status']}\n"
            f"Winner objective: {winner['metrics']['objective']}\n"
            f"Winner primary_score: {winner['metrics'].get('primary_score')}\n"
            f"Winner tie_break_score: {winner['metrics'].get('tie_break_score')}\n"
            f"Winner error: {winner['metrics'].get('error')}\n"
            f"Failed tests: {json.dumps(failed_tests)}\n"
            f"Rejection reason: {rejection_reason or 'n/a'}\n"
            f"delta_primary_score: {delta_primary_score}\n"
            f"Winner process trace summary: {process_trace_summary or 'n/a'}\n"
            f"{_reflection_interaction_guidance(task)}\n"
            f"{outcome_instructions}\n"
            f"{REFLECTION_FRAGMENT_INSTRUCTION}"
        )
    payload, trace = runtime.complete_json(
        purpose="memory_reflection",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        queue_priority=_reflection_queue_priority(generation),
    )
    return _normalize_reflection_payload(payload, trace), trace
