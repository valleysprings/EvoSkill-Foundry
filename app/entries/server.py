from __future__ import annotations

import argparse
import errno
import json
import multiprocessing
import os
import queue as queue_module
import signal
import socket
import subprocess
import threading
import time
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse

from app.codegen.catalog import list_codegen_task_summaries, list_missing_local_dataset_warnings
from app.codegen.errors import AutoresearchError, ConfigError
from app.codegen.llm import ProposalRuntime
from app.configs.prompts import MODEL_COMPLETION_MAX_ATTEMPTS
from app.entries.runner import ROOT, load_cached_discrete_payload, write_discrete_artifacts
from app.bench.personalization_references import load_personalization_reference_benchmarks
from app.memory.skills import annotate_task_catalog_with_skills


UI_DIR = ROOT / "ui" / "dist"
JOB_LOCK = threading.Lock()
JOBS: dict[str, dict[str, object]] = {}
PORT_CONFLICT_MODES = ("auto", "next", "kill", "error")
JOB_PROCESS_START_METHOD = os.getenv("AUTORESEARCH_JOB_PROCESS_START_METHOD", "spawn").strip().lower() or "spawn"
DEFAULT_JOB_STALL_TIMEOUT_S = 180.0
JOB_STALL_TIMEOUT_S = DEFAULT_JOB_STALL_TIMEOUT_S
JOB_STALL_PROGRESS_GRACE_S = 60.0
JOB_STALL_REASONER_EXTRA_S = 120.0
QUIET_ACCESS_LOG_PATHS = frozenset({"/api/latest-run", "/api/job", "/api/health"})


def _runtime_for_request(model: str | None = None, llm_concurrency: int | None = None) -> ProposalRuntime:
    runtime = ProposalRuntime.from_env()
    runtime = runtime.with_llm_concurrency(llm_concurrency)
    return runtime.with_model(model)


def _effective_llm_concurrency(llm_concurrency: int | None, item_workers: int | None) -> int | None:
    if llm_concurrency is not None:
        return llm_concurrency
    if item_workers is not None:
        return item_workers
    return None


def _retry_backoff_budget_s() -> float:
    return float(sum(min(2 ** (attempt - 1), 5) for attempt in range(1, MODEL_COMPLETION_MAX_ATTEMPTS)))


def _job_stall_timeout_s(proposal_runtime: ProposalRuntime) -> float:
    configured_timeout = float(JOB_STALL_TIMEOUT_S)
    if configured_timeout != DEFAULT_JOB_STALL_TIMEOUT_S:
        return configured_timeout

    derived_timeout = (
        float(proposal_runtime.config.timeout_s) * MODEL_COMPLETION_MAX_ATTEMPTS
        + _retry_backoff_budget_s()
        + JOB_STALL_PROGRESS_GRACE_S
    )
    if "reasoner" in proposal_runtime.active_model.lower():
        derived_timeout += JOB_STALL_REASONER_EXTRA_S
    return max(configured_timeout, derived_timeout)


def _json_response(handler: SimpleHTTPRequestHandler, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store, max-age=0")
    handler.send_header("Pragma", "no-cache")
    handler.send_header("Expires", "0")
    handler.end_headers()
    handler.wfile.write(body)


def _error_payload(exc: Exception) -> dict[str, object]:
    if isinstance(exc, AutoresearchError):
        return exc.as_payload()
    return {
        "terminal": True,
        "error_type": "runtime_error",
        "error": str(exc),
        "model": None,
    }


def _parse_positive_int(raw_value: str | None, field: str) -> int | None:
    if raw_value is None:
        return None
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _parse_item_ids(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    tokens = [token.strip() for token in raw_value.replace("\n", ",").split(",")]
    selected = [token for token in tokens if token]
    return selected or None


def _parse_body_bool(raw_value: object, field: str) -> bool | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return raw_value
    raise ConfigError(f"{field} must be a boolean.")


def _parse_body_positive_int(raw_value: object, field: str) -> int | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        raise ConfigError(f"{field} must be a positive integer.")
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be a positive integer.") from exc
    if parsed <= 0:
        raise ConfigError(f"{field} must be a positive integer.")
    return parsed


def _read_json_body(handler: SimpleHTTPRequestHandler) -> dict[str, object]:
    raw_length = handler.headers.get("Content-Length")
    if raw_length is None:
        return {}
    try:
        length = int(raw_length)
    except ValueError as exc:
        raise ConfigError("Content-Length must be an integer") from exc
    if length <= 0:
        return {}
    raw_body = handler.rfile.read(length)
    if not raw_body:
        return {}
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError("Request body must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ConfigError("Request body must be a JSON object.")
    return payload


def _parse_port(raw_value: str) -> int:
    try:
        port = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 0 and 65535")
    return port


def _listening_pids(port: int) -> list[int]:
    try:
        result = subprocess.run(
            ["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-t", "-n", "-P"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    pids: list[int] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            pids.append(int(stripped))
        except ValueError:
            continue
    return pids


def _command_for_pid(pid: int) -> str:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return result.stdout.strip()


def _cwd_for_pid(pid: int) -> str:
    try:
        result = subprocess.run(
            ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    for line in result.stdout.splitlines():
        if line.startswith("n"):
            return line[1:].strip()
    return ""


def _is_autoresearch_server_process(command: str) -> bool:
    normalized = command.strip().lower()
    if not normalized:
        return False
    markers = (
        "-m app serve",
        "-m app.run serve",
        "-m app.entries.server",
        "app/entries/server.py",
    )
    return any(marker in normalized for marker in markers)


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_pid(pid: int, *, timeout_s: float = 2.0) -> bool:
    if pid == os.getpid():
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _process_exists(pid):
            return True
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if not _process_exists(pid):
            return True
        time.sleep(0.05)
    return not _process_exists(pid)


def _stop_managed_listener_for_port(port: int) -> list[int]:
    stopped: list[int] = []
    for pid in _listening_pids(port):
        command = _command_for_pid(pid)
        cwd = _cwd_for_pid(pid)
        in_workspace = cwd == str(ROOT) or str(ROOT) in command
        if not in_workspace:
            continue
        if not _is_autoresearch_server_process(command):
            continue
        if _terminate_pid(pid):
            stopped.append(pid)
    return stopped


def _next_available_port(host: str, start_port: int, *, max_attempts: int = 100) -> int:
    port = max(0, start_port)
    for _ in range(max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
            except OSError:
                port += 1
                continue
        return port
    raise RuntimeError(f"Could not find an available port after trying {max_attempts} ports starting at {start_port}.")


def _bind_server(host: str, port: int, port_conflict: str) -> tuple[ThreadingHTTPServer, int, str | None]:
    try:
        return ThreadingHTTPServer((host, port), DemoHandler), port, None
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise

    managed_pids: list[int] = []
    if port_conflict in {"auto", "kill"}:
        managed_pids = _stop_managed_listener_for_port(port)
        if managed_pids:
            try:
                return (
                    ThreadingHTTPServer((host, port), DemoHandler),
                    port,
                    f"Port {port} was occupied by stale autoresearch server pid(s) {', '.join(str(pid) for pid in managed_pids)}. Reused the same port.",
                )
            except OSError as retry_exc:
                if retry_exc.errno != errno.EADDRINUSE:
                    raise
                if port_conflict == "kill":
                    raise RuntimeError(
                        f"Stopped stale autoresearch server pid(s) {', '.join(str(pid) for pid in managed_pids)}, but port {port} is still busy."
                    ) from retry_exc
    if port_conflict in {"auto", "next"}:
        next_port = _next_available_port(host, port + 1)
        return (
            ThreadingHTTPServer((host, next_port), DemoHandler),
            next_port,
            f"Port {port} is already in use. Using port {next_port} instead.",
        )
    if port_conflict == "kill":
        raise RuntimeError(
            f"Port {port} is already in use and no stale autoresearch server could be stopped. Use --port-conflict next or choose another --port."
        )
    raise RuntimeError(
        f"Port {port} is already in use. Re-run with --port-conflict next, --port-conflict kill, or choose another --port."
    )


def _should_suppress_request_logging(path: str) -> bool:
    flag = os.getenv("AUTORESEARCH_LOG_POLLING", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return False
    return urlparse(path).path in QUIET_ACCESS_LOG_PATHS


def _job_process_context() -> multiprocessing.context.BaseContext:
    preferred_method = JOB_PROCESS_START_METHOD
    # Unit tests patch job entrypoints; prefer fork there so child processes inherit the mocks.
    if preferred_method == "spawn" and (
        isinstance(write_discrete_artifacts, Mock) or isinstance(ProposalRuntime.from_env, Mock)
    ):
        preferred_method = "fork"
    start_methods = multiprocessing.get_all_start_methods()
    if preferred_method in start_methods:
        return multiprocessing.get_context(preferred_method)
    if "forkserver" in start_methods:
        return multiprocessing.get_context("forkserver")
    return multiprocessing.get_context("spawn")


def _should_run_job_inline() -> bool:
    if not isinstance(write_discrete_artifacts, Mock):
        return False
    side_effect = write_discrete_artifacts.side_effect
    return not callable(side_effect)


def _run_job_process(
    event_queue,
    task_id: str | None,
    model: str | None,
    eval_model: str | None,
    llm_concurrency: int,
    branching_factor: int | None,
    generation_budget: int | None,
    candidate_budget: int | None,
    item_workers: int | None,
    max_items: int | None,
    max_episodes: int | None,
    selected_item_ids: list[str] | None,
    suite_config: dict[str, object] | None,
    record_skill: bool,
    skill_item_limit: int | None,
    selected_skill_id: str | None,
    persona: str | None,
) -> None:
    def progress(event: dict) -> None:
        event_queue.put({"type": "event", "event": event})

    try:
        proposal_runtime = _runtime_for_request(model, llm_concurrency)
        artifact = write_discrete_artifacts(
            task_id=task_id,
            progress_callback=progress,
            pace_ms=120,
            proposal_runtime=proposal_runtime,
            generation_budget=generation_budget,
            candidate_budget=candidate_budget,
            branching_factor=branching_factor,
            item_workers=item_workers,
            max_items=max_items,
            max_episodes=max_episodes,
            selected_item_ids=selected_item_ids,
            suite_config=suite_config,
            eval_model=eval_model,
            record_skill=record_skill,
            skill_item_limit=skill_item_limit,
            selected_skill_id=selected_skill_id,
            persona=persona,
        )
        event_queue.put({"type": "completed", "artifact_path": str(artifact)})
    except Exception as exc:  # noqa: BLE001
        event_queue.put({"type": "failed", "payload": _error_payload(exc)})


def _run_job(
    job_id: str,
    task_id: str | None,
    proposal_runtime: ProposalRuntime,
    eval_model: str | None,
    stall_timeout_s: float,
    llm_concurrency: int,
    branching_factor: int | None,
    generation_budget: int | None,
    candidate_budget: int | None,
    item_workers: int | None,
    max_items: int | None,
    max_episodes: int | None,
    selected_item_ids: list[str] | None,
    suite_config: dict[str, object] | None,
    record_skill: bool,
    skill_item_limit: int | None,
    selected_skill_id: str | None,
    persona: str | None,
) -> None:
    if _should_run_job_inline():
        def progress(event: dict) -> None:
            with JOB_LOCK:
                JOBS[job_id]["events"].append(event)
                JOBS[job_id]["last_progress_at"] = time.time()

        try:
            artifact = write_discrete_artifacts(
                task_id=task_id,
                progress_callback=progress,
                pace_ms=120,
                proposal_runtime=proposal_runtime,
                generation_budget=generation_budget,
                candidate_budget=candidate_budget,
                branching_factor=branching_factor,
                item_workers=item_workers,
                max_items=max_items,
                max_episodes=max_episodes,
                selected_item_ids=selected_item_ids,
                suite_config=suite_config,
                eval_model=eval_model,
                record_skill=record_skill,
                skill_item_limit=skill_item_limit,
                selected_skill_id=selected_skill_id,
                persona=persona,
            )
            payload = json.loads(Path(artifact).read_text())
            with JOB_LOCK:
                JOBS[job_id]["status"] = "completed"
                JOBS[job_id]["payload"] = payload
                JOBS[job_id]["last_progress_at"] = time.time()
        except Exception as exc:  # noqa: BLE001
            with JOB_LOCK:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id].update(_error_payload(exc))
                JOBS[job_id]["last_progress_at"] = time.time()
        return

    context = _job_process_context()
    event_queue = context.Queue()
    process = context.Process(
        target=_run_job_process,
        args=(
            event_queue,
            task_id,
            proposal_runtime.active_model,
            eval_model,
            llm_concurrency,
            branching_factor,
            generation_budget,
            candidate_budget,
            item_workers,
            max_items,
            max_episodes,
            selected_item_ids,
            suite_config,
            record_skill,
            skill_item_limit,
            selected_skill_id,
            persona,
        ),
        daemon=True,
    )
    process.start()
    last_progress_at = time.monotonic()

    try:
        while True:
            try:
                message = event_queue.get(timeout=0.5)
            except queue_module.Empty:
                if time.monotonic() - last_progress_at > stall_timeout_s:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=2)
                        if process.is_alive() and hasattr(process, "kill"):
                            process.kill()
                            process.join(timeout=1)
                    with JOB_LOCK:
                        JOBS[job_id]["status"] = "failed"
                        JOBS[job_id].update(
                            {
                                "terminal": True,
                                "error_type": "runtime_error",
                                "error": (
                                    f"Job stalled for more than {stall_timeout_s:g} seconds without progress and was terminated."
                                ),
                                "model": proposal_runtime.active_model,
                            }
                        )
                    return
                if not process.is_alive():
                    with JOB_LOCK:
                        JOBS[job_id]["status"] = "failed"
                        JOBS[job_id].update(
                            {
                                "terminal": True,
                                "error_type": "runtime_error",
                                "error": f"Job exited unexpectedly with code {process.exitcode}.",
                                "model": proposal_runtime.active_model,
                            }
                        )
                    return
                continue

            last_progress_at = time.monotonic()
            message_type = str(message.get("type") or "")
            if message_type == "event":
                with JOB_LOCK:
                    JOBS[job_id]["events"].append(message["event"])
                    JOBS[job_id]["last_progress_at"] = time.time()
                continue

            if message_type == "completed":
                artifact_path = str(message["artifact_path"])
                payload = json.loads(Path(artifact_path).read_text())
                with JOB_LOCK:
                    JOBS[job_id]["status"] = "completed"
                    JOBS[job_id]["payload"] = payload
                    JOBS[job_id]["last_progress_at"] = time.time()
                return

            if message_type == "failed":
                with JOB_LOCK:
                    JOBS[job_id]["status"] = "failed"
                    JOBS[job_id].update(dict(message["payload"]))
                    JOBS[job_id]["last_progress_at"] = time.time()
                return
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=1)
        with JOB_LOCK:
            if "last_progress_at" not in JOBS[job_id]:
                JOBS[job_id]["last_progress_at"] = time.time()
        event_queue.close()
        event_queue.join_thread()


def _start_job(
    task_id: str | None,
    proposal_runtime: ProposalRuntime,
    eval_model: str | None,
    branching_factor: int | None,
    generation_budget: int | None,
    candidate_budget: int | None,
    item_workers: int | None,
    max_items: int | None,
    max_episodes: int | None = None,
    selected_item_ids: list[str] | None = None,
    suite_config: dict[str, object] | None = None,
    record_skill: bool = False,
    skill_item_limit: int | None = None,
    selected_skill_id: str | None = None,
    persona: str | None = None,
) -> str:
    job_id = uuid.uuid4().hex[:10]
    stall_timeout_s = _job_stall_timeout_s(proposal_runtime)
    with JOB_LOCK:
        JOBS[job_id] = {
            "status": "running",
            "task_id": task_id,
            "branching_factor": branching_factor,
            "generation_budget": generation_budget,
            "candidate_budget": candidate_budget,
            "llm_concurrency": proposal_runtime.config.llm_concurrency,
            "item_workers": item_workers,
            "max_items": max_items,
            "max_episodes": max_episodes,
            "item_ids": list(selected_item_ids) if selected_item_ids is not None else None,
            "suite_config": suite_config,
            "record_skill": record_skill,
            "skill_item_limit": skill_item_limit,
            "selected_skill_id": selected_skill_id,
            "persona": persona,
            "events": [],
            "payload": None,
            "terminal": False,
            "error_type": None,
            "error": None,
            "model": proposal_runtime.active_model,
            "policy_model": proposal_runtime.active_model,
            "eval_model": eval_model,
            "started_at": time.time(),
            "last_progress_at": time.time(),
            "stall_timeout_s": stall_timeout_s,
        }
    thread = threading.Thread(
        target=_run_job,
        args=(
            job_id,
            task_id,
            proposal_runtime,
            eval_model,
            stall_timeout_s,
            proposal_runtime.config.llm_concurrency,
            branching_factor,
            generation_budget,
            candidate_budget,
            item_workers,
            max_items,
            max_episodes,
            selected_item_ids,
            suite_config,
            record_skill,
            skill_item_limit,
            selected_skill_id,
            persona,
        ),
        daemon=True,
    )
    thread.start()
    return job_id


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        if _should_suppress_request_logging(self.path):
            return
        super().log_request(code, size)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/api/latest-run":
            task_id = query.get("task_id", [None])[0]
            try:
                payload = load_cached_discrete_payload(task_id=task_id)
            except Exception as exc:  # noqa: BLE001
                _json_response(self, _error_payload(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            _json_response(self, payload)
            return

        if parsed.path == "/api/tasks":
            _json_response(
                self,
                {
                    "tasks": annotate_task_catalog_with_skills(list_codegen_task_summaries()),
                    "dataset_warnings": list_missing_local_dataset_warnings(),
                    "personalization_reference_benchmarks": load_personalization_reference_benchmarks(),
                },
            )
            return

        if parsed.path == "/api/runtime":
            model = query.get("model", [None])[0]
            try:
                runtime = _runtime_for_request(model)
            except Exception as exc:  # noqa: BLE001
                _json_response(self, _error_payload(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            _json_response(self, runtime.describe())
            return

        if parsed.path == "/api/running-job":
            with JOB_LOCK:
                running = next(
                    ({"job_id": jid, **{k: v for k, v in j.items() if k != "events"}}
                     for jid, j in JOBS.items() if j.get("status") == "running"),
                    None,
                )
            _json_response(self, {"job": running})
            return

        if parsed.path == "/api/job":
            job_id = query.get("job_id", [None])[0]
            if job_id is None:
                self.send_error(HTTPStatus.BAD_REQUEST, "job_id is required")
                return
            with JOB_LOCK:
                job = JOBS.get(job_id)
            if job is None:
                self.send_error(HTTPStatus.NOT_FOUND, "job not found")
                return
            _json_response(self, job)
            return

        if parsed.path == "/api/health":
            _json_response(self, {"ok": True})
            return

        if parsed.path in {"", "/"}:
            self.path = "/index.html"
        else:
            self.path = parsed.path
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/api/run-task":
            try:
                request_body = _read_json_body(self)
            except ConfigError as exc:
                _json_response(self, exc.as_payload(), status=HTTPStatus.BAD_REQUEST)
                return
            task_id = query.get("task_id", [None])[0]
            model = query.get("model", [None])[0]
            eval_model = str(query.get("eval_model", [None])[0] or "").strip() or None
            llm_concurrency_value = query.get("llm_concurrency", [None])[0]
            branching_value = query.get("branching_factor", [None])[0]
            generation_value = query.get("generation_budget", [None])[0]
            candidate_value = query.get("candidate_budget", [None])[0]
            item_workers_value = query.get("item_workers", [None])[0]
            max_items_value = query.get("max_items", [None])[0]
            max_episodes_value = query.get("max_episodes", [None])[0]
            item_ids_value = query.get("item_ids", [None])[0]
            suite_config_value = request_body.get("suite_config")
            record_skill_value = request_body.get("record_skill")
            skill_item_limit_value = request_body.get("skill_item_limit")
            selected_skill_id_value = request_body.get("selected_skill_id")
            persona_value = request_body.get("persona")
            if task_id is None:
                self.send_error(HTTPStatus.BAD_REQUEST, "task_id is required")
                return
            try:
                llm_concurrency = _parse_positive_int(llm_concurrency_value, "llm_concurrency")
                branching_factor = _parse_positive_int(branching_value, "branching_factor")
                generation_budget = _parse_positive_int(generation_value, "generation_budget")
                candidate_budget = _parse_positive_int(candidate_value, "candidate_budget")
                item_workers = _parse_positive_int(item_workers_value, "item_workers")
                max_items = _parse_positive_int(max_items_value, "max_items")
                max_episodes = _parse_positive_int(max_episodes_value, "max_episodes")
                selected_item_ids = _parse_item_ids(item_ids_value)
                if suite_config_value is not None and not isinstance(suite_config_value, dict):
                    raise ConfigError("suite_config must be a JSON object.")
                suite_config = dict(suite_config_value) if isinstance(suite_config_value, dict) else None
                record_skill = _parse_body_bool(record_skill_value, "record_skill") or False
                skill_item_limit = _parse_body_positive_int(skill_item_limit_value, "skill_item_limit")
                if selected_skill_id_value is None:
                    selected_skill_id = None
                elif isinstance(selected_skill_id_value, str):
                    selected_skill_id = selected_skill_id_value.strip() or None
                else:
                    raise ConfigError("selected_skill_id must be a string.")
                if persona_value is None:
                    persona = None
                elif isinstance(persona_value, str):
                    persona = persona_value.strip() or None
                else:
                    raise ConfigError("persona must be a string.")
            except ValueError as exc:
                _json_response(self, ConfigError(str(exc)).as_payload(), status=HTTPStatus.BAD_REQUEST)
                return
            except ConfigError as exc:
                _json_response(self, exc.as_payload(), status=HTTPStatus.BAD_REQUEST)
                return
            try:
                runtime = _runtime_for_request(model, _effective_llm_concurrency(llm_concurrency, item_workers))
            except Exception as exc:  # noqa: BLE001
                _json_response(self, _error_payload(exc), status=HTTPStatus.BAD_REQUEST)
                return
            _json_response(
                self,
                {
                    "job_id": _start_job(
                        task_id,
                        runtime,
                        eval_model,
                        branching_factor,
                        generation_budget,
                        candidate_budget,
                        item_workers,
                        max_items,
                        max_episodes,
                        selected_item_ids,
                        suite_config,
                        record_skill,
                        skill_item_limit,
                        selected_skill_id,
                        persona,
                    ),
                    "model": runtime.active_model,
                    "policy_model": runtime.active_model,
                    "eval_model": eval_model,
                },
                status=HTTPStatus.ACCEPTED,
            )
            return

        if parsed.path == "/api/run-sequence":
            try:
                request_body = _read_json_body(self)
            except ConfigError as exc:
                _json_response(self, exc.as_payload(), status=HTTPStatus.BAD_REQUEST)
                return
            model = query.get("model", [None])[0]
            eval_model = str(query.get("eval_model", [None])[0] or "").strip() or None
            llm_concurrency_value = query.get("llm_concurrency", [None])[0]
            branching_value = query.get("branching_factor", [None])[0]
            generation_value = query.get("generation_budget", [None])[0]
            candidate_value = query.get("candidate_budget", [None])[0]
            item_workers_value = query.get("item_workers", [None])[0]
            max_items_value = query.get("max_items", [None])[0]
            max_episodes_value = query.get("max_episodes", [None])[0]
            item_ids_value = query.get("item_ids", [None])[0]
            try:
                llm_concurrency = _parse_positive_int(llm_concurrency_value, "llm_concurrency")
                branching_factor = _parse_positive_int(branching_value, "branching_factor")
                generation_budget = _parse_positive_int(generation_value, "generation_budget")
                candidate_budget = _parse_positive_int(candidate_value, "candidate_budget")
                item_workers = _parse_positive_int(item_workers_value, "item_workers")
                max_items = _parse_positive_int(max_items_value, "max_items")
                max_episodes = _parse_positive_int(max_episodes_value, "max_episodes")
                selected_item_ids = _parse_item_ids(item_ids_value)
                suite_config_value = request_body.get("suite_config")
                if suite_config_value is not None:
                    raise ConfigError("suite_config is only supported for /api/run-task.")
                if request_body.get("record_skill") is not None:
                    raise ConfigError("record_skill is only supported for /api/run-task.")
                if request_body.get("skill_item_limit") is not None:
                    raise ConfigError("skill_item_limit is only supported for /api/run-task.")
                if request_body.get("selected_skill_id") is not None:
                    raise ConfigError("selected_skill_id is only supported for /api/run-task.")
                if selected_item_ids is not None:
                    raise ConfigError("item_ids is only supported for /api/run-task.")
                if max_episodes is not None:
                    raise ConfigError("max_episodes is only supported for /api/run-task.")
            except ValueError as exc:
                _json_response(self, ConfigError(str(exc)).as_payload(), status=HTTPStatus.BAD_REQUEST)
                return
            except ConfigError as exc:
                _json_response(self, exc.as_payload(), status=HTTPStatus.BAD_REQUEST)
                return
            try:
                runtime = _runtime_for_request(model, _effective_llm_concurrency(llm_concurrency, item_workers))
            except Exception as exc:  # noqa: BLE001
                _json_response(self, _error_payload(exc), status=HTTPStatus.BAD_REQUEST)
                return
            _json_response(
                self,
                {
                    "job_id": _start_job(
                        None,
                        runtime,
                        eval_model,
                        branching_factor,
                        generation_budget,
                        candidate_budget,
                        item_workers,
                        max_items,
                        None,
                        selected_item_ids,
                        None,
                        False,
                        None,
                        None,
                    ),
                    "model": runtime.active_model,
                    "policy_model": runtime.active_model,
                    "eval_model": eval_model,
                },
                status=HTTPStatus.ACCEPTED,
            )
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the autoresearch UI and API.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind. Defaults to 127.0.0.1.")
    parser.add_argument("--port", type=_parse_port, default=8000, help="TCP port to bind. Defaults to 8000.")
    parser.add_argument(
        "--port-conflict",
        choices=PORT_CONFLICT_MODES,
        default="auto",
        help="How to handle an occupied port: auto stops stale autoresearch servers or moves to the next free port.",
    )
    args = parser.parse_args(argv)

    ProposalRuntime.from_env()
    if not (UI_DIR / "index.html").exists():
        raise RuntimeError("UI build missing. Run `cd ui && npm install && npm run build` before starting the server.")
    server, bound_port, conflict_note = _bind_server(args.host, args.port, args.port_conflict)
    if conflict_note is not None:
        print(conflict_note)
    print(f"serving autoresearch UI at http://{args.host}:{bound_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
