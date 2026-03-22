from __future__ import annotations

import json
import threading
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from app.codegen.catalog import list_codegen_task_summaries
from app.codegen.errors import AutoresearchError, ConfigError
from app.codegen.llm import ProposalRuntime
from app.entries.discrete_demo import ROOT, load_cached_discrete_payload, write_discrete_artifacts


UI_DIR = ROOT / "ui" / "dist"
JOB_LOCK = threading.Lock()
JOBS: dict[str, dict[str, object]] = {}


def _runtime_for_request(model: str | None = None) -> ProposalRuntime:
    runtime = ProposalRuntime.from_env()
    return runtime.with_model(model)


def _json_response(handler: SimpleHTTPRequestHandler, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
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


def _run_job(
    job_id: str,
    task_id: str | None,
    proposal_runtime: ProposalRuntime,
    branching_factor: int | None,
    generation_budget: int | None,
    candidate_budget: int | None,
    item_workers: int | None,
    max_items: int | None,
) -> None:
    def progress(event: dict) -> None:
        with JOB_LOCK:
            JOBS[job_id]["events"].append(event)

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
        )
        payload = json.loads(artifact.read_text())
        with JOB_LOCK:
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["payload"] = payload
    except Exception as exc:  # noqa: BLE001
        with JOB_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id].update(_error_payload(exc))


def _start_job(
    task_id: str | None,
    proposal_runtime: ProposalRuntime,
    branching_factor: int | None,
    generation_budget: int | None,
    candidate_budget: int | None,
    item_workers: int | None,
    max_items: int | None,
) -> str:
    job_id = uuid.uuid4().hex[:10]
    with JOB_LOCK:
        JOBS[job_id] = {
            "status": "running",
            "task_id": task_id,
            "branching_factor": branching_factor,
            "generation_budget": generation_budget,
            "candidate_budget": candidate_budget,
            "item_workers": item_workers,
            "max_items": max_items,
            "events": [],
            "payload": None,
            "terminal": False,
            "error_type": None,
            "error": None,
            "model": proposal_runtime.active_model,
        }
    thread = threading.Thread(
        target=_run_job,
        args=(job_id, task_id, proposal_runtime, branching_factor, generation_budget, candidate_budget, item_workers, max_items),
        daemon=True,
    )
    thread.start()
    return job_id


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

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
            _json_response(self, {"tasks": list_codegen_task_summaries()})
            return

        if parsed.path == "/api/runtime":
            try:
                runtime = _runtime_for_request()
            except Exception as exc:  # noqa: BLE001
                _json_response(self, _error_payload(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            _json_response(self, runtime.describe())
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
            task_id = query.get("task_id", [None])[0]
            model = query.get("model", [None])[0]
            branching_value = query.get("branching_factor", [None])[0]
            generation_value = query.get("generation_budget", [None])[0]
            candidate_value = query.get("candidate_budget", [None])[0]
            item_workers_value = query.get("item_workers", [None])[0]
            max_items_value = query.get("max_items", [None])[0]
            if task_id is None:
                self.send_error(HTTPStatus.BAD_REQUEST, "task_id is required")
                return
            try:
                branching_factor = _parse_positive_int(branching_value, "branching_factor")
                generation_budget = _parse_positive_int(generation_value, "generation_budget")
                candidate_budget = _parse_positive_int(candidate_value, "candidate_budget")
                item_workers = _parse_positive_int(item_workers_value, "item_workers")
                max_items = _parse_positive_int(max_items_value, "max_items")
            except ValueError as exc:
                _json_response(self, ConfigError(str(exc)).as_payload(), status=HTTPStatus.BAD_REQUEST)
                return
            try:
                runtime = _runtime_for_request(model)
            except Exception as exc:  # noqa: BLE001
                _json_response(self, _error_payload(exc), status=HTTPStatus.BAD_REQUEST)
                return
            _json_response(
                self,
                {
                    "job_id": _start_job(
                        task_id,
                        runtime,
                        branching_factor,
                        generation_budget,
                        candidate_budget,
                        item_workers,
                        max_items,
                    ),
                    "model": runtime.active_model,
                },
                status=HTTPStatus.ACCEPTED,
            )
            return

        if parsed.path == "/api/run-sequence":
            model = query.get("model", [None])[0]
            branching_value = query.get("branching_factor", [None])[0]
            generation_value = query.get("generation_budget", [None])[0]
            candidate_value = query.get("candidate_budget", [None])[0]
            item_workers_value = query.get("item_workers", [None])[0]
            max_items_value = query.get("max_items", [None])[0]
            try:
                branching_factor = _parse_positive_int(branching_value, "branching_factor")
                generation_budget = _parse_positive_int(generation_value, "generation_budget")
                candidate_budget = _parse_positive_int(candidate_value, "candidate_budget")
                item_workers = _parse_positive_int(item_workers_value, "item_workers")
                max_items = _parse_positive_int(max_items_value, "max_items")
            except ValueError as exc:
                _json_response(self, ConfigError(str(exc)).as_payload(), status=HTTPStatus.BAD_REQUEST)
                return
            try:
                runtime = _runtime_for_request(model)
            except Exception as exc:  # noqa: BLE001
                _json_response(self, _error_payload(exc), status=HTTPStatus.BAD_REQUEST)
                return
            _json_response(
                self,
                {
                    "job_id": _start_job(
                        None,
                        runtime,
                        branching_factor,
                        generation_budget,
                        candidate_budget,
                        item_workers,
                        max_items,
                    ),
                    "model": runtime.active_model,
                },
                status=HTTPStatus.ACCEPTED,
            )
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")


def main() -> None:
    ProposalRuntime.from_env()
    if not (UI_DIR / "index.html").exists():
        raise RuntimeError("UI build missing. Run `cd ui && npm install && npm run build` before starting the server.")
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DemoHandler)
    print("serving codegen demo at http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
