from __future__ import annotations

import ast
import contextlib
import faulthandler
import json
import multiprocessing
import platform
import signal
import subprocess
import sys
import time
from decimal import Decimal
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import mock_open, patch


OFFICIAL_TIMEOUT_S = 6
REPO_ROOT = Path(__file__).resolve().parents[2]
IMPORT_STRING = (
    "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\n"
    "from heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\n"
    "from statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\n"
    "from io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\n"
    "import string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\n"
    "import math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\n"
    "import io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"
)


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):  # noqa: ARG001
    raise TimeoutException


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        self._stringio = StringIO()
        self._stringio.close = lambda x=None: 1
        sys.stdout = self._stringio
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


class MockBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs.encode("utf-8")

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self.inputs.split(b"\n")[0] + b"\n"


class MockStdinWithBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs
        self._stringio = StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self.inputs.split("\n")

    def __getattr__(self, name):
        return getattr(self._stringio, name)


def truncatefn(value: Any, length: int = 300) -> str:
    text = value if isinstance(value, str) else str(value)
    if len(text) <= length:
        return text
    return text[: length // 2] + "...(truncated) ..." + text[-length // 2 :]


def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)  # type: ignore[arg-type]
    except Exception:
        pass
    return code


def _split_future_imports(code: str) -> tuple[str, str]:
    future_lines: list[str] = []
    body_lines: list[str] = []
    for line in code.splitlines():
        if line.strip().startswith("from __future__ import "):
            future_lines.append(line)
        else:
            body_lines.append(line)
    future_block = "\n".join(future_lines)
    body_block = "\n".join(body_lines)
    return future_block, body_block


def make_function(code: str) -> str:
    try:
        future_block, code = _split_future_imports(code)
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)
        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        prefix = future_block + ("\n" if future_block else "")
        return prefix + IMPORT_STRING + "\n" + ast.unparse(import_stmts) + "\n" + ast.unparse(function_ast)  # type: ignore[arg-type]
    except Exception:
        return code


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)
    inputs_line_iterator = iter(inputs.split("\n"))
    mock_stdin = MockStdinWithBuffer(inputs)

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", mock_stdin)
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            return None

    return _inner_call_method(method)


def get_function(compiled_sol, fn_name: str):
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception:
        return None


def compile_code(code: str, timeout: int):
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        compiled_sol = tmp_sol.Solution() if hasattr(tmp_sol, "Solution") else tmp_sol
        assert compiled_sol is not None
        return compiled_sol
    finally:
        signal.alarm(0)


def convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    try:
        return True, [Decimal(elem) for elem in line.split()]
    except Exception:
        return False, []


def get_stripped_lines(value: str) -> list[str]:
    value = value.strip()
    return [line.strip() for line in value.split("\n")]


def grade_call_based(code: str, all_inputs: list[str], all_outputs: list[str], fn_name: str, timeout: int):
    future_block, code = _split_future_imports(code)
    code = (future_block + "\n" if future_block else "") + IMPORT_STRING + "\n\n" + code
    compiled_sol = compile_code(code, timeout)
    method = get_function(compiled_sol, fn_name)
    if method is None:
        return [-4], {"error_code": -4, "error_message": f"Missing callable {fn_name}"}
    parsed_inputs = [[json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs]
    parsed_outputs = [json.loads(output) for output in all_outputs]
    total_execution = 0.0
    all_results: list[Any] = []
    for gt_inp, gt_out in zip(parsed_inputs, parsed_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            start = time.time()
            prediction = method(*gt_inp)
            total_execution += time.time() - start
            signal.alarm(0)
            if isinstance(prediction, tuple):
                prediction = list(prediction)
            ok = prediction == gt_out
            all_results.append(ok)
            if not ok:
                return all_results, {
                    "output": truncatefn(prediction),
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                    "error_code": -2,
                    "error_message": "Wrong Answer",
                }
        except Exception as exc:
            signal.alarm(0)
            if "timeoutexception" in repr(exc).lower():
                all_results.append(-3)
                return all_results, {
                    "error": repr(exc),
                    "error_code": -3,
                    "error_message": "Time Limit Exceeded",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }
            all_results.append(-4)
            return all_results, {
                "error": repr(exc),
                "error_code": -4,
                "error_message": "Runtime Error",
                "inputs": truncatefn(gt_inp),
                "expected": truncatefn(gt_out),
            }
        finally:
            signal.alarm(0)
            faulthandler.disable()
    return all_results, {"execution time": total_execution}


def grade_stdio(code: str, all_inputs: list[str], all_outputs: list[str], timeout: int):
    code = clean_if_name(code)
    code = make_function(code)
    compiled_sol = compile_code(code, timeout)
    method = get_function(compiled_sol, "wrapped_function")
    if method is None:
        return [-4], {"error_code": -4, "error_message": "Missing wrapped_function"}
    all_results: list[Any] = []
    total_execution_time = 0.0
    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        with Capturing() as captured_output:
            try:
                start = time.time()
                call_method(method, gt_inp)
                total_execution_time += time.time() - start
                signal.alarm(0)
            except Exception as exc:
                signal.alarm(0)
                if "timeoutexception" in repr(exc).lower():
                    all_results.append(-3)
                    return all_results, {
                        "error": repr(exc),
                        "error_code": -3,
                        "error_message": "Time Limit Exceeded",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                    }
                all_results.append(-4)
                return all_results, {
                    "error": repr(exc),
                    "error_code": -4,
                    "error_message": "Runtime Error",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }
            finally:
                signal.alarm(0)
                faulthandler.disable()
        prediction = captured_output[0]
        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(gt_out)
        wa_payload = {
            "output": truncatefn(prediction),
            "inputs": truncatefn(gt_inp),
            "expected": truncatefn(gt_out),
            "error_code": -2,
        }
        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            all_results.append(-2)
            wa_payload["error_message"] = "Wrong answer: mismatched output length"
            return all_results, wa_payload
        for output_line_idx, (pred_line, exp_line) in enumerate(zip(stripped_prediction_lines, stripped_gt_out_lines)):
            wa_payload["error_message"] = f"Wrong answer at output_line_idx={output_line_idx}: {truncatefn(pred_line)} != {truncatefn(exp_line)}"
            if pred_line == exp_line:
                continue
            ok_pred, pred_decimals = convert_line_to_decimals(pred_line)
            ok_exp, exp_decimals = convert_line_to_decimals(exp_line)
            if not ok_pred or not ok_exp or pred_decimals != exp_decimals:
                all_results.append(-2)
                return all_results, wa_payload
        all_results.append(True)
    return all_results, {"execution time": total_execution_time}


def reliability_guard(maximum_memory_bytes=None):
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()
    import builtins
    import os
    import shutil
    import subprocess

    builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"
    for name in (
        "kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid", "fork", "forkpty",
        "killpg", "rename", "renames", "truncate", "replace", "unlink", "fchmod", "fchown", "chmod", "chown",
        "chroot", "lchflags", "lchmod", "lchown", "getcwd", "chdir",
    ):
        setattr(os, name, None)
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    subprocess.Popen = None  # type: ignore[assignment]
    if isinstance(__builtins__, dict):
        __builtins__["help"] = None
    else:
        setattr(__builtins__, "help", None)
    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def run_test(sample, test: str, timeout: int = OFFICIAL_TIMEOUT_S):
    signal.signal(signal.SIGALRM, timeout_handler)
    reliability_guard()
    in_outs = json.loads(sample["input_output"])
    method_name = in_outs.get("fn_name")
    if method_name is None:
        return grade_stdio(
            code=test,
            all_inputs=in_outs["inputs"],
            all_outputs=in_outs["outputs"],
            timeout=timeout,
        )
    return grade_call_based(
        code=test,
        all_inputs=in_outs["inputs"],
        all_outputs=in_outs["outputs"],
        fn_name=method_name,
        timeout=timeout,
    )


def _temp_run(sample: dict[str, Any], generation: str, timeout: int, result, metadata_list) -> None:
    res, metadata = run_test(sample, test=generation, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except Exception:
            pass
    return str(value)


def _global_timeout_seconds(sample: dict[str, Any], timeout: int) -> tuple[int, int]:
    num_inputs = len(json.loads(sample["input_output"]).get("inputs") or [])
    return max(num_inputs, 1), (timeout + 1) * max(num_inputs, 1) + 5


def _global_timeout_result(num_inputs: int) -> tuple[list[Any], dict[str, Any]]:
    return ([-1 for _ in range(max(num_inputs, 1))], {"error_code": -1, "error_message": "Global Timeout"})


def _worker_payload(sample: dict[str, Any], generation: str, timeout: int) -> str:
    return json.dumps(
        {
            "sample": _json_safe(sample),
            "generation": generation,
            "timeout": timeout,
        }
    )


def _run_subprocess_worker(sample: dict[str, Any], generation: str, timeout: int) -> tuple[list[Any], dict[str, Any]]:
    num_inputs, global_timeout = _global_timeout_seconds(sample, timeout)
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "app.bench.livecodebench_official_support", "--worker"],
            input=_worker_payload(sample, generation, timeout),
            text=True,
            capture_output=True,
            cwd=str(REPO_ROOT),
            timeout=global_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _global_timeout_result(num_inputs)
    except Exception as exc:
        return ([-5 for _ in range(num_inputs)], {"error_code": -5, "error_message": "TestRunnerError", "error": repr(exc)})

    stdout = completed.stdout.strip()
    if completed.returncode != 0:
        details = completed.stderr.strip() or stdout or f"worker exited with status {completed.returncode}"
        return ([-5 for _ in range(num_inputs)], {"error_code": -5, "error_message": "TestRunnerError", "error": details})
    if not stdout:
        return ([-5 for _ in range(num_inputs)], {"error_code": -5, "error_message": "TestRunnerError", "error": "worker returned empty output"})
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        details = completed.stderr.strip() or stdout
        return (
            [-5 for _ in range(num_inputs)],
            {
                "error_code": -5,
                "error_message": "TestRunnerError",
                "error": f"Invalid worker payload: {exc}. Output: {truncatefn(details)}",
            },
        )
    result = list(payload.get("result") or [])
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    if not result:
        return _global_timeout_result(num_inputs)
    return result, dict(metadata)


def _should_use_subprocess_runner() -> bool:
    if bool(getattr(multiprocessing.current_process(), "daemon", False)):
        return True
    main_module = sys.modules.get("__main__")
    main_path = getattr(main_module, "__file__", None)
    if not isinstance(main_path, str) or not main_path.strip():
        return True
    main_path = main_path.strip()
    if main_path.startswith("<") and main_path.endswith(">"):
        return True
    return not Path(main_path).exists()


def check_correctness(sample: dict[str, Any], generation: str, timeout: int = OFFICIAL_TIMEOUT_S) -> tuple[list[Any], dict[str, Any]]:
    if _should_use_subprocess_runner():
        return _run_subprocess_worker(sample, generation, timeout)

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    process = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, timeout, result, metadata_list),
    )
    process.start()
    num_inputs, global_timeout = _global_timeout_seconds(sample, timeout)
    process.join(timeout=global_timeout)
    if process.is_alive():
        process.kill()
    if not result:
        return _global_timeout_result(num_inputs)
    return list(result[0]), dict(metadata_list[0])


def _problem_cases(problem: dict[str, Any]) -> list[dict[str, Any]]:
    return list(problem.get("public_test_cases") or []) + list(problem.get("private_test_cases") or [])


def problem_to_official_sample(problem: dict[str, Any]) -> dict[str, Any]:
    cases = _problem_cases(problem)
    metadata = dict(problem.get("metadata") or {})
    evaluation_mode = str(problem.get("evaluation_mode") or "").strip().lower()
    fn_name = problem.get("function_name") or metadata.get("func_name")
    if evaluation_mode != "functional":
        fn_name = None
    normalized_fn_name = str(fn_name).strip() if isinstance(fn_name, str) else None
    return {
        "input_output": json.dumps(
            {
                "inputs": [str(case.get("input") or "") for case in cases],
                "outputs": [str(case.get("output") or "") for case in cases],
                "fn_name": normalized_fn_name or None,
            }
        )
    }


def official_case_rows(problem: dict[str, Any], code: str, timeout: int = OFFICIAL_TIMEOUT_S) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = _problem_cases(problem)
    raw_results, metadata = check_correctness(problem_to_official_sample(problem), generation=code, timeout=timeout)
    first_failed_index = next((index for index, value in enumerate(raw_results) if value is not True), None)
    failure_actual = str(metadata.get("output") or "").rstrip("\n")
    failure_error = str(metadata.get("error_message") or metadata.get("error") or "").strip() or None
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        case_name = str(case.get("name") or f"case-{index + 1}")
        expected = str(case.get("output") or "").rstrip("\n")
        result_value = raw_results[index] if index < len(raw_results) else False
        passed = bool(result_value is True)
        error = None
        actual = expected if passed else ""
        if not passed and first_failed_index is not None and index == first_failed_index:
            actual = failure_actual
            error = failure_error
        elif not passed and index >= len(raw_results):
            error = "not executed after earlier failure"
        rows.append(
            {
                "name": case_name,
                "expected": expected,
                "actual": actual,
                "actual_raw": actual,
                "passed": passed,
                "error": error,
            }
        )
    return rows, metadata


def evaluate_livecodebench_problem(problem: dict[str, Any], candidate_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    code = candidate_path.read_text()
    rows, metadata = official_case_rows(problem, code)
    passed = sum(1 for row in rows if row["passed"])
    total = len(rows)
    solved = passed == total
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    status = "pass" if solved else "fail"
    return {
        "status": status,
        "verifier_status": status,
        "correctness": 1.0 if solved else 0.0,
        "passed_tests": passed,
        "total_tests": total,
        "benchmark_ms": round(elapsed_ms, 3),
        "benchmark_samples_ms": [round(elapsed_ms, 3)],
        "objective": 1.0 if solved else 0.0,
        "objective_score": 1.0 if solved else 0.0,
        "objective_signal": 1.0 if solved else 0.0,
        "stability": 1.0,
        "error": None if solved else str(metadata.get("error_message") or metadata.get("error") or "").strip() or None,
        "platform": problem.get("platform"),
        "evaluation_mode": problem.get("evaluation_mode"),
        "test_results": rows,
        "official_metadata": metadata,
    }


def _worker_main() -> int:
    payload: dict[str, Any] = {}
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        sample = payload.get("sample")
        generation = payload.get("generation")
        timeout = int(payload.get("timeout") or OFFICIAL_TIMEOUT_S)
        if not isinstance(sample, dict):
            raise ValueError("worker payload is missing sample")
        if not isinstance(generation, str):
            raise ValueError("worker payload is missing generation")
        result, metadata = run_test(sample, test=generation, timeout=timeout)
        sys.stdout.write(json.dumps({"result": _json_safe(result), "metadata": _json_safe(metadata)}))
        return 0
    except Exception as exc:
        num_inputs = 1
        try:
            raw_sample = payload.get("sample") if isinstance(payload, dict) else None
            if isinstance(raw_sample, dict):
                num_inputs, _ = _global_timeout_seconds(raw_sample, OFFICIAL_TIMEOUT_S)
        except Exception:
            pass
        sys.stdout.write(
            json.dumps(
                {
                    "result": [-5 for _ in range(max(num_inputs, 1))],
                    "metadata": {
                        "error_code": -5,
                        "error_message": "TestRunnerError",
                        "error": repr(exc),
                    },
                }
            )
        )
        return 0


if __name__ == "__main__":
    if "--worker" in sys.argv:
        raise SystemExit(_worker_main())
