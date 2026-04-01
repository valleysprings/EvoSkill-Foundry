from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = ROOT / "benchmark"
REGISTRY_PATH = BENCHMARK_ROOT / "registry.json"
RUNS_ROOT = ROOT / "runs"
RUNTIME_ROOT = RUNS_ROOT / "runtime"
RUNTIME_REPOS_ROOT = RUNTIME_ROOT / "repos"
RUNTIME_BENCHMARKS_ROOT = RUNTIME_ROOT / "benchmarks"
