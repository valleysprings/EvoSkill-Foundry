from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = ROOT / "benchmark"
REGISTRY_PATH = BENCHMARK_ROOT / "registry.json"
RUNS_ROOT = ROOT / "runs"
