from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = ROOT / "external" / "safety"
REPOS: tuple[tuple[str, str], ...] = (
    ("harmbench", "https://github.com/centerforaisafety/HarmBench.git"),
    ("hallulens", "https://github.com/facebookresearch/HalluLens.git"),
    ("jailbreakbench", "https://github.com/JailbreakBench/jailbreakbench.git"),
    ("jailbreakbench-artifacts", "https://github.com/JailbreakBench/artifacts.git"),
    ("longsafety", "https://github.com/thu-coai/longsafety.git"),
    ("tau2-bench", "https://github.com/sierra-research/tau2-bench.git"),
)


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip())
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}")


def ensure_repo_checkout(slug: str, *, refresh: bool = False) -> Path:
    repo_map = dict(REPOS)
    if slug not in repo_map:
        raise KeyError(f"Unknown safety external repo: {slug}")

    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    destination = EXTERNAL_ROOT / slug
    if (destination / ".git").exists():
        if refresh:
            print(f"[update] {slug}")
            _run(["git", "pull", "--ff-only"], cwd=destination)
        return destination
    if destination.exists():
        shutil.rmtree(destination)

    url = repo_map[slug]
    print(f"[clone] {slug}")
    _run(["git", "clone", "--depth", "1", "--filter=blob:none", url, str(destination)])
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone or refresh mirrored safety benchmark repos.")
    parser.add_argument("--refresh", action="store_true", help="Run git pull --ff-only for existing mirrors.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for slug, _url in REPOS:
        ensure_repo_checkout(slug, refresh=bool(args.refresh))


if __name__ == "__main__":
    main()
