from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = ROOT / "external" / "personalization"
REFERENCE_CATALOG_PATH = Path(__file__).with_name("reference_benchmarks.json")


def _load_repos() -> tuple[tuple[str, str], ...]:
    payload = json.loads(REFERENCE_CATALOG_PATH.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected a top-level list in {REFERENCE_CATALOG_PATH}")

    repos: list[tuple[str, str]] = []
    seen: set[str] = set()
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        mirror = entry.get("mirror")
        if not isinstance(mirror, dict):
            continue
        slug = str(mirror.get("slug") or "").strip()
        url = str(mirror.get("url") or "").strip()
        if not slug or not url or slug in seen:
            continue
        repos.append((slug, url))
        seen.add(slug)
    return tuple(repos)


REPOS: tuple[tuple[str, str], ...] = _load_repos()


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
        raise KeyError(f"Unknown personalization external repo: {slug}")

    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    destination = EXTERNAL_ROOT / slug
    if (destination / ".git").exists():
        if refresh:
            print(f"[update] {slug}")
            _run(["git", "pull", "--ff-only"], cwd=destination)
        return destination

    url = repo_map[slug]
    print(f"[clone] {slug}")
    _run(["git", "clone", "--depth", "1", "--filter=blob:none", url, str(destination)])
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone or refresh mirrored personalization benchmark repos.")
    parser.add_argument("--refresh", action="store_true", help="Run git pull --ff-only for existing mirrors.")
    parser.add_argument("--list", action="store_true", help="Print the repo slugs known to the personalization catalog and exit.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list:
        for slug, _url in REPOS:
            print(slug)
        return
    for slug, _url in REPOS:
        ensure_repo_checkout(slug, refresh=bool(args.refresh))


if __name__ == "__main__":
    main()
