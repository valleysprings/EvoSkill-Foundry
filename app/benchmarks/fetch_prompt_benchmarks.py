#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional dependency at runtime
    load_dataset = None

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "benchmarks" / "prompt" / "configs"
DATA_DIR = ROOT / "benchmarks" / "prompt" / "data"
HF_DATASET_VIEWER_API = "https://datasets-server.huggingface.co"


def read_json(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError:
        result = subprocess.run(
            ["curl", "-fsSL", url],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)


def build_url(path: str, **params: Any) -> str:
    encoded = urllib.parse.urlencode(params)
    return f"{HF_DATASET_VIEWER_API}{path}?{encoded}"


def normalize_record(benchmark_id: str, row: dict[str, Any]) -> dict[str, Any]:
    if benchmark_id == "banking77_small":
        return {
            "input": row["text"],
            "target": row["label_text"],
            "metadata": {
                "label_id": row["label"],
            },
        }
    if benchmark_id == "boolq_small":
        return {
            "input": {
                "question": row["question"],
                "passage": row["passage"],
            },
            "target": "yes" if row["answer"] else "no",
            "metadata": {},
        }
    if benchmark_id == "ag_news_small":
        label_names = {
            1: "world",
            2: "sports",
            3: "business",
            4: "sci_tech",
        }
        text = row["title"].strip()
        description = row["description"].strip()
        return {
            "input": {
                "title": text,
                "description": description,
                "article": f"{text}\n\n{description}",
            },
            "target": label_names[row["label"]],
            "metadata": {
                "label_id": row["label"],
            },
        }
    raise ValueError(f"Unknown benchmark id: {benchmark_id}")


def fetch_rows(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict[str, Any]]:
    if load_dataset is not None:
        split_spec = f"{split}[{offset}:{offset + length}]"
        config_name = None if config in {"", "default"} else config
        dataset_slice = load_dataset(dataset, name=config_name, split=split_spec)
        return [dataset_slice[index] for index in range(len(dataset_slice))]

    rows: list[dict[str, Any]] = []
    remaining = length
    cursor = offset
    while remaining > 0:
        batch = min(remaining, 100)
        url = build_url(
            "/rows",
            dataset=dataset,
            config=config,
            split=split,
            offset=cursor,
            length=batch,
        )
        payload = read_json(url)
        chunk = [item["row"] for item in payload["rows"]]
        rows.extend(chunk)
        remaining -= len(chunk)
        cursor += len(chunk)
        if not chunk:
            break
    return rows


def available_configs(dataset: str) -> list[dict[str, str]]:
    url = build_url("/splits", dataset=dataset)
    payload = read_json(url)
    return payload.get("splits", [])


def fetch_benchmark(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    benchmark_id = config["id"]
    dataset = config["dataset"]
    requested_config = config.get("config", "default")
    if load_dataset is None:
        split_info = available_configs(dataset)
        known_configs = {item["config"] for item in split_info}
        if requested_config not in known_configs and known_configs:
            requested_config = sorted(known_configs)[0]

    target_dir = DATA_DIR / benchmark_id
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "id": benchmark_id,
        "task_family": config["task_family"],
        "dataset": dataset,
        "source_url": config["source_url"],
        "license": config["license"],
        "config": requested_config,
        "metrics": config["metrics"],
        "record_format": config["record_format"],
        "notes": config["notes"],
        "splits": {},
    }

    for split_name, spec in config["splits"].items():
        rows = fetch_rows(
            dataset=dataset,
            config=requested_config,
            split=spec["source_split"],
            offset=int(spec["offset"]),
            length=int(spec["length"]),
        )
        normalized = [
            {
                "id": f"{benchmark_id}-{split_name}-{index}",
                **normalize_record(benchmark_id, row),
            }
            for index, row in enumerate(rows)
        ]

        out_path = target_dir / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for record in normalized:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        manifest["splits"][split_name] = {
            "path": str(out_path.relative_to(ROOT)),
            "num_rows": len(normalized),
            "source_split": spec["source_split"],
            "offset": spec["offset"],
            "length": spec["length"],
        }
        print(f"wrote {out_path.relative_to(ROOT)} ({len(normalized)} rows)")

    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch small local prompt benchmark slices from Hugging Face.")
    parser.add_argument("--benchmark", help="Only fetch one benchmark id.")
    args = parser.parse_args()

    config_paths = sorted(CONFIG_DIR.glob("*.json"))
    if args.benchmark:
        config_paths = [path for path in config_paths if path.stem == args.benchmark]
        if not config_paths:
            print(f"unknown benchmark: {args.benchmark}", file=sys.stderr)
            raise SystemExit(1)

    for config_path in config_paths:
        fetch_benchmark(config_path)


if __name__ == "__main__":
    main()
