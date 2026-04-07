from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any

from app.codegen.llm import ProposalRuntime
from app.configs.paths import RUNS_ROOT


SKILLS_DIR_NAME = "skills"
_SKILL_FILENAME_RE = re.compile(r"^(?P<dataset>.+)-(?P<model>.+)-task(?P<count>\d+)-(?P<timestamp>\d{8}_\d{6})(?:-(?P<suffix>\d+))?\.md$")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "artifact"


def _stringify(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _preview(text: str, *, limit: int = 220) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def skills_root(*, runs_root: Path | None = None) -> Path:
    return (runs_root or RUNS_ROOT) / SKILLS_DIR_NAME


def task_skills_dir(task_id: str, *, runs_root: Path | None = None) -> Path:
    return skills_root(runs_root=runs_root) / task_id


def _parse_skill_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return metadata
    for line in lines[:24]:
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        key, separator, value = stripped[2:].partition(":")
        if not separator:
            continue
        metadata[key.strip()] = value.strip()
    return metadata


def _skill_title(path: Path) -> str:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return path.stem
    for line in lines[:12]:
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or path.stem
    return path.stem


def list_task_skills(task_id: str, *, runs_root: Path | None = None) -> list[dict[str, Any]]:
    directory = task_skills_dir(task_id, runs_root=runs_root)
    if not directory.exists():
        return []
    root = skills_root(runs_root=runs_root)
    artifacts: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.md"), reverse=True):
        metadata = _parse_skill_metadata(path)
        match = _SKILL_FILENAME_RE.match(path.name)
        source_items: int | None = None
        raw_count = metadata.get("source_items") or (match.group("count") if match else None)
        if raw_count is not None:
            try:
                source_items = int(raw_count)
            except ValueError:
                source_items = None
        generated_at = metadata.get("generated_at")
        artifacts.append(
            {
                "id": str(path.relative_to(root)),
                "filename": path.name,
                "title": _skill_title(path),
                "path": str(path),
                "task_id": metadata.get("task_id") or task_id,
                "dataset_id": metadata.get("dataset_id") or task_id,
                "source_model": metadata.get("source_model"),
                "source_items": source_items,
                "generated_at": generated_at,
            }
        )
    return artifacts


def annotate_task_summary_with_skills(summary: dict[str, Any], *, runs_root: Path | None = None) -> dict[str, Any]:
    task_id = str(summary.get("id") or "").strip()
    if not task_id:
        return dict(summary)
    return {
        **summary,
        "available_skills": list_task_skills(task_id, runs_root=runs_root),
    }


def annotate_task_catalog_with_skills(task_catalog: list[dict[str, Any]], *, runs_root: Path | None = None) -> list[dict[str, Any]]:
    return [annotate_task_summary_with_skills(summary, runs_root=runs_root) for summary in task_catalog]


def _resolve_skill_path(task_id: str, skill_id: str, *, runs_root: Path | None = None) -> Path:
    root = skills_root(runs_root=runs_root).resolve()
    candidate = (root / skill_id).resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Skill id must stay inside {root}.") from exc
    if not relative.parts or relative.parts[0] != task_id:
        raise ValueError(f"Skill {skill_id!r} does not belong to task {task_id!r}.")
    if candidate.suffix.lower() != ".md" or not candidate.is_file():
        raise FileNotFoundError(f"Skill markdown not found: {candidate}")
    return candidate


def load_task_skill_markdown(task_id: str, skill_id: str, *, runs_root: Path | None = None) -> str:
    return _resolve_skill_path(task_id, skill_id, runs_root=runs_root).read_text()


def append_distilled_skill_prompt_context(
    prompt_context: str | None,
    *,
    skill_markdown: str,
    skill_label: str | None = None,
) -> str:
    sections = [_stringify(prompt_context)]
    label = _stringify(skill_label)
    sections.extend(
        [
            "Distilled prior skill",
            label or None,
            "Treat the following markdown as reusable prior guidance from earlier runtime-memory traces on the same dataset.",
            "Follow it when relevant, but prefer the current item's evidence whenever there is a conflict.",
            skill_markdown.strip(),
        ]
    )
    return "\n\n".join(section for section in sections if section)


def _distillation_prompt(task: dict[str, Any], item_runs: list[dict[str, Any]]) -> str:
    sections = [
        f"Task id: {task['id']}",
        f"Task title: {task['title']}",
        f"Track: {task['track']}",
        f"Dataset id: {task['dataset_id']}",
        f"Interaction mode: {task.get('interaction_mode') or 'single_turn'}",
        "Goal: distill a reusable markdown skill from runtime-memory traces so a smaller model can reuse process-level feedback on future items from the same dataset.",
        "Requirements:",
        "- Focus on reusable process priors, not just single-question answers.",
        "- Surface common failure modes, repair hints, and decision rules that transfer across items.",
        "- Keep the result concise but specific.",
        "- Include a final section named `## Prompt Snippet` with short prompt-ready guidance.",
        "- Do not include YAML or JSON frontmatter.",
    ]
    for index, item_run in enumerate(item_runs, start=1):
        question = dict(item_run.get("question") or {})
        winner = dict(item_run.get("winner") or {})
        winner_metrics = dict(winner.get("metrics") or {})
        sections.extend(
            [
                "",
                f"### Runtime memory sample {index}",
                f"- item_id: {item_run.get('item_id')}",
                f"- item_name: {item_run.get('item_name')}",
                f"- prompt_preview: {_preview(_stringify(question.get('raw_prompt') or question.get('prompt')))}",
                f"- verifier_status: {winner_metrics.get('verifier_status')}",
                f"- delta_primary_score: {item_run.get('run_delta_primary_score') or item_run.get('delta_primary_score')}",
                f"- winner_summary: {_stringify(winner.get('candidate_summary'))}",
                f"- selection_reason: {_stringify(item_run.get('selection_reason'))}",
                "Runtime memory markdown:",
                _stringify(item_run.get("memory_markdown")) or "(no runtime memory markdown captured)",
            ]
        )
    return "\n".join(sections)


def _skill_document(
    *,
    task: dict[str, Any],
    runtime_model: str,
    source_item_count: int,
    generated_at: str,
    session_id: str,
    body_markdown: str,
) -> str:
    title = f"Distilled Skill for {task['title']}"
    metadata_lines = [
        f"- task_id: {task['id']}",
        f"- dataset_id: {task.get('dataset_id') or task['id']}",
        f"- source_model: {runtime_model}",
        f"- source_items: {source_item_count}",
        f"- generated_at: {generated_at}",
        f"- source_session_id: {session_id}",
    ]
    return "\n".join(
        [
            f"# {title}",
            "",
            *metadata_lines,
            "",
            body_markdown.strip(),
            "",
        ]
    )


def _unique_skill_path(task: dict[str, Any], runtime_model: str, source_item_count: int, *, runs_root: Path | None = None) -> Path:
    directory = task_skills_dir(str(task["id"]), runs_root=runs_root)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    dataset_slug = _slugify(_stringify(task.get("dataset_id") or task["id"]))
    model_slug = _slugify(runtime_model)
    stem = f"{dataset_slug}-{model_slug}-task{source_item_count}-{timestamp}"
    candidate = directory / f"{stem}.md"
    suffix = 2
    while candidate.exists():
        candidate = directory / f"{stem}-{suffix}.md"
        suffix += 1
    return candidate


def distill_dataset_skill(
    runtime: ProposalRuntime,
    *,
    task: dict[str, Any],
    item_runs: list[dict[str, Any]],
    skill_item_limit: int | None,
    session_id: str,
    runs_root: Path | None = None,
) -> dict[str, Any] | None:
    if not item_runs:
        return None
    limit = max(1, skill_item_limit) if isinstance(skill_item_limit, int) and skill_item_limit > 0 else len(item_runs)
    selected_item_runs = item_runs[:limit]
    user_prompt = _distillation_prompt(task, selected_item_runs)
    response, _trace = runtime.chat(
        purpose="distill_dataset_skill",
        messages=[
            {
                "role": "system",
                "content": (
                    "You distill reusable benchmark skills from runtime-memory traces into concise markdown. "
                    "Output markdown only."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=min(int(runtime.config.max_tokens), 2400),
        queue_priority=900,
    )
    body_markdown = _stringify(response.get("message"))
    if not body_markdown:
        raise ValueError("Skill distillation returned an empty markdown response.")
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    artifact_path = _unique_skill_path(task, runtime.active_model, len(selected_item_runs), runs_root=runs_root)
    document = _skill_document(
        task=task,
        runtime_model=runtime.active_model,
        source_item_count=len(selected_item_runs),
        generated_at=generated_at,
        session_id=session_id,
        body_markdown=body_markdown,
    )
    artifact_path.write_text(document)
    return {
        "id": str(artifact_path.relative_to(skills_root(runs_root=runs_root))),
        "filename": artifact_path.name,
        "title": f"Distilled Skill for {task['title']}",
        "path": str(artifact_path),
        "task_id": str(task["id"]),
        "dataset_id": str(task.get("dataset_id") or task["id"]),
        "source_model": runtime.active_model,
        "source_items": len(selected_item_runs),
        "generated_at": generated_at,
    }
