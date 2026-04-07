from __future__ import annotations

from datetime import datetime
from typing import Any


def render_memory_markdown(
    memories: list[dict[str, Any]],
    *,
    title: str,
    generated_at: str | None = None,
) -> str:
    timestamp = generated_at or datetime.now().astimezone().isoformat(timespec="seconds")
    success_count = sum(1 for item in memories if item.get("experience_outcome", "success") == "success")
    failure_count = sum(1 for item in memories if item.get("experience_outcome") == "failure")
    lines = [
        f"# {title}",
        "",
        f"- generated_at: {timestamp}",
        f"- num_memories: {len(memories)}",
        f"- success_memories: {success_count}",
        f"- failure_memories: {failure_count}",
        "",
        "## Experience Units",
        "",
    ]

    for item in memories:
        delta_primary_score = item.get("delta_primary_score", item.get("delta_J", 0.0))
        lines.extend(
            [
                f"### {item.get('experience_id', 'unknown-memory')}",
                "",
                f"- source_task: {item.get('source_task', 'unknown')}",
                f"- source_session_id: {item.get('source_session_id', 'unknown')}",
                f"- family: {item.get('family', 'agnostic')}",
                f"- experience_outcome: {item.get('experience_outcome', 'success')}",
                f"- verifier_status: {item.get('verifier_status', '')}",
                f"- rejection_reason: {item.get('rejection_reason', '')}",
                f"- delta_primary_score: {delta_primary_score}",
                f"- task_signature: {', '.join(item.get('task_signature', []))}",
                f"- failure_pattern: {item.get('failure_pattern', '')}",
                f"- strategy_hypothesis: {item.get('strategy_hypothesis', '')}",
                f"- successful_strategy: {item.get('successful_strategy', '')}",
                f"- prompt_fragment: {item.get('prompt_fragment', item.get('successful_strategy', ''))}",
                f"- candidate_summary: {item.get('candidate_summary', item.get('code_pattern', ''))}",
                f"- tool_trace_summary: {item.get('tool_trace_summary', '')}",
                f"- process_failure_mode: {item.get('process_failure_mode', '')}",
                f"- process_repair_hint: {item.get('process_repair_hint', '')}",
                f"- process_trace_summary: {item.get('process_trace_summary', '')}",
                f"- knowledge_scope: {item.get('knowledge_scope', 'episode_strategy')}",
                f"- distilled_skill: {item.get('distilled_skill', '')}",
                f"- applicability_notes: {item.get('applicability_notes', '')}",
                f"- source_dataset_ids: {', '.join(item.get('source_dataset_ids', [])) or 'none'}",
                f"- proposal_model: {item.get('proposal_model', '')}",
                f"- reusable_rules: {', '.join(item.get('reusable_rules', []))}",
                f"- supporting_memory_ids: {', '.join(item.get('supporting_memory_ids', [])) or 'none'}",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"
