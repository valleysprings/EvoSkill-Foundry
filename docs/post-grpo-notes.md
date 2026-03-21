# Post-GRPO Notes

These are the 2025-2026 sources that currently matter most for this repo's direction.

## Most Relevant Sources

1. `AlphaEvolve`
   - paper date: `2025-06-16`
   - why it matters: closest reference for code- and operator-level evolutionary search with automatic evaluation
   - links:
     - <https://arxiv.org/abs/2506.13131>
     - <https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/>

2. `GEPA`
   - paper date: `2025-07-25`
   - why it matters: reflective prompt evolution with a stronger outer-loop framing than plain RL prompt tuning
   - link:
     - <https://arxiv.org/abs/2507.19457>

3. `C-Evolve`
   - paper date: `2025-09-27`
   - why it matters: prompt groups, consensus, and island-style evolution map cleanly onto multi-lane proposal systems
   - link:
     - <https://arxiv.org/abs/2509.23331>

4. `AlphaResearch`
   - paper date: `2025-11-11`
   - why it matters: pushes closer to autonomous research workflows instead of isolated optimization loops
   - link:
     - <https://arxiv.org/abs/2511.08522>

5. `ImprovEvolve`
   - paper date: `2026-02-10`
   - why it matters: gives a useful operator framing around `init / improve / perturb`, which is directly relevant to discrete optimization tasks
   - link:
     - <https://arxiv.org/abs/2602.10233>

6. `VISTA`
   - paper date: `2026-03-19`
   - why it matters: emphasizes interpretable reflective prompt optimization and clearer multi-agent decomposition
   - link:
     - <https://arxiv.org/abs/2603.18388>

## How To Use Them In This Repo

- `AlphaEvolve` and `ImprovEvolve` should shape the discrete optimization track.
- `GEPA` and `C-Evolve` should shape the future prompt optimization track.
- `AlphaResearch` should shape the longer-term outer-loop research workflow.
- `VISTA` should shape auditability, memory visibility, and process decomposition.
