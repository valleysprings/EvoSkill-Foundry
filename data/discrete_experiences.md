# Discrete Seed Memory

- generated_at: repo-seed
- num_memories: 4

## Experience Units

### exp-discrete-feasibility-first

- source_task: seed
- family: agnostic
- delta_J: 0.16
- task_signature: discrete-opt, feasibility, objective-tracking
- failure_pattern: operators proposed invalid tours or partitions that looked promising but broke the task constraints
- successful_strategy: check feasibility first, then compare the objective on valid candidates only
- tool_trace_summary: candidate solution -> feasibility check -> objective -> normalized quality -> selective write-back
- reusable_rules: feasibility_first, objective_after_validity, deterministic_selection
- supporting_memory_ids: none

### exp-tsp-construct-then-repair

- source_task: seed
- family: route-search
- delta_J: 0.24
- task_signature: discrete-opt, tsp, euclidean, clustered
- failure_pattern: starting local search from the index-order route wastes generations on obvious edge crossings
- successful_strategy: build a nearest-neighbor seed, then intensify with 2-opt or crossing repair
- tool_trace_summary: constructive seed -> route crossing repair -> local improvement loop
- reusable_rules: construct_then_intensify, repair_crossings, two_opt_after_seed
- supporting_memory_ids: none

### exp-discrete-plateau-escape

- source_task: seed
- family: agnostic
- delta_J: 0.19
- task_signature: discrete-opt, local-search, plateau-escape
- failure_pattern: single-move local search plateaus early even when better solutions are nearby
- successful_strategy: use a non-local kick or pair move, then re-run greedy improvement from the perturbed state
- tool_trace_summary: local optimum -> perturbation -> intensification -> keep only measured gains
- reusable_rules: escape_plateau, pair_move_then_refine, perturb_then_intensify
- supporting_memory_ids: none

### exp-maxcut-greedy-flips

- source_task: seed
- family: graph-partition
- delta_J: 0.22
- task_signature: discrete-opt, max-cut, weighted-graph, community-structure
- failure_pattern: static parity partitions miss large gains from heavy-edge aware flips
- successful_strategy: seed the partition from heavy edges, then apply greedy node flips and occasional pair escapes
- tool_trace_summary: heavy-edge seed -> greedy gain flips -> pair perturbation on plateau
- reusable_rules: heavy_edge_seed, greedy_flip_gain, pair_flip_escape
- supporting_memory_ids: none
