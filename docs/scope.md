# Scope

This repo is an `autoresearch` workbench only when it is operating on the **outer loop**:

- selecting prompts
- selecting code patches
- selecting operator families
- selecting solver structures
- selecting optimizer schedules or training recipes
- selecting what experiment to run next

## In Scope

These are the right targets for the current system:

- discrete artifacts with a verifier
- code or prompt mutations with measurable outcomes
- evolutionary search over operator libraries
- memory-guided proposal generation
- replay-conditioned selection and write-back
- hybrid outer-loop search around a training loop

Examples:

- Euclidean TSP operator search
- Weighted Max-Cut operator search
- prompt architecture search with deterministic evaluation
- optimizer schedule search for a small model under fixed budget

## Out Of Scope

These are not the core object of `autoresearch` here:

- raw model weight updates by themselves
- large-scale pretraining loops without a clear verifier
- open-ended generation with no keep/discard metric
- pure continuous optimization that never changes a higher-level artifact

If the system is only running SGD on weights, that is `inner-loop training`, not the main autoresearch loop.

## Hybrid Scope

Some problems sit between the two:

- learning-rate schedule search
- optimizer family search
- loss shaping or reward shaping
- data curriculum search
- augmentation policy search
- architecture and hyperparameter search

These count as `autoresearch` because the **outer loop** is still choosing a research artifact, even though each candidate runs a continuous inner loop.

## Why The Demo Starts With Discrete Optimization

Discrete optimization is the cleanest first scope because it gives:

- deterministic feasibility checks
- clear objectives
- low local runtime on a Mac
- visually obvious optimization paths
- easy memory write-back rules

That makes it the right stage for:

- memory visualization
- operator lineage
- `delta_J` interpretation
- audit trails

## Continuous Optimization Position

Continuous optimization is not excluded forever. It is just not the first-class story for this repo yet.

The correct framing is:

- `inner loop`: model weights learn
- `outer loop`: the system chooses how the model should learn

So the future trainer integration belongs in a `hybrid` tier, not in the core discrete demo tier.
