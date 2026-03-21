# Benchmarks

This repo should treat benchmarks as a staged ladder, not a single mixed pool.

## Tier 0: Verifier Bench

Purpose:

- validate the task/evaluator/memory pipeline
- catch correctness bugs early
- keep scoring stable

Requirements:

- deterministic verifier
- cheap local runtime
- obvious feasibility constraints

Metrics:

- feasible rate
- objective value
- `delta_J`
- runtime per generation
- write-back rate

## Tier 1: Mac Demo Bench

Purpose:

- power the local visual demo
- make memory and operator effects obvious
- run repeatedly on a laptop without cluster dependencies

Current recommended tasks:

1. `Clustered Euclidean TSP`
2. `Weighted Max-Cut with Community Structure`

Why these two:

- they are visually inspectable
- they are NP-hard enough to be interesting
- they expose clear operator families
- they make memory reuse legible

Metrics:

- baseline objective
- best-so-far curve
- generation count to plateau
- relative improvement vs baseline
- `delta_J`
- memory hit rate
- memory write-back count

## Tier 2: Family Generalization Bench

Purpose:

- test whether a learned operator or memory rule transfers across instance families
- prevent overfitting to one hand-crafted instance

Example pools:

- multiple Euclidean TSP instance families
- multiple weighted Max-Cut families
- bin packing or knapsack families

Metrics:

- average rank across instances
- cross-instance reuse uplift
- archive coverage
- transfer stability

## Tier 3: Hybrid Outer-Loop Bench

Purpose:

- search training recipes or inference policies around a fixed inner loop

Examples:

- prompt architecture search
- optimizer schedule search for a tiny model
- reward/loss shaping search
- retrieval policy search

Metrics:

- task accuracy or pass rate
- cost
- latency
- sample efficiency
- reproducibility under fixed seeds

## Audit Requirements

Every benchmark tier should expose:

- deterministic task id
- exact instance definition
- exact operator used
- exact objective before and after mutation
- `J` and `delta_J`
- retrieved memories
- written memories
- replayable run artifact

The discrete demo should also expose:

- memory markdown ledger
- candidate lineage
- objective curve
- operator family comparison
