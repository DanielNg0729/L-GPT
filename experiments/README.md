# Experiment Suite

This directory contains the complete experimental record behind the submitted agent. It is
organized as a reproducible research package rather than as exploratory analysis.

## Contents

| Path | Purpose |
|---|---|
| [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md) | One-line question, result, and decision for every experiment |
| [`EXPERIMENT_DECISION_LOG.md`](EXPERIMENT_DECISION_LOG.md) | Full one-by-one result, ruling, and final-design impact for every experiment |
| [`EXPERIMENT_FINDINGS.md`](EXPERIMENT_FINDINGS.md) | Detailed methods, measurements, corrections, and interpretation |
| [`scripts/`](scripts/) | Runnable experiment and validation programs |
| [`results/`](results/) | Versioned raw JSON outputs used by the reports |
| [`archive/`](archive/) | Preserved original baseline implementation |
| `studies/` | Ignored local optimizer databases, caches, and temporary model artifacts |

## Evidence policy

Results are labeled according to their source:

- Official: produced by the unchanged local evaluator on the released 200 sessions.
- Internal proxy: produced by the official simulator over participant-generated target
  samples that follow disclosed organizer statistics.
- Stress test: deliberately disturbed input or population assumptions.
- Diagnostic: component-level analysis that is not an end-to-end score claim.

Negative and invalidated experiments remain in the registry. A mechanism is described as
shipped only if it is present in `submission/agent.py`. Historical gains can be superseded
by later joint optimization, so the final configuration is always reported separately from
the path used to discover it.

## Running experiments

Run programs from the repository root, for example:

```bash
python experiments/scripts/04_ablation.py
python experiments/scripts/30_robustness_benchmark.py
python experiments/scripts/57_independent_validation.py
```

Scripts write versionable outputs to `experiments/results/`. Large caches, Optuna databases,
and temporary training artifacts belong in `experiments/studies/` and are excluded from Git.

The later experiments depend on decisions and utilities introduced by earlier ones. The
registry is therefore the recommended entry point before reproducing an individual script.
