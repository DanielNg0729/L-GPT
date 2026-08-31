# Independent validation of Optuna v2 candidates

## Protocol

Candidate parameters were frozen in `experiments/validation_candidates.json` before any
held-out fold was evaluated. Both Optuna workers were stopped. No result in this report was
fed back into tuning.

Stage 1 tests ordinary sample variance with four independent 800-session folds generated
from the same disclosed-cardinality population process used by the optimization fold. Each
fold has 800 distinct targets, zero public overlap, ten equal popularity strata and the
official 40/40/15/5 behavior mix.

Stage 2 changes only the target-population popularity statistic. It moves 5%, 10% or 20%
total-variation probability mass between the lower and upper five popularity deciles, in
both directions, with two independent replicates per condition. Target count, distinctness,
candidate-pool size and behavior mix remain fixed.

## Stage 1: Unseen4x800, same population and different samples

| Candidate | Primary-fold delta | Held-out mean score | Held-out paired delta | Delta stdev | Worst fold |
|---|---:|---:|---:|---:|---:|
| Shipped |  -  | 0.948681 | 0 | 0 | 0 |
| Conservative trial 44 | +0.002250 | 0.948353 | **−0.000328** | 0.001770 | −0.001937 |
| Balanced trial 38 | +0.004137 | **0.949894** | **+0.001213** | 0.001587 | −0.000237 |
| Aggressive trial 106 | +0.004500 | **0.949894** | **+0.001213** | 0.001579 | −0.000187 |

Trial 44 fails replication. Trials 38 and 106 retain a positive mean gain, but the gain
shrinks by roughly 71% relative to the optimization fold. This is evidence of selection
overfit, not total failure. With only four independent folds, the positive mean is not a
high-confidence statistical proof; it is a candidate-selection signal.

Trial 38 fold deltas were `+0.003063`, `−0.000237`, `+0.000025`, and `+0.001999`.

## Stage 2: Shifted12x800, controlled population disturbance

Mean paired TechnicalScore delta versus shipped:

| Candidate | Less popular TV 5% | TV 10% | TV 20% | More popular TV 5% | TV 10% | TV 20% |
|---|---:|---:|---:|---:|---:|---:|
| Conservative trial 44 | −0.000369 | −0.000469 | −0.000007 | −0.000088 | +0.000619 | +0.000819 |
| Balanced trial 38 | −0.000337 | −0.000275 | +0.001338 | +0.001399 | +0.002062 | +0.002287 |
| Aggressive trial 106 | −0.000306 | −0.000294 | +0.001344 | +0.001531 | +0.002118 | +0.002275 |

The disturbance response is asymmetric. Trials 38 and 106 improve as the population moves
toward more popular targets. Mild shifts toward less-popular targets produce a small mean
regression. The 20% less-popular result becomes positive, so the curve is not monotonic;
with two replicates per condition, individual levels should not be overinterpreted. The
worst individual disturbed-fold delta for trial 38 was `−0.001150`.

## Decision

Select **balanced trial 38** for any final candidate-vs-shipped bake-off:

- it preserves the official public score exactly (`0.969600`);
- it improves the independent same-population mean by `+0.001213`;
- it is effectively tied with trial 106 across every independent condition; and
- it is safer than trial 106's small public regression.

Do not tune trial 38 using these folds. They are now consumed validation evidence. If a
shipping change is considered, the remaining legitimate checks are deterministic runtime,
packaging, and contract tests - not another parameter search over this validation suite.

Raw results: `experiments/results/out_57_independent_validation.json`.
