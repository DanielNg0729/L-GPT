# V2.01 Semantic Feature Grounding

## Question

Can a local semantic encoder map an attribute paraphrase to a visible catalogue feature phrase
without changing the exact-form V1 behavior?

## Fixed internal test sets

| Suite | Purpose | Status in this experiment |
|---|---|---|
| Official200 | Exact-form no-regression control | Evaluated |
| SemanticShift-Dev200 | Development set of target-disjoint attribute paraphrases | Evaluated |
| SemanticShift-Holdout800 | Target-disjoint semantic holdout | Not opened |

`SemanticShift-Dev200` and `SemanticShift-Holdout800` are documented in
[`experiments/studies/README.md`](../../experiments/studies/README.md). The held-out set remains sealed
because no candidate met the development and clean-control requirements.

## Candidates

| Candidate | Description |
|---|---|
| Literal V1 | Existing deterministic literal resolver. |
| Development phrase map | Deliberately weak control that maps development paraphrases to their known canonical phrases after a full literal miss. It proves the harness and the recovery opportunity, but is not admissible. |
| Semantic feature grounding | `all-MiniLM-L6-v2` nearest-neighbour retrieval over visible frozen catalogue feature strings. It may only add a catalogue-attested feature phrase. |

## Results

| Candidate | SemanticShift-Dev200 | Official200 | Clean semantic activations |
|---|---:|---:|---:|
| Literal V1, canonical-value replay | 0.780750 | Not applicable | 0 |
| Literal V1 | 0.063100 | 0.969600 | 0 |
| Development phrase map | 0.616667 | 0.969600 | 0 |
| Semantic feature, full-phrase gate | 0.078700 | 0.968600 | 13 |
| Semantic feature, strict lexical-fallback gate | 0.063100 | 0.969600 | 0 |

The canonical-value replay uses the exact same Dev200 targets, profile, scenario type, message
wrappers, turn timing, probe sequence, candidate generation, ranking, and popularity prior. It
changes only `paraphrase` back to `canonical`, producing 0.780750. This is the relevant
upper reference for the present card construction. It is lower than Official200 because these
are broad target-disjoint targets with a deliberately reduced semantic card, not the released
review-weighted target distribution with its full organiser card.

The phrase map recovers 0.553567 of the 0.717650 semantic-loss gap, or 77.1 percent, which
demonstrates that semantic attribute recovery can materially improve this synthetic task. It is
intentionally built from development answers and is therefore only a control. The genuine
encoder is not competitive: with a full-phrase gate it accepts 618 of 623 unresolved development
phrases, mostly incorrectly, and causes a clean-score regression. With the strict gate it
correctly preserves Official200 but never activates because the lexical resolver finds
incidental globally-attested n-grams in every paraphrase.

## Diagnosis

The grounding inventory is broad catalogue feature text. It contains semantically adjacent but
non-equivalent strings. A nearest feature phrase is therefore not sufficient evidence of the
customer's intended attribute. The lexical resolver also treats phrases such as `repels light`
as catalogue-attested even though they are not a useful semantic interpretation of `repels light
moisture`.

## Decision

Reject broad nearest-feature grounding as a standalone node. Do not evaluate it on
SemanticShift-Holdout800.

The next V2 node is attribute-family recognition. Its output must be independently guarded and
may not modify state. If it can assign a high-confidence family such as material, colour,
closure, weather resistance, or care, a later grounding node can search only the matching visible
catalogue family rather than the full feature inventory.
