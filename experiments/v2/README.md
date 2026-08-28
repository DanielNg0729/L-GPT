# V2 Semantic-Robustness Program

## Purpose

V1 is optimized for the released evaluator, where customer values are derived literally
from the target catalogue record. V2 tests the stronger product requirement stated in the
competition materials: an agent should preserve intent when message structure changes,
attribute values are semantically rephrased, or an intent override supersedes prior
preferences.

V2 does not modify `submission/agent.py`. That file remains the frozen V1 submission
candidate. Every V2 mechanism is evaluated in an isolated implementation and must preserve
the Official200 score before it can be considered for release.

## Evaluation protocol

| Suite | Target source | Language change | Role |
|---|---|---|---|
| `Official200` | Organizer-released public targets | None | Clean no-regression guard. |
| `FormatShift-Dev` | Public development targets | Reworded message templates, values copied literally | Develop format-robust parsing. |
| `SemanticShift-Dev` | Public development targets | Values rewritten into semantically equivalent language | Develop semantic grounding. |
| `FormatShift-Holdout800` | Target-disjoint catalogue targets | Unseen templates, literal values | Hold out format generalization. |
| `SemanticShift-Holdout800` | Target-disjoint catalogue targets | Unseen value paraphrases and templates | Sealed semantic generalization test. |
| `OverrideConflict-Holdout` | Target-disjoint catalogue targets | Explicit incompatible replacement at turn 3 or 4 | Test true intent replacement. |

The holdout suites are not organizer-private score estimates. They deliberately make no
assumption about the organizer's unknown target pool or its popularity distribution.

## Benchmark construction rules

1. A transformed message must preserve the target intent and the scenario timing.
2. Format shifts must remove dependence on released literal wrappers such as “What matters
   is”.
3. Semantic shifts must remove the original catalogue phrase or a direct lexical anchor.
   “Buckle closure” becoming “fastens with a buckle” is not sufficient because `buckle`
   remains an easy lexical bridge. “Fastens using a clasp” is a stronger test.
4. Every semantic rewrite is stored with its original source span, target, transformation
   family, and provenance. A human review sample is required before reporting results.
5. Development and holdout paraphrase families must be disjoint. A prompt or rewrite rule
   used to create holdout language cannot be used for tuning.
6. Override conflicts must replace a comparable attribute slot, retain category, and use a
   new target whose revised constraints are internally consistent.

## Candidate directions

### 1. Grounded semantic candidate generation

Embed the active customer intent and visible product documents. Retrieve a semantic top-N
candidate set only when the recognition gate sees unfamiliar wording. Union that set with
the lexical FTS5 ladder. The exact lexical ranker remains a feature of the hybrid, not the
only retriever.

This reopens dense retrieval under a different condition. V1 rejected dense fusion on
literal provenance messages, where semantic similarity blurred a signal that was already
exact. That result does not answer whether semantic retrieval restores recall after the
literal signal is removed.

### 2. Semantic-to-grounded evidence resolution

For each unfamiliar customer clause, retrieve a small set of catalogue-attested phrases or
product passages semantically. Only retrieved visible catalogue text may enter the evidence
ledger. This is preferable to accepting an ungrounded model interpretation.

### 3. Dual intent branches for overrides

On a high-confidence replacement cue, maintain two hypotheses:

- accumulated evidence, which protects V1-compatible sessions;
- category plus post-override evidence, which represents a true replacement.

Retrieve candidates from both hypotheses, then score the revised branch preferentially only
when the new evidence conflicts with a prior inferred slot. This avoids the blunt
category-only reset rejected in experiment 58.

### 4. Optional semantic parser

An offline or cached external parser may identify clauses and replacement cues, but it may
not fabricate catalogue facts. Its output must be linked to a semantic retrieval result in
the visible frozen catalogue.

## Decision rule

A V2 candidate is admissible only when it:

1. does not regress Official200 beyond a preregistered noise tolerance;
2. improves both semantic development and unseen semantic holdout conditions;
3. does not depend on organizer-private labels, target-pool assumptions, or network access
   in the default submission path; and
4. documents its latency, local asset size, and fallback behavior.

The V1 result ledger remains evidence about literal-provenance behavior. V2 reruns only the
experiments whose premise changes under semantic language, beginning with retrieval recall,
candidate generation, grounding, and override state.
