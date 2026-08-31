# System Architecture

This is the per-component rationale behind the pipeline. For the shape of the pipeline and
what runs when, start with the [README's Architecture section](../../README.md#architecture);
this document assumes it and explains *why* each part is built the way it is.

## Design principle

The official simulator discloses constraints derived from the hidden target's catalogue
record. The central task is therefore grounded provenance recovery: identify the product
that contains the accumulated evidence, then rank ambiguous matches using a calibrated
target-selection prior.

This finding determines the architecture. Exact lexical evidence remains authoritative.
Learned and external models may help interpret unfamiliar wording, but they cannot introduce
requirements that are absent from the visible message or frozen catalogue.

## Request flow

```text
customer message
    -> official-form recognition
    -> deterministic extraction                       (recognised messages)
    -> dialogue-act routing                           (unfamiliar wording only)
    -> exact catalogue span and category recovery     (unfamiliar wording only)
    -> gated scaffolding tagging                      (unfamiliar wording only)
    -> catalogue-attested n-gram mining
    -> attribute deparaphrasing                       (unattested VALUES only)
    -> session-state update and override handling
    -> conjunctive, backoff, and disjunctive retrieval
    -> evidence-coverage ranking and calibrated prior
    -> rejection demotion and sequential disclosure
    -> validated response
```

Two of those gates are governed by different questions, and conflating them is the easiest
mistake to make here. Recognition asks whether the **message** has a familiar shape;
attestation asks whether a **value** exists in the frozen catalogue. A perfectly recognised
message can carry a value the catalogue has never seen, which is why the deparaphraser is
reachable on clean traffic while the two classifiers are not.

## Components

### Safety envelope

`Agent.respond()` is total over expected evaluator inputs. Optional model, network, and
parsing failures return to the deterministic ranking. Recommendations are normalized by the
evaluator contract, and the agent retains a last-good fallback.

### Recognition and extraction

Anchored patterns recognize every released simulator message form. Recognized messages use
deterministic parsing only. An unrecognized message may enter three further mechanisms, in
order of cost: a DistilBERT **dialogue-act router**, which reads what the turn *means* when
the wording is unfamiliar; **exact span recovery**, which finds catalogue-attested one- to
three-token values and the longest matching category; and a DistilBERT **scaffolding
tagger**, which strips conversational filler so mining sees only product text. Each runs
only if its dependencies and feature flags are available, and every failure path returns the
pre-model behaviour.

The router exists because the recognition gate is a *detector*, not a classifier: it reports
that a message is unfamiliar without saying what it means. Two behaviours depend on the
meaning -- clearing the rejection set when the customer changes their mind, and contributing
nothing when they state no preference. Hand-written lexical cues reached 37.5% and 0.0% on
held-out templates against the model's 100%; the no-evidence row is decisive, because the
held-out templates share zero vocabulary with the training ones.

A hosted span extractor was also built and is retained **disabled**: the local tagger beat
it on the hardest transform, so it was superseded rather than adopted.

Every optional extracted phrase must pass two hard checks:

1. it is a verbatim span of the visible customer message;
2. it is attested in the frozen product catalogue.

This prevents a model from inventing a constraint or product identifier.

### Catalogue-grounded mining

The miner enumerates bounded n-grams from visible text and retains only phrases with a
catalogue document frequency between 1 and `DF_CAP`. Longer matching phrases take
precedence. This channel is nearly neutral on official messages but becomes the principal
fallback under wording changes.

### Attribute deparaphrasing

Reached only where a value fails catalogue attestation -- exactly where the agent would
otherwise suppress the clause -- so its floor is the shipped behaviour. A hosted model is
given the unattested phrase alone, with no catalogue context and no candidate list, and
proposes the trade term it names. The catalogue then decides: a proposal with no document
frequency is discarded.

The absence of a candidate list is deliberate and measured. A retrieval-augmented variant
scored below doing nothing, because a list captures the answer whatever the instruction
says: offered candidates as hints it should ignore when wrong, the model answered off-list
twice in 21 attempts, against 23 of 23 unaided.

Accepted proposals enter at a **reduced weight**, not at constraint strength. This is the
single largest decision in this layer: the same knowledge at full weight recovers 81.5% of a
perfect resolver against roughly 96% attenuated. The difference is entirely what a wrong
proposal costs. A later audit confirmed it from the other direction -- deleting 41
confidently wrong proposals changed the score by exactly 0.000000, because at the attenuated
weight they were already unable to outrank real evidence.

### Session ledger

Per-session state tracks active evidence, asked attributes, known-wrong recommendations,
turn number, and intent override status. An override replaces incompatible evidence rather
than appending contradictory constraints.

### Probe policy

The agent asks for useful undisclosed evidence while returning recommendations in the same
response. The official interface does not require an exclusive ask-or-recommend decision,
so a learned dialogue policy was unnecessary and measured alternatives did not improve
MTTC.

### Retrieval

An in-memory SQLite FTS5 index supports three candidate-generation stages:

1. conjunctive phrase queries over selective evidence;
2. shorter conjunctive backoffs when the strict query is empty;
3. a disjunctive phrase query as a recall floor.

Candidate generation is lexical and does not use the population prior. This separation
allows the prior to be calibrated from retrieved-pool statistics without circularity.

### Reranking

Each candidate receives weighted coverage for attested evidence. Phrase weight combines
evidence source, document frequency, and a bounded length factor. The remaining ambiguity
is resolved with a popularity prior based on `log1p(rating_number)`.

The prior is the only component with material population dependence. Its effective weight
is scaled by the aggregate popularity of retrieved pools, which uses neither target labels
nor product identity. A fresh agent or insufficient observations leaves the static prior
unchanged.

### Rejection and disclosure

If a session continues, previously presented candidates are demoted because they did not
produce a valid hit. Override guards prevent pre-override turns from being interpreted as
negative evidence.

The agent returns one highest-confidence candidate during turns 1 through 9, then widens to
ten candidates at turn 10. This preserves the final recall budget while placing successful
hits at rank 1.

## Model and network boundary

The submitted path is offline and deterministic. The local tagger is unreachable on
official message forms and fails closed to lexical extraction. Groq extraction and
reranking require an explicit feature flag and credentials; reranking remains disabled
because it performed worse than the deterministic popularity tie-breaker.

## Evidence for excluded alternatives

| Alternative | Measurement |
|---|---|
| Dense bi-encoder with rank fusion | Tuning TechnicalScore delta `-0.047` |
| Local cross-encoder reranking | Held-out delta `-0.030` |
| Groq listwise tie reranking | End-to-end delta approximately `-0.027`; 41.2% target-first versus 57.4% for popularity |
| Public-only learning-to-rank | Held-out delta `-0.040` |
| Synthetic learning-to-rank | Tune and holdout effects disagreed around zero |
| RM3 query expansion | Held-out delta `-0.004` |
| Learned probe policy | Worse MTTC and negative end-to-end deltas |

See the [experiment registry](../../experiments/INDEX.md) and
[complete findings](../../experiments/FINDINGS.md) for the full record.
