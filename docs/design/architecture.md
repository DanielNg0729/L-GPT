# System Architecture

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
    -> deterministic extraction
    -> optional gated scaffolding extraction
    -> catalogue-attested n-gram mining
    -> session-state update and override handling
    -> conjunctive, backoff, and disjunctive retrieval
    -> evidence-coverage ranking and calibrated prior
    -> rejection demotion and sequential disclosure
    -> validated response
```

## Components

### Safety envelope

`Agent.respond()` is total over expected evaluator inputs. Optional model, network, and
parsing failures return to the deterministic ranking. Recommendations are normalized by the
evaluator contract, and the agent retains a last-good fallback.

### Recognition and extraction

Anchored patterns recognize every released simulator message form. Recognized messages use
deterministic parsing only. An unrecognized message may enter the local DistilBERT
scaffolding tagger and the optional Groq extractor, but only if their dependencies and
feature flags are available.

Every optional extracted phrase must pass two hard checks:

1. it is a verbatim span of the visible customer message;
2. it is attested in the frozen product catalogue.

This prevents a model from inventing a constraint or product identifier.

### Catalogue-grounded mining

The miner enumerates bounded n-grams from visible text and retains only phrases with a
catalogue document frequency between 1 and `DF_CAP`. Longer matching phrases take
precedence. This channel is nearly neutral on official messages but becomes the principal
fallback under wording changes.

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
