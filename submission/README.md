# Submission Package

This directory contains the canonical TechJam Track 4 agent and its optional local and
external model integrations. The official evaluator imports `starter.agent`, which **re-exports** the class defined
here rather than copying it. `tests/test_submission_contract.py` asserts object identity
(`starter.agent.Agent is submission.agent.Agent`), so the scored entry point and the
implementation cannot drift. An earlier byte-copy arrangement had already drifted when it
was audited, which is why identity is asserted instead.

## Verified performance

| Evaluation | Sessions | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Official public development set | 200 | 0.9950 | 0.9950 | 2.3200 | **0.969600** |
| Tune800 fixed selection fold | 800 | 0.9775 | 0.976875 | 2.94875 | **0.942837** |
| Unseen4x800 independent mean | 3,200 | 0.983438 | 0.981563 | 2.81469 | **0.949894** |

Balanced trial 38 was frozen before independent evaluation. It preserved the public score
and improved the untouched-fold mean by `+0.001213` over the preceding configuration. See
the [independent validation report](https://github.com/DanielNg0729/L-GPT/blob/full/experiments/notes/independent_validation.md).

These internal proxy results are generalization evidence, not private-label estimates.

## Architecture

| Layer | Responsibility |
|---|---|
| Safety envelope | Converts failures into a valid ranking response rather than raising an exception |
| Evidence extraction | Parses official templates and mines only catalogue-attested spans from visible text |
| Recognition gate | Keeps official messages on the deterministic path and gates unfamiliar wording |
| Session ledger | Accumulates active evidence, intent changes, questions, and rejected products |
| Probe policy | Requests the highest-value undisclosed information |
| Retrieval | Combines conjunctive, backoff, and disjunctive FTS5 candidate generation |
| Reranking | Scores weighted evidence coverage and a self-calibrating target-population prior |
| Disclosure | Presents the best candidate while collecting evidence, then widens on the final turn |

The design follows the simulator's grounded information channel: disclosed constraints are
derived from target catalogue fields. Exact phrase provenance is therefore more reliable
than general semantic similarity for the official task.

## Package contents

| File or directory | Purpose |
|---|---|
| [`agent.py`](agent.py) | Canonical `Agent`: the deterministic pipeline and every gate |
| [`span_node.py`](span_node.py) | Exact catalogue span recovery from unrecognised wording |
| [`route_node.py`](route_node.py) | Dialogue-act router; decides override vs no-evidence |
| [`bert_extract.py`](bert_extract.py) | Scaffolding tagger: strips filler before mining |
| [`llm_resolve.py`](llm_resolve.py) | Deparaphraser: unattested value to catalogue-attested value |
| [`llm_rescue.py`](llm_rescue.py) | Whole-transcript recovery, once per session, on a stall |
| [`llm_message.py`](llm_message.py) | Optional phrasing writer; off by default, no score effect |
| [`llm_extract.py`](llm_extract.py) | Optional Groq extraction; off, superseded by the span node |
| [`llm_rerank.py`](llm_rerank.py) | Optional reranker; off, measured negative and rejected |
| [`catalogue_attribute_dictionary.jsonl`](catalogue_attribute_dictionary.jsonl) | Frozen attribute vocabulary used by the tagger's other half |
| [`demo.py`](demo.py) | Annotated single-session walkthrough |
| [`requirements.txt`](requirements.txt) | Submission runtime dependencies |

The two learned checkpoints are **not in this repository**. They are gitignored and resolve
from the Hugging Face Hub on first use — see [`../docs/MODEL_ARTIFACT_POLICY.md`](../docs/MODEL_ARTIFACT_POLICY.md).
An earlier revision of this file described them as "stored through Git LFS", which was true
before the migration and is not true now: a Git-LFS pointer left in place of real weights
loaded "successfully" and scored as if healthy, which is the failure the migration ended.

## Reproduction

Install dependencies from the repository root:

```bash
python -m pip install -r submission/requirements.txt
```

Run the unchanged official evaluator:

```bash
python -m evaluator.local_evaluator
```

Run contract and safety tests:

```bash
python -m unittest discover -s tests -v
```

Run a demonstrated multi-turn intent-override session:

```bash
python -m submission.demo --sample-id public_0002
```

See the annotated [`demo transcript`](../docs/DEMO.md).

The complete investigation history is available in the
[experiment registry](https://github.com/DanielNg0729/L-GPT/blob/full/experiments/INDEX.md) and
[experiment-by-experiment decision log](https://github.com/DanielNg0729/L-GPT/blob/full/experiments/DECISION_LOG.md).
The root [README](../README.md#final-shipped-pipeline) specifies the complete final
execution path, including the gated BERT fallback and disabled optional API paths.

## Network, model, and cost disclosure

The span node is unconditional deterministic core. Of the five configurable layers
(route classifier, BERT tagger, deparaphraser, transcript rescue, and message writer),
the configuration enables four when a Groq key is available; the message writer remains
off. Without a key, only the two local layers are active.

| Item | Shipped behavior |
|---|---|
| External network required | No |
| External API calls | Zero |
| External-model cost | $0.00 |
| Public evaluation latency | 17.93 seconds for 200 sessions, including index construction, in the final local audit environment |
| Public token usage | 0 prompt tokens and 0 completion tokens |
| Local model | Fine-tuned `distilbert-base-uncased` token tagger |
| Missing local model | Deterministic lexical fallback |
| Optional Groq deparaphraser and rescue | Flagged on, but inert without `GROQ_API_KEY`. Neither is reachable on the scored suites: the recognition gate matches every official message, and the rescue additionally needs a stalled session, which does not occur at MTTC 2.225 |
| Optional Groq extraction | Disabled; superseded by the offline span node |
| Optional Groq reranking | Disabled; experiments showed a negative effect |

Optional extraction is constrained by two checks. It runs only on a message that fails the
official-form recognition gate, and every returned requirement must be both a verbatim span
of the visible message and attested in the frozen catalogue. Invalid output is discarded.

Failure tests cover missing credentials, DNS failure, timeouts, retryable and terminal HTTP
errors, malformed JSON, empty output, hallucinated spans, and rate-limit exhaustion. Every
tested failure returns the deterministic result without raising. See
[`44_llm_failure_modes.py`](https://github.com/DanielNg0729/L-GPT/blob/full/experiments/log/44_llm_failure_modes.py).

Credentials must be supplied through the environment or a local ignored `.env` file. They
must never be committed.

Supported environment variables are documented in [`.env.example`](../.env.example):

Local layers — no credential, no network, no per-call cost:

| Variable | Default | Purpose |
|---|---|---|
| `V2_ROUTE` | `1` | Dialogue-act router on unrecognised messages |
| `BERT_EXTRACT` | `1` | Scaffolding tagger on unrecognised messages |
| `BERT_KEEP` | `0.30` | Content-token retention threshold |
| `BERT_DEVICE` | `auto` | Forces `cpu` if set to `cpu` |
| `MESSAGE_VARIETY` | `1` | Deterministic phrasing variety; pure string work |

Hosted layers — flagged on, but inert until `GROQ_API_KEY` is present:

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | unset | Activates every hosted layer; without it all are inert |
| `LLM_RESOLVE` | `1` | Deparaphraser for unattested values |
| `LLM_RESCUE` | `1` | Whole-transcript recovery on a stalled session |
| `LLM_RESCUE_TURN` | `5` | First turn the rescue may fire |
| `LLM_RESCUE_REJECTS` | `4` | Rejected candidates required before it may fire |
| `LLM_MESSAGE` | `0` | Phrasing writer; no score effect, off by design |
| `LLM_EXTRACT` | `0` | Superseded by the span node |
| `LLM_RERANK` | `0` | Measured negative, rejected |
| `LLM_RPM` / `LLM_TPM` | `25` / `5500` | Request- and token-rate limits |
| `LLM_TIME_BUDGET` | `5400` | Network-time budget in seconds |
| `LLM_CACHE`, `LLM_EXTRACT_CACHE` | submission-local | Cache paths |

**Two variables are documented but should be left unset.** `BERT_TAGGER_DIR` and
`V2_ROUTE_MODEL_DIR` are absolute overrides, returned ahead of both the local-weights check
and the Hub fallback. `submission/models/` is gitignored, so on a fresh clone the directory
they normally name does not exist; setting them there makes the load raise and the layer
disable itself **silently**. Unset, resolution is local weights if genuinely present,
otherwise the Hub — which is what a clone needs. `BERT_TAGGER_HUB` and `V2_ROUTE_HUB`
override the Hub ids and are equally unnecessary.

**Do not set `GROQ_MODEL` globally.** Each layer selects its own model in code — `20b` for
resolve, rescue and message; `120b` for extract and rerank. A global `GROQ_MODEL` overrides
all five at once, moving the three `20b` layers onto a model they were never measured on.

## Sequential disclosure

The agent returns its highest-confidence candidate during turns 1 through 9 and returns up
to ten candidates at turn 10. Rejection feedback moves past unsuccessful candidates while
preserving full final-turn recall. This policy is valid under the official contract, which
specifies a maximum list size and no minimum list size.

The resulting public MRR equals HitRate at `0.995`, which means every successful session
hits at rank 1. This benefit and the metric effect of shorter lists are reported explicitly
in the [findings ledger](https://github.com/DanielNg0729/L-GPT/blob/full/experiments/FINDINGS.md).

## Robustness and limitations

- The exact private target identities are unavailable. Proxy tests match disclosed
  cardinalities, disjointness, scenario mix, and observed target-popularity statistics.
- The popularity prior is the only component with meaningful population dependence. It is
  calibrated from aggregate retrieved-pool statistics without using labels or product
  identity.
- Message and constraint rewrite tests are retained as stress characterization. Project Q&A
  notes report that official testing will not paraphrase disclosed values; this statement is
  not reproduced in the written specification, so it is not used as an official-score claim.
- LLM reranking, cross-encoders, dense retrieval, and multiple learned rankers were rejected
  after negative held-out measurements. Their code and outputs remain versioned.
- The optional local model is approximately 265 MB before dependencies. It is loaded lazily
  and never loaded during official-form evaluation, but its package size remains a practical
  deployment limitation.

Given more time, we would quantize or distill the optional tagger, validate against an
organizer-provided eligible-target pool, and collect independently authored paraphrase data.
We would not continue optimizing against the consumed validation folds.

See the [robustness benchmark](https://github.com/DanielNg0729/L-GPT/blob/full/experiments/README.md) and
[robustness audit](https://github.com/DanielNg0729/L-GPT/blob/full/experiments/notes/robustness_audit.md) for the full evidence.
