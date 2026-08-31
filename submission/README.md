# Submission Package

This directory contains the canonical TechJam Track 4 agent and its optional local and
external model integrations. The official evaluator imports `starter.agent`; release
validation requires `starter/agent.py` and `submission/agent.py` to be byte-identical.

## Verified performance

| Evaluation | Sessions | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Official public development set | 200 | 0.9950 | 0.9950 | 2.3200 | **0.969600** |
| Tune800 fixed selection fold | 800 | 0.9775 | 0.976875 | 2.94875 | **0.942837** |
| Unseen4x800 independent mean | 3,200 | 0.983438 | 0.981563 | 2.81469 | **0.949894** |

Balanced trial 38 was frozen before independent evaluation. It preserved the public score
and improved the untouched-fold mean by `+0.001213` over the preceding configuration. See
the [independent validation report](../experiments/notes/independent_validation.md).

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
| [`agent.py`](agent.py) | Canonical submitted `Agent` implementation |
| [`llm_extract.py`](llm_extract.py) | Optional Groq-based extraction for unrecognized wording |
| [`llm_rerank.py`](llm_rerank.py) | Optional experimental reranker, disabled and rejected for official use |
| [`models/scaffolding_tagger/`](models/scaffolding_tagger/) | Fine-tuned local DistilBERT tagger stored through Git LFS |
| [`requirements.txt`](requirements.txt) | Submission runtime dependencies |

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
[experiment registry](../experiments/INDEX.md) and
[experiment-by-experiment decision log](../experiments/DECISION_LOG.md).
The root [README](../README.md#final-shipped-pipeline) specifies the complete final
execution path, including the gated BERT fallback and disabled optional API paths.

## Network, model, and cost disclosure

The official configuration is offline.

| Item | Shipped behavior |
|---|---|
| External network required | No |
| External API calls | Zero |
| External-model cost | $0.00 |
| Public evaluation latency | 17.93 seconds for 200 sessions, including index construction, in the final local audit environment |
| Public token usage | 0 prompt tokens and 0 completion tokens |
| Local model | Fine-tuned `distilbert-base-uncased` token tagger |
| Missing local model | Deterministic lexical fallback |
| Optional Groq extraction | Disabled unless `LLM_EXTRACT=1` and `GROQ_API_KEY` are both present |
| Optional Groq reranking | Disabled; experiments showed a negative effect |

Optional extraction is constrained by two checks. It runs only on a message that fails the
official-form recognition gate, and every returned requirement must be both a verbatim span
of the visible message and attested in the frozen catalogue. Invalid output is discarded.

Failure tests cover missing credentials, DNS failure, timeouts, retryable and terminal HTTP
errors, malformed JSON, empty output, hallucinated spans, and rate-limit exhaustion. Every
tested failure returns the deterministic result without raising. See
[`44_llm_failure_modes.py`](../experiments/log/44_llm_failure_modes.py).

Credentials must be supplied through the environment or a local ignored `.env` file. They
must never be committed.

Supported environment variables are documented in [`.env.example`](../.env.example):

| Variable | Default | Purpose |
|---|---|---|
| `BERT_EXTRACT` | `1` | Enables the gated local tagger on unrecognized messages |
| `BERT_TAGGER_DIR` | bundled model directory | Overrides the local model path |
| `BERT_KEEP` | `0.30` | Content-token retention threshold |
| `LLM_EXTRACT` | `0` | Enables optional Groq extraction |
| `LLM_RERANK` | `0` | Enables the rejected experimental Groq reranker |
| `GROQ_API_KEY` | unset | Supplies optional API credentials |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Selects the optional Groq model |
| `LLM_RPM` | `25` | Optional extraction request-rate limit |
| `LLM_TPM` | `5500` | Optional extraction token-rate limit |
| `LLM_TIME_BUDGET` | `5400` | Optional extraction network-time budget in seconds |
| `LLM_EXTRACT_CACHE` | submission-local cache | Overrides the extraction cache path |
| `LLM_CACHE` | submission-local cache | Overrides the reranking cache path |

## Sequential disclosure

The agent returns its highest-confidence candidate during turns 1 through 9 and returns up
to ten candidates at turn 10. Rejection feedback moves past unsuccessful candidates while
preserving full final-turn recall. This policy is valid under the official contract, which
specifies a maximum list size and no minimum list size.

The resulting public MRR equals HitRate at `0.995`, which means every successful session
hits at rank 1. This benefit and the metric effect of shorter lists are reported explicitly
in the [findings ledger](../experiments/FINDINGS.md).

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

See the [robustness benchmark](../experiments/README.md) and
[robustness audit](../experiments/notes/robustness_audit.md) for the full evidence.
