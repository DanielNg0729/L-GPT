# Grounded Multi-Turn Product Search Agent

An offline-first conversational retrieval agent for the TechJam multi-turn e-commerce
search challenge. It searches a frozen 50,000-product Amazon Clothing catalogue, asks
clarifying questions, maintains active constraints and rejection history, handles intent
overrides, and returns ranked `parent_asin` values through the official Python interface.

## Results

| Evaluation | Sessions | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Public development set | 200 | 0.995 | 0.995 | 2.320 | **0.96960** |
| Optuna v2 primary proxy | 800 | 0.9775 | 0.976875 | 2.94875 | **0.942837** |
| Independent same-population mean | 4 × 800 | 0.983438 | 0.981563 | 2.81469 | **0.949894** |
| Published weak BM25 starter | 200 | 0.125 | 0.068034 | 9.81 | 0.10671 |

The internal results are not private-score claims. Their targets are participant-safe proxies
that match the disclosed 1,406-target universe and 800-target distinctness constraints.
Trial 38 was frozen before the four independent folds were evaluated. See
[notes/independent_validation_report.md](notes/independent_validation_report.md).

## Architecture

1. Recognize official message shapes and extract category/constraint spans.
2. For unfamiliar wording only, optionally strip conversational scaffolding with a local
   DistilBERT token tagger; any failure returns to the deterministic path.
3. Mine catalogue-attested n-grams from visible customer text.
4. Accumulate evidence, overrides, asked attributes, and rejected products per session.
5. Retrieve through conjunctive, backoff, and disjunctive FTS5 rungs.
6. Rank by grounded evidence coverage plus a population-aware purchase prior.
7. Return the highest-confidence candidate while asking for more evidence, then widen on
   the final turn.

The optional Groq extraction/reranking modules are disabled unless explicitly configured.
The default official messages make zero external calls.

## Repository map

```text
submission/              canonical agent and local model integration
starter/                 evaluator entry point mirroring the submission
evaluator/               official local simulator and scorer
data/                    public sessions and catalogue release/checksum
robustness/              reproducible internal proxy and population-shift suite
tests/                   evaluator, model-gate, API-failure, and robustness tests
notes/                   experiment ledger, ablations, literature and slide transcription
docs/                    official contract, specification, scoring and submission rules
```

## Setup

Python 3.10 or newer is required.

```bash
python -m pip install -r submission/requirements.txt
```

Download and decompress the frozen catalogue if `data/catalog.jsonl` is absent:

```bash
curl -L -o data/catalog.jsonl.gz https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz
gzip -dk data/catalog.jsonl.gz
```

Verify it using `data/SHA256SUMS`. The catalogue is release data and is not duplicated in
the Git history.

## Run the official public evaluator

```bash
python -m evaluator.local_evaluator
```

The evaluator imports `starter.agent`. `starter/` and `submission/` are kept synchronized
for the release branch; `submission/agent.py` is the canonical entry file for packaging.

## Run the internal robustness benchmark

```bash
python -m robustness.build_sets
python -m robustness.run_benchmark
```

Run only the realistic proxy:

```bash
python -m robustness.run_benchmark --only organizer_proxy_800
```

The benchmark also includes catalogue-wide review-weighted, uniform, and inverse-popularity
sets. Those are stress tests rather than estimates of the private score.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Models, network and cost

- Local tagger: fine-tuned `distilbert-base-uncased`, stored with Git LFS.
- Local inference dependencies: PyTorch and Transformers.
- External API required: no.
- Default external-model cost: $0.00.
- Optional Groq extraction/reranking: requires `GROQ_API_KEY` plus its explicit feature
  flag; all returned spans are verbatim-checked and catalogue-attested.
- Deterministic fallback: if model dependencies, weights, credentials, or network are
  unavailable, the agent continues through its lexical retrieval pipeline.

Never commit credentials. Use environment variables or a local ignored `.env` file.

## Evidence and limitations

The complete experiment ledger contains more than 50 passes covering dense retrieval,
cross-encoders, learning-to-rank, query expansion, LLM extraction/reranking, disclosure
policies, population shift, and local NLP extraction. Rejected approaches are retained so
the final architecture is auditable rather than reconstructed from successes alone.

The exact 1,206 unseen eligible target ASINs cannot be recovered from participant-visible
data. The internal proxy therefore models disclosed cardinalities and purchase likelihood
without claiming to reproduce organizer-private labels. Private targets are user- and
target-disjoint, so public-only hyperparameter gains remain subject to distribution shift.

Catalogue and sessions derive from Amazon Reviews 2023 by McAuley Lab, UCSD. See
[DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md).
