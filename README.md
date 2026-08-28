# Grounded Multi-Turn Product Search

A submission-ready conversational retrieval agent for the TechJam multi-turn e-commerce
search challenge. The system searches a frozen 50,000-product Amazon Clothing catalogue,
asks targeted clarification questions, maintains active constraints and rejection history,
handles intent changes, and returns ranked `parent_asin` values through the official Python
interface.

The shipped path is offline, deterministic, and costs $0.00 per evaluation.

## Verified results

| Evaluation | Sessions | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Official public development set | 200 | 0.9950 | 0.9950 | 2.3200 | **0.969600** |
| Fixed Optuna v2 proxy fold | 800 | 0.9775 | 0.976875 | 2.94875 | **0.942837** |
| Untouched same-population folds, mean | 4 x 800 | 0.983438 | 0.981563 | 2.81469 | **0.949894** |
| Published weak BM25 baseline | 200 | 0.1250 | 0.068034 | 9.8100 | 0.106710 |

Trial 38 was frozen before the four independent folds were evaluated. It preserved the
public score and improved the untouched-fold mean by `+0.001213` over the previous shipped
configuration. The gain is small and is reported as a candidate-selection signal, not as a
private-score claim. See the [independent validation report](docs/validation/independent_validation.md).

## System design

The simulator reveals constraints derived from the target product's catalogue record. The
agent treats this as grounded provenance recovery:

1. Recognize official message forms and extract literal category and constraint spans.
2. Use a gated local DistilBERT tagger only for unfamiliar wording.
3. Mine catalogue-attested n-grams from visible customer text.
4. Maintain evidence, overrides, asked attributes, and rejected products per session.
5. Retrieve through conjunctive, backoff, and disjunctive FTS5 stages.
6. Rank by grounded evidence coverage and a self-calibrating purchase prior.
7. Return the highest-confidence candidate while gathering evidence, then widen at the
   final turn.

Optional Groq extraction and reranking modules are disabled by default. Official message
forms make zero external calls even when credentials are present.

## Repository guide

| Path | Purpose |
|---|---|
| [`submission/`](submission/) | Canonical agent, optional model integrations, and packaging documentation |
| [`starter/`](starter/) | Evaluator entry point, kept identical to the canonical agent |
| [`evaluator/`](evaluator/) | Official local simulator and scorer |
| [`robustness/`](robustness/) | Reproducible private-like proxy and population-shift suites |
| [`experiments/`](experiments/) | Experiment registry, runnable scripts, raw results, and complete findings |
| [`docs/`](docs/) | Competition evidence, design rationale, research, and validation reports |
| [`tests/`](tests/) | Contract, determinism, fallback, model-gate, and robustness tests |
| [`data/`](data/) | Released sessions, checksums, and catalogue download instructions |

Start with the [experiment registry](experiments/EXPERIMENT_INDEX.md) for a concise record
of every investigation. The [complete findings ledger](experiments/EXPERIMENT_FINDINGS.md)
contains methods, measurements, corrections, and negative results.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -r submission/requirements.txt
```

If `data/catalog.jsonl` is absent, download and decompress the frozen release catalogue:

```bash
curl -L -o data/catalog.jsonl.gz https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz
gzip -dk data/catalog.jsonl.gz
```

Verify the download against `data/SHA256SUMS`. The catalogue is not duplicated in Git.

## Reproduce the evaluation

Run the official public evaluator:

```bash
python -m evaluator.local_evaluator
```

Run the internal robustness suite:

```bash
python -m robustness.build_sets
python -m robustness.run_benchmark
```

Run all automated tests:

```bash
python -m unittest discover -s tests -v
```

The evaluator imports `starter.agent`. `starter/agent.py` and `submission/agent.py` must
remain byte-identical for a release.

## Model, network, and cost disclosure

- Required external API: none.
- Default external-model cost: $0.00.
- Local model: fine-tuned `distilbert-base-uncased`, stored through Git LFS.
- Optional API: Groq extraction or reranking requires both `GROQ_API_KEY` and an explicit
  feature flag.
- Safety boundary: optional model output must be a verbatim span from the visible message
  and must be attested in the frozen catalogue.
- Fallback: missing dependencies, weights, credentials, network, or valid model output
  returns control to the lexical pipeline.

Never commit credentials. Use environment variables or an ignored local `.env` file.

## Scope and limitations

The exact organizer-private target identities are unavailable. Internal proxies reproduce
the disclosed 1,406-target universe, 800 distinct private targets, public-target
disjointness, scenario proportions, and observed popularity distribution without claiming
to reconstruct private labels.

The agent benefits from the confirmed literal relationship between disclosed constraints
and target catalogue text. The repository also retains paraphrase and population-shift
tests as robustness characterization, even though paraphrasing was confirmed absent from
the official evaluation.

Catalogue and sessions derive from Amazon Reviews 2023 by McAuley Lab, UCSD. See
[`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md).
