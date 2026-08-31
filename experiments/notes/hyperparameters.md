# Final Hyperparameter Configuration

This document records the exact balanced trial 38 values shipped in
`submission/agent.py`. Historical search spaces and candidate results remain in the Optuna
programs and raw study artifacts.

## Retrieval index

| Parameter | Final value | Role |
|---|---:|---|
| BM25 title weight | 3.264 | Title-field contribution |
| BM25 category weight | 1.202 | Category-field contribution |
| BM25 feature weight | 1.869 | Feature-bullet contribution |
| BM25 details weight | 2.561 | Structured-details contribution |
| BM25 store weight | 1.659 | Store-field contribution |
| BM25 description weight | 2.153 | Description contribution |
| `DF_CAP` | 2,715 | Maximum document frequency accepted by grounded mining |
| mining `maxn` | 12 | Longest candidate n-gram |
| mining `minn` | 4 | Shortest candidate n-gram |
| `POOL` | 1,200 | Candidate pool limit per retrieval stage |
| `STRONG_DF` | 400 | Selectivity threshold for conjunctive evidence |
| `STRONG_CAP` | 13 | Maximum phrases in conjunctive construction |
| `OR_CAP` | 8 | Maximum phrases in disjunctive construction |
| `RESOLVE_CAP` | 22 | Token cap for phrase-resolution backoff |

The complete FTS5 expression is:

```text
bm25(p, 0.0, 3.264, 1.202, 1.869, 2.561, 1.659, 2.153)
```

The initial `0.0` is the unindexed identifier column.

## Ranking

| Parameter | Final value | Role |
|---|---:|---|
| `W_CATEGORY` | 0.4541399437579685 | Category evidence weight |
| `W_MINED` | 0.47960403849856215 | Catalogue-mined evidence weight |
| `IDF_POW` | 0.08825136552256256 | Document-frequency attenuation exponent |
| `W_POP` | 0.5114555220952501 | Maximum static popularity-prior weight |
| `MINED_LEN_DIV` | 7.067577463426672 | Length normalization for mined phrases |

`W_POP` is further scaled by the label-free retrieved-pool population detector. The table
records the maximum configured value, not necessarily the effective value after calibration.

## Dialogue and safety constants

| Parameter | Final value | Role |
|---|---:|---|
| disclosure schedule | nine turns at 1, final turn at 10 | Rank-first sequential disclosure with final recall width |
| optional LLM extraction | off | Requires `LLM_EXTRACT=1` and a Groq key |
| optional LLM reranking | off | Retained only as rejected experimental code |
| local tagger gate | official-form miss only | Prevents any model effect on official message forms |

## Optimization protocol

The v2 study optimized the exact aggregate official reward over 1,000 fixed sessions: the
released public 200 and one stratified private-like 800. Every trial saw identical data,
paraphrasing was absent from the objective, and only completed trials were eligible.

Candidate selection was intentionally separate from optimization:

1. freeze conservative trial 44, balanced trial 38, and aggressive trial 106;
2. stop all optimizer workers;
3. evaluate four untouched same-population 800-session folds;
4. evaluate controlled 5%, 10%, and 20% popularity-distribution disturbances;
5. select trial 38 because it preserved the public score and matched the independent mean
   of trial 106 without the latter's public regression.

The final independent mean delta was `+0.001213` over the preceding shipped configuration.
This is smaller than individual-fold noise and is treated as a cautious selection signal.
The consumed folds are not eligible for further tuning.

See [`../validation/independent_validation.md`](../validation/independent_validation.md).
