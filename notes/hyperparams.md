# Tunable hyper-parameter registry — the search space for the final Optuna pass

Every knob in the shipped pipeline, its current value, its plausible range, and what we
already know about it. **Nothing here is being tuned yet** — this is the inventory the
final global optimisation will consume, kept current so that pass does not have to
rediscover the space.

Status column:
- **fitted-public** — value came from coordinate ascent on the 200 public sessions only.
  This is the class that produced the `IDF_POW` defect; treat every one as suspect.
- **fitted-multi** — validated across public + unseen + stress conditions (passes 33–34).
- **derived** — read off the evaluator's source or the scoring formula, not fitted.
- **untuned** — never swept. An unknown, not a safe default.

---

## A. Ranking weights — `submission/agent.py`, class `Agent`

| Param | Current | Range to search | Status | What we know |
|---|---|---|---|---|
| `W_CONSTRAINT` | 1.00 | **fix at 1.0** | derived | Scale anchor. Only ratios matter, so searching it duplicates the other three and wastes trials. |
| `W_CATEGORY` | 0.75 | 0.2 – 1.4 | fitted-public | Sweep spread 0.012 on unseen; broad optimum, argmax(unseen) was 1.0 — mildly under-set. |
| `W_MINED` | 0.15 | 0.02 – 0.8 | fitted-public | Spread **0.00018** on unseen. Flattest knob in the system. Low value to search. |
| `IDF_POW` | **0.00** | 0.0 – 0.5 | fitted-multi | Was 0.35; found actively harmful on *every* axis (pass 34). Keep in the space but expect 0. |
| `W_POP` | **0.25** | 0.0 – 0.6 | fitted-multi | **The one population bet.** +0.051 real / −0.059 uniform / −0.086 inverse. See pass 37. |
| `W_TITLE` | 0.0 | 0.0 – 0.6 | fitted-public | Measured harmful: 0.0→0.8964, 0.6→0.8504, 1.6→0.8268. Include a narrow range to re-confirm. |
| `W_PROFILE` | 0.0 | 0.0 – 0.3 | fitted-public | Measured harmful: 0.12→0.8599, 0.25→0.8160. Same. |
| `_weight` MINED length divisor | 8.0 | 3 – 16 | **untuned** | `base *= min(1, len(phrase.split())/8.0)`. Never swept. |

## B. Retrieval — `Agent` and `CatalogIndex`

| Param | Current | Range to search | Status | What we know |
|---|---|---|---|---|
| `POOL` | 400 | 150 – 1200 | fitted-public | 700 won 6 of 7 conditions, lost 0.0006 on public-hold; largest paraphrase gains of any candidate. **Best-known unclaimed gain.** |
| `STRONG_DF` | 500 | 100 – 1500 | fitted-public | Spread 0.012 on unseen; 150/300 slightly better than 500. |
| `CatalogIndex.DF_CAP` | 4000 | 1000 – 20000 | **untuned** | Caps the df count *and* gates n-gram mining (`0 < df <= DF_CAP`). Affects mining recall directly — and mining is the paraphrase floor. |
| BM25 `title` weight | 6.0 | 0 – 10 | fitted-public | Never swept on unseen sessions. Gap #3 in the audit. |
| BM25 `categories` | 4.0 | 0 – 10 | fitted-public | ditto |
| BM25 `features` | 2.5 | 0 – 10 | fitted-public | ditto |
| BM25 `details` | 2.5 | 0 – 10 | fitted-public | ditto |
| BM25 `store` | 1.5 | 0 – 5 | fitted-public | ditto |
| BM25 `description` | 1.0 | 0 – 5 | fitted-public | ditto |
| `_candidates` strong-phrase cap | `strong[:8]` | 3 – 16 | **untuned** | How many phrases enter the conjunctive rung. |
| `_candidates` OR-rung cap | `ev[:14]` | 6 – 30 | **untuned** | |
| `_candidates` fallback token cap | `[:40]` | 15 – 80 | **untuned** | Last-resort rung only; rarely fires. |

## C. Evidence extraction

| Param | Current | Range to search | Status | What we know |
|---|---|---|---|---|
| `mine()` `maxn` | 9 | 5 – 14 | **untuned** | Longest n-gram tried. Mining is the paraphrase floor (+0.585) — this is the highest-leverage untuned knob in the registry. |
| `mine()` `minn` | 3 | 2 – 5 | **untuned** | Shortest n-gram kept. `minn=2` would admit far more, and much noisier, evidence. |
| `_resolve()` `cap` | 12 | 6 – 24 | **untuned** | Max tokens of a constraint considered. |
| `_resolve()` window floor | 2 | 1 – 3 | **untuned** | Shortest fallback window. |
| `_resolve()` hits kept | `[:2]` | 1 – 4 | **untuned** | |

## D. Dialogue policy

| Param | Current | Range to search | Status | What we know |
|---|---|---|---|---|
| `DISCLOSURE` schedule | `(1,)*9+(10,)` | schedule family, not 10 free ints | fitted-multi | Net positive under **every** stress axis (pass 36): +0.062 nominal … +0.035 inverse-pop. Search as a parameterised family (start width, growth, final width) or the search space explodes to 10^10. |
| `PROBE_ORDER` | 7-attribute tuple | permutation, or a per-attribute priority score | fitted-public | Effect attenuates to <0.003 at the tuned config. Low expected value; searching a permutation is expensive. Consider fixing. |
| `DEAD_ATTRIBUTES` | `(category, brand, budget)` | subset of 3 | derived | Zero measured effect either way. Fix it. |
| `LLM_TIE_DEPTH` | 1 | — | derived | Dead while the LLM layer is off. Exclude. |

## E. Structural toggles (booleans — categorical, not continuous)

| Toggle | Current | Status | What we know |
|---|---|---|---|
| Rejection feedback | on | derived | +0.029 public / +0.042 unseen; +0.054 under paraphrase. |
| Demote vs. filter rejected | demote | derived | Scores **identically**; demote chosen because it cannot be mistaken for denominator-shrinking. Do not search. |
| Category part-split | on | derived | +0.004 public, +0.008 unseen; census-grounded. |
| `_resolve` backoff | on | derived | Now 0.000 — fully subsumed. Candidate for deletion, not tuning. |
| Retrieval ladder | on | derived | −0.0001 / −0.0011: removing it is *neutral to slightly better* nominally. Its behaviour under paraphrase is **unmeasured** (audit gap #4). |
| Popularity form | additive | pass 37 | Additive lets the prior override evidence. Alternatives under test now. |

---

## Design notes for the final Optuna pass

**1. The objective must not be the public-200 score.** That objective is exactly what
produced `IDF_POW = 0.35`, a value that was actively harmful on every other axis and
invisible on the 200. Proposal: optimise the **mean over rotating unseen draws**, then
validate the top-k trials on the full seven-condition worst-case grid. Resample which
draws a trial sees, so a configuration cannot memorise one synthetic set the way the
earlier ones memorised the public 200.

**2. Budget arithmetic.** 100 public sessions ≈ 5 s; 800 synthetic ≈ 45–90 s. A trial
touching four conditions is ~3–5 min ⇒ **~300–500 trials in 24 h** single-threaded. The
machine has 32 cores, and trials are independent, so an Optuna SQLite study with parallel
workers gets this to several thousand. Prune on the cheap 100-session condition first.

**3. Roughly 25 live parameters.** With aggressive pruning that is tractable, but two
cuts are worth making before starting: fix `W_CONSTRAINT` (pure scale duplication) and
fix `PROBE_ORDER` and `DEAD_ATTRIBUTES` (both measured at zero effect). That takes the
effective space to ~20 continuous knobs.

**4. Multiple comparisons are the real hazard.** Thousands of trials against a noisy
objective will produce apparent winners by luck; between-draw variance alone is ~0.012.
Any adopted configuration must clear the same pre-registered rule used in pass 34 — **no
regression on any condition** — evaluated on draws the study never optimised against.
