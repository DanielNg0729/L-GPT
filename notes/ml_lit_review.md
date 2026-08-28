# ML / AI Literature Review — every learned approach considered for Track 4

Companion to [`lit_review.md`](lit_review.md), which covers the task-level literature
(CRS, retrieval, dialogue). **This document covers only the learned/AI approaches**, with
each one's status on this benchmark and the measurement that produced it.

Terminology, as agreed with the team:

- **LLM** — a cloud API call to a general-purpose model (Groq, Anthropic, …). Remote,
  needs a credential, needs network.
- **ML** — a model we train locally on this task and ship as a file. Deterministic,
  offline, no credential.

**Bars, fixed before any of these ran:**

| Measurement | Bar |
|---|---|
| End-to-end TechnicalScore | **+0.005 held-out** to adopt (fold stdev is 0.0168) |
| Within-tie target-first rate | must beat **popularity: 57.4%** (tune) / 55.1% (hold) |
| Reference points | LLM 41.2% · feature-LTR 32.4% · random 16.2% |

---

## A. Classical conversational-recommendation learning

### A1. Factorization Machine preference estimation — EAR's *Estimation* stage
`arXiv:2002.09102` (WSDM 2020)

EAR trains an FM over user–item and user–attribute interactions to rank both items and
attributes before deciding what to ask.

**Status: structurally inapplicable.** An FM needs collaborative signal — repeated users
with interaction histories. Our harness gives a *fresh anonymous user every session*, and
the profile it does supply is near-constant (measured: `purchase_frequency` has **1
distinct value** across all 200 sessions; `preference_tags` draws from 9 generic words).
There is no user-item matrix to factorise.

### A2. RL dialogue policy — CRM / EAR / SCPR *Action* stage
`arXiv:2002.09102`, `arXiv:2007.00194`

All three learn a policy over "ask an attribute" vs "recommend items".

**Status: dissolved, not merely unnecessary.** Our `respond()` returns `ask_attribute`
**and** `recommendations` in one payload, and the harness scores the list every turn. The
decision these methods exist to optimise does not exist here. See `lit_review.md` §6.

Worth recording: SCPR's own Table 3 shows **Max Entropy — an untrained heuristic — beating
both RL methods** in the large-attribute setting (0.669 vs EAR 0.595, CRM 0.580 on
LastFM\*). Published evidence, from the authors of the method that beat all of them, that
learned policies are not automatically better than rules here.

### A3. Graph attribute pruning — SCPR
`arXiv:2007.00194` (KDD 2020)

**Status: untested, low expected value.** Our candidate set already performs the pruning
role the graph plays, and our "attributes" are free-text bullets rather than a curated
facet schema.

---

## B. Neural retrieval (first stage)

### B1. Dense bi-encoder, off-the-shelf — `BAAI/bge-small-en-v1.5`
**Status: TESTED → REJECTED. −0.047.**

Dense-only 0.5659, sparse+dense RRF 0.7784, sparse alone 0.8257 — fusion *costs* 0.047 for
958 s of CPU embedding. Mechanism: the task is provenance recovery over verbatim
substrings, and embeddings deliberately blur the lexical precision that solves it.

### B2. Dense bi-encoder **fine-tuned on our data**
**Status: untested.** Now unblocked (torch 2.13.0+cu130 installed, RTX 4060). Lower
priority than B4/C2 because the first stage is not the bottleneck — recall is already
99.0%, so a better retriever has almost nothing to win.

### B3. ColBERT / late interaction
`arXiv:2112.01488`; OOD analysis `arXiv:2302.06589`

**Status: untested, predicted to fail.** Attractive in principle because it "prefers exact
lexical matches for rare high-IDF terms" — but the OOD study finds its generalisation gains
are **not** driven by exact-match signals, which is the only signal that works here.

### B4. SPLADE / learned sparse retrieval
**Status: untested.** The most interesting untested retriever: sparse (so it preserves
exact matching) but learned (so term weights are fitted rather than hand-set). Same caveat
as B2 — first-stage recall is 99.0%, so the ceiling is small.

### B5. doc2query / docTTTTTquery document expansion
**Status: untested, infeasible here.** Requires generating predicted queries for **50,000
documents**. Far beyond Groq's ~1,000 requests/day free tier, and a local generator would
need a model we do not have.

---

## C. Neural reranking (second stage)

### C1. Cross-encoder, off-the-shelf — `cross-encoder/ms-marco-MiniLM-L-6-v2`
**Status: TESTED → REJECTED. −0.030 held-out.**

Pure CE ordering drops HR@10 from 99.0% to 96.0%. Within-tie target-first: **32.4%**
against popularity's 57.4%. Feasibility was *not* the problem — 65 ms/rerank at depth 20,
819 MB RSS, ~1.7 min projected across 800 sessions.

An earlier draft of the report declined this component on three stated grounds, **all of
which collapsed under review** (wrong headroom cited, a disk restriction that the rules do
not contain, and a bi-encoder/cross-encoder conflation). The conclusion survived; the
reasoning did not. Recorded because the error is instructive.

### C2. Cross-encoder **fine-tuned on our synthetic data**
**Status: TESTED -> REJECTED, and it got WORSE with training.**
Within-tie target-first fell monotonically as training loss fell: zero-shot 32.4% ->
16.2% -> 13.2% -> **8.8%**, i.e. below random (16.2%). Falling loss with falling accuracy
is real learning on the wrong thing: the candidates are text-tied by construction, so no
legitimate text signal exists and the model fits incidental correlations that invert.
A measured contributor: the target sits at position 0 in 45.6% of synthetic training
groups but 57.4% of real ones, so the listwise positional prior is mismatched by 11.7
points. Original text: The domain mismatch in C1 is real — MS MARCO is natural-language
web queries; ours is a bag of catalogue phrases. Fine-tuning on this task's own
distribution, with a **listwise** objective over tie groups, is the direct test.

### C3. LLM listwise reranking (RankGPT-style)
`arXiv:2304.09542`; open-weight variants `arXiv:2312.02724`

**Status: TESTED → REJECTED. −0.027 end-to-end.**

Within-tie target-first 41.2% (at 320-char context; 29.4% at 110 chars — our truncation was
itself a confound worth correcting). RRF blending with popularity degrades
**monotonically** as LLM weight rises — the signature of noise, not complementary signal.

### C4. Permutation distillation (RankGPT → small model)
**Status: pointless here.** Distillation transfers a strong teacher's rankings into a
cheap student. Our LLM teacher scores 41.2% against popularity's 57.4% — distilling it
would teach a student to be worse than the rule it replaces.

---

## D. Feature-based learning to rank

### D1. LTR on public data (LambdaMART family)
`Ranking Distillation`, KDD 2018 · pointwise/pairwise/listwise taxonomy

**Status: TESTED → REJECTED. −0.040 held-out, overfit gap +0.065.**

Beat the hand-tuned scorer by +0.013 *on the half it trained on* and lost 0.065 on unseen
sessions. Cause: 30,726 rows **looks** like plenty but is **147 positives across 100
independent queries** — ranking models are governed by query count, not row count.

### D2. LTR on **synthetic** data — weak supervision
`Neural Ranking Models with Weak Supervision`, SIGIR 2017 (`arXiv:1704.08803`);
`Generalized Weak Supervision for Neural IR`, TOIS 2024; Gecko-style generate-then-relabel

**Status: TESTED → REJECTED. −0.011 tune.**

The enabling insight is real and verified: because the simulator's `intent_card` can be
reconstructed **exactly** (200/200), we can mint a labelled session from any catalogue
product — **~49,650 independent queries, a 496× increase**. Our supervision is *stronger*
than the cited literature's: those papers use a weak labeler (BM25 as noisy teacher), we
*choose* the target, so labels are exact.

**Critical detail, verified by distribution matching:** real targets come from a 5-core
leave-last-out review split, so P(target) ∝ review count, sampled *with replacement*.

| | p10 | median | p90 |
|---|---|---|---|
| Real targets | 5.20 | **8.80** | 10.61 |
| ∝ reviews, with replacement | 5.28 | **8.84** | 11.30 |
| Uniform sampling | 0.69 | 2.64 | 5.56 |

Uniform minting would have taught the model a popularity prior inverted relative to
reality — miscalibrating the single decisive feature.

### D3. Gated LTR — apply the model only where confidence is low
**Status: TESTED → noise.** −0.0057 tune, +0.0043 held-out. The gate itself *works*
(consistently +0.0016 to +0.0052 over ungated) but gating can only limit the damage of a
weaker ranker, never create gain.

### D4. Tie-specialised LTR with within-group normalisation
**Status: TESTED -> the hypothesis was RIGHT and the conclusion is still REJECT.**
Within-group features moved accuracy 32.4% -> 39.7% and `pop_z` became the top feature by
2x, confirming the diagnosis. But the feature-count ablation is decisive: **k=1 (pop_z
alone) reproduces popularity exactly at 57.4%**, and every added feature costs accuracy
(k=2: 42.6%, k=3: 33.8%). The signal was never missing -- capacity consumed it. Original text: Two defects being fixed: training rows drawn from *all* turns
while evaluation happens only inside ties, and **every feature being absolute with none
relative to the group**. The latter is the leading explanation for D1–D3: the model has
popularity as a feature yet loses to popularity alone, which is what you would expect if
it cannot express "take the maximum in *this* group". Per-query normalisation is standard
in LTR precisely for this. 18 features → 34.

### D5. Learned term/phrase weighting ("learned BM25")
**Status: untested, sklearn-feasible.** Narrow: it learns per-phrase-property weights
where D1–D4 already learn over a superset of that information.

---

## E. Probe-policy learning

### E1. Entropy / information-gain attribute selection
IDSS `arXiv:2603.11399`; *Ask to Be Sure* `arXiv:2608.15949`

**Status: partially tested, effect vanishes.** Probe ordering mattered at weak ranking
configurations (0.8285 vs 0.8257) and **attenuates to <0.003 — inside noise — at the tuned
configuration**. Once retrieval is strong, evidence arrives fast enough that ordering stops
binding. Notably, *Ask to Be Sure* also keeps question **selection** in a scoring rule and
uses the LLM only to phrase it.

---

## Summary

| Direction | Status | Result |
|---|---|---|
| A1 FM preference estimation | inapplicable | no collaborative signal exists |
| A2 RL ask-vs-recommend policy | dissolved | the decision doesn't exist in this API |
| A3 Graph attribute pruning | untested | candidate set already prunes |
| B1 Dense bi-encoder (off-the-shelf) | **rejected** | −0.047 |
| B2 Dense bi-encoder (fine-tuned) | untested | first stage isn't the bottleneck (99.0% recall) |
| B3 ColBERT / late interaction | untested | predicted to fail on stated mechanism |
| B4 SPLADE / learned sparse | untested | most interesting untested retriever |
| B5 doc2query | infeasible | 50k generations, past quota |
| C1 Cross-encoder (off-the-shelf) | **rejected** | −0.030; 32.4% within-tie |
| C2 Cross-encoder (fine-tuned) | **in progress** | the domain-mismatch fix |
| C3 LLM listwise rerank | **rejected** | −0.027; 41.2% within-tie |
| C4 Permutation distillation | pointless | teacher worse than the rule |
| D1 LTR on public data | **rejected** | −0.040, overfit gap +0.065 |
| D2 LTR on synthetic data | **rejected** | −0.011 |
| D3 Gated LTR | **noise** | −0.006 / +0.004 |
| D4 Tie-specialised + within-group | **in progress** | fixes the leading explanation |
| D5 Learned term weighting | untested | narrow |
| E1 Learned probe policy | attenuates | <0.003 at tuned config |

**The pattern across every completed row:** learned methods land *above random and below
the hand-built rule*. Popularity 57.4% > LLM 41.2% > feature-LTR 32.4% ≈ zero-shot
cross-encoder 32.4% > random 16.2%. Three independent learned approaches converge near the
same number, which is more likely one shared cause than three coincidences — and D4 names
a candidate cause that is testable.

---

## F. ML for ROBUSTNESS — a different layer, a different failure

Directions A–E all attacked **retrieval or reranking** — choosing among candidates — and all
lost to a one-dimensional popularity statistic. This family targets the layers where
robustness actually fails: **extraction (Layer 1)** and **probe policy (Layer 3)**. It also
has something the ranking problem never had: `intent_card()` is a pure function, so calling
it across all 50,000 catalogue products enumerates **every string the generator can emit**.
Exact labels, unlimited, no weak-supervision noise.

Benchmarked on the full grid: clean · 800 unseen · uniform-pop · inverse-pop · T1/T2/T5
paraphrase.

### F1. Constraint-likeness scorer for n-gram mining
`mine()` is the paraphrase floor (0.838 vs 0.164 without it) and contains **no learning** —
greedy longest-match with a `df` gate. Learn P(this n-gram is something `intent_card` emits)
and let mining rank rather than take-longest.

**Status: TESTED → REJECTED, in both possible directions.** Held-out accuracy **0.637**.

| use | T1 | T2 | T5 |
|---|---|---|---|
| none (shipped) | **0.85230** | **0.86915** | **0.84770** |
| hard filter ≥ 0.30 | 0.77715 | 0.81778 | 0.78055 |
| hard filter ≥ 0.50 | 0.65270 | 0.73650 | 0.65200 |
| hard filter ≥ 0.70 | 0.21730 | 0.49365 | 0.16370 |
| soft weight, floor 0.5 | −0.0013 | — | −0.0109 |
| soft weight, floor 0.0 | −0.0136 | — | −0.0274 |

Monotone degradation as the model gains influence, **as a filter and as a weight**. That is
the signature of noise rather than complementary signal — the same shape as the LLM /
popularity RRF blend in C3. At 0.637 the classifier is far too weak to gate on: filtering
trades a little precision for a lot of recall, and recall is the entire job of the floor.

**A label leak was caught here and is worth recording.** The first version scored 0.978
held-out with `cap_ratio` at **+28.0**, ten times any other coefficient. Positives came from
`intent_card()` (original casing, punctuation); negatives from `ix.blob`, which `raw_toks`
had already lowercased and stripped. The model was classifying *which pipeline built the
string*. It would also have shipped broken — `mine()` runs on `raw_toks` output, so both
leaked features are identically zero at inference. Normalising both classes through the
inference transform dropped accuracy 0.978 → 0.637. Third leak of this shape in the project;
inspecting coefficients rather than trusting accuracy caught all three.

### F2. Local paraphrase-robust extractor (the one meant to replace the API call)
Token-level: which tokens of a message are scaffolding vs product text? Trained on minted
sessions with paraphrase transforms applied and **exact** token labels. Would move the
extraction channel O2 → O1 at zero cost and zero latency.

**Status: TESTED → REJECTED.** Train accuracy 0.837, but the decisive measurement was
**held-out-transform** accuracy — train on one paraphrase family, test on families never
seen:

| held-out transform | accuracy | majority class | lift |
|---|---|---|---|
| T5 realistic | 0.852 | 0.744 | +0.108 |
| T4 case/punct churn | 0.778 | 0.582 | +0.196 |
| **T2 scaffolding stripped** | 0.443 | 0.879 | **−0.435** |

End-to-end, every threshold loses: T1 **−0.098**, T2 −0.060, T5 −0.067.

The T2 row is the diagnosis. T2 removes scaffolding entirely, leaving bare values — so the
correct behaviour is "keep everything", and the model instead discards content. **It learned
our specific filler vocabulary, not the general shape of scaffolding.** That was the
pre-registered risk, and the held-out-transform protocol is what exposed it; training
accuracy alone would have looked like success.

### F3. State-conditioned probe policy (aimed at MTTC)
`PROBE_ORDER` is a fixed 7-tuple. Pass 04 swept fixed *orders* and found <0.003 at tuned
weights — but a permutation is not a policy. This fits an expected-yield table conditioned
on session state (turn bucket × evidence held × attributes unasked) over minted sessions.
This is the EAR/SCPR *Action* stage (`arXiv:2002.09102`, `arXiv:2007.00194`), the one CRS
component previously dismissed as "dissolved".

**Status: TESTED → REJECTED.** −0.0015 clean, −0.0026 unseen-800, −0.0047 uniform-pop, and
**MTTC 2.395 against the fixed order's 2.320** — worse on the exact metric it was built to
improve. The earlier conclusion survives, now for a stronger reason: it was tested as a
policy, not just as an ordering.

---

## What the F family establishes

**Sixteen learned approaches, sixteen rejections.** But F2 explains *why the LLM works where
local ML does not*, which is the useful result:

The LLM knows "Jewelry Necklaces" is a product category and "Appreciate it" is filler from
**pretraining**. Our local models see only catalogue `df` statistics and 70k tokens of
synthetic data. More importantly, F2's T2 collapse shows a locally-trained extractor
overfits to *the paraphrase family it was trained on* — and the organizer's paraphraser, if
any, is precisely the thing we cannot anticipate. The LLM has never seen our transforms
either, which is exactly why it generalises across all of them.

So the API call is not a shortcut taken for convenience. It is the only extraction channel
measured to generalise to paraphrase styles we cannot enumerate in advance — and the
recognition gate confines it to the case where that property is the one that matters.

One structural result worth keeping: **clean scored exactly 0.96960 under every F1/F2
variant**, including the ones that destroyed the paraphrase floor. The recognition gate
protects the clean path from *any* experimental extraction channel, not just the LLM.


---

### Addendum: the "full-model training" scope concern is withdrawn

Section F flagged that `full-model training` appears in the specification's "Out of scope"
list, making the fine-tuned distilbert tagger arguably ambiguous. The organizer's slide 5
reframes the identical list as **"NOT REQUIRED"** -- not required is not prohibited, and the
same list contains "User interface", which is plainly a statement about effort expectations
rather than a ban. Combined with "IN SCOPE: ... semantic reranking" and the specification's
"legally accessible LLM APIs or local models", a fine-tuned local tagger is permitted.

Also from the Q&A: **there will be no paraphrasing**. The F-family results stand as
measured, but their practical stakes drop -- the extraction channels they concern are now
insurance against a scenario the organizer has ruled out. They are retained because the
recognition gate makes them cost exactly zero on clean traffic (0 calls, measured), and
because "degrades gracefully outside its assumptions" is a property worth demonstrating
even when the assumption holds.
