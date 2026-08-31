# Literature Review  -  Conversational Product Search under a Fixed 10-Turn Budget

Scope: the research relevant to TechJam Track 4 (Shopping Copilot), organised by pipeline
stage. Every numeric claim below was read out of the primary source (PDF or publisher
HTML), not from a search-result summary. Where a claim could only be verified at abstract
level, it is marked **[abstract-only]**.

Companion documents: [`../competition/track4_brief.md`](../../docs/competition/track4_brief.md)
(the official problem statement),
[`../../experiments/FINDINGS.md`](../../experiments/FINDINGS.md)
(empirical results), and [`../../experiments/log/`](../../experiments/log/)
(reproducible programs).

---

## 0. The framing question

The brief's Pillar I-III read as a specification for a classical conversational recommender
system (CRS): intent routing, a dialogue state machine, information-gain clarification, and
an "LLM Semantic Ranking" stage. Two decades of CRS research exist to solve exactly that
problem. The purpose of this review is to establish **which of those results transfer to
this harness and which are dissolved by it**  -  because, as Section 6 shows, one structural
property of the official API removes the central research problem of the classical line.

---

## 1. Classical conversational recommendation (the ask-vs-recommend line)

### 1.1 EAR  -  Estimation-Action-Reflection (Lei et al., WSDM 2020)

`arXiv:2002.09102`

Three stages: *Estimation* builds predictive models over both items and item attributes;
*Action* learns a dialogue policy deciding whether to ask an attribute or recommend items;
*Reflection* updates the recommender when the user rejects a recommendation. The estimation
component is a Factorization Machine; the action component is reinforcement-learned.
**[abstract-only]** for the internal formulae; the results below are read from SCPR's
reproduction tables (§1.2), which use EAR as their primary baseline.

| Strength | Weakness |
|---|---|
| Directly optimises "successful recommendation in fewer turns"  -  the same pressure our MTTC term creates. | Requires a trained FM updated online per session. Out of scope for us (no fine-tuning) and unnecessary given our evidence is verbatim text, not latent collaborative signal. |
| The ask-vs-recommend framing is the canonical formulation of the problem. | That framing presupposes ask and recommend are **mutually exclusive per turn**  -  see §6. |

### 1.2 SCPR / CPR  -  Interactive Path Reasoning on Graph (Lei et al., KDD 2020)

`arXiv:2007.00194`. Full result tables read from the PDF.

CPR models conversational recommendation as path reasoning over an attribute graph, walking
attribute vertices by user feedback. The graph adjacency constraint prunes the candidate
attribute set, so the RL policy's action space collapses from `|P|+1` to just 2
(ask vs. recommend). Reward has five components: `r_rec_suc`, `r_rec_fail`, `r_ask_suc`,
`r_ask_fail`, `r_quit`.

**Table 2  -  original attribute space** (SR@15 / Average Turns; higher SR, lower AT better):

| Method | LastFM SR@15 | LastFM AT | Yelp SR@15 | Yelp AT |
|---|---|---|---|---|
| Abs Greedy | 0.222 | 13.48 | 0.264 | 12.57 |
| Max Entropy | 0.283 | 13.91 | 0.921 | 6.59 |
| CRM | 0.325 | 13.75 | 0.923 | 6.25 |
| EAR | 0.429 | 12.88 | 0.967 | 5.74 |
| **SCPR** | **0.465** | **12.86** | **0.973** | **5.67** |

**Table 3  -  large attribute space** (LastFM\*/Yelp\*):

| Method | LastFM\* SR@15 | LastFM\* AT | Yelp\* SR@15 | Yelp\* AT |
|---|---|---|---|---|
| Abs Greedy | 0.635 | 8.66 | 0.189 | 13.43 |
| Max Entropy | 0.669 | 9.33 | 0.398 | 13.42 |
| CRM | 0.580 | 10.79 | 0.177 | 13.69 |
| EAR | 0.595 | 10.51 | 0.182 | 13.63 |
| **SCPR** | **0.709** | **8.43** | **0.489** | **12.62** |

**Two findings in this table matter more to us than SCPR's own contribution:**

1. **Max Entropy  -  a rule-based heuristic with no training  -  beats both RL methods (CRM,
   EAR) in the large-attribute-space setting**, on both datasets (0.669 vs 0.595/0.580 on
   LastFM\*; 0.398 vs 0.182/0.177 on Yelp\*). Published evidence, from the authors of the
   method that beat all of them, that an untrained entropy heuristic is competitive with
   learned policies once the attribute space is non-trivial. This substantially de-risks
   choosing a deterministic clarification policy over a learned one.
2. **Abs Greedy  -  which never asks, only ever recommends  -  scores 0.635 on LastFM\***,
   beating both EAR and CRM. The paper notes it "can achieve the best results on the first
   few turns but plunges in further turns", because it is the only method that spends every
   turn recommending. Under our harness we get Abs Greedy's early-turn behaviour *for free
   and permanently* (§6), which is why our MTTC lands near 3 rather than near 12.

### 1.3 Where this line is measured

Note the scale mismatch that limits direct transfer: LastFM has 33 attributes and 7,432
items; Yelp has 29 attributes and 70,311 items. Our setting has a 9-value closed attribute
enum and 50,000 items, but  -  critically  -  our "attributes" are not categorical facets over
a curated schema. They are free-text bullets lifted verbatim from the product listing. That
difference is the central argument of the
[`experiment findings`](../../experiments/FINDINGS.md), section 2.

---

## 2. Information-gain clarification

### 2.1 Entropy-guided elicitation (IDSS, `arXiv:2603.11399`)

Selection criterion, verbatim in structure: normalised Shannon entropy over the current
candidate set, `H(d) = -Σ p(v) log₂ p(v)`, normalised as `H_norm(d) = H(d)/log₂|Val(d)|` so
dimensions of different cardinality compare; then `d* = argmax H(d)` over unspecified,
previously-unasked dimensions, asked only if `H_norm > τ_H = 0.3`.

Ablation (short queries, embedding-similarity variant): removing entropy-guided questioning
drops Precision@9 from 0.903 → 0.880; on long queries 0.801 → 0.753 under coverage-risk
optimisation. Question novelty 94.6% with entropy selection vs 60.2% without.

> **Strength**: needs no training data at all  -  it is a pure function of the current
> candidate distribution, which is exactly what we can compute in-memory.
> **Weakness**: assumes a *categorical* attribute schema with enumerable values per
> dimension. Our attribute channel is a 9-value enum whose payload is free text, so entropy
> must be computed over the candidate set induced by *phrases*, not over facet values.

### 2.2 The classic critique of pure entropy

The earlier feature-selection literature (IEEE, *Feature Selection Methods for
Conversational Recommender Systems*) documents the failure mode that pure entropy selects
questions that are statistically informative but that the user has no reason to answer
usefully; the recommended fix is to combine feature entropy with a relevance measure. We
observe the same pathology in a different form: our highest-frequency attribute
(`material`) is also our *least selective* channel (§3 of findings). Frequency ≠ information.

### 2.3 Ask to Be Sure (`arXiv:2608.15949`)

Turn-level information gain defined as `I_T(A₁) = H_w(C₁) − H_w(C₂)`, the weighted entropy
of sampled recommendation lists before and after an interaction. Notably the **entropy rule
selects; the LLM only realises the question in language**  -  the reward guides SFT/DPO
training rather than the model freelancing question choice at inference.

Reported turns-to-ground-truth: **3.07** (INSPIRED) and **2.75** (ReDial) for the DPO +
turn-entropy variant, against 3.12-4.12 for baselines. Hit@1 ≈ 2-3%, Hit@5 ≈ 5%.

> Useful calibration: state-of-the-art multi-turn LLM recommendation converges in ~2.8-3.1
> turns on curated dialogue benchmarks. Our measured MTTC of 2.98 (§findings) is in exactly
> that band  -  but at HR@10 ≈ 0.93 rather than Hit@5 ≈ 0.05, because our task leaks verbatim
> provenance and theirs does not. The comparison is a sanity check on turn count, **not** a
> claim of superiority.

---

## 3. Sparse, dense, and hybrid retrieval

### 3.1 Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009)

`RRFscore(d) = Σ_i 1/(k + r_i(d))`, conventionally `k = 60`. Rank-only: no score
normalisation needed, which is why it is the default hybrid fusion in OpenSearch and
Elasticsearch. Larger `k` flattens the score distribution; smaller `k` sharpens rank
differences.

**Counter-evidence worth recording**: OpenSearch's own benchmark reports RRF hybrid scoring
**3.86% lower** than score-based normalisation methods averaged over six BEIR datasets, and
on NFCorpus BM25 alone (NDCG@10 0.3065) beats RRF hybrid (0.2977). Hybrid is not a free win
even in the literature that popularised it.

### 3.2 Embedding models

- `BAAI/bge-small-en-v1.5`  -  33M params, retrieval-tuned, runs on CPU via ONNX (fastembed).
- Multilingual E5 (`arXiv:2402.05672`) gives the size/quality ladder: mE5-small 57.9,
  mE5-base 59.5, mE5-large 61.5 MTEB average over 56 datasets. The small→large gap is real
  (~3.6 points) but bounded and budgetable.

> **Our empirical result contradicts the default hybrid prior for this task**: dense-only
> scored 0.5659 and sparse+dense RRF scored 0.7784, both *below* sparse-only at 0.8257, at a
> cost of 958s to embed the catalog on CPU. Mechanism: the task is provenance recovery over
> verbatim substrings, and dense embeddings deliberately blur the lexical precision that
> solves it. See the
> [`experiment findings`](../../experiments/FINDINGS.md), section 5.

---

## 4. LLM reranking

### 4.1 RankGPT (Sun et al., EMNLP 2023, `arXiv:2304.09542`)

Established listwise permutation generation for LLM reranking, plus permutation distillation
into smaller models  -  a distilled 440M model outperformed a 3B supervised model on BEIR. The
paper itself names the core tension: "the discrepancy between the pre-training objectives of
LLMs and the ranking objective". Introduced NovelEval to control for data contamination.

### 4.2 RankZephyr (Pradeep et al., `arXiv:2312.02724`)

Open-weight listwise reranker distilled from RankGPT supervision; closes most of the gap to
RankGPT-4 and in places exceeds it. 7B-class, so a real latency/memory footprint per turn.

### 4.3 How Good are LLM-based Rerankers? (EMNLP 2025 Findings)

22 methods / 40 variants across TREC DL19, DL20, BEIR, plus **FutureQueryEval**, built from
queries postdating model pretraining. Headline: LLM rerankers "demonstrate superior
performance on familiar queries", but "their generalization ability to novel queries varies,
with lightweight models offering comparable efficiency". On FutureQueryEval, listwise
Zephyr-7B reaches NDCG@10 62.65 and Vicuna-7B 58.63, while ListT5-3B collapses below 12.
The paper attributes LLM reranking difficulty to "computational complexity, API reliance,
and prediction inconsistencies across pointwise, pairwise, and listwise methods".

> Decisive for us on three counts: (a) gains concentrate on *familiar* queries and our
> private eval set is by construction unseen; (b) "API reliance" is precisely the risk the
> submission rules flag; (c) prediction inconsistency is a reliability problem across 800
> unattended sessions (§5).

---

## 5. LLM agents as recommenders  -  and their reliability

### 5.1 Chat-REC (`arXiv:2303.14524`, 2023)

Converts user profiles and interaction history into prompts, using in-context learning for
interactivity and explainability. The LLM is the generative surface; recommendation quality
still depends on what is fed to it. Addresses "poor interactivity and explainability"  -
note that neither is a scored quantity in our harness.

### 5.2 InteRecAgent (Huang et al., ACM TOIS 2025, `arXiv:2308.16505`)

LLM as "brain", recommender models as tools, with three tool classes (retrieval, filtering,
ranking), a **candidate memory bus** through which candidate items stream between tool
calls, dynamic demonstration-augmented planning, and reflection.

The single most on-point sentence we found in the entire review, from their Amazon results:

> "some LLMs underperform compared to random and popularity methods in ranking tasks,
> particularly in the Amazon dataset. This can be primarily attributed to LLMs not adhering
> to the ranking instructions, which arise due to LLMs' uncertainty and produce out-of-scope
> items, especially for smaller LLMs."

Our task *is* ranking Amazon catalogue items, and our harness silently discards out-of-scope
`parent_asin` values (`normalize_recommendations`), so this failure mode would be **invisible
in our metrics and simply cost us the slot**.

### 5.3 Large Language Models as Zero-Shot Conversational Recommenders (CIKM 2023, `arXiv:2308.10053`)

The strongest counter-evidence to a non-LLM design, and it should be engaged honestly.
Verbatim: "even without fine-tuning, large language models can outperform existing
fine-tuned conversational recommendation models", attributed to content/context knowledge
rather than collaborative knowledge. **[abstract-only]** for per-metric numbers.

> Why it does not overturn our design: their benchmarks (Reddit-Movie, ReDial) require
> *world knowledge about items*  -  an LLM knows what "a heist movie like Inception" means
> without retrieval. Our benchmark supplies verbatim substrings of the target document and
> requires exact `parent_asin` recovery from a frozen 50k catalogue the model has never
> memorised. Content knowledge is exactly the capability that does not transfer, and §5.2
> documents what happens when it is asked to.

### 5.4 Agent reliability  -  the case against freeform loops

**Canonical Path Deviation (`arXiv:2602.19008`)**  -  over 515 mixed-outcome units (same model,
same task, different outcomes across runs): successful runs adhere more closely to the
canonical tool path, mean within-unit gap **+0.060 Jaccard (p < 0.0001, n = 488)**, worth
**+5.3 percentage points** of success probability. The drift is gradual and self-reinforcing:
"An off-canonical call at position t increases the probability that position t+1 is also
off-canonical by 22.7pp." Mixed-outcome units show higher adherence variance (0.109) than
always-succeed units (0.086)  -  the signature of a *reliability*, not capability, failure. A
monitor restarting the worst-adhering tercile lifts success **+8.8pp**.

**The Long-Horizon Task Mirage (`arXiv:2604.11978`)**  -  "long-horizon failure is not merely a
drop in success rate, but a structural shift in failure composition", with planning and
memory failures becoming dominant as horizon grows; degradation is non-linear with a sharp
collapse past small compositional depth. Explicitly: "model scaling alone is unlikely to
resolve the dominant failure mechanisms."

**ToolFailBench (`arXiv:2607.04686`)**  -  taxonomy across 19 frontier models: Tool-Skip,
Result-Ignore, Output-Fabrication, Unnecessary-Tool-Use. Best CTUR 86.33% (Grok-4.3);
Result-Ignore median 12.24% in finance; Llama-3.1-70B showed 77.73% unnecessary tool use
where a same-scale peer differed by 89 percentage points on control accuracy.

> Together these bound the risk of putting an LLM on the scored critical path across 800
> unattended sessions where an exception, malformed output, or timeout "may count as a miss"
> (competition_specification.md) and no human is present to notice.

---

## 6. The structural observation that reorganises all of the above

Every method in §1 exists to answer: *this turn, do I ask or do I recommend?* CRM, EAR and
SCPR all learn a policy over that decision; SCPR's stated advantage is precisely that its
"more dedicated RL model … only decides to recommend or to ask".

**Our API does not pose that question.** `respond()` returns `message`, `ask_attribute`
*and* `recommendations` in a single payload, and the harness scores the recommendation list
on **every** turn while the customer simultaneously answers the probe. Ask and recommend are
not alternatives here; they are concurrent free actions.

Consequences:

1. The policy-learning contribution of the CRM→EAR→SCPR line is **inapplicable**, not merely
   unnecessary  -  there is no decision left for it to optimise.
2. "Abs Greedy" behaviour (recommend every turn) is strictly dominant and costless, so its
   documented early-turn advantage is available permanently rather than at the cost of never
   asking.
3. What remains of the dialogue problem is narrow: *which* probe to send, given that sending
   one is free. That is the §2 information-gain question and nothing more.

This is why our design spends its engineering budget on retrieval rather than on dialogue
policy, and it is a claim about the harness, not a claim about the field.

---

## 7. Summary of what transfers

| Literature contribution | Transfers? | Basis |
|---|---|---|
| Ask-vs-recommend policy learning (CRM/EAR/SCPR) | **No**  -  dissolved by the API | §6 |
| Entropy / information-gain probe *ordering* | **Yes, adapted** | §2.1, confirmed empirically |
| Max Entropy ≳ learned RL at scale | **Yes**  -  licenses a deterministic policy | SCPR Table 3 |
| Abs Greedy early-turn advantage | **Yes, for free** | SCPR Table 3 + §6 |
| Graph pruning of attribute space (SCPR) | **Partially**  -  our candidate set substitutes for the graph | §1.2 |
| RRF sparse+dense hybrid | **No** for this task  -  measured regression | §3.1, findings §5 |
| LLM listwise reranking | **Not on the critical path** | §4.3, §5.2 |
| LLM-as-brain, tools-as-execution (InteRecAgent) | **Yes, as the shape of any LLM use** | §5.2 |
| Short-term/long-term memory split | **Yes, trivially**  -  sessions are independent | §5.1 |
| Agent reliability constraints | **Yes**  -  bounds the architecture | §5.4 |
