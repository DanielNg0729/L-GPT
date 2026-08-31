# Submission Report — Grounded Multi-Turn Product Search

TikTok TechJam 2026, Track 4 (Shopping Copilot). This is the short report required by
the submission rules: method, model choice, cost and latency disclosure, limitations,
and team contributions. Full engineering detail, the experiment programme, and
reproduction commands live in [`README.md`](README.md) and
[`experiments/`](https://github.com/DanielNg0729/L-GPT/tree/experimental/experiments).

## 1. Problem insight

The released simulator builds every customer utterance from the target product's own
catalogue record, so every spoken constraint is a **verbatim substring of the target
document**. The benchmark is therefore string-provenance recovery, not semantic search
— and it additionally **favours popular products**: targets derive from real purchase
records, so probability mass concentrates on heavily-reviewed items. Both observations
are measured, not assumed, and both shape the design: exact phrase matching is the
estimator matched to the generative process, and a review-count prior
(`log1p(rating_number)`) is load-bearing in ranking. The cost of the popularity lean is
also measured: on a population re-drawn *inverse* to popularity, the score drops from
0.954 to 0.868 (§3).

Both observations trace to the pre-agent research phase: the catalogue/session profile
([`docs/research/data_profile.md`](https://github.com/DanielNg0729/L-GPT/blob/experimental/docs/research/data_profile.md), 2026-08-26) measured
the target-vs-catalogue popularity gap (median 6,846 vs 12 reviews) and the expected
value of every clarifying question — the table the shipped ask policy encodes — and
[`docs/research/industry_notes.md`](https://github.com/DanielNg0729/L-GPT/blob/experimental/docs/research/industry_notes.md) records the
production practices the design adopts (clarification budgeting, slot-state with
override reset, popularity priors, lexical-first retrieval with semantic assist) and
the two places it deliberately departs from them for the benchmark.

## 2. Method

A hybrid escalation ladder — cheap exact mechanisms first; learned components are
reachable only when the cheap ones cannot see the answer, enforced by control flow
rather than confidence thresholds:

1. **Recognition gate** (lexical): is the message one of the simulator's own shapes?
   463/463 clean messages recognised; 0 perturbed ones.
2. **Template extraction** (lexical) on recognised messages; on unrecognised wording, a
   **DistilBERT dialogue-act router** and a **scaffolding tagger** (both local, lazily
   loaded, fail-closed) plus catalogue-attested span recovery and n-gram mining.
3. **Attribute deparaphraser** (hosted LLM): consulted only for a value the catalogue
   cannot attest (df = 0); it proposes a canonical term and the catalogue admits or
   drops it. The model proposes; the catalogue disposes.
4. **Session ledger** → SQLite FTS5 retrieval ladder (conjunctive → backoff →
   disjunctive) → coverage ranking with the popularity prior → rejection feedback
   (shown-and-not-hit items demoted, never dropped) → disclosure.
5. **LLM relevance filter** (optional, off by default): judges ranked candidates in
   windows of ten and demotes ones that clearly contradict an explicit requirement,
   refilling from the next survivors. Kept demo-only after measurement (§4).

Two disclosure modes are shipped and documented (`DISCLOSURE` in `.env`):
`sequential` (the default — one candidate per turn, ten on the last; maximises the
official metric and is what a fresh clone reproduces) and `full` (ten every turn — the
realistic shopping surface, used for demos).

## 3. Verified results

Scored through the organizer's unmodified `evaluator/local_evaluator.py`:

| Evaluation | Sessions | TechnicalScore |
|---|---:|---:|
| Official public set, `DISCLOSURE=sequential` | 200 | **0.971500** |
| Official public set, `DISCLOSURE=full` | 200 | 0.900181 |
| Organizer-proxy population | 800 | 0.954163 |
| Review-weighted unseen population | 800 | 0.947263 |
| Uniform-target population | 800 | 0.885581 |
| Inverse-popularity population | 800 | 0.867775 |
| Published weak BM25 baseline | 200 | 0.106710 |

Robustness on self-generated paraphrase suites (characterisation, not leaderboard
claims): template paraphrase 0.667 → **0.920** with the hybrid; attribute paraphrase
0.847 → 0.864 with the resolver; both at once 0.605 → **0.810**.

## 4. Model choice

- **Local**: two DistilBERT checkpoints (dialogue-act router, scaffolding tagger),
  hosted on the Hugging Face Hub, fetched on first use, only ever reached by wording
  that fails the recognition gate. On the official path they record 0 loads and 0
  inferences.
- **Hosted**: `openai/gpt-oss-20b` through any OpenAI-compatible endpoint
  (`LLM_ENDPOINT`): Groq (default), OpenRouter, or Cloudflare Workers AI
  (`@cf/openai/gpt-oss-20b`). Measured operationally: Groq free tier is fastest per
  call (~0.5 s) but throttles hard under load; OpenRouter ran 135 calls with zero
  failures at ~2–4 s per call.
- **A measured negative result**: the LLM relevance filter, run at full throughput on
  two 30-session suites, *costs* score on simulator traffic (clean −0.009, paraphrase
  −0.020, entirely MRR; HitRate untouched because flagged items are demoted with
  refill, never dropped). This replicates the repository's earlier finding that LLM
  relevance judgment loses to the popularity prior here (41.2% vs 57.4% target-first).
  It therefore ships **off by default** and exists for real-shopper wording — the
  "for my husband" class of contradiction that exact matching cannot see — which it
  handles correctly in live tests.

## 5. Cost, latency, and token disclosure

- **Official configuration** (offline, deterministic): **0 prompt and 0 completion
  tokens, $0.00**, no network, no API key. Complete 200-session public evaluation in
  ~18 s including index construction (~0.09 s per session) on a laptop CPU.
- **Optional hosted layers**, when a key is provided: the deparaphraser is reachable 0
  times on the public set and 2 times per 800 unseen sessions; the demo-only filter
  measured ~2,100–2,750 tokens and ~8–11 s per session (`LLM_FILTER_CALLS=3`, via
  OpenRouter); a 30-session filtered run measured 62k–82k tokens (~1–2 US cents at
  current gpt-oss-20b rates).
  Hosted usage is reported per turn in the response `usage` field, is never required
  for the reported scores, and every failure path degrades to the deterministic agent.

## 6. Limitations

- **Simulator-shaped.** The verbatim-substring property is the organizer's generative
  process; against real humans the exact layers degrade to the measured paraphrase
  numbers (§3), which rely on self-generated synthetic variation — a weaker claim than
  generalising to real users.
- **Popularity-leaning.** The benchmark rewards popular targets and so does the agent;
  the inverse-popularity fold (0.868) bounds what that bias costs. A deployment over a
  different demand distribution would need the prior re-calibrated (the agent already
  self-calibrates its prior weight from label-free pool statistics).
- **Hosted calls are not bit-reproducible** even at temperature 0; all reported scores
  therefore come from the deterministic path.
- **The sequential disclosure schedule optimises the metric, not the experience**; the
  `full` mode exists for exactly that reason, and both numbers are published.

## 7. Team contributions

| Member | Responsibility |
|---|---|
| **Khiêm** — lead engineer & experimentation | The shipped agent end to end: exact-match core (recognition gate, template extraction, span recovery, grounded mining), FTS5 retrieval ladder, coverage ranking, session ledger, disclosure policy — the components behind the 0.9715 headline — plus the ~70-experiment programme, robustness and population-shift suites, release tests, and reproducibility infrastructure. |
| **Dương** — LLM layer | Attribute deparaphraser (generate-then-verify), transcript rescue, and the gating that keeps hosted calls off the scored path and fail-safe. |
| **Thanh Duy** — final architecture | The escalation-ladder design, component boundaries, integration review. |
| **Huy** — industry research | Production conversational-commerce practice (clarification budgeting, dialogue state, popularity priors, lexical-first retrieval) grounding the design — [`docs/research/industry_notes.md`](https://github.com/DanielNg0729/L-GPT/blob/experimental/docs/research/industry_notes.md). |
| **Tài** — main research & relevance filter | Participant-kit setup and checksum verification, exact baseline reproduction (0.10671) and the scenario breakdown that directed effort; the early data profile whose probe expected-value and target-popularity analyses anticipated the shipped ask policy and ranking prior — [`docs/research/data_profile.md`](https://github.com/DanielNg0729/L-GPT/blob/experimental/docs/research/data_profile.md); the windowed LLM contradiction filter and its measured A/B verdict — [`experiments/LLM_FILTER_NOTES.md`](https://github.com/DanielNg0729/L-GPT/blob/experimental/experiments/LLM_FILTER_NOTES.md). |

Catalogue and sessions derive from Amazon Reviews 2023 (McAuley Lab, UCSD) — see
[`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md).
