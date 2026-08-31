# Team Contributions

> TikTok TechJam 2026 — Track 4: Shopping Copilot.
> This section is a submission requirement ("each member's contributions", judging criteria for non-solo entries).
> Paste into the `## Team contributions` section of the submission README before the deadline.

| Member | Role | Focus |
|---|---|---|
| **Khiêm** | String matching & retrieval | Exact-match pipeline, retrieval ladder, experiment programme |
| **Dương** | LLM layer | LLM components, gating, and failure handling |
| **Thanh Duy** | Final architecture | System design and component integration |
| **Huy** | Industry research | Domain research grounding the design |
| **Tài** | LLM filter & main research | Research lead, baseline & evaluation analysis, LLM-based filtering |

## Khiêm — String matching & retrieval (GitHub: [KhiemGOM](https://github.com/KhiemGOM))

- Built the deterministic exact-match core: recognition gate for the simulator's message
  shapes, template slot extraction, catalogue-attested span recovery, and
  catalogue-grounded n-gram mining under a document-frequency cap.
- Implemented the in-process SQLite FTS5 retrieval ladder (conjunctive → backoff →
  disjunctive → bag-of-words floor) and the coverage + popularity ranking, plus the
  session ledger (evidence, overrides, rejection history with demote-not-drop).
- Ran the numbered experiment programme (~70 studies with recorded verdicts), the
  robustness/population-shift suites, the release test suite, and the reproducibility
  work (integrity checker, runbook, Hugging Face Hub checkpoint resolution).
- These components are what the 0.9715 headline score rests on; Khiêm is the lead
  engineer of the shipped agent.

## Dương — LLM layer

- Designed and built the LLM components: the attribute deparaphraser
  (generate-then-verify — the model proposes a canonical catalogue term, the catalogue
  attests it before it enters retrieval at attenuated weight) and the optional
  turn-5 LLM rescue path.
- Established the gating discipline that keeps the LLM unreachable on clean traffic,
  so the scored path stays at 0 tokens, and the failure handling (timeouts, malformed
  output, missing credentials all degrade to the deterministic path).
- Evaluated where LLMs help and where they hurt: reranking and retrieval-augmented
  variants measured and rejected; deparaphrasing kept.

## Thanh Duy — Final architecture

- Drove the agreed system architecture after group discussion: the hybrid escalation
  design where cheap exact mechanisms run first and learned components are reachable
  only when they cannot see the answer.
- Specified the component boundaries (understanding → retrieval → ranking → ask
  policy → disclosure) and the session-state contract that both prototypes implement,
  and reviewed the integration of the final pipeline against that design.

## Huy — Industry research

- Researched how production e-commerce conversational search and recommendation
  systems approach the problem (clarification budgeting, dialogue state, ranking
  priors, lexical-first retrieval) and distilled it into the design requirements the
  team built against — `docs/research/industry_notes.md`.
- Grounded the "beyond the simulator" story for judging: which behaviours matter for
  real shoppers (budget/brand questions, natural phrasing) versus what the benchmark
  rewards, informing the demo and the report's impact narrative.

## Tài — LLM filter & main research (GitHub: huynhchitai)

- Research lead: set up the participant kit with checksum verification, reproduced the
  official BM25 baseline to exact published numbers (TechnicalScore 0.10671), and
  produced the scenario breakdown that identified browsing/boundary sessions as the
  main scoring opportunity.
- Profiled the catalogue and public sessions (`docs/research/data_profile.md`):
  the probe expected-value table (`other` 100% / `budget`·`brand`·`category` 0%) and
  the target-popularity analysis (median 6,846 vs 12 reviews; `log1p` soft-prior
  recommendation) — both later mirrored by the shipped ask policy and ranking prior.
- Built the windowed LLM contradiction filter (`submission/llm_filter.py`): judges
  ranked candidates in windows of ten, demotes ones that clearly contradict an explicit
  requirement, refills from the next survivors. Measured A/B on two 30-session suites
  (clean and value-paraphrase): negative on simulator traffic (−0.009 / −0.020, all
  MRR; HitRate untouched by the demote-not-drop design), so it ships off by default as
  a demo-only layer for real-shopper wording, where it correctly handles cases exact
  matching cannot see ("for my husband" vs a men's/women's catalogue split).

---

*Catalogue and sessions derive from Amazon Reviews 2023 by McAuley Lab, UCSD — see
`DATA_ATTRIBUTION.md`.*
