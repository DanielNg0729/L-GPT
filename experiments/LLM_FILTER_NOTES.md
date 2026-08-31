# LLM Relevance Filter & Disclosure Modes — Experiment Notes

Continuation of the decision log for work done on top of the frozen trial-38 agent
(branch `feat/llm-filter`, 2026-09-01, author: Tài). Same rules as
[`DECISION_LOG.md`](DECISION_LOG.md): every ruling cites a number that was actually
measured, negative results are retained, and nothing here changes the official scored
path — the filter is off by default and the disclosure default stays `sequential`.

## What was built

`submission/llm_filter.py` — windowed contradiction demotion over the ranked list.
Judge the first ten candidates against the shopper's stated requirements; candidates
that *clearly contradict* an explicit requirement (wrong department/gender, wrong
stated colour, far over a stated price cap) are demoted to the tail and the window
refills from the next ranked survivors ("judge ten, demote the wrong ones, pull the
next ten"). Design rules inherited from the core agent: gated by control flow
(`LLM_FILTER=1` + a key, default off), demote-never-drop (the returned list never
shrinks; a wrongly flagged target stays reachable), fail-open on every error path
(failed call / malformed JSON / whole-window flag ⇒ the lexical order stands), token
usage reported through the contract `usage` field, shared `RateLimiter` budget.

Context handed to the model per call: the shopper's last 8 messages verbatim, the
evidence phrases from the session ledger, and one ~240-char catalogue snippet
(`CatalogIndex.doc`) per candidate. Output contract: `{"remove": [indices]}`.

## Measurements

All through the organizer's unmodified evaluator, 30-session stratified subsets
(10 buying / 10 browsing / 7 intent_override / 3 boundary), `DISCLOSURE=full`.

| Suite | Filter OFF | Filter ON | Δ | Ruling |
|---|---:|---:|---:|---|
| Clean (public-set draw) | 0.877084 | 0.868584 | **−0.0085** | Rejected on scored traffic |
| Value-paraphrase (`review800_open_vocab_paraphrase` draw) | 0.806456 | 0.786305 | **−0.0202** | Rejected on scored traffic |

- The loss is **entirely MRR**; HitRate@10 and MTTC were identical in every arm — the
  demote-with-refill design bounded the worst case to rank shuffling, exactly as
  intended. No session was lost to a filter error in ~135 live calls.
- An earlier Groq free-tier run showed a spurious **+0.0017**: 102 of ~161 call
  attempts failed on throttling and fell open, so the filter barely ran. A conclusion
  from a rate-limited arm is a conclusion about the rate limit. Re-measured at full
  throughput (OpenRouter, 0 failures) before ruling.
- The result replicates experiment 41-era findings one layer up: LLM relevance
  judgment loses to the popularity prior on this benchmark (41.2% vs 57.4%
  target-first), because simulator constraints — even value-paraphrased ones — are
  near-listing text, so most "contradictions" the model sees are false positives.
- Where it does earn its keep: live smoke tests on real-shopper wording
  ("looking for something for women, black") correctly demoted men's items and a
  white-only item — the class exact matching cannot see ("men" ⊂ "women" at the
  substring level; department is often implicit). **Kept as a demo-only layer.**

## Disclosure schedule, re-measured

| Mode | HR@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| `sequential` (default) | 0.995 | 0.995 | 2.225 | **0.971500** |
| `full` (demo opt-in) | 0.995 | 0.7189 | 1.650 | 0.900181 |

Full 200 public sessions, same evaluator. The entire gap is rank position; the
default stays `sequential` so a fresh clone reproduces the runbook number.

## Provider operations (hosted `openai/gpt-oss-20b`)

| | Groq free tier | OpenRouter |
|---|---|---|
| Latency per call | ~0.5 s | ~1.8–4 s |
| Throttling under load | severe (102 failures / 59 calls) | none observed (135/135) |
| 30 filtered sessions | ~11 min | ~4–5 min |
| Cost | $0 | ~1–2 US cents |

`LLM_ENDPOINT` now selects any OpenAI-compatible provider (Groq default, OpenRouter,
Cloudflare Workers AI `@cf/openai/gpt-oss-20b`); `GROQ_API_KEY` is the Bearer token
whatever the provider.

## Repository defects found and fixed along the way

- `experiments/studies/evaluate_full_pipeline.py` pointed at
  `experiments/studies/open_vocabulary/`; the datasets moved to
  `experiments/datasets/open_vocabulary/` in the restructure. Fixed.
- `tests/test_disclosure.py` (new) initially broke the contract test's class-identity
  assertion by reloading `submission.agent`; fixed by reloading `starter.agent` in
  teardown.
- Pre-existing on `origin/full`: `test_upstream_integrity` and `test_population_sets`
  failed — every "modified" hash mismatch. **Diagnosed and fixed:** the manifests were
  written on a Windows checkout whose autocrlf materialized the text files with CRLF,
  so the manifests pinned CRLF hashes while the repository stores LF. Every protected
  file was verified byte-identical to the organizer kit under `provided/` AND to the
  organizer's GitHub upstream (`raw.githubusercontent.com` hash match) before touching
  any manifest. Fix: both hashers (`tools/verify_upstream_integrity.py`,
  `experiments/studies/build_sets.py`) and the population test now hash after
  CRLF→LF normalization — platform-independent, still catches real content changes —
  and both manifests were regenerated. Suite: 43/43 green.
