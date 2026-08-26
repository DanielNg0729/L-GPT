# TikTok TechJam 2026 — Track 4: Shopping Copilot

An AI conversational search & recommendation agent over an Amazon catalog (50,000 products,
`Clothing_Shoes_and_Jewelry`). The agent has to surface the correct hidden target product
within **at most 10 conversation turns**.

## Status

- [x] Download + verify all organizer-provided resources → [provided/](provided/)
- [x] Reproduce the BM25 baseline with the official evaluator — **100% match** with the published numbers
- [ ] Intent router (Buying / Browsing)
- [ ] Multi-route retrieval (BM25 + category + vector, in-memory)
- [ ] Dialog state machine (slot accumulation + intent override)
- [ ] LLM semantic ranking
- [ ] MTTC optimization

## Baseline reproduced

`starter/agent.py` (weak BM25) over the 200 public sessions:

| Metric | Ours | Published |
|---|---|---|
| Hit Rate@10 | 0.125 | 0.125 |
| MRR | 0.068034 | 0.068034 |
| MTTC | 9.81 | 9.81 |
| Efficiency | 0.119 | 0.119 |
| **TechnicalScore** | **0.10671** | **0.10671** |

Broken down by scenario — this is where the baseline's main weakness shows up:

| Scenario | n | Hit@10 | MRR | MTTC |
|---|---|---|---|---|
| buying | 80 | 0.2375 | 0.1265 | 8.63 |
| intent_override | 30 | 0.1333 | 0.1042 | 10.07 |
| browsing | 80 | **0.025** | 0.0045 | 10.75 |
| boundary | 10 | **0.000** | 0.0000 | 11.00 |

> **browsing + boundary = 90/200 sessions score almost nothing.** This is the biggest scoring
> opportunity: simply lifting browsing from 0.025 up to the buying level nearly doubles TechnicalScore.

## Reproducing

```bash
cd provided && ./fetch.sh                          # download the catalog (not tracked in git)
cd techjam-conversational-search
python3 -m evaluator.local_evaluator               # -> results.json
```

No dependencies to install — the evaluator and starter agent use the standard library only.

## Documentation

- [TRACK4_SHOPPING_COPILOT.md](TRACK4_SHOPPING_COPILOT.md) — problem statement, rules, and judging criteria
- [TechJam2026_Tom_tat.md](TechJam2026_Tom_tat.md) — overview of all tracks
- [provided/README.md](provided/README.md) — provenance & checksums of the organizer-provided resources

## Rules that will cost you points

- Exceeding **10 turns** → that session scores **zero**. You need a hard counter in the agent loop.
- The catalog is **read-only** — no mutation, no injecting fake ASINs.
- The vector index must be **in-memory**; no external service.
- **Never commit API keys.** The organizers do not provide or reimburse API credits.
