# `provided/` — Resources supplied by the TikTok TechJam 2026 organizers

> **READ-ONLY.** Do not modify any file in this directory. Team code lives outside and imports
> from here. The catalog is read-only per the rules (no structural mutation, no fake ASINs).

## Source

| | |
|---|---|
| Organizer repo | https://github.com/TechJam2026/techjam-conversational-search |
| Release | `participant-kit` (published 2026-08-24) |
| Upstream commit | see `UPSTREAM_COMMIT.txt` |
| Original data | Amazon Reviews 2023 — https://amazon-reviews-2023.github.io/ |

## Layout

```
provided/
├── SHA256SUMS                       # official organizer checksums
├── UPSTREAM_COMMIT.txt              # organizer repo commit that was snapshotted
├── fetch.sh                         # re-download + verify (use after a fresh clone)
├── release/                         # original assets (NOT committed — see .gitignore)
│   ├── catalog.jsonl.gz             # 18MB
│   └── techjam-participant-kit.zip  # 18MB
└── techjam-conversational-search/   # unpacked kit (= organizer repo + catalog)
    ├── README.md                    # the repo version, newer than the one in the zip
    ├── DATA_ATTRIBUTION.md
    ├── data/
    │   ├── catalog.jsonl            # 50,000 products, 60MB — NOT committed
    │   └── public_set.jsonl         # 200 labeled dev sessions
    ├── docs/                        # spec, API contract, eval config, baseline, rules
    ├── evaluator/local_evaluator.py # official evaluator, deterministic
    ├── starter/agent.py             # weak BM25 baseline
    └── tests/test_evaluator.py      # only in the repo, not in the zip
```

## After cloning this repo

`catalog.jsonl`, `catalog.jsonl.gz`, and `techjam-participant-kit.zip` are gitignored (~96MB total).
Run this to restore them:

```bash
cd provided && ./fetch.sh
```

Requires an authenticated `gh` CLI. The script verifies SHA-256 itself and checks for exactly 50,000 lines.

## Verified

- `shasum -a 256 -c SHA256SUMS` → both assets **OK**
- `catalog.jsonl` from the zip **hash-matches** the decompressed `catalog.jsonl.gz`
- `catalog.jsonl` = 50,000 lines · `public_set.jsonl` = 200 lines
- Scenario distribution: buying 80 · browsing 80 · intent_override 30 · boundary 10

## Important note

**The README in the zip differs from the one in the repo** — the repo version is newer and is the one kept here:

> ~~The organizer may reimburse model costs through prizes instead of issuing API keys.~~
> → **The organizer does not provide or reimburse model API credits; teams are responsible
> for any costs incurred through optional external services.**

In other words, **the organizers neither provide nor reimburse API credits**. Using a paid LLM is not required.

**Never** put API keys, private eval data, or agent output in this directory.
