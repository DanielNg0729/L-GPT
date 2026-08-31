"""EDA pass 55: re-test DF_CAP at the CURRENT (trial-38) configuration.

WHY RE-TEST SOMETHING ALREADY TUNED
-----------------------------------
Pass 42 measured `DF_CAP` 4000 -> 12000 as a free gain (clean and unseen-800 unchanged,
paraphrase T1 +0.0056, T5 +0.0097). The shipped agent now carries `DF_CAP = 2715`, adopted
as part of "frozen trial 38" together with the BM25 column weights and `maxn=12 / minn=4`.

Those two facts are not contradictory -- they were measured on different configurations --
but they cannot both be optimal, and 2715 is a strange value to arrive at on purpose. Its
job is twofold:

  * bound the cost of a document-frequency scan (a performance concern), and
  * gate n-gram mining, which keeps an n-gram only when `0 < df <= DF_CAP`.

The second is substantive: at 2715, any phrase occurring in more than 2,715 products is
excluded from mining entirely. Mining is the paraphrase floor, so that is a real decision
rather than a tuning detail.

WHAT THIS DOES NOT ASSUME. `DF_CAP` is swept AT the current configuration -- trial-38 BM25
weights, maxn/minn as shipped -- so the interaction is respected. A value that wins here
wins in the presence of the rest of trial 38, which is the only claim worth making.

The pre-registered rule is unchanged: adopt only if NO condition regresses. Conditions are
the ones that survive the organizer's "no paraphrase" confirmation as decision criteria
(public, unseen, populations), plus the paraphrase suites reported as characterisation.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/scripts/68_dfcap_recheck.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notes" / "eda"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent  # noqa: E402

# Uses the FROZEN release sets in robustness/sets rather than minting on the fly, so the
# grid matches the published robustness suite exactly and is byte-reproducible.

CAPS = (1500, 2715, 6000, 12000, 25000)


def main() -> None:
    t0 = time.time()
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    pub_t = {str(s["ground_truth"]["parent_asin"]) for s in samples}
    profiles = [s["user_profile"] for s in samples]
    shipped = base.ix.DF_CAP
    print(f"shipped DF_CAP = {shipped}  (trial 38)")

    rs = ROOT / "robustness" / "sets"
    pvo = ROOT / "robustness" / "v2" / "public_value_only"
    sets = {
        "public200": samples,
        "org-proxy": load_jsonl(rs / "organizer_proxy_800.jsonl"),
        "review800": load_jsonl(rs / "catalog_review_distinct_800.jsonl"),
        "uniform": load_jsonl(rs / "catalog_uniform_800.jsonl"),
        "inverse": load_jsonl(rs / "catalog_inverse_800.jsonl"),
        "attr-para": load_jsonl(pvo / "official200_attribute_paraphrase_dev.jsonl"),
    }
    COLS = list(sets)

    def share():
        o = object.__new__(Agent)
        o.ix, o.sessions = base.ix, {}
        o.llm = o.llm_extract = o.tagger = None
        return o

    def row(cap):
        base.ix.DF_CAP = cap
        base.ix.df.cache_clear()          # df is lru_cached and the cap changes its value
        r = {}
        for name, sub in sets.items():
            r[name] = evaluate(share(), sub, cid, cats, prods)["recommended_technical_score"]
        return r

    print(f"\n{'DF_CAP':>8}" + "".join(f"{c:>12}" for c in COLS))
    print("-" * (8 + 12 * len(COLS)))
    out, ref = {}, None
    try:
        for cap in CAPS:
            r = row(cap)
            out[cap] = r
            if cap == shipped:
                ref = r
            print(f"{cap:>8}" + "".join(f"{r[c]:>12.5f}" for c in COLS), flush=True)
    finally:
        base.ix.DF_CAP = shipped
        base.ix.df.cache_clear()

    print(f"\ndeltas vs shipped ({shipped})")
    print(f"{'DF_CAP':>8}" + "".join(f"{c:>12}" for c in COLS) + "   verdict")
    print("-" * (8 + 12 * len(COLS) + 12))
    for cap, r in out.items():
        d = {c: r[c] - ref[c] for c in COLS}
        decisive = ["public200", "org-proxy", "review800", "uniform", "inverse"]
        worst = min(d[c] for c in decisive)
        verdict = ("shipped" if cap == shipped else
                   "ADOPT -- no regression on decision criteria" if worst >= 0 else
                   "inside noise" if worst > -0.005 else "REJECT")
        print(f"{cap:>8}" + "".join(f"{d[c]:>+12.5f}" for c in COLS) + f"   {verdict}")

    print("\n  decision criteria are public200/unseen800/uniform/inverse. The paraphrase")
    print("  columns are characterisation only -- the organizer confirmed no paraphrase.")
    (ROOT / "experiments" / "results" / "out_68_dfcap_recheck.json").write_text(
        json.dumps({"shipped": shipped, "results": {str(k): v for k, v in out.items()}},
                   indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] experiments/results/out_68_dfcap_recheck.json   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
