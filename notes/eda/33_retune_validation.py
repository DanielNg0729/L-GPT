"""EDA pass 33: the robustness benchmark found two constants tuned to the WRONG population.

Pass 30's sensitivity sweep, run on 800 sessions whose targets appear in no public
session, disagreed with the shipped configuration on two parameters:

    IDF_POW    shipped 0.35 -> 0.9436   but 0.0 -> 0.9518 and 0.2 -> 0.9516
    W_POP      shipped 0.35 -> 0.9436   but 0.15 -> 0.9501
    POOL       shipped 400  -> 0.9436   but 700 -> 0.9492

Both were fitted by coordinate ascent over the public 200 (pass 07). That is exactly the
failure the benchmark exists to catch: a constant that encodes the sampling noise of 200
sessions rather than a property of the task.

THE TRAP THIS PASS AVOIDS. Having found a better value on synth800, adopting it because
synth800 likes it would just move the overfitting to a new population. So every candidate
is scored on FOUR sets, and adopted only if it does not lose on any of them:

    public-tune     100 sessions, index even   (the half everything was fitted on)
    public-hold     100 sessions, index odd    (never used for fitting)
    synth-A         800 unseen sessions, seed A -- the set that raised the flag
    synth-B         800 unseen sessions, DIFFERENT seed and disjoint draw -- so a win
                    on synth-A cannot simply be synth-A's own sampling noise

Run:  PYTHONIOENCODING=utf-8 python -u notes/eda/33_retune_validation.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notes" / "eda"))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent  # noqa: E402

SEEDS = {"reviews": 1001, "uniform": 1002, "inverse": 1003}
mint = __import__("30_robustness_benchmark").mint


def main() -> None:
    t0 = time.time()
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    pub_t = {str(s["ground_truth"]["parent_asin"]) for s in samples}
    profiles = [s["user_profile"] for s in samples]

    SETS = {
        "public-tune": [s for i, s in enumerate(samples) if i % 2 == 0],
        "public-hold": [s for i, s in enumerate(samples) if i % 2 == 1],
        "synth-A":     mint(prods, pub_t, profiles, "reviews", 800, seed=SEEDS["reviews"]),
        "synth-B":     mint(prods, pub_t, profiles, "reviews", 800, seed=987_654),
    }
    print(f"synth-A / synth-B target overlap: "
          f"{len({s['ground_truth']['parent_asin'] for s in SETS['synth-A']} & {s['ground_truth']['parent_asin'] for s in SETS['synth-B']})} "
          f"products (independent draws from 49,800 candidates)\n")

    def share(**kw):
        o = object.__new__(Agent)
        o.ix, o.sessions, o.llm = base.ix, {}, None
        for k, v in kw.items():
            setattr(o, k, v)
        return o

    def run(kw, sub):
        r = evaluate(share(**kw), sub, cid, cats, prods)
        return r["recommended_technical_score"], r["hit_rate_at_10"]

    CANDIDATES = {
        "shipped (IDF .35, POP .35, POOL 400)": {},
        "IDF_POW 0.20":                         {"IDF_POW": 0.20},
        "IDF_POW 0.00":                         {"IDF_POW": 0.00},
        "W_POP 0.25":                           {"W_POP": 0.25},
        "W_POP 0.15":                           {"W_POP": 0.15},
        "POOL 700":                             {"POOL": 700},
        "IDF .20 + POOL 700":                   {"IDF_POW": 0.20, "POOL": 700},
        "IDF .20 + POP .25":                    {"IDF_POW": 0.20, "W_POP": 0.25},
        "IDF .20 + POP .25 + POOL 700":         {"IDF_POW": 0.20, "W_POP": 0.25, "POOL": 700},
        "IDF .00 + POP .15 + POOL 700":         {"IDF_POW": 0.00, "W_POP": 0.15, "POOL": 700},
    }

    cols = list(SETS)
    print(f"{'configuration':<38}" + "".join(f"{c:>13}" for c in cols) + f"{'minΔ':>9}")
    print("-" * (38 + 13 * len(cols) + 9))
    OUT, ref = {}, None
    for name, kw in CANDIDATES.items():
        res = {c: run(kw, SETS[c]) for c in cols}
        if ref is None:
            ref = {c: res[c][0] for c in cols}
        deltas = [res[c][0] - ref[c] for c in cols]
        OUT[name] = {"kw": kw, "scores": {c: res[c][0] for c in cols},
                     "hr": {c: res[c][1] for c in cols},
                     "deltas": dict(zip(cols, deltas)), "min_delta": min(deltas)}
        cells = "".join(f"{res[c][0]:>13.5f}" for c in cols)
        print(f"{name:<38}{cells}{min(deltas):>+9.5f}")

    print(f"\n{'configuration':<38}" + "".join(f"{c:>13}" for c in cols) + "   verdict")
    print("-" * (38 + 13 * len(cols) + 12))
    for name, v in OUT.items():
        cells = "".join(f"{v['deltas'][c]:>+13.5f}" for c in cols)
        if name.startswith("shipped"):
            verdict = "reference"
        elif v["min_delta"] >= 0:
            verdict = "ADOPT (wins/ties everywhere)"
        elif v["min_delta"] > -0.005:
            verdict = "inside noise"
        else:
            verdict = "REJECT"
        print(f"{name:<38}{cells}   {verdict}")

    best = max((k for k in OUT if not k.startswith("shipped")),
               key=lambda k: (OUT[k]["min_delta"],
                              sum(OUT[k]["deltas"].values())))
    print(f"\n  best by worst-case delta: {best}")
    print(f"    HR@10 -- " + "  ".join(f"{c} {OUT[best]['hr'][c]:.1%}" for c in cols))
    print(f"    shipped  " + "  ".join(
        f"{c} {OUT['shipped (IDF .35, POP .35, POOL 400)']['hr'][c]:.1%}" for c in cols))

    (ROOT / "notes" / "eda" / "out_33.json").write_text(
        json.dumps(OUT, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] notes/eda/out_33.json   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
