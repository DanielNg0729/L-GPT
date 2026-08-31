"""
Experiment 15: turn-adaptive prior weighting.

Pass 14 establishes that P(item is target) is strongly popularity-biased. But our prior
weight is a CONSTANT, and that is wrong on its face. Ranking is, in effect:

    score(d)  =  log P(evidence | d)   +   log P(d is the target)
                 ^ likelihood              ^ prior

The relative influence of the prior should fall as the likelihood sharpens. Our evidence
strength varies enormously across a session:

    turn 1, browsing : one category phrase, median 145 matching products  -> weak
    turn 3, any      : four verbatim constraints, median 1 match          -> overwhelming

A single W_POP cannot serve both. At turn 1 it is too timid (the prior is nearly all the
information we have); by turn 3 it is actively harmful (it can outvote exact provenance).
This pass makes the prior weight a decreasing function of measured evidence strength.

Evidence strength is taken as the total achievable coverage weight -- the sum of phrase
weights the agent could award -- which grows as constraints accumulate and is naturally
larger for rare (low-df) phrases.

    w_eff = W_BASE / (1 + DECAY * strength)

DECAY = 0 recovers the current fixed-weight behaviour, so the sweep contains the shipped
configuration as a special case and cannot do worse than it by construction on the
tuning half.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/log/15_adaptive_prior.py
"""
from __future__ import annotations

import bisect
import collections
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, coarse_category, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent  # noqa: E402

CATALOG = ROOT / "data" / "catalog.jsonl"
PUBLIC = ROOT / "data" / "public_set.jsonl"
OUT: dict = {}

print("loading ...")
samples = load_jsonl(PUBLIC)
cid, cats, prods = catalog_index(CATALOG)
TUNE = [s for i, s in enumerate(samples) if i % 2 == 0]
HOLD = [s for i, s in enumerate(samples) if i % 2 == 1]
base = Agent(CATALOG)

# rebuild the prior shapes (same definitions as pass 14)
def _rn(a):
    try:
        return float(prods[a].get("rating_number") or 0)
    except (TypeError, ValueError):
        return 0.0

RN = {a: _rn(a) for a in prods}
LOG = {a: math.log1p(v) for a, v in RN.items()}
MAXLOG = max(LOG.values()) or 1.0
bycat = collections.defaultdict(list)
CATOF = {}
for a, d in prods.items():
    c = coarse_category([str(x) for x in (d.get("categories") or [])])
    CATOF[a] = c
    bycat[c].append(RN[a])
for c in bycat:
    bycat[c].sort()
CPCT = {}
for a in prods:
    peers = bycat[CATOF[a]]
    CPCT[a] = (bisect.bisect_left(peers, RN[a]) / len(peers)) if len(peers) > 1 else 0.5

SHAPES = {
    "log1p": lambda a: LOG[a] / MAXLOG,
    "cat_pct": lambda a: CPCT[a],
    "blend": lambda a: 0.5 * CPCT[a] + 0.5 * (LOG[a] / MAXLOG),
}

# pick up pass 14's winner if it ran
try:
    prev = json.loads((ROOT / "experiments" / "results" / "out_14.json").read_text(encoding="utf-8"))
    won = prev.get("holdout", {}).get("cfg", ["", 0.35])
    print(f"  pass 14 winner: {won[0]} @ W_POP={won[1]}")
    OUT["pass14_winner"] = won
except Exception:
    won = ["", 0.35]


class Adaptive(Agent):
    SHAPE = staticmethod(SHAPES["log1p"])
    W_BASE = 0.35
    DECAY = 0.0

    def _rank(self, st, pool, top_k):
        wmap = {p: self._weight(p, df, tier) for p, (df, tier) in st.evidence.items()}
        order = {a: i for i, a in enumerate(pool)}
        cover = self.ix.covers
        strength = sum(wmap.values())
        w_eff = self.W_BASE / (1.0 + self.DECAY * strength)

        def score(asin: str):
            s = 0.0
            for phrase, w in wmap.items():
                if cover(asin, phrase):
                    s += w
            s += w_eff * self.SHAPE(asin)
            return (-s, order[asin])

        return sorted(pool, key=score)[:top_k]


def share(shape, w, decay):
    o = object.__new__(Adaptive)
    o.ix, o.sessions = base.ix, {}
    o.SHAPE = staticmethod(SHAPES[shape])
    o.W_BASE, o.DECAY = w, decay
    return o


def run(ag, subset, tag):
    r = evaluate(ag, subset, cid, cats, prods)
    print(f"    {tag:<44} HR@10 {r['hit_rate_at_10']:>6.1%}  MRR {r['mrr']:>6.4f}  "
          f"MTTC {r['mttc']:>5.2f}  SCORE {r['recommended_technical_score']:>7.5f}")
    return {"hr": r["hit_rate_at_10"], "mrr": r["mrr"], "mttc": r["mttc"],
            "score": r["recommended_technical_score"]}


SHAPE = won[0].split()[-1] if won[0] and won[0].split()[-1] in SHAPES else "log1p"
BASE_W = float(won[1]) if won[1] else 0.35
print(f"  building on shape='{SHAPE}', base W_POP={BASE_W}\n")

print("=" * 100)
print("A. DECAY SWEEP (DECAY=0 reproduces the fixed-weight shipped behaviour) -- tuning half")
print("=" * 100)
res = {}
for decay in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
    res[decay] = run(share(SHAPE, BASE_W, decay), TUNE, f"DECAY={decay}")
OUT["decay_tune"] = {str(k): v for k, v in res.items()}
best_decay = max(res, key=lambda k: res[k]["score"])
print(f"\n  best DECAY={best_decay} ({res[best_decay]['score']:.5f})")

print("\n" + "=" * 100)
print("B. RAISE THE BASE WEIGHT NOW THAT IT DECAYS -- tuning half")
print("=" * 100)
res2 = {}
for w in (BASE_W, BASE_W * 2, BASE_W * 4, BASE_W * 8):
    res2[w] = run(share(SHAPE, w, best_decay), TUNE, f"W_BASE={w:.2f}, DECAY={best_decay}")
OUT["base_sweep"] = {str(k): v for k, v in res2.items()}
best_w = max(res2, key=lambda k: res2[k]["score"])
print(f"\n  best W_BASE={best_w:.2f} ({res2[best_w]['score']:.5f})")

print("\n" + "=" * 100)
print("C. HELD-OUT ADJUDICATION")
print("=" * 100)
hb = run(share(SHAPE, BASE_W, 0.0), HOLD, f"fixed weight (DECAY=0, W={BASE_W})")
hn = run(share(SHAPE, best_w, best_decay), HOLD, f"adaptive (W={best_w:.2f}, DECAY={best_decay})")
d = hn["score"] - hb["score"]
print(f"\n  HELD-OUT DELTA: {d:+.5f}  -> "
      f"{'ADOPT' if d > 0.005 else 'inside noise' if d > -0.005 else 'REJECT'}")
OUT["holdout"] = {"fixed": hb, "adaptive": hn, "delta": d,
                  "cfg": {"shape": SHAPE, "w_base": best_w, "decay": best_decay}}

full = evaluate(share(SHAPE, best_w, best_decay), samples, cid, cats, prods)
print(f"\n    ALL 200 with adaptive prior: SCORE {full['recommended_technical_score']:.5f}")
for k, v in sorted(full["scenario_metrics"].items()):
    print(f"      {k:<18} HR@10 {v['hit_rate_at_10']:>6.1%}  MRR {v['mrr']:>6.3f}  MTTC {v['mttc']:>5.2f}")
OUT["full"] = {"score": full["recommended_technical_score"], "hr": full["hit_rate_at_10"],
               "mrr": full["mrr"], "mttc": full["mttc"]}

Path(ROOT / "experiments" / "results" / "out_15.json").write_text(
    json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print("\n[saved] experiments/results/out_15.json")
