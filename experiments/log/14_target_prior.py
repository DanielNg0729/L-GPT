"""
Experiment 14: modelling the TARGET SELECTION PRIOR.

The literature passes are exhausted -- every IR/RAG technique we tested is null or
negative, because the task is provenance recovery and semantic machinery blurs exactly
the signal that solves it. The remaining edge is not in the retrieval literature; it is
in the BENCHMARK'S DATA-GENERATION PROCESS, which the README documents and we had never
exploited.

Measured (pass 14 diagnostic):

    rating_number    catalog median 12        TARGET median 6,846
    rating_number>=50  catalog 27.4%          targets 96.5%
    target's popularity percentile WITHIN ITS OWN CATEGORY: median 97.8%

Mechanism: the README states sessions are "sampled deterministically from the official
Clothing 5-core leave-last-out split". Sampling a REVIEW and taking its item samples
items in proportion to their review count, so P(item is target) ~ rating_number. Our
shipped prior is W_POP * log1p(rating_number)/max -- heavily compressed, and normalised
GLOBALLY rather than within the category the customer just named.

Is this legitimate, or dataset gaming? Popularity is a canonical recommender prior and
"the item someone actually bought tends to be a popular item" is true of real commerce,
not just of this split. But the STRENGTH here (97.8th percentile) is inflated by 5-core
filtering, and would be weaker in a deployment with long-tail purchases. That caveat
belongs in the write-up; it does not make the prior invalid.

Prior shapes tested, all deterministic and offline:
    P0  shipped:  log1p(rn)/max_log1p
    P1  steeper:  (log1p(rn)/max)^2
    P2  global popularity percentile
    P3  CATEGORY-CONDITIONAL popularity percentile
    P4  log(rn) direct, i.e. proportional-to-count in log space
    P5  P3 blended with P0

Run:  PYTHONIOENCODING=utf-8 python -u experiments/log/14_target_prior.py
"""
from __future__ import annotations

import bisect
import collections
import json
import math
import statistics
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


def rn(a: str) -> float:
    try:
        return float(prods[a].get("rating_number") or 0)
    except (TypeError, ValueError):
        return 0.0


print("precomputing prior shapes ...")
t0 = time.time()
RN = {a: rn(a) for a in prods}
LOG = {a: math.log1p(v) for a, v in RN.items()}
MAXLOG = max(LOG.values()) or 1.0

# global percentile
gsorted = sorted(RN.values())
GPCT = {a: bisect.bisect_left(gsorted, v) / len(gsorted) for a, v in RN.items()}

# category-conditional percentile
bycat: dict[str, list[float]] = collections.defaultdict(list)
CATOF: dict[str, str] = {}
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
print(f"  done in {time.time()-t0:.0f}s   ({len(bycat):,} categories)")

SHAPES = {
    "P0 shipped log1p/max": lambda a: LOG[a] / MAXLOG,
    "P1 steeper (log1p/max)^2": lambda a: (LOG[a] / MAXLOG) ** 2,
    "P2 global percentile": lambda a: GPCT[a],
    "P3 category percentile": lambda a: CPCT[a],
    "P4 log(rn) direct": lambda a: LOG[a] / 12.0,
    "P5 0.5*cat_pct + 0.5*log1p": lambda a: 0.5 * CPCT[a] + 0.5 * (LOG[a] / MAXLOG),
}


class PriorAgent(Agent):
    SHAPE = staticmethod(SHAPES["P0 shipped log1p/max"])

    def _rank(self, st, pool, top_k):
        wmap = {p: self._weight(p, df, tier) for p, (df, tier) in st.evidence.items()}
        order = {a: i for i, a in enumerate(pool)}
        cover = self.ix.covers

        def score(asin: str):
            s = 0.0
            for phrase, w in wmap.items():
                if cover(asin, phrase):
                    s += w
            s += self.W_POP * self.SHAPE(asin)
            return (-s, order[asin])

        return sorted(pool, key=score)[:top_k]


def share(shape, w_pop):
    o = object.__new__(PriorAgent)
    o.ix, o.sessions = base.ix, {}
    o.SHAPE = staticmethod(shape)
    o.W_POP = w_pop
    return o


def run(ag, subset, tag):
    r = evaluate(ag, subset, cid, cats, prods)
    print(f"    {tag:<40} HR@10 {r['hit_rate_at_10']:>6.1%}  MRR {r['mrr']:>6.4f}  "
          f"MTTC {r['mttc']:>5.2f}  SCORE {r['recommended_technical_score']:>7.5f}")
    return {"hr": r["hit_rate_at_10"], "mrr": r["mrr"], "mttc": r["mttc"],
            "score": r["recommended_technical_score"]}


print("\n" + "=" * 96)
print("A. PRIOR SHAPE, at the shipped weight W_POP=0.35 -- tuning half")
print("=" * 96)
res = {}
for name, fn in SHAPES.items():
    res[name] = run(share(fn, 0.35), TUNE, name)
OUT["shape_at_035"] = res
best_shape = max(res, key=lambda k: res[k]["score"])
print(f"\n  best shape: {best_shape} ({res[best_shape]['score']:.5f})")

print("\n" + "=" * 96)
print(f"B. WEIGHT SWEEP for the two leading shapes -- tuning half")
print("=" * 96)
wres: dict = {}
for name in (best_shape, "P0 shipped log1p/max"):
    if name in wres:
        continue
    wres[name] = {}
    for w in (0.2, 0.35, 0.6, 1.0, 1.6, 2.5):
        wres[name][w] = run(share(SHAPES[name], w), TUNE, f"{name}  W_POP={w}")
    print()
OUT["weight_sweep"] = wres

best_cfg = max(((n, w) for n in wres for w in wres[n]),
               key=lambda nw: wres[nw[0]][nw[1]]["score"])
print(f"  best configuration on tuning half: {best_cfg[0]} @ W_POP={best_cfg[1]} "
      f"({wres[best_cfg[0]][best_cfg[1]]['score']:.5f})")

print("\n" + "=" * 96)
print("C. HELD-OUT ADJUDICATION")
print("=" * 96)
hb = run(share(SHAPES["P0 shipped log1p/max"], 0.35), HOLD, "shipped (P0 @ 0.35)")
hn = run(share(SHAPES[best_cfg[0]], best_cfg[1]), HOLD, f"best ({best_cfg[0]} @ {best_cfg[1]})")
d = hn["score"] - hb["score"]
print(f"\n  HELD-OUT DELTA: {d:+.5f}  -> "
      f"{'ADOPT' if d > 0.005 else 'inside noise' if d > -0.005 else 'REJECT'}")
OUT["holdout"] = {"shipped": hb, "best": hn, "delta": d, "cfg": [best_cfg[0], best_cfg[1]]}

print("\n" + "=" * 96)
print("D. TAIL SAFETY: does an aggressive prior lose UNPOPULAR targets?")
print("=" * 96)
r_full = evaluate(share(SHAPES[best_cfg[0]], best_cfg[1]), samples, cid, cats, prods)
byid = {s["sample_id"]: s for s in samples}
miss = [x for x in r_full["sessions"] if not x["hit"]]
print(f"    misses under the new prior: {len(miss)}")
for m in miss[:8]:
    a = str(byid[m["sample_id"]]["ground_truth"]["parent_asin"])
    print(f"      {m['sample_id']}  rating_number={RN[a]:>8.0f}  "
          f"cat_pct={CPCT[a]:>5.1%}  scenario={byid[m['sample_id']]['scenario_type']}")
tar_rn = sorted(RN[str(s['ground_truth']['parent_asin'])] for s in samples)
print(f"    target rating_number p05={tar_rn[len(tar_rn)//20]:.0f}  "
      f"min={tar_rn[0]:.0f}  -- an over-aggressive prior would drop the low tail first")
print(f"\n    ALL 200 with best prior: SCORE {r_full['recommended_technical_score']:.5f}")
OUT["full"] = {"score": r_full["recommended_technical_score"],
               "hr": r_full["hit_rate_at_10"], "mrr": r_full["mrr"], "mttc": r_full["mttc"]}

Path(ROOT / "experiments" / "results" / "out_14.json").write_text(
    json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print("\n[saved] experiments/results/out_14.json")
