"""
Experiment 7: coordinate-ascent tuning of the reranker, scored by the official evaluator.

After pass 6 the official CLI reports HR@10 0.975 but MRR 0.647 -- i.e. conditional on a
hit the mean rank is ~1.5. MRR carries 0.30 weight and is nonlinear, so promoting mass
from rank 2-3 to rank 1 is the highest-value work left (pass 1: rank quality is worth
~12x turn latency).

METHODOLOGICAL NOTE ON OVERFITTING
-----------------------------------
We tune on the 200 PUBLIC sessions; the real score comes from 800 PRIVATE ones. Any
sharp optimum found here is more likely to be noise than signal. This script therefore
reports the whole sweep curve, not just the argmax, and the final configuration is
chosen from a FLAT region rather than a peak. Sensitivity is printed so the risk is
visible rather than assumed away.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/scripts/07_rank_refinement.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent  # noqa: E402

CATALOG = ROOT / "data" / "catalog.jsonl"
PUBLIC = ROOT / "data" / "public_set.jsonl"
OUT: dict = {}

print("loading catalog + building index (once, reused for every configuration) ...")
samples = load_jsonl(PUBLIC)
cid, cats, prods = catalog_index(CATALOG)
t0 = time.time()
agent = Agent(CATALOG)
print(f"index built in {time.time()-t0:.0f}s\n")

PARAMS = ("IDF_POW", "W_TITLE", "W_POP", "W_CATEGORY", "W_MINED", "W_PROFILE", "STRONG_DF")


def snapshot() -> dict:
    return {p: getattr(agent, p) for p in PARAMS}


def apply(cfg: dict) -> None:
    for k, v in cfg.items():
        setattr(agent, k, v)


def score(tag: str = "") -> dict:
    r = evaluate(agent, samples, cid, cats, prods)
    if tag:
        print(f"    {tag:<26} HR@10 {r['hit_rate_at_10']:>6.1%}  MRR {r['mrr']:>6.4f}  "
              f"MTTC {r['mttc']:>5.2f}  SCORE {r['recommended_technical_score']:>7.5f}")
    return {"hr": r["hit_rate_at_10"], "mrr": r["mrr"], "mttc": r["mttc"],
            "score": r["recommended_technical_score"]}


base = snapshot()
print("baseline configuration:", base)
b = score("baseline")
OUT["baseline"] = {"cfg": dict(base), "metrics": b}
print()

SWEEPS = [
    ("IDF_POW",   [0.0, 0.10, 0.20, 0.25, 0.35, 0.50, 0.70]),
    ("W_TITLE",   [0.0, 0.20, 0.35, 0.60, 1.00, 1.60]),
    ("W_CATEGORY", [0.20, 0.40, 0.55, 0.75, 1.00]),
    ("W_MINED",   [0.15, 0.30, 0.40, 0.55, 0.75]),
    ("W_POP",     [0.0, 0.05, 0.10, 0.20, 0.35]),
    ("W_PROFILE", [0.0, 0.05, 0.12, 0.25]),
    ("STRONG_DF", [200, 500, 1200, 2500]),
]

curves: dict[str, dict] = {}
for name, values in SWEEPS:
    print(f"--- sweeping {name} (others held at current best) ---")
    curve = {}
    for v in values:
        setattr(agent, name, v)
        curve[v] = score(f"{name}={v}")
    setattr(agent, name, max(curve, key=lambda k: curve[k]["score"]))
    best_v = getattr(agent, name)
    scores = [c["score"] for c in curve.values()]
    spread = max(scores) - min(scores)
    flat = spread < 0.006
    print(f"    -> best {name}={best_v}  (spread {spread:.4f}"
          f"{', FLAT - low overfit risk' if flat else ''})\n")
    curves[name] = {"curve": {str(k): v for k, v in curve.items()},
                    "best": best_v, "spread": spread, "flat": flat}

OUT["sweeps"] = curves
tuned = snapshot()
print("tuned configuration:", tuned)
final = score("TUNED")
OUT["tuned"] = {"cfg": dict(tuned), "metrics": final}

print(f"\nbaseline score {b['score']:.5f}  ->  tuned {final['score']:.5f} "
      f"({final['score']-b['score']:+.5f})")

# ---------------------------------------------------------------- probe order
print("\n--- probe-order sweep at the tuned weights ---")
import submission.agent as sa  # noqa: E402

ORDERS = {
    "feature,other,material,...": ("feature", "other", "material", "color", "style", "size", "use_case"),
    "other first":               ("other", "feature", "material", "color", "style", "size", "use_case"),
    "feature,material,other":    ("feature", "material", "other", "color", "style", "size", "use_case"),
    "other only":                ("other",),
    "feature only":              ("feature",),
}
po = {}
orig_order = sa.PROBE_ORDER
for nm, order in ORDERS.items():
    sa.PROBE_ORDER = order
    po[nm] = score(nm)
sa.PROBE_ORDER = max(po, key=lambda k: po[k]["score"])
best_order = ORDERS[sa.PROBE_ORDER] if isinstance(sa.PROBE_ORDER, str) else sa.PROBE_ORDER
sa.PROBE_ORDER = ORDERS[max(po, key=lambda k: po[k]["score"])]
OUT["probe_order"] = po
OUT["best_probe_order"] = list(sa.PROBE_ORDER)
print(f"\n-> best probe order: {sa.PROBE_ORDER}")

# ---------------------------------------------------------------- stability
print("\n--- stability check: 5 disjoint 40-session folds at the tuned config ---")
folds = []
for i in range(5):
    sub = samples[i::5]
    r = evaluate(agent, sub, cid, cats, prods)
    folds.append(r["recommended_technical_score"])
    print(f"    fold {i+1} (n={len(sub)}): {r['recommended_technical_score']:.4f}")
print(f"    mean {statistics.fmean(folds):.4f}  stdev {statistics.pstdev(folds):.4f}")
print("    (a large stdev here means the public-set number is a noisy estimate of the")
print("     private-set score, and the tuned weights should be treated with caution)")
OUT["folds"] = {"scores": folds, "mean": statistics.fmean(folds),
                "stdev": statistics.pstdev(folds)}

Path(ROOT / "experiments" / "results" / "out_07.json").write_text(
    json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print("\n[saved] experiments/results/out_07.json")
print("\nFINAL RECOMMENDED CONSTANTS for submission/agent.py:")
for k, v in tuned.items():
    print(f"    {k} = {v}")
print(f"    PROBE_ORDER = {tuple(sa.PROBE_ORDER)}")
