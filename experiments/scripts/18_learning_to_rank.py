"""
Experiment 18: learning-to-rank over the coverage features.

Our reranker is a hand-tuned linear sum of ~5 constants I chose by coordinate ascent:

    score = SUM_phrase w(phrase)*covers(d, phrase)  +  W_POP * popularity

A linear sum cannot express INTERACTIONS. A tree ensemble can learn things a fixed weight
cannot, e.g. "popularity matters only when evidence count <= 2" -- which is exactly the
adaptive-prior idea that failed in pass 15 as a hand-coded decay curve, and exactly the
title-anchoring feature that had to be zeroed in pass 7 because it helped only sometimes.

Rules check: the brief bans "training or full-parameter fine-tuning of base foundational
LLMs". A gradient-boosted tree over hand-built features is not an LLM, and the brief
explicitly lists "fine-tuning ... local scoring logic" as IN SCOPE.

PROTOCOL. Training data comes ONLY from the tuning half; the model is scored ONLY on the
held-out half, so no session contributes to both. Rows are grouped by session, never split
across it. Note the real constraint is 100 independent QUERIES, not 40,000 rows -- the
BM25 sweep in pass 8 showed 6 free parameters already overfit 100 sessions, and a tree
ensemble has far more capacity.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/scripts/18_learning_to_rank.py
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    behavior_for, catalog_index, coarse_category, customer_reply, evaluate,
    initial_message, intent_card, load_jsonl,
)
from submission.agent import Agent, CONSTRAINT, CAT, MINED  # noqa: E402

CATALOG = ROOT / "data" / "catalog.jsonl"
PUBLIC = ROOT / "data" / "public_set.jsonl"
OUT: dict = {}

print("loading ...")
samples = load_jsonl(PUBLIC)
cid, cats, prods = catalog_index(CATALOG)
TUNE = [s for i, s in enumerate(samples) if i % 2 == 0]
HOLD = [s for i, s in enumerate(samples) if i % 2 == 1]
base = Agent(CATALOG)

RATE, POP, DLEN = {}, {}, {}
for a, d in prods.items():
    try:
        RATE[a] = float(d.get("average_rating") or 0)
    except (TypeError, ValueError):
        RATE[a] = 0.0
    POP[a] = base.ix.pop.get(a, 0.0)
    DLEN[a] = max(1, len(base.ix.blob.get(a, "")))
MAXPOP = base.ix.max_pop or 1.0

FEATS = ["n_con", "n_cat", "n_mined", "sum_w", "frac_cov", "hand_score", "pool_rank",
         "log_pop", "rating", "log_len", "turn", "n_ev", "tot_w", "in_title", "min_df"]


def featurize(agent: Agent, st, pool: list[str]) -> np.ndarray:
    wmap = {p: agent._weight(p, df, tier) for p, (df, tier) in st.evidence.items()}
    tiers = {p: t for p, (_, t) in st.evidence.items()}
    dfs = {p: d for p, (d, _) in st.evidence.items()}
    tot_w = sum(wmap.values()) or 1.0
    n_ev = len(st.evidence) or 1
    rows = np.zeros((len(pool), len(FEATS)), dtype=np.float32)
    for i, a in enumerate(pool):
        n_con = n_cat = n_mined = 0
        sw = 0.0
        intitle = 0
        mind = 1e9
        for p, w in wmap.items():
            if agent.ix.covers(a, p):
                sw += w
                t = tiers[p]
                n_con += t == CONSTRAINT
                n_cat += t == CAT
                n_mined += t == MINED
                if agent.ix.in_title(a, p):
                    intitle += 1
                mind = min(mind, dfs[p])
        hand = sw + agent.W_POP * (POP[a] / MAXPOP)
        rows[i] = (n_con, n_cat, n_mined, sw, sw / tot_w, hand, i,
                   POP[a], RATE[a], math.log1p(DLEN[a]), st.turn, n_ev, tot_w,
                   intitle, math.log1p(mind if mind < 1e9 else 5000))
    return rows


def replay(subset, collect=True):
    """Replay sessions the way the harness does, capturing pool features per turn."""
    X, y, grp = [], [], []
    ag = object.__new__(Agent)
    ag.ix, ag.sessions = base.ix, {}
    for s in subset:
        tgt = str(s["ground_truth"]["parent_asin"])
        card = intent_card(prods[tgt])
        rng = random.Random(f"{s['sample_id']}\0{s['scenario_type']}")
        eff = {**s, "intent_card": card,
               "behavior": behavior_for(str(s["scenario_type"]), card, rng)}
        disclosed, bu = set(), False
        sid = s["sample_id"]
        ag.reset(sid, s["user_profile"])
        msg = initial_message(eff, coarse_category(
            [str(v) for v in (prods[tgt].get("categories") or [])]), disclosed)
        st = ag.sessions[sid]
        applied = s["scenario_type"] != "intent_override"
        for turn in range(1, 11):
            st.turn += 1
            try:
                ag._observe(st, msg)
                pool = ag._candidates(st, msg)
            except Exception:
                pool = []
            if pool and collect:
                keep = pool[:200]
                if tgt in pool and tgt not in keep:
                    keep = keep[:199] + [tgt]
                F = featurize(ag, st, keep)
                X.append(F)
                y.append(np.array([1 if a == tgt else 0 for a in keep], dtype=np.int8))
                grp.append(np.full(len(keep), hash(sid) % (2**31), dtype=np.int64))
            probe = ag._next_probe(st)
            st.asked.append(probe)
            ranked = ag._rank(st, pool, 10) if pool else []
            if applied and tgt in ranked:
                break
            ov = eff.get("behavior", {}).get("override") or {}
            if not applied and turn + 1 == int(ov.get("turn", 3)):
                applied = True
                nv = str(ov.get("new_value", ""))
                if nv:
                    disclosed.add(nv)
                msg = str(ov.get("message", ""))
            else:
                msg, bu = customer_reply(eff, probe, disclosed, bu)
    if not X:
        return None, None, None
    return np.vstack(X), np.concatenate(y), np.concatenate(grp)


print("collecting training rows from the TUNING half only ...")
t0 = time.time()
Xtr, ytr, gtr = replay(TUNE)
print(f"  {Xtr.shape[0]:,} rows, {Xtr.shape[1]} features, {int(ytr.sum())} positives, "
      f"{len(set(gtr.tolist()))} independent queries  [{time.time()-t0:.0f}s]")
OUT["train"] = {"rows": int(Xtr.shape[0]), "positives": int(ytr.sum()),
                "queries": len(set(gtr.tolist()))}

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

MODELS = {
    "shallow (depth 3, 100 trees)": dict(max_depth=3, max_iter=100, learning_rate=0.1),
    "medium (depth 5, 300 trees)": dict(max_depth=5, max_iter=300, learning_rate=0.08),
    "deep (depth 8, 600 trees)": dict(max_depth=8, max_iter=600, learning_rate=0.05),
}
fitted = {}
for name, kw in MODELS.items():
    t0 = time.time()
    m = HistGradientBoostingClassifier(random_state=0, class_weight="balanced", **kw)
    m.fit(Xtr, ytr)
    fitted[name] = m
    print(f"  trained {name} in {time.time()-t0:.0f}s")


class LTRAgent(Agent):
    MODEL = None
    BLEND = 0.0          # 0 = pure model score; >0 = model + BLEND*hand score

    def _rank(self, st, pool, top_k):
        if self.MODEL is None or not pool or not st.evidence:
            return super()._rank(st, pool, top_k)
        keep = pool[:200]
        try:
            F = featurize(self, st, keep)
            p = self.MODEL.predict_proba(F)[:, 1]
        except Exception:
            return super()._rank(st, pool, top_k)
        if self.BLEND:
            hand = F[:, FEATS.index("hand_score")]
            hs = (hand - hand.min()) / (hand.ptp() + 1e-9)
            p = p + self.BLEND * hs
        idx = np.argsort(-p)[:top_k]
        out = [keep[i] for i in idx]
        return (out + [a for a in pool if a not in set(out)])[:top_k]


def share(cls, **kw):
    o = object.__new__(cls)
    o.ix, o.sessions = base.ix, {}
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def run(ag, subset, tag):
    r = evaluate(ag, subset, cid, cats, prods)
    print(f"    {tag:<48} HR@10 {r['hit_rate_at_10']:>6.1%}  MRR {r['mrr']:>6.4f}  "
          f"MTTC {r['mttc']:>5.2f}  SCORE {r['recommended_technical_score']:>7.5f}")
    return {"hr": r["hit_rate_at_10"], "mrr": r["mrr"], "mttc": r["mttc"],
            "score": r["recommended_technical_score"]}


print("\n" + "=" * 108)
print("A. IN-SAMPLE (tuning half -- the half the model TRAINED on; expect inflation)")
print("=" * 108)
ins = {"baseline": run(share(Agent), TUNE, "baseline")}
for name, m in fitted.items():
    ins[name] = run(share(LTRAgent, MODEL=m), TUNE, f"LTR {name}")
OUT["in_sample"] = ins

print("\n" + "=" * 108)
print("B. HELD-OUT (sessions the model never saw) -- the only number that counts")
print("=" * 108)
hb = run(share(Agent), HOLD, "baseline")
hold = {"baseline": hb}
for name, m in fitted.items():
    hold[name] = run(share(LTRAgent, MODEL=m), HOLD, f"LTR {name}")
for b in (0.5, 2.0):
    nm = f"medium + {b}x hand score"
    hold[nm] = run(share(LTRAgent, MODEL=fitted["medium (depth 5, 300 trees)"], BLEND=b),
                   HOLD, f"LTR {nm}")
OUT["holdout"] = hold

best = max((k for k in hold if k != "baseline"), key=lambda k: hold[k]["score"])
d = hold[best]["score"] - hb["score"]
print(f"\n  best held-out LTR variant: {best}  ({hold[best]['score']:.5f})")
print(f"  delta vs baseline: {d:+.5f}  -> "
      f"{'ADOPT' if d > 0.005 else 'inside noise' if d > -0.005 else 'REJECT'}")

gap = ins["medium (depth 5, 300 trees)"]["score"] - hold["medium (depth 5, 300 trees)"]["score"]
print(f"\n  in-sample minus held-out for the medium model: {gap:+.5f}")
print("  (a large positive gap is memorisation of the 100 training queries)")
OUT["overfit_gap"] = gap

Path(ROOT / "experiments" / "results" / "out_18.json").write_text(
    json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print("\n[saved] experiments/results/out_18.json")
