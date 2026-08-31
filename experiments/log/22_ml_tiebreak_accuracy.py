"""Experiment 22: is the ML model better than popularity INSIDE ties? The decisive test.

Pass 21 showed the gated learned ranker lands at -0.0057 (tune) / +0.0043 (held-out) --
a tune/holdout disagreement that reads as noise, not signal. But end-to-end score is a
blunt instrument for a component that only acts on tie groups. Pass 20 built the sharp
one, so use it.

The bar was fixed in advance, from pass 20:

    within a coverage tie containing the target, target-first rate
        popularity (shipped)   57.4%
        LLM (Groq, 320 chars)  41.2%
        random shuffle         16.2%   (group size 6.28 -> 20.1% expected)

The learned ranker has to beat 57.4% to be worth anything, because popularity is what it
would replace. This measures exactly that, on the same 68 tie groups, with no end-to-end
noise in the way.

Reuses the cached synthetic features, so training is seconds rather than minutes.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/log/22_ml_tiebreak_accuracy.py
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    behavior_for, catalog_index, coarse_category, customer_reply, initial_message,
    intent_card, load_jsonl,
)

CACHE = ROOT / "experiments" / "studies" / ".ltr_feats.npz"
if not CACHE.exists():
    print(f"missing {CACHE} -- run 21_synthetic_ltr.py first")
    raise SystemExit(1)

print("loading ...")
samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
TUNE = [s for i, s in enumerate(samples) if i % 2 == 0]
cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("p21", ROOT / "experiments" / "log" / "21_synthetic_ltr.py")

from submission.agent import Agent, CAT, CONSTRAINT, MINED  # noqa: E402

base = Agent(ROOT / "data" / "catalog.jsonl")

# --- rebuild the exact feature computation used at training time -------------
FEATS = ["cov_con", "cov_cat", "cov_mined", "sum_w", "frac_cov", "hand", "pool_rank",
         "log_pop", "cat_pop_pct", "rating", "log_len", "turn", "n_ev", "tot_w",
         "in_title", "log_min_df", "n_price", "cov_ratio"]

import bisect  # noqa: E402


def _rn(a):
    try:
        return float(prods[a].get("rating_number") or 0)
    except (TypeError, ValueError):
        return 0.0


RN = {a: _rn(a) for a in prods}
LOGP = {a: math.log1p(v) for a, v in RN.items()}
RATE = {}
for a, d in prods.items():
    try:
        RATE[a] = float(d.get("average_rating") or 0)
    except (TypeError, ValueError):
        RATE[a] = 0.0
HASPRICE = {a: 1.0 if prods[a].get("price") not in (None, "") else 0.0 for a in prods}
bycat, CATOF = {}, {}
for a, d in prods.items():
    c = coarse_category([str(x) for x in (d.get("categories") or [])])
    CATOF[a] = c
    bycat.setdefault(c, []).append(RN[a])
for c in bycat:
    bycat[c].sort()
CPCT = {a: (bisect.bisect_left(bycat[CATOF[a]], RN[a]) / len(bycat[CATOF[a]]))
        if len(bycat[CATOF[a]]) > 1 else 0.5 for a in prods}
DLEN = {a: max(1, len(base.ix.blob.get(a, ""))) for a in prods}
MAXPOP = base.ix.max_pop or 1.0


def featurize(agent, st, pool, ranks):
    wmap = {p: agent._weight(p, df, t) for p, (df, t) in st.evidence.items()}
    tiers = {p: t for p, (_, t) in st.evidence.items()}
    dfs = {p: d for p, (d, _) in st.evidence.items()}
    tot = sum(wmap.values()) or 1.0
    n_ev = len(st.evidence) or 1
    rows = np.zeros((len(pool), len(FEATS)), dtype=np.float32)
    for i, a in enumerate(pool):
        cc = ct = cm = 0
        sw = 0.0
        tit = 0
        mind = 1e9
        for p, w in wmap.items():
            if agent.ix.covers(a, p):
                sw += w
                t = tiers[p]
                cc += t == CONSTRAINT
                ct += t == CAT
                cm += t == MINED
                if agent.ix.in_title(a, p):
                    tit += 1
                mind = min(mind, dfs[p])
        hand = sw + agent.W_POP * (LOGP[a] / MAXPOP)
        rows[i] = (cc, ct, cm, sw, sw / tot, hand, ranks[i], LOGP[a], CPCT[a], RATE[a],
                   math.log1p(DLEN[a]), st.turn, n_ev, tot, tit,
                   math.log1p(mind if mind < 1e9 else 5000), HASPRICE[a],
                   sw / max(1.0, math.log1p(DLEN[a])))
    return rows


z = np.load(CACHE)
Xs, ys = z["X"], z["y"]
print(f"cached features: {Xs.shape[0]:,} rows, {int(ys.sum()):,} positives")

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

model = HistGradientBoostingClassifier(random_state=0, class_weight="balanced",
                                       max_depth=6, max_iter=600, learning_rate=0.06)
model.fit(Xs, ys)
print("model trained\n")

# --- walk the tune half, collect tie groups containing the target ------------
rng = random.Random(0)
det_pos, ml_pos, rnd_pos, sizes = [], [], [], []
ag = base

for s in TUNE:
    tgt = str(s["ground_truth"]["parent_asin"])
    card = intent_card(prods[tgt])
    eff = {**s, "intent_card": card,
           "behavior": behavior_for(str(s["scenario_type"]), card,
                                    random.Random(f"{s['sample_id']}\0{s['scenario_type']}"))}
    disclosed, bu = set(), False
    sid = s["sample_id"]
    ag.reset(sid, s["user_profile"])
    st = ag.sessions[sid]
    msg = initial_message(eff, coarse_category(
        [str(v) for v in (prods[tgt].get("categories") or [])]), disclosed)
    applied = s["scenario_type"] != "intent_override"
    for turn in range(1, 11):
        st.turn += 1
        try:
            ag._observe(st, msg)
            pool = ag._candidates(st, msg)
            ranked = ag._rank(st, pool, 10) if pool else []
        except Exception:
            ranked = []
        if len(ranked) >= 2 and st.evidence:
            wm = {p: ag._weight(p, df, t) for p, (df, t) in st.evidence.items()}

            def cov(a):
                return sum(w for p, w in wm.items() if ag.ix.covers(a, p))

            i = 0
            while i < len(ranked):
                j = i + 1
                while j < len(ranked) and abs(cov(ranked[j]) - cov(ranked[i])) < 1e-12:
                    j += 1
                if j - i >= 2 and i == 0 and tgt in ranked[i:j]:
                    grp = ranked[i:j]
                    sizes.append(len(grp))
                    det_pos.append(grp.index(tgt))
                    rnd_pos.append(rng.randrange(len(grp)))
                    F = featurize(ag, st, grp, list(range(i, j)))
                    p = model.predict_proba(F)[:, 1]
                    ml_order = [grp[k] for k in np.argsort(-p)]
                    ml_pos.append(ml_order.index(tgt))
                i = j
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
            msg, bu = customer_reply(eff, ag._next_probe(st), disclosed, bu)
    ag.sessions.pop(sid, None)

if not det_pos:
    print("no tie groups found")
    raise SystemExit(0)


def rep(name, pos):
    first = sum(1 for p in pos if p == 0) / len(pos)
    mrr = statistics.fmean(1.0 / (p + 1) for p in pos)
    print(f"  {name:<30} target-first {first:>6.1%}   mean pos {statistics.fmean(pos)+1:>5.2f}"
          f"   within-MRR {mrr:.4f}")
    return {"target_first": first, "mrr": mrr}


print(f"tie groups containing the target: {len(det_pos)}   "
      f"mean size {statistics.fmean(sizes):.2f}\n")
out = {"groups": len(det_pos), "mean_size": statistics.fmean(sizes),
       "deterministic": rep("deterministic (popularity)", det_pos),
       "ml": rep("ML learned ranker", ml_pos),
       "random": rep("random shuffle", rnd_pos)}
up = sum(1 for d, m in zip(det_pos, ml_pos) if m < d)
dn = sum(1 for d, m in zip(det_pos, ml_pos) if m > d)
print(f"\n  ML moved the target UP in {up}, DOWN in {dn}, unchanged in {len(det_pos)-up-dn}")
out["moved_up"], out["moved_down"] = up, dn
bar = out["deterministic"]["target_first"]
print(f"\n  BAR (popularity): {bar:.1%}   ML: {out['ml']['target_first']:.1%}   "
      f"-> {'BEATS the bar' if out['ml']['target_first'] > bar else 'FAILS the bar'}")
print(f"  reference: LLM scored 41.2% on this same measurement (pass 20)")

(ROOT / "experiments" / "results" / "out_22.json").write_text(
    json.dumps(out, indent=2) + "\n", encoding="utf-8")
print("\n[saved] experiments/results/out_22.json")
