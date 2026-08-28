"""EDA pass 52: WHICH product features actually predict the target inside a tie?

WHY THIS WAS MISSING. "Popularity is the only usable signal" has been asserted since pass
14 and repeated ever since, but it was never MEASURED directly -- it was inferred from the
failure of models (D1-D4, pass 24) that happened to use popularity plus features I chose.
That is a weak basis: a model failing could mean no signal exists, or that the model was
wrong, or that my feature set missed it. This measures the dependency itself, before any
model is involved.

WHAT A "TIE" IS. Retrieval scores candidates by weighted phrase coverage. Frequently
several candidates cover EXACTLY the same evidence, so the evidence cannot order them --
59% of rank>1 hits are of this form. Something else must break the tie, and today that is
popularity. The question is whether anything else would break it better, or additionally.

THE MEASUREMENT. For each tie group containing the true target, and each candidate feature
f, compute the within-group AUC: the probability that a randomly chosen non-target scores
BELOW the target under f.

    AUC = 0.5   the feature is unrelated to being the target
    AUC > 0.5   higher f makes a candidate more likely to be the target
    AUC < 0.5   the relationship is inverted (still usable, with a sign flip)

Reported alongside the target-first rate, which is directly comparable to the 57.4% figure
popularity earns today.

TWO THINGS THAT MAKE THIS HONEST, both of which the earlier work lacked:

  1. A NULL BAND. Labels are permuted within each group to produce the AUC distribution
     under no signal. Any feature inside that band is noise, however good its point
     estimate looks. Without this, 0.52 looks like a finding.

  2. A CONDITIONAL TEST. The target is "a real purchase record" from a 5-core
     leave-last-out split, so P(target) ~ review count and ANY feature correlated with
     review count will show marginal signal that is really popularity in disguise. So each
     feature is re-measured WITHIN popularity strata. Only conditional signal is new
     information; marginal signal may be double-counting what W_POP already exploits.

Every feature is computed from fields the rules make visible: title, features, description,
price, categories, details, average_rating, rating_number, store.

Run:  PYTHONIOENCODING=utf-8 python -u notes/eda/52_tie_feature_dependency.py
"""
from __future__ import annotations

import json
import math
import pickle
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notes" / "eda"))

from evaluator.local_evaluator import (  # noqa: E402
    behavior_for, catalog_index, coarse_category, customer_reply, initial_message,
    intent_card, load_jsonl,
)
from submission.agent import CAT, CONSTRAINT, Agent  # noqa: E402

_p30 = __import__("30_robustness_benchmark")
mint, SEEDS = _p30.mint, _p30.SEEDS

N_SESSIONS = 2500
MAX_TURNS = 6


# --------------------------------------------------------------------------- features
def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def build_features(prods, ix):
    """Per-product feature vectors, from participant-visible fields only."""
    store_count: dict[str, int] = defaultdict(int)
    for a, d in prods.items():
        store_count[str(d.get("store") or "")] += 1

    feats: dict[str, dict[str, float]] = {}
    for a, d in prods.items():
        title = str(d.get("title") or "")
        bullets = d.get("features") or []
        desc = d.get("description") or []
        details = d.get("details") or {}
        cats = d.get("categories") or []
        rn = _num(d.get("rating_number"))
        ar = _num(d.get("average_rating"))
        price = _num(d.get("price"))
        doc = ix.blob.get(a, "")
        n_words = doc.count(" ")
        feats[a] = {
            "pop_log_reviews": math.log1p(rn),
            "avg_rating": ar,
            "bayes_rating": (ar * rn + 4.0 * 20.0) / (rn + 20.0),   # smoothed toward 4.0
            "rating_x_log_reviews": ar * math.log1p(rn),
            "price": price,
            "log_price": math.log1p(max(price, 0.0)),
            "has_price": 1.0 if price > 0 else 0.0,
            "title_words": float(len(title.split())),
            "n_feature_bullets": float(len(bullets) if isinstance(bullets, list) else 0),
            "n_desc_chars": float(sum(len(str(x)) for x in desc)
                                  if isinstance(desc, list) else len(str(desc))),
            "n_details_keys": float(len(details) if isinstance(details, dict) else 0),
            "n_categories": float(len(cats) if isinstance(cats, list) else 0),
            "store_size": float(store_count[str(d.get("store") or "")]),
            "log_store_size": math.log1p(store_count[str(d.get("store") or "")]),
            "doc_words": float(n_words),
            "completeness": float(bool(title) + bool(bullets) + bool(desc)
                                  + bool(details) + bool(price > 0)),
        }
    return feats


# --------------------------------------------------------------------------- tie groups
def collect_ties(agent, sessions, prods, cats, max_groups=40_000):
    """Replay sessions; record groups the coverage score cannot separate that hold the target."""
    groups = []
    for s in sessions:
        tgt = str(s["ground_truth"]["parent_asin"])
        card = intent_card(prods[tgt])
        rng = random.Random(f"{s['sample_id']}\0{s['scenario_type']}")
        eff = {**s, "intent_card": card,
               "behavior": behavior_for(str(s["scenario_type"]), card, rng)}
        ov = eff.get("behavior", {}).get("override") or {}
        sid = f"tie_{s['sample_id']}"
        agent.reset(sid, s["user_profile"])
        st = agent.sessions[sid]
        disclosed, bu = set(), False
        applied = s["scenario_type"] != "intent_override"
        msg = initial_message(eff, coarse_category(cats.get(tgt, [])), disclosed)

        for turn in range(1, MAX_TURNS + 1):
            st.turn += 1
            agent._observe(st, msg)
            pool = agent._candidates(st, msg)
            if tgt in pool and st.evidence:
                wmap = {p: agent._weight(p, df, tier)
                        for p, (df, tier) in st.evidence.items()}

                def cov(a):
                    return round(sum(w for ph, w in wmap.items()
                                     if agent.ix.covers(a, ph)), 10)
                tv = cov(tgt)
                tied = [a for a in pool[:300] if cov(a) == tv]
                # The target can sit beyond the 300-candidate window we scan, in which
                # case it is absent from its own tie group and every feature lookup for it
                # is undefined. Require its presence rather than patch it in afterwards.
                if tgt in tied and 2 <= len(tied) <= 60:
                    groups.append((tgt, tied))
                    if len(groups) >= max_groups:
                        agent.sessions.pop(sid, None)
                        return groups
            probe = agent._next_probe(st)
            st.asked.append(probe)
            if not applied and turn + 1 == int(ov.get("turn", 3)):
                applied = True
                nv = str(ov.get("new_value", ""))
                if nv:
                    disclosed.add(nv)
                msg = str(ov.get("message", ""))
            else:
                msg, bu = customer_reply(eff, probe, disclosed, bu)
        agent.sessions.pop(sid, None)
    return groups


def auc_and_first(groups, feats, name, key=None):
    """Within-group AUC and target-first rate for one feature."""
    aucs, first = [], []
    for tgt, members in groups:
        vals = [(a, feats[a][name] if key is None else key(a)) for a in members]
        tv = dict(vals)[tgt]
        others = [v for a, v in vals if a != tgt]
        if not others:
            continue
        below = sum(1 for v in others if v < tv) + 0.5 * sum(1 for v in others if v == tv)
        aucs.append(below / len(others))
        first.append(1.0 if tv > max(others) else 0.0)
    return (statistics.fmean(aucs) if aucs else 0.5,
            statistics.fmean(first) if first else 0.0, len(aucs))


def null_band(groups, seed=0, reps=40):
    """AUC distribution when the label is assigned at random inside each group."""
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        aucs = []
        for _tgt, members in groups:
            n = len(members)
            if n < 2:
                continue
            pos = rng.randrange(n)
            vals = [rng.random() for _ in range(n)]
            tv = vals[pos]
            others = vals[:pos] + vals[pos + 1:]
            aucs.append(sum(1 for v in others if v < tv) / len(others))
        means.append(statistics.fmean(aucs))
    return statistics.fmean(means), statistics.pstdev(means)


def main() -> None:
    t0 = time.time()
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    pub_t = {str(s["ground_truth"]["parent_asin"]) for s in samples}
    profiles = [s["user_profile"] for s in samples]

    feats = build_features(prods, base.ix)
    names = sorted(next(iter(feats.values())).keys())
    print(f"{len(names)} features over {len(feats):,} products  [{time.time()-t0:.0f}s]")

    agent = object.__new__(Agent)
    agent.ix, agent.sessions = base.ix, {}
    agent.llm = agent.llm_extract = agent.tagger = None

    sessions = samples + mint(prods, pub_t, profiles, "reviews", N_SESSIONS,
                              seed=SEEDS["reviews"])
    cache = ROOT / "notes" / "eda" / ".tie_groups_v2.pkl"
    groups = None
    if cache.exists():
        try:
            cand = pickle.loads(cache.read_bytes())
            # Validate the SHAPE, not just that a file exists. A cache holding the wrong
            # object silently poisons every number downstream, which is worse than no cache.
            ok = (isinstance(cand, list) and cand
                  and all(isinstance(g, tuple) and len(g) == 2
                          and isinstance(g[0], str) and isinstance(g[1], list)
                          for g in cand[:50]))
            if ok:
                groups = cand
                print(f"loaded {len(groups):,} cached tie groups")
            else:
                print("cache present but wrong shape -- ignoring and recollecting")
        except Exception:
            print("cache unreadable -- recollecting")
    if groups is None:
        print(f"collecting tie groups from {len(sessions):,} sessions ...")
        groups = collect_ties(agent, sessions, prods, cats)
        cache.write_bytes(pickle.dumps(groups))
    sizes = [len(m) for _t, m in groups]
    print(f"  {len(groups):,} tie groups, median size {statistics.median(sizes):.0f}, "
          f"mean {statistics.fmean(sizes):.1f}  [{time.time()-t0:.0f}s]")

    nmean, nsd = null_band(groups)
    lo, hi = nmean - 2.5 * nsd, nmean + 2.5 * nsd
    print(f"\n  NULL BAND (labels permuted): AUC {nmean:.4f} +/- {nsd:.4f}  "
          f"-> anything in [{lo:.4f}, {hi:.4f}] is NOISE\n")

    print(f"{'feature':<24}{'AUC':>9}{'|dev|':>8}{'first%':>9}{'verdict':>12}")
    print("-" * 62)
    rows = []
    for nm in names:
        auc, first, n = auc_and_first(groups, feats, nm)
        dev = abs(auc - nmean)
        verdict = "SIGNAL" if (auc < lo or auc > hi) else "noise"
        rows.append((nm, auc, dev, first, verdict))
    for nm, auc, dev, first, verdict in sorted(rows, key=lambda r: -r[2]):
        print(f"{nm:<24}{auc:>9.4f}{dev:>8.4f}{first:>9.1%}{verdict:>12}")

    # ---------------------------------------------------- conditional on popularity
    print(f"\n  CONDITIONAL TEST -- signal remaining INSIDE popularity strata.")
    print(f"  Targets are drawn ~ review count, so marginal signal may just be popularity")
    print(f"  wearing a different hat. Only conditional signal is genuinely new.\n")
    strata = defaultdict(list)
    for tgt, members in groups:
        b = min(int(feats[tgt]["pop_log_reviews"] // 2), 4)
        strata[b].append((tgt, members))
    print(f"{'feature':<24}{'cond AUC':>10}{'|dev|':>8}{'verdict':>12}")
    print("-" * 56)
    crows = []
    for nm in names:
        vals, wts = [], 0
        for b, gs in strata.items():
            if len(gs) < 100:
                continue
            a, _f, n = auc_and_first(gs, feats, nm)
            vals.append(a * n)
            wts += n
        cauc = (sum(vals) / wts) if wts else 0.5
        dev = abs(cauc - nmean)
        crows.append((nm, cauc, dev,
                      "SIGNAL" if (cauc < lo or cauc > hi) else "noise"))
    for nm, cauc, dev, verdict in sorted(crows, key=lambda r: -r[2]):
        print(f"{nm:<24}{cauc:>10.4f}{dev:>8.4f}{verdict:>12}")

    OUT = {"null": {"mean": nmean, "sd": nsd, "lo": lo, "hi": hi},
           "n_groups": len(groups),
           "marginal": [{"feature": r[0], "auc": r[1], "first": r[3], "verdict": r[4]}
                        for r in rows],
           "conditional": [{"feature": r[0], "auc": r[1], "verdict": r[3]} for r in crows]}
    (ROOT / "notes" / "eda" / "out_52.json").write_text(
        json.dumps(OUT, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] notes/eda/out_52.json   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
