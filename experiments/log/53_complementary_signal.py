"""Experiment 53: on the ties popularity gets WRONG, does any other feature get them right?

WHY PASS 52's CONDITIONAL TEST DID NOT WORK. It stratified on `log1p(rating_number) // 2`,
which gives ~4 strata each spanning a factor of e^2 ~ 7.4 in review count. Popularity still
varies enormously inside a stratum, so the "conditional" AUC of popularity itself came back
0.8722 against a marginal 0.8697 -- essentially unchanged, which is proof the control was
not binding rather than proof of anything about the features. Recorded as an error.

THE SHARPER TEST, and the one that is actually decision-relevant. Popularity puts the
target first in ~59.6% of tie groups. The other ~40.4% are precisely the groups the shipped
agent gets wrong today. So:

    Split the groups by whether popularity already wins.
    On the groups where it LOSES, measure every other feature.

A feature that carries the same information as popularity will score at chance there,
because those groups are defined by popularity being wrong. A feature that scores above
chance on that subset carries genuinely COMPLEMENTARY information -- and that is the only
kind that can improve the agent, since the rest is already exploited by W_POP.

Chance is not 0.5 for the first-rate, it is ~1/group-size, so the null band is computed by
permutation on the same subset rather than assumed.

Also tested: simple two-signal fusion (popularity + each candidate, by rank) measured on ALL
groups, because a feature can be complementary and still not survive being combined.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/log/53_complementary_signal.py
"""
from __future__ import annotations

import json
import pickle
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "log"))

from evaluator.local_evaluator import catalog_index, load_jsonl  # noqa: E402
from submission.agent import Agent  # noqa: E402

_p52 = __import__("52_tie_feature_dependency")

POP = "pop_log_reviews"


def first_rate(groups, feats, name):
    hits = []
    for tgt, members in groups:
        vals = {a: feats[a][name] for a in members}
        best = max(vals.values())
        # ties in the feature itself count fractionally, so a constant feature scores
        # 1/n rather than 1.0
        winners = [a for a, v in vals.items() if v == best]
        hits.append((1.0 / len(winners)) if tgt in winners else 0.0)
    return statistics.fmean(hits) if hits else 0.0


def auc(groups, feats, name):
    out = []
    for tgt, members in groups:
        vals = [(a, feats[a][name]) for a in members]
        tv = dict(vals)[tgt]
        others = [v for a, v in vals if a != tgt]
        if not others:
            continue
        out.append((sum(1 for v in others if v < tv)
                    + 0.5 * sum(1 for v in others if v == tv)) / len(others))
    return statistics.fmean(out) if out else 0.5


def null_first(groups, seed=0, reps=60):
    """Permutation null for the first-rate on THIS subset (depends on group sizes)."""
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        hits = []
        for _tgt, members in groups:
            n = len(members)
            hits.append(1.0 if rng.randrange(n) == 0 else 0.0)
        means.append(statistics.fmean(hits))
    return statistics.fmean(means), statistics.pstdev(means)


def fuse_first(groups, feats, a_name, b_name, w=0.5, k=10.0):
    """Reciprocal-rank fusion of two features; first-rate over all groups."""
    hits = []
    for tgt, members in groups:
        ra = {a: i for i, a in enumerate(
            sorted(members, key=lambda x: -feats[x][a_name]))}
        rb = {a: i for i, a in enumerate(
            sorted(members, key=lambda x: -feats[x][b_name]))}
        score = {a: 1.0 / (k + ra[a]) + w / (k + rb[a]) for a in members}
        best = max(score.values())
        winners = [a for a, v in score.items() if v == best]
        hits.append((1.0 / len(winners)) if tgt in winners else 0.0)
    return statistics.fmean(hits)


def main() -> None:
    t0 = time.time()
    _samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    _cid, _cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    feats = _p52.build_features(prods, base.ix)

    cache = ROOT / "experiments" / "studies" / ".tie_groups_v2.pkl"
    groups = pickle.loads(cache.read_bytes())
    names = sorted(next(iter(feats.values())).keys())
    print(f"{len(groups):,} tie groups  [{time.time()-t0:.0f}s]")

    # ---------------------------------------------------------------- split by popularity
    won, lost = [], []
    for tgt, members in groups:
        vals = {a: feats[a][POP] for a in members}
        best = max(vals.values())
        (won if vals[tgt] == best else lost).append((tgt, members))
    print(f"  popularity puts the target first in {len(won):,}/{len(groups):,} "
          f"= {len(won)/len(groups):.1%}")
    print(f"  it FAILS on {len(lost):,} groups -- median size "
          f"{statistics.median([len(m) for _t, m in lost]):.0f}. Those are the ones the")
    print(f"  agent gets wrong today, and the only place a new feature can help.\n")

    nm, nsd = null_first(lost)
    lo, hi = nm - 2.5 * nsd, nm + 2.5 * nsd
    print(f"  NULL first-rate on that subset: {nm:.4f} +/- {nsd:.4f} "
          f"-> noise band [{lo:.4f}, {hi:.4f}]\n")

    print(f"{'feature':<24}{'first% (pop-fails)':>20}{'AUC':>9}{'verdict':>14}")
    print("-" * 68)
    rows = []
    for n in names:
        if n == POP:
            continue
        f = first_rate(lost, feats, n)
        a = auc(lost, feats, n)
        verdict = "COMPLEMENTARY" if f > hi else ("inverted" if f < lo else "noise")
        rows.append((n, f, a, verdict))
    for n, f, a, verdict in sorted(rows, key=lambda r: -r[1]):
        print(f"{n:<24}{f:>19.1%}{a:>9.4f}{verdict:>14}")

    # ---------------------------------------------------------------- fusion, all groups
    base_first = first_rate(groups, feats, POP)
    print(f"\n  FUSION on ALL groups -- popularity alone: {base_first:.1%}")
    print(f"{'  + feature':<26}{'w=0.3':>9}{'w=0.6':>9}{'w=1.0':>9}{'best delta':>12}")
    print("-" * 66)
    fus = []
    for n in names:
        if n == POP:
            continue
        vals = [fuse_first(groups, feats, POP, n, w=w) for w in (0.3, 0.6, 1.0)]
        best = max(vals)
        fus.append((n, vals, best - base_first))
    for n, vals, d in sorted(fus, key=lambda r: -r[2])[:8]:
        print(f"{'  + ' + n:<26}" + "".join(f"{v:>9.1%}" for v in vals) + f"{d:>+12.1%}")

    OUT = {"n_groups": len(groups), "pop_first": base_first,
           "pop_wins": len(won), "pop_fails": len(lost),
           "null_first_on_fails": {"mean": nm, "sd": nsd},
           "complementary": [{"feature": r[0], "first": r[1], "auc": r[2],
                              "verdict": r[3]} for r in rows],
           "fusion": [{"feature": r[0], "first_by_w": r[1], "delta": r[2]} for r in fus]}
    (ROOT / "experiments" / "results" / "out_53.json").write_text(
        json.dumps(OUT, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] experiments/results/out_53.json   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
