"""Experiment 21: learning-to-rank on SYNTHETIC sessions minted from the known generator.

WHY THIS REOPENS A CLOSED DIRECTION
-----------------------------------
Pass 18 rejected learning-to-rank: +0.013 on the half it trained on, -0.065 held out,
overfit gap +0.065. The stated cause was "147 positives across 100 independent queries" --
and that was treated as a fundamental limit. It is not. Pass 16 established that the
simulator's `intent_card` can be reconstructed EXACTLY (200/200 on public targets), so a
labelled session can be minted for ANY catalogue product. Measured here: 99.3% of the
catalogue is usable, i.e. ~49,650 independent training queries -- a 496x increase.

This is the weak-supervision recipe (Dehghani et al., SIGIR 2017; Gecko-style
generate-then-relabel), except our supervision is not weak: we CHOOSE the target when
minting the session, so labels are exact.

THE DETAIL THAT WOULD SILENTLY RUIN IT
--------------------------------------
Real targets come from a 5-core leave-last-out review split, so they are drawn roughly in
proportion to review count: median rating_number 6,846 against a catalogue median of 12,
and the 97.8th popularity percentile within their own category. Minting synthetic targets
UNIFORMLY would train the model on a population whose popularity prior is inverted
relative to reality. Synthetic targets are therefore sampled to MATCH the measured real
target distribution (stratified by log-popularity decile).

Everything else follows the established protocol: train on synthetic data only, tune on
the tune half, adjudicate on the held-out half. Public targets are excluded from minting
so no evaluation session leaks into training.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/log/21_synthetic_ltr.py --mint 12000
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    behavior_for, catalog_index, classify_constraint, coarse_category, customer_reply,
    evaluate, initial_message, intent_card, load_jsonl,
)
from submission.agent import Agent, CAT, CONSTRAINT, MINED  # noqa: E402

CATALOG = ROOT / "data" / "catalog.jsonl"
PUBLIC = ROOT / "data" / "public_set.jsonl"
OUT: dict = {}

FEATS = ["cov_con", "cov_cat", "cov_mined", "sum_w", "frac_cov", "hand", "pool_rank",
         "log_pop", "cat_pop_pct", "rating", "log_len", "turn", "n_ev", "tot_w",
         "in_title", "log_min_df", "n_price", "cov_ratio"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mint", type=int, default=12000, help="synthetic sessions to mint")
    ap.add_argument("--pool", type=int, default=120, help="candidates kept per turn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="experiments/studies/.ltr_feats.npz",
                    help="reuse collected features if present")
    ap.add_argument("--negs", type=int, default=15,
                    help="negatives sampled per turn during TRAINING collection")
    args = ap.parse_args()

    print("loading ...")
    samples = load_jsonl(PUBLIC)
    cid, cats, prods = catalog_index(CATALOG)
    TUNE = [s for i, s in enumerate(samples) if i % 2 == 0]
    HOLD = [s for i, s in enumerate(samples) if i % 2 == 1]
    base = Agent(CATALOG)
    public_targets = {str(s["ground_truth"]["parent_asin"]) for s in samples}

    # ---------------------------------------------------------------- priors
    def rn(a: str) -> float:
        try:
            return float(prods[a].get("rating_number") or 0)
        except (TypeError, ValueError):
            return 0.0

    RN = {a: rn(a) for a in prods}
    LOGP = {a: math.log1p(v) for a, v in RN.items()}
    RATE = {}
    for a, d in prods.items():
        try:
            RATE[a] = float(d.get("average_rating") or 0)
        except (TypeError, ValueError):
            RATE[a] = 0.0
    HASPRICE = {a: 1.0 if prods[a].get("price") not in (None, "") else 0.0 for a in prods}
    bycat: dict[str, list[float]] = {}
    CATOF: dict[str, str] = {}
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

    # --------------------------------- synthetic sampling matched to the generator
    # The real split samples a REVIEW and takes its item, so P(item is target) is
    # proportional to review count, WITH replacement. Verified by distribution match
    # (log-popularity p10/median/p90):
    #     real targets                    5.20 / 8.80 / 10.61
    #     uniform sampling                0.69 / 2.64 /  5.56   <- badly wrong
    #     proportional, with replacement  5.28 / 8.84 / 11.30   <- matches
    # Stratified-without-replacement was tried first and could not reach the top decile
    # (only ~4,959 of 14,000 requested, median 6.06): the catalogue does not hold enough
    # distinct ultra-popular products, because the real process REPEATS them. Getting
    # this wrong miscalibrates exactly the feature that decides ties.
    real_pop = sorted(LOGP[str(s["ground_truth"]["parent_asin"])] for s in samples)
    candidates = [a for a in prods if a not in public_targets]
    weights = [RN[a] for a in candidates]
    rng = random.Random(args.seed)
    minted = rng.choices(candidates, weights=weights, k=args.mint)
    mp = sorted(LOGP[a] for a in minted)

    def _q(v, p):
        return v[int(p * (len(v) - 1))]

    print(f"minted {len(minted):,} synthetic targets  "
          f"({len(set(minted)):,} distinct), sampled proportional to review count")
    print(f"  real      log-pop p10/median/p90: "
          f"{_q(real_pop,.1):.2f} / {_q(real_pop,.5):.2f} / {_q(real_pop,.9):.2f}")
    print(f"  synthetic log-pop p10/median/p90: "
          f"{_q(mp,.1):.2f} / {_q(mp,.5):.2f} / {_q(mp,.9):.2f}")

    SCEN = ["buying"] * 40 + ["browsing"] * 40 + ["intent_override"] * 15 + ["boundary"] * 5

    def synth_session(asin: str, i: int) -> dict:
        return {"sample_id": f"synth_{i}", "scenario_type": SCEN[i % len(SCEN)],
                "user_profile": {"preference_tags": []},
                "ground_truth": {"parent_asin": asin}}

    # ---------------------------------------------------------------- features
    def featurize(agent: Agent, st, pool: list[str],
                  ranks: list[int] | None = None) -> np.ndarray:
        """`ranks` are TRUE positions in the retrieval pool.

        BUG THIS FIXES: the row index within `pool` was previously used as the
        `pool_rank` feature. With negative sampling the target was appended LAST, so it
        carried pool_rank == n_negs in EVERY training row while negatives carried
        0..n_negs-1. The model learned "last position => positive" -- a reverse label
        leak. At inference the list is the hand ranker's top-10 (ranks 0..9), so that
        signal never occurs and the model ranks near-randomly. Symptom: 160x more data
        made SELECT collapse from HR 96% to HR 38%.
        """
        wmap = {p: agent._weight(p, df, t) for p, (df, t) in st.evidence.items()}
        tiers = {p: t for p, (_, t) in st.evidence.items()}
        dfs = {p: d for p, (d, _) in st.evidence.items()}
        tot = sum(wmap.values()) or 1.0
        n_ev = len(st.evidence) or 1
        if ranks is None:
            ranks = list(range(len(pool)))
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
            # cov_ratio: how much of THIS product's text the evidence explains
            rows[i] = (cc, ct, cm, sw, sw / tot, hand, ranks[i], LOGP[a], CPCT[a], RATE[a],
                       math.log1p(DLEN[a]), st.turn, n_ev, tot, tit,
                       math.log1p(mind if mind < 1e9 else 5000), HASPRICE[a],
                       sw / max(1.0, math.log1p(DLEN[a])))
        return rows

    def replay(subset, collect: bool):
        X, y, grp = [], [], []
        srng = random.Random(args.seed + 7)
        t_start = time.time()
        ag = object.__new__(Agent)
        ag.ix, ag.sessions = base.ix, {}
        for n, s in enumerate(subset):
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
                except Exception:
                    pool = []
                if pool and collect:
                    # NEGATIVE SAMPLING. Keeping all `pool` candidates per turn produced
                    # ~2.3M rows across 14k sessions and the collector was OOM-killed
                    # before writing a line. It is also poor practice: 1:80 imbalance
                    # trains worse than 1:15. Keep the positive plus a random sample of
                    # negatives, preserving pool_rank as a feature so ordering survives.
                    # HARD NEGATIVES. Random negatives teach the model to separate the
                    # target from easy pool members, but at inference it only ever sees
                    # the hand ranker's top-10 -- all hard. Train on that distribution:
                    # take the head of the pool, plus a few sampled deeper, and carry
                    # each candidate's TRUE pool rank through as a feature.
                    head = pool[:args.pool]
                    hard = [(r, a) for r, a in enumerate(head[:args.negs]) if a != tgt]
                    deep = [(r, a) for r, a in enumerate(head) if a != tgt][args.negs:]
                    srng.shuffle(deep)
                    chosen = hard + deep[:max(0, args.negs // 3)]
                    if tgt in pool:
                        chosen.append((pool.index(tgt), tgt))
                    if len(chosen) < 2:
                        chosen = [(r, a) for r, a in enumerate(head[:2])]
                    chosen.sort(key=lambda ra: ra[0])
                    ranks = [r for r, _ in chosen]
                    keep = [a for _, a in chosen]
                    X.append(featurize(ag, st, keep, ranks))
                    y.append(np.array([1 if a == tgt else 0 for a in keep], dtype=np.int8))
                    grp.append(np.full(len(keep), n, dtype=np.int32))
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
            ag.sessions.pop(sid, None)
            if collect and n and n % 2000 == 0:
                rows_so_far = sum(a.shape[0] for a in X)
                print(f"    {n:,}/{len(subset):,} sessions  {rows_so_far:,} rows  "
                      f"[{time.time()-t_start:.0f}s]", flush=True)
        if not X:
            return None, None, None
        return np.vstack(X), np.concatenate(y), np.concatenate(grp)

    # Cache the collected features. Collection is the expensive step (~8 min); every
    # later iteration on the gate or the model only needs the matrix.
    cache_p = ROOT / "experiments" / "studies" / ".ltr_feats.npz"
    if cache_p.exists():
        z = np.load(cache_p)
        Xs, ys, gs = z["X"], z["y"], z["g"]
        print(f"\nloaded cached features from {cache_p.name}")
    else:
        print(f"\nreplaying {len(minted):,} synthetic sessions to collect features ...")
        t0 = time.time()
        synth = [synth_session(a, i) for i, a in enumerate(minted)]
        Xs, ys, gs = replay(synth, collect=True)
        np.savez_compressed(cache_p, X=Xs, y=ys, g=gs)
        print(f"  cached -> {cache_p.name}")
    print(f"  {Xs.shape[0]:,} rows  {int(ys.sum()):,} positives  "
          f"{len(set(gs.tolist())):,} independent queries")
    OUT["train"] = {"rows": int(Xs.shape[0]), "positives": int(ys.sum()),
                    "queries": len(set(gs.tolist()))}

    from sklearn.ensemble import HistGradientBoostingClassifier

    MODELS = {
        "d4/300": dict(max_depth=4, max_iter=300, learning_rate=0.08),
        "d6/600": dict(max_depth=6, max_iter=600, learning_rate=0.06),
        "d8/900": dict(max_depth=8, max_iter=900, learning_rate=0.05),
    }
    fitted = {}
    for name, kw in MODELS.items():
        t0 = time.time()
        m = HistGradientBoostingClassifier(random_state=0, class_weight="balanced", **kw)
        m.fit(Xs, ys)
        fitted[name] = m
        print(f"  trained {name} in {time.time()-t0:.0f}s")

    class LTR(Agent):
        """Two placement modes, and the difference between them is the whole result.

        SELECT mode lets the model choose WHICH items make the top-10. That moved MRR
        0.7507 -> 0.8139 (the model really does rank better) but cost 3 hits and 0.55
        turns, because it also evicted targets the hand scorer had correctly included --
        a net loss.

        REORDER mode keeps the hand scorer's top-k SET and only permutes it. HitRate and
        MTTC then depend solely on set membership, so both are identical to baseline BY
        CONSTRUCTION, and MRR is the only metric that can move. The learned ranker is
        applied exactly where it is better and nowhere else.
        """
        MODEL = None
        BLEND = 0.0
        MODE = "reorder"          # "reorder" | "select"
        GATE = False              # act only where coverage ties

        def _score_rows(self, st, keep, ranks=None):
            F = featurize(self, st, keep, ranks)
            p = self.MODEL.predict_proba(F)[:, 1]
            if self.BLEND:
                h = F[:, FEATS.index("hand")]
                p = p + self.BLEND * (h - h.min()) / (np.ptp(h) + 1e-9)
            return p

        def _rank(self, st, pool, top_k):
            hand = super()._rank(st, pool, top_k)
            if self.MODEL is None or not st.evidence:
                return hand
            try:
                if self.MODE == "reorder":
                    if len(hand) < 2:
                        return hand
                    # CONFIDENCE GATE: only reorder where the evidence does NOT already
                    # separate the candidates. Where coverage is decisive the hand
                    # ranker is near-perfect and any perturbation can only cost; where
                    # it ties, ranking is measurably worst (turn-1 hits: 57.8% rank-1
                    # against 90.5% at turn 4). Spend the model only there.
                    if self.GATE:
                        # GROUP-WISE gate. An earlier version skipped only when ALL ten
                        # coverage values were distinct -- with discrete coverage sums
                        # that essentially never happens, so the gate never fired and
                        # gated runs were byte-identical to ungated ones. Correct form:
                        # leave separated candidates in place and permute only WITHIN
                        # each tied block, exactly as the LLM tie-break does.
                        wm = {q: self._weight(q, d, t)
                              for q, (d, t) in st.evidence.items()}
                        cv = [sum(w for q, w in wm.items() if self.ix.covers(a, q))
                              for a in hand]
                        out = list(hand)
                        i = 0
                        while i < len(out):
                            j = i + 1
                            while j < len(out) and abs(cv[j] - cv[i]) < 1e-12:
                                j += 1
                            if j - i >= 2:
                                blk = out[i:j]
                                q = self._score_rows(st, blk, list(range(i, j)))
                                out[i:j] = [blk[k] for k in np.argsort(-q)]
                            i = j
                        return out
                    p = self._score_rows(st, hand, list(range(len(hand))))
                    return [hand[i] for i in np.argsort(-p)]
                keep = pool[:args.pool]
                p = self._score_rows(st, keep, list(range(len(keep))))
                out = [keep[i] for i in np.argsort(-p)[:top_k]]
                seen = set(out)
                return (out + [a for a in pool if a not in seen])[:top_k]
            except Exception:
                return hand

    def share(**kw):
        o = object.__new__(LTR)
        # __init__ is bypassed to reuse the prebuilt index, so every attribute it would
        # have set must be set here. Missing `llm` raises inside _rerank_exact_ties, the
        # safety envelope swallows it, and EVERY turn returns an empty list -- a silent
        # 0.00000 that looks like a modelling result rather than a wiring bug.
        o.ix, o.sessions, o.llm = base.ix, {}, None
        for k, v in kw.items():
            setattr(o, k, v)
        return o

    def run(ag, subset, tag):
        r = evaluate(ag, subset, cid, cats, prods)
        print(f"    {tag:<40} HR@10 {r['hit_rate_at_10']:>6.1%}  MRR {r['mrr']:>6.4f}  "
              f"MTTC {r['mttc']:>5.2f}  SCORE {r['recommended_technical_score']:>7.5f}")
        return {"hr": r["hit_rate_at_10"], "mrr": r["mrr"], "mttc": r["mttc"],
                "score": r["recommended_technical_score"]}

    print("\n" + "=" * 96)
    print("A. TUNE HALF -- no public session was used in training")
    print("=" * 96)
    res = {"baseline": run(share(MODEL=None), TUNE, "baseline (hand-tuned)")}
    for name, m in fitted.items():
        res[f"select {name}"] = run(share(MODEL=m, MODE="select"), TUNE,
                                    f"SELECT top-10  {name}")
    for name, m in fitted.items():
        res[f"reorder {name}"] = run(share(MODEL=m, MODE="reorder"), TUNE,
                                     f"REORDER within top-10  {name}")
    for b in (0.3, 1.0):
        res[f"reorder-blend {b}"] = run(
            share(MODEL=fitted["d6/600"], BLEND=b, MODE="reorder"), TUNE,
            f"REORDER d6/600 + {b}x hand")
    for name, m in fitted.items():
        res[f"gated {name}"] = run(share(MODEL=m, MODE="reorder", GATE=True), TUNE,
                                   f"GATED reorder (ties only)  {name}")
    for b in (0.3, 1.0):
        res[f"gated-blend {b}"] = run(
            share(MODEL=fitted["d6/600"], BLEND=b, MODE="reorder", GATE=True), TUNE,
            f"GATED reorder + {b}x hand")
    OUT["tune"] = res

    best = max((k for k in res if k != "baseline"), key=lambda k: res[k]["score"])
    print(f"\n  best on tune: {best} ({res[best]['score']:.5f}) vs "
          f"baseline {res['baseline']['score']:.5f} "
          f"({res[best]['score']-res['baseline']['score']:+.5f})")

    print("\n" + "=" * 96)
    print("B. HELD-OUT ADJUDICATION")
    print("=" * 96)
    hb = run(share(MODEL=None), HOLD, "baseline")
    parts = best.split()
    if parts[0] in ("reorder-blend", "gated-blend"):
        cfg = {"MODEL": fitted["d6/600"], "BLEND": float(parts[1]), "MODE": "reorder",
               "GATE": parts[0] == "gated-blend"}
    elif parts[0] == "gated":
        cfg = {"MODEL": fitted[parts[1]], "MODE": "reorder", "GATE": True}
    else:
        cfg = {"MODEL": fitted[parts[1]], "MODE": parts[0]}
    hn = run(share(**cfg), HOLD, best)
    d = hn["score"] - hb["score"]
    print(f"\n  HELD-OUT DELTA {d:+.5f}  -> "
          f"{'ADOPT' if d > 0.005 else 'inside noise' if d > -0.005 else 'REJECT'}")
    OUT["holdout"] = {"baseline": hb, "best": hn, "delta": d, "cfg": best}

    if d > -0.005:
        full = evaluate(share(**cfg), samples, cid, cats, prods)
        print(f"\n  ALL 200: SCORE {full['recommended_technical_score']:.5f}  "
              f"HR@10 {full['hit_rate_at_10']:.1%}  MRR {full['mrr']:.4f}")
        OUT["full"] = {"score": full["recommended_technical_score"],
                       "hr": full["hit_rate_at_10"], "mrr": full["mrr"]}

    (ROOT / "experiments" / "results" / "out_21.json").write_text(
        json.dumps(OUT, indent=2, default=str), encoding="utf-8")
    print("\n[saved] experiments/results/out_21.json")


if __name__ == "__main__":
    main()
