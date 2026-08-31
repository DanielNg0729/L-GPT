"""Experiment 24: tie-specialised feature model with WITHIN-GROUP normalisation.

Three defects in the earlier feature model, fixed here.

DEFECT 1 -- trained on the wrong decision.
Pass 21 collected rows from every turn, so coverage ties were a minority of training
data. The model optimised average-case ranking (where coverage dominates and the prior
barely matters) and was then evaluated only inside ties. This pass trains ONLY on tie
groups.

DEFECT 2 -- every feature was ABSOLUTE; none was relative to the group.
This is the likely explanation for the pass-22 paradox: the model HAS popularity as a
feature and still loses to popularity alone (32.4% vs 57.4%). Of course it does -- it
sees `log_pop = 8.8`, never "is this the most popular candidate in THIS tie". Groups vary
enormously in scale, so expressing "take the max here" from raw magnitudes is hard.
Per-query normalisation is standard in learning-to-rank for exactly this reason, and it
was missing. Added: within-group rank, z-score and argmax indicators for popularity,
coverage, length, price and rating.

DEFECT 3 -- thin feature set.
Added: per-FIELD coverage (title / features / details / description), longest matched
phrase, distinct fields matched, UNMATCHED evidence count, first-match position, df
spread (max/mean, not just min), price magnitude and percentile, store frequency,
bullet count, and the BM25-vs-coverage disagreement.

18 features -> 34. Listwise objective on the same tie groups the cross-encoder uses.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/scripts/24_tie_features.py
"""
from __future__ import annotations

import bisect
import collections
import json
import math
import pickle
import random
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, coarse_category, load_jsonl  # noqa: E402
from submission.agent import Agent, CAT, CONSTRAINT, MINED, raw_toks  # noqa: E402

GROUPS_CACHE = ROOT / "experiments" / "studies" / ".tie_groups.pkl"

ABS = ["cov_con", "cov_cat", "cov_mined", "sum_w", "n_unmatched", "max_phr_len",
       "n_fields", "cov_title", "cov_feat", "cov_det", "cov_desc", "first_pos",
       "log_pop", "rating", "log_len", "log_price", "has_price", "cat_pop_pct",
       "n_bullets", "store_freq", "title_len", "df_min", "df_max", "df_mean"]
REL = ["pop_rank", "pop_z", "is_max_pop", "cov_rank", "len_rank", "price_rank",
       "rating_rank", "hand_rank"]
CTX = ["group_size", "turn_proxy"]
FEATS = ABS + REL + CTX


def main() -> None:
    if not GROUPS_CACHE.exists():
        print(f"missing {GROUPS_CACHE.name} -- run 23_tie_crossencoder.py first "
              "(it writes the tie-group cache)")
        raise SystemExit(1)

    print("loading ...")
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    TUNE = [s for i, s in enumerate(samples) if i % 2 == 0]
    HOLD = [s for i, s in enumerate(samples) if i % 2 == 1]
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    agent = Agent(ROOT / "data" / "catalog.jsonl")

    # ---------------- per-product statics
    def num(a, k):
        try:
            return float(prods[a].get(k) or 0)
        except (TypeError, ValueError):
            return 0.0

    RN = {a: num(a, "rating_number") for a in prods}
    LOGP = {a: math.log1p(v) for a, v in RN.items()}
    RATE = {a: num(a, "average_rating") for a in prods}
    PRICE = {a: num(a, "price") for a in prods}
    NBULL = {a: len(prods[a].get("features") or []) for a in prods}
    TITLEN = {a: len(raw_toks(str(prods[a].get("title") or ""))) for a in prods}
    DLEN = {a: max(1, len(agent.ix.blob.get(a, ""))) for a in prods}
    store_ct = collections.Counter(str(prods[a].get("store") or "") for a in prods)
    STOREF = {a: store_ct[str(prods[a].get("store") or "")] for a in prods}
    bycat, CATOF = {}, {}
    for a, d in prods.items():
        c = coarse_category([str(x) for x in (d.get("categories") or [])])
        CATOF[a] = c
        bycat.setdefault(c, []).append(RN[a])
    for c in bycat:
        bycat[c].sort()
    CPCT = {a: (bisect.bisect_left(bycat[CATOF[a]], RN[a]) / len(bycat[CATOF[a]]))
            if len(bycat[CATOF[a]]) > 1 else 0.5 for a in prods}

    # field-level normalised text, for per-field coverage
    def field_blob(a, key):
        v = prods[a].get(key)
        if isinstance(v, dict):
            v = " ".join(f"{k} {x}" for k, x in v.items())
        elif isinstance(v, list):
            v = " ".join(str(x) for x in v)
        return " " + " ".join(raw_toks(str(v or ""))) + " "

    FIELD_CACHE: dict[tuple, str] = {}

    def fblob(a, key):
        k = (a, key)
        if k not in FIELD_CACHE:
            FIELD_CACHE[k] = field_blob(a, key)
        return FIELD_CACHE[k]

    def ranks_of(vals, higher_better=True):
        order = sorted(range(len(vals)), key=lambda i: vals[i],
                       reverse=higher_better)
        r = [0] * len(vals)
        for pos, i in enumerate(order):
            r[i] = pos
        return r

    def featurize_group(g):
        """One row per candidate; absolute + WITHIN-GROUP relative features."""
        asins = g["asins"]
        phrases = [p for p in g.get("phrases", [])] or g["query"].split("; ")
        phrases = [" ".join(raw_toks(p)) for p in phrases if p.strip()]
        dfs = [agent.ix.df(p) or 1 for p in phrases] or [1]
        n = len(asins)
        rows = np.zeros((n, len(FEATS)), dtype=np.float32)
        pops = [LOGP[a] for a in asins]
        covs, hands = [], []
        base = []
        for a in asins:
            blob = agent.ix.blob.get(a, "")
            matched = [p for p in phrases if f" {p} " in blob]
            sw = float(len(matched))
            covs.append(sw)
            hands.append(sw + 0.35 * (LOGP[a] / (agent.ix.max_pop or 1.0)))
            base.append((a, blob, matched))
        pr = ranks_of(pops)
        cr = ranks_of(covs)
        lr = ranks_of([DLEN[a] for a in asins], higher_better=False)
        pricer = ranks_of([PRICE[a] for a in asins])
        rar = ranks_of([RATE[a] for a in asins])
        hr = ranks_of(hands)
        mu = statistics.fmean(pops) if pops else 0.0
        sd = statistics.pstdev(pops) or 1.0
        mx = max(pops) if pops else 0.0
        for i, (a, blob, matched) in enumerate(base):
            first = min((blob.find(f" {p} ") for p in matched), default=-1)
            row = [
                len(matched), 0, 0, float(len(matched)),
                len(phrases) - len(matched),
                max((len(p.split()) for p in matched), default=0),
                sum(1 for k in ("title", "features", "details", "description")
                    if any(f" {p} " in fblob(a, k) for p in matched)),
                sum(1 for p in matched if f" {p} " in fblob(a, "title")),
                sum(1 for p in matched if f" {p} " in fblob(a, "features")),
                sum(1 for p in matched if f" {p} " in fblob(a, "details")),
                sum(1 for p in matched if f" {p} " in fblob(a, "description")),
                (first / max(1, len(blob))) if first >= 0 else 1.0,
                LOGP[a], RATE[a], math.log1p(DLEN[a]), math.log1p(PRICE[a]),
                1.0 if PRICE[a] > 0 else 0.0, CPCT[a], NBULL[a],
                math.log1p(STOREF[a]), TITLEN[a],
                math.log1p(min(dfs)), math.log1p(max(dfs)),
                math.log1p(statistics.fmean(dfs)),
                pr[i], (pops[i] - mu) / sd, 1.0 if pops[i] >= mx else 0.0,
                cr[i], lr[i], pricer[i], rar[i], hr[i],
                float(n), float(len(phrases)),
            ]
            rows[i] = row
        return rows

    groups = pickle.loads(GROUPS_CACHE.read_bytes())
    groups = [g for g in groups if g.get("label", -1) >= 0 and len(g["asins"]) >= 2]
    print(f"training tie groups: {len(groups):,}  "
          f"mean size {statistics.fmean(len(g['asins']) for g in groups):.2f}")

    X = np.vstack([featurize_group(g) for g in groups])
    y = np.concatenate([[1 if i == g["label"] else 0 for i in range(len(g["asins"]))]
                        for g in groups]).astype(np.int8)
    print(f"  rows {X.shape[0]:,}  features {X.shape[1]}  positives {int(y.sum()):,}")

    from sklearn.ensemble import HistGradientBoostingClassifier

    models = {
        "d4/400": dict(max_depth=4, max_iter=400, learning_rate=0.08),
        "d6/800": dict(max_depth=6, max_iter=800, learning_rate=0.06),
    }

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "p23", ROOT / "experiments" / "scripts" / "23_tie_crossencoder.py")
    p23 = importlib.util.module_from_spec(spec)
    # Proper module execution. exec()'ing the source text left __file__ undefined,
    # which pass 23 needs at import time.
    spec.loader.exec_module(p23)
    collect_groups = p23.collect_groups

    print("\ncollecting PUBLIC tie groups ...")
    ev_tune = [g for g in collect_groups(agent, TUNE, prods) if g["label"] >= 0]
    ev_hold = [g for g in collect_groups(agent, HOLD, prods) if g["label"] >= 0]
    print(f"  tune {len(ev_tune)}  hold {len(ev_hold)}")

    def report(name, model, gs, tag):
        det = [g["label"] for g in gs]
        ml = []
        for g in gs:
            p = model.predict_proba(featurize_group(g))[:, 1]
            ml.append(int(np.argsort(-p).tolist().index(g["label"])))
        f_d = sum(1 for p in det if p == 0) / len(det)
        f_m = sum(1 for p in ml if p == 0) / len(ml)
        print(f"    [{tag}] popularity {f_d:>6.1%} | {name} {f_m:>6.1%}  "
              f"({'BEATS' if f_m > f_d else 'fails'})")
        return {"det_first": f_d, "ml_first": f_m,
                "det_mrr": statistics.fmean(1 / (p + 1) for p in det),
                "ml_mrr": statistics.fmean(1 / (p + 1) for p in ml)}

    OUT = {"n_features": len(FEATS), "train_groups": len(groups)}
    print()
    for name, kw in models.items():
        m = HistGradientBoostingClassifier(random_state=0, class_weight="balanced", **kw)
        m.fit(X, y)
        OUT[name] = {"tune": report(name, m, ev_tune, "tune"),
                     "hold": report(name, m, ev_hold, "hold")}
        try:
            from sklearn.inspection import permutation_importance
            sub = np.random.default_rng(0).choice(len(X), size=min(4000, len(X)),
                                                  replace=False)
            imp = permutation_importance(m, X[sub], y[sub], n_repeats=3,
                                         random_state=0, scoring="average_precision")
            top = np.argsort(-imp.importances_mean)[:8]
            print("      top features: " +
                  ", ".join(f"{FEATS[i]}({imp.importances_mean[i]:.3f})" for i in top))
        except Exception as exc:
            print(f"      (importance skipped: {type(exc).__name__})")

    # ---------------- FEATURE-COUNT ABLATION -------------------------------
    # pop_z is a monotone transform of popularity WITHIN the group, so a model using
    # pop_z alone must reproduce the popularity ordering exactly -- 57.4%. The
    # 34-feature model ranks pop_z first by 2x and still scores 39.7%. If accuracy
    # RISES as features are removed, the extra features are not merely useless: with
    # 7,745 groups a tree ensemble finds spurious splits in them that override the one
    # feature that matters. That is a statement about capacity vs data, not about
    # whether the signal exists.
    print("\n  feature-count ablation (subsets ordered by importance):")
    ORDER = ["pop_z", "log_pop", "cat_pop_pct", "df_min", "group_size", "sum_w",
             "cov_rank", "first_pos", "log_len", "log_price"]
    idx_of = {f: i for i, f in enumerate(FEATS)}
    abl = {}
    for k in (1, 2, 3, 5, 8, len(FEATS)):
        cols = ([idx_of[f] for f in ORDER[:k]] if k <= len(ORDER)
                else list(range(len(FEATS))))
        mk = HistGradientBoostingClassifier(random_state=0, class_weight="balanced",
                                            max_depth=4, max_iter=400,
                                            learning_rate=0.08)
        mk.fit(X[:, cols], y)

        def firstrate(gs, cols=cols, mk=mk):
            hit = 0
            for g in gs:
                pr = mk.predict_proba(featurize_group(g)[:, cols])[:, 1]
                if int(np.argmax(pr)) == g["label"]:
                    hit += 1
            return hit / len(gs)

        t, h = firstrate(ev_tune), firstrate(ev_hold)
        abl[k] = {"tune": t, "hold": h}
        names = ",".join(ORDER[:k]) if k <= len(ORDER) else "ALL 34"
        print(f"    k={k:<2} tune {t:>6.1%}  hold {h:>6.1%}   [{names[:50]}]")
    OUT["ablation"] = abl

    print(f"\n  BAR: popularity 57.4% tune / 55.1% hold")
    print(f"  reference: LLM 41.2%, old 18-feature LTR 32.4%, random 16.2%")
    (ROOT / "experiments" / "results" / "out_24.json").write_text(
        json.dumps(OUT, indent=2) + "\n", encoding="utf-8")
    print("\n[saved] experiments/results/out_24.json")


if __name__ == "__main__":
    main()
