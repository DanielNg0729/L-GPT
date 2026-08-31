"""Experiment 47: M2 -- a LOCAL paraphrase-robust extractor, to replace the API call.

THE ONE ML IDEA LEFT WITH REAL UPSIDE, AND WHY IT IS NOT M1 AGAIN
----------------------------------------------------------------
M1 (pass 45/46) asked "is this PHRASE constraint-shaped?" from phrase-level features alone
and reached 0.637 held-out accuracy -- too weak to act on in either direction. As a filter
it collapsed the paraphrase floor (T1 0.852 -> 0.217); as a soft weight it degraded
monotonically as the model gained influence. Phrase shape is nearly signal-free here.

This asks a different question, with context: WHICH TOKENS OF THIS MESSAGE ARE SCAFFOLDING?

    "Appreciate it. I want to find Jewelry Necklaces. It absolutely has to be
     Material:alloy. Cheers."
       scaffolding: appreciate it i want to find it absolutely has to be cheers
       content:     jewelry necklaces  material alloy

That is exactly the operation the LLM performs, and it works: +0.0688 on T1, 68.8% of the
gap to clean. So the task is demonstrably learnable. The question this pass answers is
whether a SMALL LOCAL model can learn it -- which would move the extraction channel from
O2 (needs network the organizer may disable) to O1, at zero cost and zero latency.

THE SUPERVISION IS AGAIN EXACT AND UNLIMITED. Mint a session from any catalogue product;
the intent card tells us the constraint values verbatim; apply a paraphrase transform; label
every token that belongs to a constraint value as content and everything else as
scaffolding. No teacher, no annotation, no noise.

THE HONEST RISK, STATED BEFORE MEASURING. The paraphrase transforms are OURS. A model
trained on them may learn our specific filler vocabulary rather than the general shape of
scaffolding, and the organizer's paraphraser (if any) will differ. So the decisive number
here is not training accuracy -- it is HELD-OUT-TRANSFORM accuracy: train on one family of
paraphrases, test on families never seen. That is reported separately and is what the
adopt/reject decision rests on.

SHIPPING CONSTRAINT: standard library only at inference, so a linear model whose
coefficients transcribe into plain Python.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/log/47_local_paraphrase_extractor.py
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
sys.path.insert(0, str(ROOT / "experiments" / "log"))

from evaluator.local_evaluator import (  # noqa: E402
    behavior_for, catalog_index, coarse_category, customer_reply, evaluate,
    initial_message, intent_card, load_jsonl,
)
from submission.agent import MINED, Agent, raw_toks, recognised  # noqa: E402

_p30 = __import__("30_robustness_benchmark")
_p31 = __import__("31_paraphrase_stress")
mint, SEEDS = _p30.mint, _p30.SEEDS
evaluate_transformed, TRANSFORMS = _p31.evaluate_transformed, _p31.TRANSFORMS

TRAIN_TRANSFORMS = ["T1 scaffold reworded", "T3 conversational noise"]
HELDOUT_TRANSFORMS = ["T2 scaffold stripped", "T5 realistic (T1+T3)",
                      "T4 case/punctuation churn"]
N_SESSIONS = 900


def gen_examples(prods, ix, asins, transforms, seed=0):
    """(token, context) -> is it part of a constraint value? Labels are exact."""
    rng = random.Random(seed)
    rows: list[tuple[list[str], int, set[str]]] = []
    for asin in asins:
        card = intent_card(prods[asin])
        values = [str(v) for v in
                  list(card["hard_constraints"]) + list(card["soft_preferences"])]
        content = set()
        for v in values:
            content.update(raw_toks(v))
        cat = coarse_category([str(x) for x in (prods[asin].get("categories") or [])])
        content.update(raw_toks(cat))
        scenario = rng.choice(["buying", "browsing"])
        eff = {"sample_id": asin, "scenario_type": scenario, "intent_card": card,
               "behavior": behavior_for(scenario, card, random.Random(asin))}
        disclosed: set[str] = set()
        bu = False
        msgs = [initial_message(eff, cat, disclosed)]
        for attr in ("feature", "material", "color", "other"):
            m, bu = customer_reply(eff, attr, disclosed, bu)
            msgs.append(m)
        for m in msgs:
            fn = TRANSFORMS[rng.choice(transforms)]
            shown = fn(m)
            toks = raw_toks(shown)
            if 2 <= len(toks) <= 60:
                rows.append((toks, len(rows), content))
    return rows


def featurise_token(toks, i, ix):
    t = toks[i]
    prev = toks[i - 1] if i else ""
    nxt = toks[i + 1] if i + 1 < len(toks) else ""
    df = ix.df(t)
    dfp = ix.df(prev) if prev else 0
    dfn = ix.df(nxt) if nxt else 0
    n = max(len(toks) - 1, 1)
    return [
        math.log1p(df),                       # is the token catalogue vocabulary at all
        math.log1p(dfp), math.log1p(dfn),     # are its neighbours
        float(len(t)),
        1.0 if any(c.isdigit() for c in t) else 0.0,
        i / n,                                # position in the message
        1.0 if df == 0 else 0.0,              # unattested -> almost certainly filler
        float(len(toks)),
    ]


def main() -> None:
    t0 = time.time()
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    pub_t = {str(s["ground_truth"]["parent_asin"]) for s in samples}
    profiles = [s["user_profile"] for s in samples]

    pool = [a for a in prods if a not in pub_t]
    random.Random(3).shuffle(pool)
    train_asins, test_asins = pool[:N_SESSIONS], pool[N_SESSIONS:N_SESSIONS + 300]

    print(f"minting token-labelled data (train transforms: {TRAIN_TRANSFORMS}) ...")
    tr_rows = gen_examples(prods, base.ix, train_asins, TRAIN_TRANSFORMS, seed=1)
    print(f"  {len(tr_rows):,} messages  [{time.time()-t0:.0f}s]")

    def build(rows):
        X, y = [], []
        for toks, _idx, content in rows:
            for i in range(len(toks)):
                X.append(featurise_token(toks, i, base.ix))
                y.append(1 if toks[i] in content else 0)
        return np.array(X, dtype=np.float64), np.array(y, dtype=np.int8)

    Xtr, ytr = build(tr_rows)
    print(f"  {Xtr.shape[0]:,} tokens, {ytr.mean():.1%} are content  [{time.time()-t0:.0f}s]")

    from sklearn.linear_model import LogisticRegression
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    model = LogisticRegression(max_iter=3000).fit((Xtr - mu) / sd, ytr)
    W = model.coef_[0] / sd
    B = float(model.intercept_[0] - float(np.dot(model.coef_[0], mu / sd)))
    print(f"  train accuracy {model.score((Xtr - mu) / sd, ytr):.3f}")

    print("\n  HELD-OUT-TRANSFORM accuracy (paraphrase styles never trained on):")
    for tname in HELDOUT_TRANSFORMS:
        rows = gen_examples(prods, base.ix, test_asins, [tname], seed=2)
        Xte, yte = build(rows)
        acc = model.score((Xte - mu) / sd, yte)
        majority = max(yte.mean(), 1 - yte.mean())
        print(f"    {tname:<28} acc {acc:.3f}   majority-class {majority:.3f}   "
              f"lift {acc - majority:+.3f}")

    def p_content(toks, i) -> float:
        z = B + float(np.dot(W, featurise_token(toks, i, base.ix)))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    # ---------------------------------------------------------------- agent integration
    def make(threshold: float):
        class LocalExtract(Agent):
            """Strip predicted scaffolding, then mine the surviving text."""
            KEEP = threshold

            def _observe(self, st, msg):
                if recognised(msg):
                    return super()._observe(st, msg)      # clean path untouched
                toks = raw_toks(msg)
                if len(toks) >= 3:
                    kept = [t for i, t in enumerate(toks) if p_content(toks, i) >= self.KEEP]
                    if len(kept) >= 2:
                        msg = " ".join(kept)
                return super()._observe(st, msg)
        return LocalExtract

    sets = {
        "clean":       samples,
        "unseen800":   mint(prods, pub_t, profiles, "reviews", 800, seed=SEEDS["reviews"]),
        "uniform-pop": mint(prods, pub_t, profiles, "uniform", 800, seed=SEEDS["uniform"]),
    }
    PARA = ["T1 scaffold reworded", "T2 scaffold stripped", "T5 realistic (T1+T3)"]
    COLS = list(sets) + [p.split()[0] for p in PARA]

    def share(cls=Agent):
        o = object.__new__(cls)
        o.ix, o.sessions, o.llm, o.llm_extract = base.ix, {}, None, None
        return o

    def row(cls):
        r = {}
        for name, sub in sets.items():
            r[name] = evaluate(share(cls), sub, cid, cats, prods)[
                "recommended_technical_score"]
        for p in PARA:
            r[p.split()[0]] = evaluate_transformed(
                share(cls), samples, cid, cats, prods,
                TRANSFORMS[p])["recommended_technical_score"]
        return r

    VARIANTS = {"shipped": Agent,
                "M2 keep p>=0.30": make(0.30),
                "M2 keep p>=0.50": make(0.50),
                "M2 keep p>=0.70": make(0.70)}

    print(f"\n{'variant':<24}" + "".join(f"{c:>12}" for c in COLS))
    print("-" * (24 + 12 * len(COLS)))
    OUT, ref = {}, None
    for name, cls in VARIANTS.items():
        r = row(cls)
        if ref is None:
            ref = r
        OUT[name] = r
        print(f"{name:<24}" + "".join(f"{r[c]:>12.5f}" for c in COLS))

    print(f"\n{'variant':<24}" + "".join(f"{c:>12}" for c in COLS) + "   verdict")
    print("-" * (24 + 12 * len(COLS) + 12))
    llm_t1 = 0.0688
    for name in VARIANTS:
        r = OUT[name]
        d = {c: r[c] - ref[c] for c in COLS}
        if r is ref:
            verdict = "reference"
        elif d["clean"] < -1e-9 or d["unseen800"] < -0.005:
            verdict = "REJECT (clean/unseen regressed)"
        elif max(d[p.split()[0]] for p in PARA) > 0.005:
            verdict = "ADOPT -- local paraphrase gain"
        else:
            verdict = "no material gain"
        print(f"{name:<24}" + "".join(f"{d[c]:>+12.5f}" for c in COLS) + f"   {verdict}")
    best_t1 = max(OUT[n]["T1"] - ref["T1"] for n in VARIANTS if n != "shipped")
    print(f"\n  best local gain on T1: {best_t1:+.5f}   "
          f"LLM gain on the same axis: +{llm_t1:.4f}   "
          f"({best_t1/llm_t1:.0%} of the API layer, offline)")

    (ROOT / "experiments" / "results" / "out_47.json").write_text(
        json.dumps(OUT, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n[saved] experiments/results/out_47.json   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
