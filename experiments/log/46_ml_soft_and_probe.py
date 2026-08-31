"""Experiment 46: M1 used SOFTLY, and M2 -- a learned state-conditioned probe policy.

PART A -- M1 revisited. Pass 45 trained a constraint-likeness classifier on exact labels
(every string `intent_card()` can emit, across all 50,000 products) and used it as a hard
FILTER inside `mine()`. It degraded monotonically with the threshold:

    threshold   T1        T2        T5
    none        0.85230   0.86915   0.84770     <- shipped
    >= 0.30     0.77715   0.81778   0.78055
    >= 0.50     0.65270   0.73650   0.65200
    >= 0.70     0.21730   0.49365   0.16370     <- identical to deleting mining entirely

At 0.637 held-out accuracy the classifier is far too weak to gate on: filtering trades a
little precision for a lot of recall, and recall is the entire job of the paraphrase floor.

But a weak signal can still be used WITHOUT discarding anything. Instead of dropping
low-scoring n-grams, keep every one and let likeness modulate its WEIGHT:

    w(phrase) = W_MINED * (FLOOR + (1 - FLOOR) * likeness(phrase))

With FLOOR > 0 no evidence can ever be lost, so the catastrophic recall collapse above is
structurally impossible; the model can only reorder emphasis. If 0.637 accuracy carries
real information this is the form that can exploit it.

PART B -- M2, the probe policy, aimed at MTTC. `PROBE_ORDER` is a fixed 7-tuple. Pass 04
swept fixed ORDERS and found the effect attenuates to under 0.003 at tuned weights, and I
have been citing that ever since as "probe order does not matter". That conclusion is
narrower than the way I have used it: a fixed permutation is not a policy. This tests a
STATE-CONDITIONED rule -- which attribute to ask given what the session has already
learned -- which is the EAR/SCPR *Action* stage (arXiv:2002.09102, arXiv:2007.00194), the
one CRS component this project dismissed as dissolved.

The policy is fitted offline on minted sessions, where the payout of each attribute is
directly observable: ask it, and see whether the simulator discloses anything new. No
gradient method, no RL machinery -- an expected-yield table conditioned on the session
state, which is what the entropy/IDSS line (arXiv:2603.11399) actually amounts to here.

MTTC is the only remaining lever: 2.320 against a 1.39 floor, worth 0.20 of the score.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/scripts/46_ml_soft_and_probe.py
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

from evaluator.local_evaluator import (  # noqa: E402
    behavior_for, catalog_index, coarse_category, customer_reply, evaluate,
    initial_message, intent_card, load_jsonl,
)
from submission.agent import DEAD_ATTRIBUTES, MINED, Agent  # noqa: E402

_p30 = __import__("30_robustness_benchmark")
_p31 = __import__("31_paraphrase_stress")
_p45 = __import__("45_ml_constraint_likeness")
mint, SEEDS = _p30.mint, _p30.SEEDS
evaluate_transformed, TRANSFORMS = _p31.evaluate_transformed, _p31.TRANSFORMS

ATTRS = ("feature", "material", "color", "style", "size", "use_case", "other")


def train_likeness(base, prods):
    pos, neg = _p45.build_training(base.ix, prods)
    X = [_p45.featurise(p, base.ix.df(p)) for p in pos]
    X += [_p45.featurise(n, base.ix.df(n)) for n in neg]
    y = [1] * len(pos) + [0] * len(neg)
    X, y = np.array(X, dtype=np.float64), np.array(y, dtype=np.int8)
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(max_iter=2000).fit((X - mu) / sd, y)
    W = m.coef_[0] / sd
    B = float(m.intercept_[0] - float(np.dot(m.coef_[0], mu / sd)))

    def likeness(phrase: str, df: int) -> float:
        z = B + float(np.dot(W, _p45.featurise(phrase, df)))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
    return likeness


def learn_probe_policy(prods, agent, n_sessions=400, seed=7):
    """Expected NEW-EVIDENCE yield of each attribute, conditioned on session state.

    State is coarse on purpose -- (turn bucket, how much evidence we already hold, which
    attributes are still unasked). Pass 24's lesson governs here too: capacity is what
    destroys signal on this task, so the policy is a small table, not a network.
    """
    rng = random.Random(seed)
    asins = [a for a in prods]
    rng.shuffle(asins)
    yields: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for asin in asins[:n_sessions]:
        card = intent_card(prods[asin])
        for scenario in ("buying", "browsing"):
            eff = {"sample_id": f"p_{asin}", "scenario_type": scenario,
                   "intent_card": card,
                   "behavior": behavior_for(scenario, card, random.Random(asin))}
            disclosed: set[str] = set()
            bu = False
            for turn in range(1, 8):
                bucket = (min(turn, 4), min(len(disclosed), 4))
                for attr in ATTRS:
                    probe_disclosed = set(disclosed)
                    msg, _ = customer_reply(eff, attr, probe_disclosed, bu)
                    gained = len(probe_disclosed) - len(disclosed)
                    yields[bucket][attr].append(float(gained))
                # advance the real session with the currently best attribute
                best = max(ATTRS, key=lambda a: (
                    sum(yields[bucket][a]) / len(yields[bucket][a])
                    if yields[bucket][a] else 0.0))
                _, bu = customer_reply(eff, best, disclosed, bu)
    table = {k: {a: (sum(v) / len(v) if v else 0.0) for a, v in d.items()}
             for k, d in yields.items()}
    return table


def main() -> None:
    t0 = time.time()
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    pub_t = {str(s["ground_truth"]["parent_asin"]) for s in samples}
    profiles = [s["user_profile"] for s in samples]

    print("training constraint-likeness model ...")
    likeness = train_likeness(base, prods)
    print(f"  done [{time.time()-t0:.0f}s]")

    print("fitting state-conditioned probe policy on minted sessions ...")
    table = learn_probe_policy(prods, base)
    print(f"  {len(table)} states [{time.time()-t0:.0f}s]")
    for state in sorted(table)[:5]:
        best = sorted(table[state].items(), key=lambda kv: -kv[1])[:3]
        print(f"    turn<={state[0]} eviden<={state[1]} -> "
              + ", ".join(f"{a}:{v:.2f}" for a, v in best))

    # ------------------------------------------------------------------ A: soft weighting
    def make_soft(floor: float):
        class SoftML(Agent):
            """Likeness modulates mined-evidence WEIGHT; nothing is ever discarded."""
            LIKE_FLOOR = floor

            def _weight(self, phrase, df, tier):
                w = super()._weight(phrase, df, tier)
                if tier == MINED:
                    s = likeness(phrase, df)
                    w *= self.LIKE_FLOOR + (1.0 - self.LIKE_FLOOR) * s
                return w
        return SoftML

    # ------------------------------------------------------------------ B: probe policy
    class LearnedProbe(Agent):
        def _next_probe(self, st):
            bucket = (min(st.turn, 4), min(len(st.evidence), 4))
            scores = table.get(bucket) or table.get((4, 4)) or {}
            options = [a for a in ATTRS
                       if a not in st.asked and a not in DEAD_ATTRIBUTES]
            if not options:
                return "other"
            return max(options, key=lambda a: scores.get(a, 0.0))

    sets = {
        "clean":       samples,
        "unseen800":   mint(prods, pub_t, profiles, "reviews", 800, seed=SEEDS["reviews"]),
        "uniform-pop": mint(prods, pub_t, profiles, "uniform", 800, seed=SEEDS["uniform"]),
        "inverse-pop": mint(prods, pub_t, profiles, "inverse", 800, seed=SEEDS["inverse"]),
    }
    PARA = ["T1 scaffold reworded", "T5 realistic (T1+T3)"]
    COLS = list(sets) + [p.split()[0] for p in PARA]

    def share(cls=Agent):
        o = object.__new__(cls)
        o.ix, o.sessions, o.llm, o.llm_extract = base.ix, {}, None, None
        return o

    def row(cls):
        r = {}
        for name, sub in sets.items():
            res = evaluate(share(cls), sub, cid, cats, prods)
            r[name] = res["recommended_technical_score"]
            if name == "clean":
                r["_mttc"] = res["mttc"]
        for p in PARA:
            r[p.split()[0]] = evaluate_transformed(
                share(cls), samples, cid, cats, prods,
                TRANSFORMS[p])["recommended_technical_score"]
        return r

    VARIANTS = {
        "shipped":                     Agent,
        "M1 soft weight (floor 0.5)":  make_soft(0.5),
        "M1 soft weight (floor 0.2)":  make_soft(0.2),
        "M1 soft weight (floor 0.0)":  make_soft(0.0),
        "M2 learned probe policy":     LearnedProbe,
    }

    print(f"\n{'variant':<30}" + "".join(f"{c:>12}" for c in COLS) + f"{'MTTC':>8}")
    print("-" * (30 + 12 * len(COLS) + 8))
    OUT, ref = {}, None
    for name, cls in VARIANTS.items():
        r = row(cls)
        if ref is None:
            ref = r
        OUT[name] = r
        print(f"{name:<30}" + "".join(f"{r[c]:>12.5f}" for c in COLS)
              + f"{r['_mttc']:>8.3f}")

    print(f"\n{'variant':<30}" + "".join(f"{c:>12}" for c in COLS) + "   verdict")
    print("-" * (30 + 12 * len(COLS) + 12))
    for name in VARIANTS:
        r = OUT[name]
        d = {c: r[c] - ref[c] for c in COLS}
        if r is ref:
            verdict = "reference"
        elif d["clean"] < -1e-9 or d["unseen800"] < -0.005:
            verdict = "REJECT (clean/unseen regressed)"
        elif max(d.values()) > 0.005:
            verdict = "ADOPT"
        else:
            verdict = "no material gain"
        print(f"{name:<30}" + "".join(f"{d[c]:>+12.5f}" for c in COLS) + f"   {verdict}")

    (ROOT / "experiments" / "results" / "out_46.json").write_text(
        json.dumps(OUT, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n[saved] experiments/results/out_46.json   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
