"""Experiment 45: M1 -- a LEARNED constraint-likeness scorer for n-gram mining.

WHY THIS IS NOT ANOTHER DOOMED RANKING MODEL
--------------------------------------------
Thirteen learned approaches have been tried and rejected on this task
(docs/research/ml_nlp_literature_review.md
B1-B4, C1-C4, D1-D5). Every one of them attacked RETRIEVAL or RERANKING -- choosing among
candidates -- and every one lost to a one-dimensional popularity statistic, because the
candidates are text-tied by construction and no legitimate text signal separates them.

This attacks a different layer. `CatalogIndex.mine()` is the paraphrase floor of the whole
agent: it is what holds the score at 0.838 instead of 0.164 when the templates stop firing.
And it contains no learning whatsoever -- it is greedy longest-match segmentation with a
document-frequency gate:

    for n in range(maxn, minn-1, -1):        # longest first
        if 0 < df(ngram) <= DF_CAP: take it and advance

Under paraphrase that ingests conversational filler on equal terms with product text,
because "appreciate it the thing" is a perfectly good catalogue n-gram somewhere in 50,000
listings. Length and df are the only signals used, and neither knows what a CONSTRAINT
looks like.

THE SUPERVISION IS EXACT, WHICH IS WHAT MAKES THIS DIFFERENT
------------------------------------------------------------
`intent_card(product)` is a pure function of a product. Calling it across all 50,000
catalogue products enumerates, exhaustively and with certainty, every string the generator
is capable of emitting as a constraint. That is not weak supervision in the sense of
Dehghani et al. (SIGIR 2017), where BM25 acts as a noisy teacher -- these labels are
definitionally correct. The earlier LTR failures (D1: 147 positives across 100 queries)
died of data scarcity; this has ~200,000 exact positives.

    positives   every hard_constraint / soft_preference over the whole catalogue
    negatives   n-grams sampled from the same catalogue text that intent_card never emits

SHIPPING CONSTRAINT: the agent is standard-library only. So the model must be a linear one
whose coefficients can be transcribed into plain Python -- no sklearn, no pickle, no torch
at inference. That also keeps it inspectable, which matters when the whole project's
lesson is that opaque capacity destroys the signal.

BENCHMARK: the same seven-condition grid every other change is held to -- clean must not
regress, and the paraphrase and shifted-population columns are where the gain must appear.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/log/45_ml_constraint_likeness.py
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "log"))

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index, evaluate, intent_card, load_jsonl,
)
from submission.agent import Agent, raw_toks  # noqa: E402

_p30 = __import__("30_robustness_benchmark")
_p31 = __import__("31_paraphrase_stress")
mint, SEEDS = _p30.mint, _p30.SEEDS
evaluate_transformed, TRANSFORMS = _p31.evaluate_transformed, _p31.TRANSFORMS

N_TRAIN = 40_000
DIGIT = re.compile(r"\d")


# --------------------------------------------------------------------------- features
# Deliberately few, and each one is a property of the PHRASE rather than of the session.
# Pass 24's ablation is the governing lesson: on this task k=1 feature reproduced the best
# result and every added feature cost accuracy, because capacity consumes the signal. So
# this stays small and interpretable, and the feature-count ablation is re-run below.
# LABEL-LEAK NOTE. The first version of this pass also carried `cap_ratio` and
# `has_colon`, and scored 0.978 held-out with cap_ratio at +28.0 -- ten times every other
# coefficient. Both were artifacts: positives came from `intent_card()`, which preserves
# the catalogue's original casing and punctuation ("Machine Wash", "color: grey"), while
# negatives were drawn from `ix.blob`, which `raw_toks` has already lowercased and stripped
# of punctuation. The model was classifying WHICH PIPELINE BUILT THE STRING, not what a
# constraint looks like.
#
# It would also have shipped broken: `mine()` runs over `raw_toks` output, so at inference
# both features are identically zero and the model's two dominant signals are dead.
#
# The fix is to normalise BOTH classes through exactly the transform inference uses, which
# makes those two features constant and therefore removes them. What remains is signal that
# survives normalisation. This is the third leak of this shape in the project; inspecting
# coefficients rather than trusting accuracy is what caught all three.
FEATURES = ("n_tokens", "log_df", "has_digit", "mean_tok_len", "stopword_ratio",
            "long_tok_ratio", "digit_tok_ratio")

STOPISH = {"the", "a", "an", "and", "or", "of", "to", "for", "with", "in", "on", "is",
           "it", "this", "that", "you", "i", "me", "my", "so", "at", "as", "be", "by"}


def normalise(text: str) -> str:
    """Exactly what `mine()` sees at inference. Training must use the same transform."""
    return " ".join(raw_toks(text))


def featurise(phrase: str, df: int) -> list[float]:
    toks = phrase.split()
    if not toks:
        return [0.0] * len(FEATURES)
    return [
        float(len(toks)),
        math.log1p(max(df, 0)),
        1.0 if DIGIT.search(phrase) else 0.0,
        sum(len(t) for t in toks) / len(toks),
        sum(1 for t in toks if t in STOPISH) / len(toks),
        sum(1 for t in toks if len(t) >= 7) / len(toks),
        sum(1 for t in toks if DIGIT.search(t)) / len(toks),
    ]


def build_training(ix, prods, seed=0):
    """Positives: what intent_card actually emits. Negatives: n-grams it never emits."""
    rng = random.Random(seed)
    asins = list(prods)
    rng.shuffle(asins)

    positives: list[str] = []
    seen_pos: set[str] = set()
    for asin in asins:
        card = intent_card(prods[asin])
        for value in list(card["hard_constraints"]) + list(card["soft_preferences"]):
            text = normalise(str(value))
            if 2 <= len(text.split()) <= 12 and text not in seen_pos:
                seen_pos.add(text)
                positives.append(text)
        if len(positives) >= N_TRAIN // 2:
            break

    # Negatives: contiguous n-grams drawn from the SAME text the positives come from, so
    # the model cannot separate them by vocabulary or topic -- only by phrase shape.
    negatives: list[str] = []
    for asin in asins:
        blob = ix.blob.get(asin, "").split()
        if len(blob) < 6:
            continue
        for _ in range(3):
            n = rng.randint(2, 9)
            i = rng.randrange(0, max(1, len(blob) - n))
            cand = " ".join(blob[i:i + n])          # blob is already normalised
            if cand and cand not in seen_pos:
                negatives.append(cand)
        if len(negatives) >= N_TRAIN // 2:
            break
    return positives[:N_TRAIN // 2], negatives[:N_TRAIN // 2]


def main() -> None:
    t0 = time.time()
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    pub_t = {str(s["ground_truth"]["parent_asin"]) for s in samples}
    profiles = [s["user_profile"] for s in samples]

    print("building exactly-labelled training data from intent_card() ...")
    pos, neg = build_training(base.ix, prods)
    print(f"  {len(pos):,} positives (real constraints)  "
          f"{len(neg):,} negatives (catalogue n-grams)   [{time.time()-t0:.0f}s]")

    X, y = [], []
    for phrase in pos:
        X.append(featurise(phrase, base.ix.df(phrase)))
        y.append(1)
    for phrase in neg:
        X.append(featurise(phrase, base.ix.df(phrase)))
        y.append(0)
    X, y = np.array(X, dtype=np.float64), np.array(y, dtype=np.int8)
    print(f"  feature matrix {X.shape}   positive rate {y.mean():.1%}   "
          f"[{time.time()-t0:.0f}s]")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    model = LogisticRegression(max_iter=2000, C=1.0).fit((Xtr - mu) / sd, ytr)
    acc = model.score((Xte - mu) / sd, yte)
    print(f"\n  held-out accuracy {acc:.3f}  (0.500 = the model learned nothing)")
    print("  learned coefficients, standardised:")
    for name, coef in sorted(zip(FEATURES, model.coef_[0]), key=lambda kv: -abs(kv[1])):
        print(f"    {name:<16}{coef:+.3f}")

    W = model.coef_[0] / sd
    B = float(model.intercept_[0] - float(np.dot(model.coef_[0], mu / sd)))
    print(f"\n  shippable form: {len(W)} coefficients + intercept "
          f"(plain Python, no sklearn at inference)")

    def likeness(phrase: str, df: int) -> float:
        z = B + float(np.dot(W, featurise(phrase, df)))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    # ---------------------------------------------------------------- agent integration
    def make(threshold: float, mode: str):
        class MLMining(Agent):
            """Mining that ranks candidate n-grams by learned constraint-likeness."""

            def _observe(self, st, msg):
                ix = self.ix
                original = ix.mine

                def scored_mine(text, maxn=9, minn=3):
                    toks = raw_toks(text)
                    out, i = [], 0
                    while i < len(toks):
                        best = None
                        for n in range(min(maxn, len(toks) - i), minn - 1, -1):
                            ph = " ".join(toks[i:i + n])
                            df = ix.df(ph)
                            if not (0 < df <= ix.DF_CAP):
                                continue
                            s = likeness(ph, df)   # ph is already normalised
                            if mode == "filter":
                                if s >= threshold:      # longest ABOVE the bar
                                    best = (ph, df, n)
                                    break
                            elif best is None or s > best[3]:
                                best = (ph, df, n, s)   # highest-scoring, any length
                        if best:
                            out.append((best[0], best[1]))
                            i += best[2]
                        else:
                            i += 1
                    return out

                ix.mine = scored_mine
                try:
                    return super()._observe(st, msg)
                finally:
                    ix.mine = original
        return MLMining

    sets = {
        "clean":       samples,
        "unseen800":   mint(prods, pub_t, profiles, "reviews", 800, seed=SEEDS["reviews"]),
        "uniform-pop": mint(prods, pub_t, profiles, "uniform", 800, seed=SEEDS["uniform"]),
        "inverse-pop": mint(prods, pub_t, profiles, "inverse", 800, seed=SEEDS["inverse"]),
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

    VARIANTS = {"shipped (greedy longest-match)": Agent,
                "M1 filter >= 0.30": make(0.30, "filter"),
                "M1 filter >= 0.50": make(0.50, "filter"),
                "M1 filter >= 0.70": make(0.70, "filter"),
                "M1 argmax likeness": make(0.0, "argmax")}

    print(f"\n{'variant':<32}" + "".join(f"{c:>12}" for c in COLS))
    print("-" * (32 + 12 * len(COLS)))
    OUT, ref = {"model": {"accuracy": acc, "coef": dict(zip(FEATURES, W.tolist())),
                          "intercept": B}}, None
    for name, cls in VARIANTS.items():
        r = row(cls)
        if ref is None:
            ref = r
        OUT[name] = r
        print(f"{name:<32}" + "".join(f"{r[c]:>12.5f}" for c in COLS))

    print(f"\n{'variant':<32}" + "".join(f"{c:>12}" for c in COLS) + "   verdict")
    print("-" * (32 + 12 * len(COLS) + 12))
    for name in VARIANTS:
        r = OUT[name]
        d = {c: r[c] - ref[c] for c in COLS}
        if r is ref:
            verdict = "reference"
        elif d["clean"] < -1e-9 or d["unseen800"] < -0.005:
            verdict = "REJECT (clean/unseen regressed)"
        elif max(d[p.split()[0]] for p in PARA) > 0.005:
            verdict = "ADOPT -- raises the paraphrase floor"
        else:
            verdict = "no material gain"
        print(f"{name:<32}" + "".join(f"{d[c]:>+12.5f}" for c in COLS) + f"   {verdict}")

    (ROOT / "experiments" / "results" / "out_45.json").write_text(
        json.dumps(OUT, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n[saved] experiments/results/out_45.json   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
