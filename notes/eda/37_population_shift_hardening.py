"""EDA pass 37: harden the popularity prior against population shift.

WHERE WE ARE. Pass 32 isolated every scrap of population exposure into one coefficient:

    targets ~ review count (the real split)   prior worth  +0.051
    targets ~ uniform                         prior worth  -0.059
    targets ~ 1/review count                  prior worth  -0.086

    with the prior OFF: 0.9009 / 0.8999 / 0.8965  -- invariant to within 0.004

THE MECHANISM WORTH ATTACKING. `W_POP` enters as an ADDITIVE term on the same scale as
phrase coverage:

    s = sum(w for matched phrases) + W_POP * pop_norm(asin)

So a sufficiently popular product can outscore a product that matches MORE of the
customer's stated evidence. Under the real population that is a good trade. Under a
shifted one it is strictly destructive -- the prior is not merely uninformative, it
OVERRIDES the evidence, which is why the downside (-0.086) is larger than the upside
(+0.051).

Nothing about the prior's usefulness requires that it be able to override evidence. These
variants keep the signal and bound the damage:

  B  weight sweep            the blunt lever, for reference
  C  tie-break ONLY          popularity cannot outrank coverage; it orders only products
                             the evidence has genuinely failed to separate. Worst case
                             should approach zero rather than -0.086.
  D  percentile rank         replaces log1p(count)/max -- a heavy-tailed quantity where a
                             few products sit near 1.0 and dominate -- with a uniform 0..1
                             rank. Same ordering, bounded leverage.
  E  evidence-gated          the prior's weight DECAYS as real evidence arrives. Pass 32
                             showed the prior's value rises when evidence is thin (+0.053
                             nominal -> +0.158 under paraphrase), so spending it early and
                             withdrawing it later matches where it actually pays.
  F  capped contribution     clip pop_norm at a ceiling so the tail cannot dominate.
  G  rank fusion (RRF)       combine coverage order and popularity order by reciprocal
                             rank, so the prior shifts position by a bounded number of
                             places regardless of scale.

SELECTION RULE, fixed before looking: adopt only if the WORST population column improves
and neither the real-population column nor the paraphrase column regresses beyond noise
(-0.005). Optimising the average would just re-buy the same bet.

Run:  PYTHONIOENCODING=utf-8 python -u notes/eda/37_population_shift_hardening.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notes" / "eda"))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent  # noqa: E402

_p30 = __import__("30_robustness_benchmark")
_p31 = __import__("31_paraphrase_stress")
mint, SEEDS = _p30.mint, _p30.SEEDS
evaluate_transformed, TRANSFORMS = _p31.evaluate_transformed, _p31.TRANSFORMS


# --------------------------------------------------------------------------- variants

class TieBreakPop(Agent):
    """C: popularity may only order candidates that coverage has tied exactly."""
    def _rank(self, st, pool, top_k):
        wmap = {p: self._weight(p, df, tier) for p, (df, tier) in st.evidence.items()}
        order = {a: i for i, a in enumerate(pool)}

        def score(asin):
            s = sum(w for phrase, w in wmap.items() if self.ix.covers(asin, phrase))
            return (-round(s, 12),
                    -(self.ix.pop.get(asin, 0.0) / (self.ix.max_pop or 1.0)),
                    order[asin])
        return sorted(pool, key=score)[:top_k]


class PercentilePop(Agent):
    """D: popularity as a uniform 0..1 percentile rank rather than log1p/max."""
    def _pctl(self):
        p = getattr(self, "_pctl_cache", None)
        if p is None:
            items = sorted(self.ix.pop.items(), key=lambda kv: kv[1])
            n = max(len(items) - 1, 1)
            p = {a: i / n for i, (a, _) in enumerate(items)}
            self._pctl_cache = p
        return p

    def _rank(self, st, pool, top_k):
        wmap = {p: self._weight(p, df, tier) for p, (df, tier) in st.evidence.items()}
        order = {a: i for i, a in enumerate(pool)}
        pct = self._pctl()

        def score(asin):
            s = sum(w for phrase, w in wmap.items() if self.ix.covers(asin, phrase))
            return (-(s + self.W_POP * pct.get(asin, 0.0)), order[asin])
        return sorted(pool, key=score)[:top_k]


class EvidenceGatedPop(Agent):
    """E: the prior's weight decays as real evidence accumulates."""
    GATE_K = 0.5

    def _rank(self, st, pool, top_k):
        wmap = {p: self._weight(p, df, tier) for p, (df, tier) in st.evidence.items()}
        order = {a: i for i, a in enumerate(pool)}
        strong = sum(1 for _, (df, _t) in st.evidence.items() if df <= self.STRONG_DF)
        w_pop = self.W_POP / (1.0 + self.GATE_K * strong)

        def score(asin):
            s = sum(w for phrase, w in wmap.items() if self.ix.covers(asin, phrase))
            return (-(s + w_pop * (self.ix.pop.get(asin, 0.0) / (self.ix.max_pop or 1.0))),
                    order[asin])
        return sorted(pool, key=score)[:top_k]


class CappedPop(Agent):
    """F: clip the normalised popularity so the tail cannot dominate."""
    POP_CAP = 0.6

    def _rank(self, st, pool, top_k):
        wmap = {p: self._weight(p, df, tier) for p, (df, tier) in st.evidence.items()}
        order = {a: i for i, a in enumerate(pool)}

        def score(asin):
            s = sum(w for phrase, w in wmap.items() if self.ix.covers(asin, phrase))
            pn = min(self.ix.pop.get(asin, 0.0) / (self.ix.max_pop or 1.0), self.POP_CAP)
            return (-(s + self.W_POP * pn), order[asin])
        return sorted(pool, key=score)[:top_k]


class RRFPop(Agent):
    """G: reciprocal-rank fusion of the coverage order and the popularity order."""
    RRF_K = 60.0
    RRF_W = 0.30

    def _rank(self, st, pool, top_k):
        wmap = {p: self._weight(p, df, tier) for p, (df, tier) in st.evidence.items()}
        order = {a: i for i, a in enumerate(pool)}
        cov = {a: sum(w for ph, w in wmap.items() if self.ix.covers(a, ph)) for a in pool}
        by_cov = sorted(pool, key=lambda a: (-cov[a], order[a]))
        by_pop = sorted(pool, key=lambda a: (-self.ix.pop.get(a, 0.0), order[a]))
        rc = {a: i + 1 for i, a in enumerate(by_cov)}
        rp = {a: i + 1 for i, a in enumerate(by_pop)}

        def score(asin):
            f = 1.0 / (self.RRF_K + rc[asin]) + self.RRF_W / (self.RRF_K + rp[asin])
            return (-f, order[asin])
        return sorted(pool, key=score)[:top_k]


def main() -> None:
    t0 = time.time()
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    pub_t = {str(s["ground_truth"]["parent_asin"]) for s in samples}
    profiles = [s["user_profile"] for s in samples]

    plain = {
        "public200":   samples,
        "real-pop":    mint(prods, pub_t, profiles, "reviews", 800, seed=SEEDS["reviews"]),
        "uniform-pop": mint(prods, pub_t, profiles, "uniform", 800, seed=SEEDS["uniform"]),
        "inverse-pop": mint(prods, pub_t, profiles, "inverse", 800, seed=SEEDS["inverse"]),
    }
    POP_COLS = ["real-pop", "uniform-pop", "inverse-pop"]

    def share(cls=Agent, **kw):
        o = object.__new__(cls)
        o.ix, o.sessions, o.llm = base.ix, {}, None
        for k, v in kw.items():
            setattr(o, k, v)
        return o

    VARIANTS = {
        "A shipped (additive, W_POP .25)": (Agent, {}),
        "B0 no prior (W_POP 0)":           (Agent, {"W_POP": 0.0}),
        "B1 additive W_POP .10":           (Agent, {"W_POP": 0.10}),
        "B2 additive W_POP .15":           (Agent, {"W_POP": 0.15}),
        "C  tie-break ONLY":               (TieBreakPop, {}),
        "D  percentile rank .25":          (PercentilePop, {}),
        "D2 percentile rank .40":          (PercentilePop, {"W_POP": 0.40}),
        "E  evidence-gated k=0.5":         (EvidenceGatedPop, {}),
        "E2 evidence-gated k=1.0":         (EvidenceGatedPop, {"GATE_K": 1.0}),
        "F  capped at 0.6":                (CappedPop, {}),
        "G  RRF fusion w=.30":             (RRFPop, {}),
    }

    COLS = list(plain) + ["para-T1"]
    print(f"{'variant':<34}" + "".join(f"{c:>13}" for c in COLS)
          + f"{'WORST-POP':>12}{'SPREAD':>9}")
    print("-" * (34 + 13 * len(COLS) + 21))

    OUT, ref = {}, None
    for name, (cls, kw) in VARIANTS.items():
        row = {}
        for c, sub in plain.items():
            row[c] = evaluate(share(cls, **kw), sub, cid, cats,
                              prods)["recommended_technical_score"]
        row["para-T1"] = evaluate_transformed(
            share(cls, **kw), samples, cid, cats, prods,
            TRANSFORMS["T1 scaffold reworded"])["recommended_technical_score"]
        worst = min(row[c] for c in POP_COLS)
        spread = max(row[c] for c in POP_COLS) - worst
        OUT[name] = {"scores": row, "worst_pop": worst, "spread": spread}
        if ref is None:
            ref = OUT[name]
        print(f"{name:<34}" + "".join(f"{row[c]:>13.5f}" for c in COLS)
              + f"{worst:>12.5f}{spread:>9.5f}")

    print(f"\ndeltas vs shipped   (selection rule: WORST-POP must rise, and neither"
          f" real-pop nor para-T1 may fall by more than 0.005)")
    print(f"{'variant':<34}" + "".join(f"{c:>13}" for c in COLS) + f"{'WORST-POP':>12}   verdict")
    print("-" * (34 + 13 * len(COLS) + 24))
    for name, v in OUT.items():
        d = {c: v["scores"][c] - ref["scores"][c] for c in COLS}
        dw = v["worst_pop"] - ref["worst_pop"]
        if v is ref:
            verdict = "reference"
        elif dw > 0 and d["real-pop"] > -0.005 and d["para-T1"] > -0.005:
            verdict = "ADOPT -- hardens without cost"
        elif dw > 0:
            verdict = f"trade-off (real {d['real-pop']:+.3f})"
        else:
            verdict = "no gain on worst population"
        print(f"{name:<34}" + "".join(f"{d[c]:>+13.5f}" for c in COLS)
              + f"{dw:>+12.5f}   {verdict}")

    (ROOT / "notes" / "eda" / "out_37.json").write_text(
        json.dumps(OUT, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] notes/eda/out_37.json   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
