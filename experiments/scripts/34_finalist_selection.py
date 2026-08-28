"""Experiment 34: pick the final configuration by WORST CASE across every stress axis.

Pass 33 found several re-tunes that beat the shipped constants on all four scored sets.
Pass 32 then showed why that is not sufficient evidence to ship them: catalogue-grounded
n-gram mining measures at -0.0001 nominally and +0.585 under paraphrase, so a criterion
that only looks at nominal score is blind to the axis that matters most for a private
run we cannot inspect.

So the finalists are scored across all seven conditions at once and chosen on the WORST
column, not the average:

  public-tune   100 sessions the constants were originally fitted on
  public-hold   100 sessions never used for fitting
  synth-A/B     800 unseen sessions each, independent draws, review-weighted targets
                (the real distribution: a 5-core leave-last-out review split)
  synth-unif    800 unseen sessions, UNIFORM targets -- a population where popularity
                carries no signal. Pass 32 measured the prior at -0.059 here.
  para-T1       public 200 with the template framings reworded
  para-T5       public 200 reworded AND wrapped in conversational filler

A configuration is only adopted if it is at least as good as shipped on the worst of the
seven. Nine candidates were searched in pass 33 over four sets, so some of those wins are
multiple-comparison luck; requiring a win on three further independent conditions is the
cheapest available guard against that.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/scripts/34_finalist_selection.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent  # noqa: E402

_p30 = __import__("30_robustness_benchmark")
_p31 = __import__("31_paraphrase_stress")
mint, SEEDS = _p30.mint, _p30.SEEDS
evaluate_transformed, TRANSFORMS = _p31.evaluate_transformed, _p31.TRANSFORMS

FINALISTS = {
    "shipped (IDF .35, POP .35)":  {},
    "IDF .00":                     {"IDF_POW": 0.00},
    "IDF .20":                     {"IDF_POW": 0.20},
    "IDF .00 + POP .25":           {"IDF_POW": 0.00, "W_POP": 0.25},
    "IDF .20 + POP .25":           {"IDF_POW": 0.20, "W_POP": 0.25},
    "IDF .00 + POP .15 + POOL 700": {"IDF_POW": 0.00, "W_POP": 0.15, "POOL": 700},
}


def main() -> None:
    t0 = time.time()
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    pub_t = {str(s["ground_truth"]["parent_asin"]) for s in samples}
    profiles = [s["user_profile"] for s in samples]

    plain = {
        "public-tune": [s for i, s in enumerate(samples) if i % 2 == 0],
        "public-hold": [s for i, s in enumerate(samples) if i % 2 == 1],
        "synth-A":     mint(prods, pub_t, profiles, "reviews", 800, seed=SEEDS["reviews"]),
        "synth-B":     mint(prods, pub_t, profiles, "reviews", 800, seed=987_654),
        "synth-unif":  mint(prods, pub_t, profiles, "uniform", 800, seed=SEEDS["uniform"]),
    }
    para = {"para-T1": "T1 scaffold reworded", "para-T5": "T5 realistic (T1+T3)"}
    COLS = list(plain) + list(para)

    def share(**kw):
        o = object.__new__(Agent)
        o.ix, o.sessions, o.llm = base.ix, {}, None
        for k, v in kw.items():
            setattr(o, k, v)
        return o

    print(f"{'configuration':<30}" + "".join(f"{c:>12}" for c in COLS) + f"{'WORST':>9}")
    print("-" * (30 + 12 * len(COLS) + 9))
    OUT, ref = {}, None
    for name, kw in FINALISTS.items():
        row = {}
        for c, sub in plain.items():
            row[c] = evaluate(share(**kw), sub, cid, cats,
                              prods)["recommended_technical_score"]
        for c, tname in para.items():
            row[c] = evaluate_transformed(share(**kw), samples, cid, cats, prods,
                                          TRANSFORMS[tname])["recommended_technical_score"]
        if ref is None:
            ref = dict(row)
        deltas = {c: row[c] - ref[c] for c in COLS}
        OUT[name] = {"kw": kw, "scores": row, "deltas": deltas,
                     "worst": min(deltas.values())}
        print(f"{name:<30}" + "".join(f"{row[c]:>12.5f}" for c in COLS)
              + f"{min(deltas.values()):>+9.5f}")

    print(f"\n{'configuration':<30}" + "".join(f"{c:>12}" for c in COLS) + "   verdict")
    print("-" * (30 + 12 * len(COLS) + 12))
    for name, v in OUT.items():
        cells = "".join(f"{v['deltas'][c]:>+12.5f}" for c in COLS)
        verdict = ("reference" if not v["kw"] else
                   "ADOPT -- no condition regresses" if v["worst"] >= 0 else
                   "conditional (regresses somewhere)" if v["worst"] > -0.005 else
                   "REJECT")
        print(f"{name:<30}{cells}   {verdict}")

    winner = max((k for k in OUT if OUT[k]["kw"]),
                 key=lambda k: (OUT[k]["worst"], sum(OUT[k]["deltas"].values())))
    print(f"\n  winner by worst-case: {winner}  {OUT[winner]['kw']}")
    print(f"    worst column {OUT[winner]['worst']:+.5f}, "
          f"sum of deltas {sum(OUT[winner]['deltas'].values()):+.5f}")

    (ROOT / "experiments" / "results" / "out_34.json").write_text(
        json.dumps(OUT, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] experiments/results/out_34.json   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
