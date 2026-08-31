"""Experiment 36: adjudicate a grading disagreement between two independent robustness audits.

An independently-run benchmark (docs/validation/robustness_benchmark.md) graded SEQUENTIAL DISCLOSURE
as population-risk 3 / organizer-risk 3 -- its joint-worst rating. This audit graded the
same component P1/O1. That is a two-grade gap on both axes, so one of us is wrong, and it
is cheap to settle by measurement rather than by argument.

The other audit's two stated reasons:

  (a) "It relies on the evaluator ending a session on any hit."
      This is verbatim in the specification -- Session Protocol step 7, "The session ends
      after a valid hit or turn 10", and step 5, "A target hit records rank and turn".
      An organizer would have to deviate from their own documented protocol. That is an
      O1 justification, not O3 -- UNLESS the policy also degrades under conditions the
      organizer CAN legitimately vary, which is what this pass tests.

  (b) "It changes the stated Top-10 shopping behaviour."
      Legitimate, but it is a product-judgement concern, not a robustness one. It cannot
      be measured by the technical score and is recorded separately, not graded here.

THE TEST. Sequential disclosure is P1/O1 only if it stays a NET POSITIVE under every
stress axis available -- including the ones where our ranking is degraded, since a
narrow-disclosure policy is only safe while the ranking it walks is good. If width-1 flips
negative anywhere, the other audit is right and the policy is a conditional bet.

  nominal            public 200, untouched
  unseen-800         review-weighted targets, no public overlap
  uniform-pop        targets drawn uniformly -- our ranking is measurably worse here
  inverse-pop        targets anti-correlated with popularity -- worse still
  paraphrase T1/T5   templates broken, evidence thin, ranking degraded

Run:  PYTHONIOENCODING=utf-8 python -u experiments/log/36_disclosure_risk_adjudication.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "log"))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent  # noqa: E402

_p30 = __import__("30_robustness_benchmark")
_p31 = __import__("31_paraphrase_stress")
mint, SEEDS = _p30.mint, _p30.SEEDS
evaluate_transformed, TRANSFORMS = _p31.evaluate_transformed, _p31.TRANSFORMS

W1 = (1,) * 9 + (10,)          # shipped
W10 = (10,) * 10               # the policy the other audit implies is safer
WMID = (1, 1, 2, 3, 4, 5, 6, 8, 9, 10)   # a middle course, for reference


def main() -> None:
    t0 = time.time()
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    pub_t = {str(s["ground_truth"]["parent_asin"]) for s in samples}
    profiles = [s["user_profile"] for s in samples]

    plain = {
        "nominal (public 200)": samples,
        "unseen-800":  mint(prods, pub_t, profiles, "reviews", 800, seed=SEEDS["reviews"]),
        "uniform-pop": mint(prods, pub_t, profiles, "uniform", 800, seed=SEEDS["uniform"]),
        "inverse-pop": mint(prods, pub_t, profiles, "inverse", 800, seed=SEEDS["inverse"]),
    }
    para = {"paraphrase-T1": "T1 scaffold reworded",
            "paraphrase-T5": "T5 realistic (T1+T3)"}

    def share(disc):
        o = object.__new__(Agent)
        o.ix, o.sessions, o.llm = base.ix, {}, None
        o.DISCLOSURE = disc
        return o

    POLICIES = {"width-1 x9, then 10 (shipped)": W1,
                "widen 1,1,2,3..10": WMID,
                "full 10 every turn": W10}

    rows: dict[str, dict] = {}
    hrs: dict[str, dict] = {}
    for pname, disc in POLICIES.items():
        r, h = {}, {}
        for cname, sub in plain.items():
            res = evaluate(share(disc), sub, cid, cats, prods)
            r[cname] = res["recommended_technical_score"]
            h[cname] = res["hit_rate_at_10"]
        for cname, tname in para.items():
            res = evaluate_transformed(share(disc), samples, cid, cats, prods,
                                       TRANSFORMS[tname])
            r[cname] = res["recommended_technical_score"]
            h[cname] = res["hit_rate_at_10"]
        rows[pname], hrs[pname] = r, h

    COLS = list(plain) + list(para)
    print("TechnicalScore by disclosure policy and stress condition")
    print(f"{'policy':<32}" + "".join(f"{c:>16}" for c in COLS))
    print("-" * (32 + 16 * len(COLS)))
    for p in POLICIES:
        print(f"{p:<32}" + "".join(f"{rows[p][c]:>16.5f}" for c in COLS))

    print(f"\nvalue OF sequential disclosure (shipped minus full-10) -- "
          f"positive means width-1 helps")
    print(f"{'':<32}" + "".join(f"{c:>16}" for c in COLS))
    ship, full = rows["width-1 x9, then 10 (shipped)"], rows["full 10 every turn"]
    deltas = {c: ship[c] - full[c] for c in COLS}
    print(f"{'delta':<32}" + "".join(f"{deltas[c]:>+16.5f}" for c in COLS))

    print(f"\nHitRate@10 -- the quantity a narrow policy could in principle destroy")
    print(f"{'policy':<32}" + "".join(f"{c:>16}" for c in COLS))
    for p in POLICIES:
        print(f"{p:<32}" + "".join(f"{hrs[p][c]:>15.1%} " for c in COLS))
    hr_d = {c: hrs['width-1 x9, then 10 (shipped)'][c] - hrs['full 10 every turn'][c]
            for c in COLS}
    print(f"{'HR delta (shipped - full10)':<32}" + "".join(f"{hr_d[c]:>+15.1%} " for c in COLS))

    worst = min(deltas.values())
    worst_c = min(deltas, key=lambda c: deltas[c])
    print(f"\n  worst condition for sequential disclosure: {worst_c}  ({worst:+.5f})")
    print(f"  HitRate ever damaged by narrowing? "
          f"{'YES' if min(hr_d.values()) < -1e-9 else 'NO -- identical in every condition'}")
    print(f"\n  VERDICT: {'P1/O1 stands -- net positive under every stress axis tested' if worst > 0 else 'the other audit is right -- width-1 is a conditional bet'}")

    (ROOT / "experiments" / "results" / "out_36.json").write_text(
        json.dumps({"score": rows, "hit_rate": hrs, "delta_vs_full10": deltas,
                    "hr_delta_vs_full10": hr_d}, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] experiments/results/out_36.json   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
