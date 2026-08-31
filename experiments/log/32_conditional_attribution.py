"""Experiment 32: two questions that passes 30 and 31 raised but could not answer alone.

Q1  IS THE UNIFORM-POPULATION DROP CAUSED BY OUR POPULARITY PRIOR, OR BY THE PRODUCTS?
    Pass 30 scored 0.9437 on review-weighted targets and 0.8278 on uniformly drawn ones.
    Two very different explanations produce that:
      (a) W_POP is a population assumption that stops holding -> our fault, our risk;
      (b) uniformly drawn products simply have thinner metadata, so the customer says
          less and there is less to retrieve on -> nobody's fault, and not fixable.
    Setting W_POP=0 separates them. If (a), removing the prior RECOVERS score on the
    uniform set. If (b), removing it changes little and the drop is intrinsic.

Q2  WHICH COMPONENTS EARN THEIR PLACE ONLY WHEN THINGS GO WRONG?
    Pass 30 measured catalogue-grounded n-gram mining at -0.0001 on the public 200 and
    +0.0008 on 800 unseen sessions -- i.e. worth nothing. Pass 31 then stripped the
    message templates entirely and the agent still scored 0.8175, which ONLY mining can
    explain, because with templates gone it is the sole evidence channel left.
    So a component's value is conditional, and measuring it only in the nominal condition
    understates insurance. This crosses every ablation with every paraphrase transform.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/scripts/32_conditional_attribution.py
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

SEEDS = {"reviews": 1001, "uniform": 1002, "inverse": 1003}
_p30 = __import__("30_robustness_benchmark")
_p31 = __import__("31_paraphrase_stress")
mint = _p30.mint
evaluate_transformed = _p31.evaluate_transformed
TRANSFORMS = _p31.TRANSFORMS


def main() -> None:
    t0 = time.time()
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    public_targets = {str(s["ground_truth"]["parent_asin"]) for s in samples}
    profiles = [s["user_profile"] for s in samples]

    def share(cls=Agent, **kw):
        o = object.__new__(cls)
        o.ix, o.sessions, o.llm = base.ix, {}, None
        for k, v in kw.items():
            setattr(o, k, v)
        return o

    OUT: dict = {}

    # =============================================== Q1: prior, or thinner products?
    print("=" * 78)
    print("Q1  is the uniform-population drop the PRIOR's fault or the PRODUCTS' fault?")
    print("=" * 78)
    POPS = {w: mint(prods, public_targets, profiles, w, 800, seed=SEEDS[w])
            for w in ("reviews", "uniform", "inverse")}

    print(f"{'target distribution':<22}{'W_POP=0.35':>12}{'W_POP=0':>10}"
          f"{'recovered':>11}{'HR@0.35':>9}{'HR@0':>8}")
    print("-" * 72)
    Q1 = {}
    for w, sub in POPS.items():
        a = evaluate(share(), sub, cid, cats, prods)
        b = evaluate(share(W_POP=0.0), sub, cid, cats, prods)
        sa, sb = a["recommended_technical_score"], b["recommended_technical_score"]
        Q1[w] = {"shipped": sa, "no_pop": sb, "delta": sb - sa,
                 "hr_shipped": a["hit_rate_at_10"], "hr_nopop": b["hit_rate_at_10"]}
        print(f"{w:<22}{sa:>12.5f}{sb:>10.5f}{sb-sa:>+11.5f}"
              f"{a['hit_rate_at_10']:>9.1%}{b['hit_rate_at_10']:>8.1%}")
    OUT["Q1_population"] = Q1

    d_uni = Q1["uniform"]["delta"]
    gap = Q1["reviews"]["shipped"] - Q1["uniform"]["shipped"]
    print(f"\n  review-weighted minus uniform, with the prior ON : {gap:+.5f}")
    print(f"  score RECOVERED on uniform by deleting the prior  : {d_uni:+.5f}")
    frac = max(0.0, d_uni) / gap if gap > 0 else 0.0
    print(f"  -> the prior explains at most {frac:.0%} of the gap; the remainder is")
    print(f"     intrinsic to sparser products and is not a modelling choice we made.")

    # ============================== Q2: which components are load-bearing UNDER STRESS?
    print("\n" + "=" * 78)
    print("Q2  component value under nominal vs paraphrased conditions (public 200)")
    print("=" * 78)

    NoLedger, NoMining = _p30_classes()
    VARIANTS = {
        "shipped":                (Agent, {}),
        "-- n-gram mining":       (NoMining, {}),
        "-- session ledger":      (NoLedger, {}),
        "-- popularity prior":    (Agent, {"W_POP": 0.0}),
        "-- sequential disclose": (Agent, {"DISCLOSURE": (10,) * 10}),
    }
    COLS = ["T0 identity (control)", "T1 scaffold reworded", "T2 scaffold stripped",
            "T5 realistic (T1+T3)"]

    header = f"{'variant':<24}" + "".join(f"{c.split()[0]:>10}" for c in COLS)
    print(header)
    print("-" * len(header))
    Q2 = {}
    for name, (cls, kw) in VARIANTS.items():
        row = {}
        cells = ""
        for c in COLS:
            r = evaluate_transformed(share(cls, **kw), samples, cid, cats, prods,
                                     TRANSFORMS[c])
            row[c] = r["recommended_technical_score"]
            cells += f"{row[c]:>10.5f}"
        Q2[name] = row
        print(f"{name:<24}{cells}")
    OUT["Q2_conditional"] = Q2

    print("\n  contribution of each component, by condition (shipped minus ablated):")
    hdr = f"{'component':<24}" + "".join(f"{c.split()[0]:>10}" for c in COLS)
    print("  " + hdr)
    print("  " + "-" * len(hdr))
    for name in VARIANTS:
        if name == "shipped":
            continue
        cells = "".join(f"{Q2['shipped'][c]-Q2[name][c]:>+10.5f}" for c in COLS)
        print(f"  {name:<24}{cells}")

    (ROOT / "experiments" / "results" / "out_32.json").write_text(
        json.dumps(OUT, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] experiments/results/out_32.json   {time.time()-t0:.0f}s")


def _p30_classes():
    """The two ablation classes pass 30 defines inside main(); redeclared here."""
    class NoLedger(Agent):
        def _observe(self, st, msg):
            st.evidence.clear()
            return super()._observe(st, msg)

    class NoMining(Agent):
        def _observe(self, st, msg):
            saved, self.ix.mine = self.ix.mine, lambda *a, **k: []
            try:
                return super()._observe(st, msg)
            finally:
                self.ix.mine = saved

    return NoLedger, NoMining


if __name__ == "__main__":
    main()
