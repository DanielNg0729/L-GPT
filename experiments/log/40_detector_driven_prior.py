"""Experiment 40: set the popularity prior from a DIRECTLY OBSERVED population statistic.

THE ROUTE THAT FAILED, AND WHY -- so the choice here is justified rather than asserted.

Passes 38/39 tried explore-then-commit bandits over W_POP in {0.25, 0.0}, rewarded first by
turns-to-close (mis-aligned: it prefers the arm that wins MTTC while losing HitRate, which
carries 2.5x the weight) and then by observed per-session score. Both misread the uniform
population. The reason is sample size, and it is worth stating numerically because it rules
out the whole family:

    per-session observed value has sd ~ 0.30 (it is ~0.3 for a long session, ~1.0 for an
    early hit), while the DIFFERENCE between the two arms is ~0.013 in the mean.
    n per arm for 80% power at alpha .05  =  16 * (0.30/0.013)^2  ~  8,500.

With 800 private sessions -- 400 per arm at best -- an outcome-feedback bandit cannot
resolve a difference this small. It is not a tuning problem; the estimator is 20x short.

THE ROUTE THAT WORKS. Do not infer the population from our own outcomes. Observe it.

Pass 39 measured four label-free per-session statistics. `pool_pop` -- the mean popularity
of the products OUR OWN retrieval returns for the customer's constraints -- separates the
populations at Cohen's d = 0.70:

    public200 3.17   real-pop 3.12   uniform-pop 2.78   inverse-pop 2.65

Pass 39's verdict line called that "weak", which applied a per-SESSION classification
threshold to what is an AGGREGATE question. For estimating a population mean over n
sessions the separation is d*sqrt(n/2): n=50 gives z=3.5, n=100 gives z=4.9. Decisive, and
~20x more sample-efficient than the bandit. The earlier verdict was wrong; this is the
correction.

Two properties make this legitimate and safe:
  * NO TARGET IS EVER READ. `pool_pop` is a property of our own retrieval output for the
    customer's own words. The specification's "ground truth ... never sent to the
    participant Agent" is untouched.
  * NO CIRCULARITY. The pool comes from FTS5/BM25 in `_candidates()`, which does not use
    W_POP. The statistic cannot be moved by the parameter it sets.

THE MAPPING is graded, not a switch, so a mis-estimate degrades smoothly:

    W_POP_eff = W_POP_max * clip((observed - LO) / (HI - LO), 0, 1)

with LO/HI calibrated offline. HI is anchored on the PUBLIC 200 -- real organizer data, not
our minting -- which is what keeps the calibration honest.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/scripts/40_detector_driven_prior.py
"""
from __future__ import annotations

import json
import statistics
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


class DetectorPrior(Agent):
    """W_POP scaled by an observed, label-free estimate of the target population.

    Per session we record the mean popularity of our own retrieved pool. Once WARMUP
    sessions have been seen, the running mean maps to a prior weight. Before that we use
    the public-set prior unchanged, so a single-session harness behaves exactly as today.
    """

    W_POP_MAX = 0.25
    LO = 2.70          # observed pool_pop where popularity carries no signal
    HI = 3.10          # observed pool_pop matching the public set / real population
    WARMUP = 40        # sessions of evidence before the estimate is trusted
    SAMPLE_TURN = 3    # record the statistic once per session, at this turn

    def _det(self):
        d = getattr(self, "_det_state", None)
        if d is None:
            d = {"obs": [], "sampled": set()}
            self._det_state = d
        return d

    def _w_pop_effective(self) -> float:
        d = self._det()
        if len(d["obs"]) < self.WARMUP:
            return self.W_POP_MAX
        m = statistics.fmean(d["obs"])
        frac = (m - self.LO) / (self.HI - self.LO)
        return self.W_POP_MAX * max(0.0, min(1.0, frac))

    def _candidates(self, st, message):
        pool = super()._candidates(st, message)
        d = self._det()
        key = id(st)
        if st.turn >= self.SAMPLE_TURN and key not in d["sampled"] and pool:
            d["sampled"].add(key)
            head = pool[:100]
            d["obs"].append(statistics.fmean(self.ix.pop.get(a, 0.0) for a in head))
        return pool

    def _rank(self, st, pool, top_k):
        w = self._w_pop_effective()
        saved = self.W_POP
        self.W_POP = w
        try:
            return super()._rank(st, pool, top_k)
        finally:
            self.W_POP = saved

    def report(self):
        d = self._det()
        return {"n_obs": len(d["obs"]),
                "mean_pool_pop": round(statistics.fmean(d["obs"]), 4) if d["obs"] else None,
                "final_W_POP": round(self._w_pop_effective(), 4)}


def main() -> None:
    t0 = time.time()
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    pub_t = {str(s["ground_truth"]["parent_asin"]) for s in samples}
    profiles = [s["user_profile"] for s in samples]

    pops = {
        "public200":   samples,
        "real-pop":    mint(prods, pub_t, profiles, "reviews", 800, seed=SEEDS["reviews"]),
        "uniform-pop": mint(prods, pub_t, profiles, "uniform", 800, seed=SEEDS["uniform"]),
        "inverse-pop": mint(prods, pub_t, profiles, "inverse", 800, seed=SEEDS["inverse"]),
    }

    def share(cls=Agent, **kw):
        o = object.__new__(cls)
        o.ix, o.sessions, o.llm = base.ix, {}, None
        for k, v in kw.items():
            setattr(o, k, v)
        return o

    print("  bandit power check: n per arm needed to resolve the arm difference")
    sd, eff = 0.30, 0.013
    print(f"    per-session value sd ~{sd}, arm difference ~{eff} "
          f"-> n/arm ~ {16*(sd/eff)**2:,.0f}  (we have <=400)")

    OUT: dict = {}
    COLS = list(pops) + ["para-T1"]
    print(f"\n{'configuration':<30}" + "".join(f"{c:>13}" for c in COLS) + f"{'WORST-POP':>12}")
    print("-" * (30 + 13 * len(COLS) + 12))

    def row_for(cls, **kw):
        r = {}
        for p, sub in pops.items():
            ag = share(cls, **kw)
            res = evaluate(ag, sub, cid, cats, prods)
            r[p] = res["recommended_technical_score"]
            if hasattr(ag, "report"):
                r.setdefault("_rep", {})[p] = ag.report()
        ag = share(cls, **kw)
        r["para-T1"] = evaluate_transformed(
            ag, samples, cid, cats, prods,
            TRANSFORMS["T1 scaffold reworded"])["recommended_technical_score"]
        return r

    for name, (cls, kw) in {
        "static .25 (shipped)":   (Agent, {"W_POP": 0.25}),
        "static 0 (no prior)":    (Agent, {"W_POP": 0.0}),
        "detector (warmup 40)":   (DetectorPrior, {}),
        "detector (warmup 80)":   (DetectorPrior, {"WARMUP": 80}),
    }.items():
        r = row_for(cls, **kw)
        worst = min(r[p] for p in pops)
        OUT[name] = {"scores": {k: v for k, v in r.items() if k != "_rep"},
                     "worst_pop": worst, "control": r.get("_rep")}
        print(f"{name:<30}" + "".join(f"{r[c]:>13.5f}" for c in COLS) + f"{worst:>12.5f}")
        if r.get("_rep"):
            for p in pops:
                d = r["_rep"][p]
                print(f"      {p:<13} observed pool_pop {d['mean_pool_pop']}  "
                      f"-> W_POP {d['final_W_POP']}")

    ship = OUT["static .25 (shipped)"]
    print(f"\n  deltas vs shipped:")
    for name, v in OUT.items():
        if name.startswith("static .25"):
            continue
        d = {c: v["scores"][c] - ship["scores"][c] for c in COLS}
        print(f"    {name:<26}" + "".join(f"{d[c]:>+13.5f}" for c in COLS)
              + f"   worst {v['worst_pop']-ship['worst_pop']:+.5f}")

    (ROOT / "experiments" / "results" / "out_40.json").write_text(
        json.dumps(OUT, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n[saved] experiments/results/out_40.json   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
