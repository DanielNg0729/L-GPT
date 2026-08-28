"""Experiment 54: validate the coarse study's candidates under the CORRECTED objective.

WHY THE STUDY'S OWN RANKING CANNOT BE TRUSTED HERE. Pass 48 optimised the mean of three
conditions -- public-tune, an unseen synthetic draw, and paraphrase-T1. The organizer's
webinar Q&A has since confirmed there will be NO PARAPHRASING, so a third of that objective
is weight spent on a condition that will never occur. Its argmax is therefore the argmax of
the wrong function, and the top trial is a candidate to test, not a result to adopt.

WHAT REPLACES IT. The conditions that still matter, all of them real:

    public-tune    100 real sessions, the half everything was originally fitted on
    public-hold    100 real sessions never used for fitting
    synth-A / B    800 unseen sessions each, independent draws, review-weighted targets
    uniform-pop    800 unseen sessions, uniform targets    -- population-shift stress
    inverse-pop    800 unseen sessions, inverted targets   -- adversarial population

CALIBRATION NOTE FROM THE SLIDES. The real target pool is 1,406 distinct products, and the
median public target sits at the 99.5th catalogue percentile (6,846 reviews against a
catalogue median of 12). Our review-weighted minting reproduces that almost exactly
(log1p median 8.84 minted vs 8.83 real, verified in pass 21), so synth-A/B are well
calibrated on the dimension that matters -- and if anything HARDER than the private set,
since they draw from ~49,800 products where the real set draws from ~1,206.

The adoption rule is unchanged and pre-registered: adopt only if NO condition regresses.
That rule is what caught IDF_POW, and a 1,600-trial search is exactly the situation it
exists for -- with that many trials some apparent winners are multiple-comparison luck.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/scripts/54_optuna_validate.py --top 6
"""
from __future__ import annotations

import argparse
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
_p48 = __import__("48_optuna_coarse")
mint, SEEDS = _p30.mint, _p30.SEEDS

SHIPPED = {
    "W_CATEGORY": 0.75, "W_MINED": 0.15, "IDF_POW": 0.0, "W_POP": 0.25,
    "MINED_LEN_DIV": 8.0, "POOL": 400, "STRONG_DF": 500, "DF_CAP": 12000,
    "STRONG_CAP": 8, "OR_CAP": 14, "MINE_MAXN": 9, "MINE_MINN": 3,
    "RESOLVE_CAP": 12, "BM_TITLE": 6.0, "BM_CATS": 4.0, "BM_FEAT": 2.5,
    "BM_DETAILS": 2.5, "BM_STORE": 1.5, "BM_DESC": 1.0,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()
    t0 = time.time()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=_p48.STUDY, storage=_p48.STORAGE)
    done = [t for t in study.trials if t.value is not None]
    done.sort(key=lambda t: -t.value)
    print(f"study has {len(done):,} complete trials; validating the top {args.top}\n")

    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    pub_t = {str(s["ground_truth"]["parent_asin"]) for s in samples}
    profiles = [s["user_profile"] for s in samples]

    _p48._G.update(
        Agent=Agent, evaluate=evaluate, cid=cid, cats=cats, prods=prods, base=base,
        tune=[s for i, s in enumerate(samples) if i % 2 == 0],
        para_set=samples[:100], pub_targets=pub_t, profiles=profiles,
        mint=mint, ev_t=None, TR=None,
    )

    SETS = {
        "pub-tune": [s for i, s in enumerate(samples) if i % 2 == 0],
        "pub-hold": [s for i, s in enumerate(samples) if i % 2 == 1],
        "synth-A": mint(prods, pub_t, profiles, "reviews", 800, seed=SEEDS["reviews"]),
        "synth-B": mint(prods, pub_t, profiles, "reviews", 800, seed=987_654),
        "uniform": mint(prods, pub_t, profiles, "uniform", 800, seed=SEEDS["uniform"]),
        "inverse": mint(prods, pub_t, profiles, "inverse", 800, seed=SEEDS["inverse"]),
    }
    COLS = list(SETS)

    def row(params):
        r = {}
        saved = base.ix.BM25
        base.ix.BM25 = (f'bm25(p, 0.0, {params["BM_TITLE"]:.3f}, {params["BM_CATS"]:.3f}, '
                        f'{params["BM_FEAT"]:.3f}, {params["BM_DETAILS"]:.3f}, '
                        f'{params["BM_STORE"]:.3f}, {params["BM_DESC"]:.3f})')
        try:
            for name, sub in SETS.items():
                agent = _p48.build_agent(params)
                agent.tagger = None
                r[name] = evaluate(agent, sub, cid, cats, prods)[
                    "recommended_technical_score"]
        finally:
            base.ix.BM25 = saved
        return r

    ref = row(SHIPPED)
    print(f"{'configuration':<30}" + "".join(f"{c:>10}" for c in COLS) + f"{'WORST':>9}")
    print("-" * (30 + 10 * len(COLS) + 9))
    print(f"{'shipped':<30}" + "".join(f"{ref[c]:>10.5f}" for c in COLS)
          + f"{min(ref.values()):>9.5f}")

    OUT = {"shipped": ref, "candidates": []}
    seen = set()
    rank = 0
    for t in done:
        key = tuple(sorted((k, round(v, 4) if isinstance(v, float) else v)
                           for k, v in t.params.items()))
        if key in seen:
            continue
        seen.add(key)
        rank += 1
        if rank > args.top:
            break
        p = dict(SHIPPED)
        p.update(t.params)
        r = row(p)
        d = {c: r[c] - ref[c] for c in COLS}
        worst = min(d.values())
        verdict = ("ADOPT -- no regression" if worst >= 0 else
                   "inside noise" if worst > -0.005 else "REJECT")
        OUT["candidates"].append({"trial": t.number, "objective": t.value,
                                  "params": t.params, "scores": r, "deltas": d,
                                  "worst_delta": worst, "verdict": verdict})
        print(f"{'trial ' + str(t.number) + f' (obj {t.value:.4f})':<30}"
              + "".join(f"{r[c]:>10.5f}" for c in COLS) + f"{min(r.values()):>9.5f}")

    print(f"\n{'configuration':<30}" + "".join(f"{c:>10}" for c in COLS) + "   verdict")
    print("-" * (30 + 10 * len(COLS) + 12))
    for c in OUT["candidates"]:
        print(f"{'trial ' + str(c['trial']):<30}"
              + "".join(f"{c['deltas'][k]:>+10.5f}" for k in COLS)
              + f"   {c['verdict']}")

    good = [c for c in OUT["candidates"] if c["worst_delta"] >= 0]
    if good:
        best = max(good, key=lambda c: sum(c["deltas"].values()))
        print(f"\n  ADOPTABLE: trial {best['trial']}, worst delta "
              f"{best['worst_delta']:+.5f}, sum {sum(best['deltas'].values()):+.5f}")
        for k, v in sorted(best["params"].items()):
            if SHIPPED.get(k) != v:
                print(f"    {k:<16}{SHIPPED.get(k)}  ->  {v}")
    else:
        print("\n  NOTHING ADOPTABLE: every candidate regresses on at least one condition.")
        print("  Expected outcome for a search whose objective included a condition that")
        print("  will not occur -- the study optimised partly for the wrong thing.")

    (ROOT / "experiments" / "results" / "out_54.json").write_text(
        json.dumps(OUT, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n[saved] experiments/results/out_54.json   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
