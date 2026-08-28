"""EDA pass 42: raise the OFFLINE paraphrase floor before betting on a network component.

An LLM extraction channel is paraphrase insurance that only pays out if the organizer
leaves network access on -- and submission_rules.md says "For official final scoring,
organizer policy may disable network access." Whatever the LLM adds, the floor beneath it
is catalogue-grounded n-gram mining, which is what already holds the paraphrased score at
0.845 / 0.838 instead of 0.217 / 0.164 (pass 32).

That floor has NEVER BEEN TUNED. `mine()` ships with maxn=9, minn=3 and DF_CAP=4000, all
chosen when mining was a fallback nobody had measured under stress. `notes/hyperparams.md`
flags maxn/minn as the highest-value untuned knobs in the registry for exactly this reason:
mining is the paraphrase floor, and its two governing constants were never swept against
paraphrase.

minn is the interesting one. At minn=3 a paraphrased message must contain a surviving
3-token catalogue n-gram to yield ANY evidence. Dropping to 2 admits far more evidence and
far more noise -- an empirical question, not a theoretical one.

The rule stays what it has been all project: adopt only if NOTHING regresses, and the clean
score is the first column checked.

Run:  PYTHONIOENCODING=utf-8 python -u notes/eda/42_mining_paraphrase_floor.py
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


def make(minn: int, maxn: int, df_cap: int | None):
    class Tuned(Agent):
        MINE_MINN = minn
        MINE_MAXN = maxn

        def _observe(self, st, msg):
            ix = self.ix
            orig_mine, orig_cap = ix.mine, ix.DF_CAP
            if df_cap is not None:
                ix.DF_CAP = df_cap
            ix.mine = lambda text, maxn=self.MINE_MAXN, minn=self.MINE_MINN: \
                orig_mine(text, maxn=maxn, minn=minn)
            try:
                return super()._observe(st, msg)
            finally:
                ix.mine, ix.DF_CAP = orig_mine, orig_cap
    return Tuned


def main() -> None:
    t0 = time.time()
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    pub_t = {str(s["ground_truth"]["parent_asin"]) for s in samples}
    profiles = [s["user_profile"] for s in samples]
    unseen = mint(prods, pub_t, profiles, "reviews", 800, seed=SEEDS["reviews"])

    def share(cls):
        o = object.__new__(cls)
        o.ix, o.sessions, o.llm = base.ix, {}, None
        return o

    VARIANTS = {
        "shipped (minn 3, maxn 9, cap 4k)": (3, 9, None),
        "minn 2":                           (2, 9, None),
        "minn 2, maxn 12":                  (2, 12, None),
        "minn 4":                           (4, 9, None),
        "maxn 6":                           (3, 6, None),
        "maxn 12":                          (3, 12, None),
        "cap 12000":                        (3, 9, 12000),
        "cap 1500":                         (3, 9, 1500),
        "minn 2 + cap 12000":               (2, 9, 12000),
    }
    PARA = ["T1 scaffold reworded", "T2 scaffold stripped", "T5 realistic (T1+T3)"]
    COLS = ["clean", "unseen800"] + PARA

    print(f"{'variant':<34}" + "".join(f"{c.split()[0]:>12}" for c in COLS) + f"{'WORST-PARA':>12}")
    print("-" * (34 + 12 * len(COLS) + 12))
    OUT, ref = {}, None
    for name, (minn, maxn, cap) in VARIANTS.items():
        cls = make(minn, maxn, cap)
        row = {"clean": evaluate(share(cls), samples, cid, cats, prods)[
            "recommended_technical_score"],
            "unseen800": evaluate(share(cls), unseen, cid, cats, prods)[
            "recommended_technical_score"]}
        for t in PARA:
            row[t] = evaluate_transformed(share(cls), samples, cid, cats, prods,
                                          TRANSFORMS[t])["recommended_technical_score"]
        wp = min(row[t] for t in PARA)
        OUT[name] = {"scores": row, "worst_para": wp, "cfg": [minn, maxn, cap]}
        if ref is None:
            ref = OUT[name]
        print(f"{name:<34}" + "".join(f"{row[c]:>12.5f}" for c in COLS) + f"{wp:>12.5f}")

    print(f"\ndeltas vs shipped (clean must not regress -- that is the whole constraint)")
    print(f"{'variant':<34}" + "".join(f"{c.split()[0]:>12}" for c in COLS) + "   verdict")
    print("-" * (34 + 12 * len(COLS) + 12))
    for name, v in OUT.items():
        d = {c: v["scores"][c] - ref["scores"][c] for c in COLS}
        if v is ref:
            verdict = "reference"
        elif d["clean"] < -1e-9 or d["unseen800"] < -0.005:
            verdict = "REJECT (clean/unseen regressed)"
        elif v["worst_para"] - ref["worst_para"] > 0.005:
            verdict = "ADOPT -- raises the floor for free"
        else:
            verdict = "no material gain"
        print(f"{name:<34}" + "".join(f"{d[c]:>+12.5f}" for c in COLS) + f"   {verdict}")

    (ROOT / "notes" / "eda" / "out_42.json").write_text(
        json.dumps(OUT, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] notes/eda/out_42.json   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
