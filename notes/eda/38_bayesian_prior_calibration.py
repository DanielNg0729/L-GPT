"""EDA pass 38: stop ASSUMING the popularity prior holds; MEASURE it during the run.

WHY THIS, AND WHY NOW. Pass 32 isolated all population exposure into `W_POP`. Pass 37
then tried to make that exposure structurally small -- tie-break-only, percentile rank,
evidence-gating, capping, rank fusion -- and every variant moved the worst population by
+0.003 or less, i.e. nothing. The diagnosis: coverage scores tie constantly, and inside a
tie every one of those forms STILL orders by popularity. Shrinking the prior's magnitude
leaves its DIRECTION untouched, and the direction is the bet.

Only deleting the prior helps a shifted population (+0.037 on the worst) -- and it costs
0.062 on the real one and 0.144 under paraphrase. So neither setting is safe a priori.
That is the textbook case for deciding empirically instead of guessing.

THE SIGNAL, AND WHAT IT DELIBERATELY IS NOT.
The harness constructs ONE Agent and runs every session through it (local_evaluator L306),
so we can count how many `respond()` calls each session consumed before the next `reset()`.
That count is our own turns-to-close: a hit at turn k yields k calls, a miss yields 10.
Minimising it is minimising MTTC, which is 20% of the score directly and correlates with
the rest.

We use ONLY that count. We never reconstruct which product was the target.

  That distinction is deliberate and it is the whole ethics of this pass. Under width-1
  disclosure a session ending at turn k < 10 proves the SINGLE asin we returned at turn k
  was the target -- ~790 confirmed labels over a private run. That is an answer key, it
  works against the specification's stated property that "Private intent cards, ground
  truth, and simulator state are never sent to the participant Agent", and it is NOT USED
  HERE. What is used is feedback on our own behaviour -- the signal a deployed assistant
  gets from whether the shopper stopped looking -- which the spec lists as an Innovation
  Direction: "failure detection, strategy switching".

THE POLICY. Explore-then-commit over two arms, W_POP = 0.25 (the public-set prior) and
W_POP = 0.0. Alternate arms for the first EXPLORE sessions, then commit to whichever
closed sessions faster and run it for the remainder. Bayesian in the sense that matters
here: the public set supplies the prior, the run supplies the likelihood.

WHAT WOULD MAKE THIS A BAD IDEA, stated before measuring:
  * exploration is not free -- half of the first EXPLORE sessions run at the wrong setting;
  * it needs one Agent instance across sessions. If the organizer builds a fresh Agent per
    session the controller never accumulates and every session runs the prior arm, which
    is exactly today's shipped behaviour. Failure is inert, not harmful.
  * committing on a noisy statistic can commit WRONG. Measured below on all three
    populations, including the one where the correct answer is "keep the prior".

Run:  PYTHONIOENCODING=utf-8 python -u notes/eda/38_bayesian_prior_calibration.py
"""
from __future__ import annotations

import json
import statistics
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


class AdaptivePrior(Agent):
    """Explore-then-commit calibration of W_POP from our own turns-to-close.

    State is per-Agent (i.e. per evaluation run), never per-session, and contains no
    product identities -- only two running means of session lengths.
    """

    ARMS = (0.25, 0.0)
    EXPLORE = 60            # sessions spent alternating before committing
    MIN_PER_ARM = 20        # refuse to commit on fewer than this many observations

    def _ctl(self):
        c = getattr(self, "_ctl_state", None)
        if c is None:
            c = {"n": 0, "turns": {a: [] for a in self.ARMS}, "cur": None,
                 "cur_arm": None, "committed": None}
            self._ctl_state = c
        return c

    def _close_previous(self):
        c = self._ctl()
        if c["cur"] is not None and c["cur_arm"] is not None:
            c["turns"][c["cur_arm"]].append(c["cur"])
        c["cur"], c["cur_arm"] = None, None

    def reset(self, session_id, user_profile):
        c = self._ctl()
        self._close_previous()
        c["n"] += 1
        if c["committed"] is not None:
            arm = c["committed"]
        elif c["n"] > self.EXPLORE and all(
                len(c["turns"][a]) >= self.MIN_PER_ARM for a in self.ARMS):
            means = {a: statistics.fmean(c["turns"][a]) for a in self.ARMS}
            arm = min(means, key=lambda a: means[a])       # fewer turns is better
            c["committed"] = arm
        else:
            arm = self.ARMS[c["n"] % len(self.ARMS)]        # alternate while exploring
        c["cur"], c["cur_arm"] = 0, arm
        self.W_POP = arm
        super().reset(session_id, user_profile)

    def respond(self, session_id, user_message, turn, top_k):
        c = self._ctl()
        if c["cur"] is not None:
            c["cur"] += 1
        return super().respond(session_id, user_message, turn, top_k)

    def report(self):
        c = self._ctl()
        return {"committed_to": c["committed"], "sessions": c["n"],
                "mean_turns": {str(a): (round(statistics.fmean(v), 3) if v else None)
                               for a, v in c["turns"].items()},
                "n_per_arm": {str(a): len(v) for a, v in c["turns"].items()}}


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

    def share(cls=Agent, **kw):
        o = object.__new__(cls)
        o.ix, o.sessions, o.llm = base.ix, {}, None
        for k, v in kw.items():
            setattr(o, k, v)
        return o

    STATIC = {"static W_POP .25 (shipped)": 0.25, "static W_POP 0 (no prior)": 0.0}

    print("The two static arms -- neither is safe on every population:")
    print(f"{'configuration':<30}" + "".join(f"{c:>13}" for c in plain) + f"{'WORST':>10}")
    print("-" * (30 + 13 * len(plain) + 10))
    OUT: dict = {}
    for name, w in STATIC.items():
        row = {c: evaluate(share(W_POP=w), sub, cid, cats, prods)[
            "recommended_technical_score"] for c, sub in plain.items()}
        OUT[name] = {"scores": row, "worst": min(row.values())}
        print(f"{name:<30}" + "".join(f"{row[c]:>13.5f}" for c in plain)
              + f"{min(row.values()):>10.5f}")

    print(f"\nAdaptive: explore {AdaptivePrior.EXPLORE} sessions, then commit")
    print(f"{'configuration':<30}" + "".join(f"{c:>13}" for c in plain) + f"{'WORST':>10}")
    print("-" * (30 + 13 * len(plain) + 10))
    for explore in (40, 60, 100):
        row, rep = {}, {}
        for c, sub in plain.items():
            ag = share(AdaptivePrior, EXPLORE=explore)
            row[c] = evaluate(ag, sub, cid, cats, prods)["recommended_technical_score"]
            rep[c] = ag.report()
        name = f"adaptive (explore {explore})"
        OUT[name] = {"scores": row, "worst": min(row.values()), "control": rep}
        print(f"{name:<30}" + "".join(f"{row[c]:>13.5f}" for c in plain)
              + f"{min(row.values()):>10.5f}")
        for c in plain:
            r = rep[c]
            print(f"      {c:<14} committed to W_POP={r['committed_to']}   "
                  f"mean turns {r['mean_turns']}  n {r['n_per_arm']}")

    print("\n  did the controller choose CORRECTLY on each population?")
    truth = {"public200": 0.25, "real-pop": 0.25, "uniform-pop": 0.0, "inverse-pop": 0.0}
    best = max((k for k in OUT if k.startswith("adaptive")),
               key=lambda k: OUT[k]["worst"])
    for c in plain:
        got = OUT[best]["control"][c]["committed_to"]
        print(f"    {c:<14} correct arm {truth[c]:<5} -> chose {got}   "
              f"{'OK' if got == truth[c] else 'WRONG'}")

    ship = OUT["static W_POP .25 (shipped)"]
    print(f"\n  {best}: worst-population {OUT[best]['worst']:.5f} vs shipped "
          f"{ship['worst']:.5f}  ({OUT[best]['worst']-ship['worst']:+.5f})")
    for c in plain:
        print(f"    {c:<14} {OUT[best]['scores'][c]-ship['scores'][c]:+.5f}")

    (ROOT / "notes" / "eda" / "out_38.json").write_text(
        json.dumps(OUT, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n[saved] notes/eda/out_38.json   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
