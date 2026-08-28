"""EDA pass 39: detect the target population WITHOUT ever reading a target.

TWO REPAIRS AND ONE NEW IDEA.

(1) THE BANDIT'S REWARD WAS WRONG.  Pass 38's controller minimised turns-to-close. On the
    uniform population that picked the WRONG arm: the prior closes faster (3.548 vs 3.850
    mean turns) while scoring worse (0.8836 vs 0.8993), because it wins on MTTC and loses
    on HitRate -- and HitRate carries 0.50 of the score against Efficiency's 0.20.
    Fix: reward each session with the score it actually earned, as far as we can observe it.
    Under width-1 disclosure a session ending at turn k < 10 is a hit at rank 1, worth
        0.50 + 0.30 + 0.20*(11-k)/10.
    A session consuming all 10 calls is a turn-10 hit OR a miss -- indistinguishable from
    call count alone -- so it is scored with a single constant for both arms. That constant
    cannot bias the COMPARISON, which is all the controller needs.

(2) A LABEL-FREE POPULATION DETECTOR.  The alternative to a bandit is to read the
    population off the messages themselves. This never touches a target's identity.

    The rationale: every constraint is lifted from the target, so the messages carry the
    target's fingerprint. Popular products are generic -- mass-market tees, boilerplate
    fabric text -- so their constraints match thousands of catalogue items. Obscure products
    carry distinctive text and match few. So per session we can observe, with no labels:

        pool_size     how many products satisfy the accumulated evidence
        mean_df       document frequency of the evidence phrases
        n_evidence    how many phrases were extracted at all
        pool_pop      mean popularity of the retrieved pool

    If those separate the three populations, a running estimate of them is a legitimate
    posterior over "which population am I in", updated from our own observations -- exactly
    the Bayesian update we want, with none of the answer-key access.

    THIS PASS ONLY ASKS WHETHER THEY SEPARATE. Building a controller on an observable that
    does not discriminate would be worse than useless, so separability is measured first,
    with the distributions printed so the decision is inspectable rather than asserted.

Run:  PYTHONIOENCODING=utf-8 python -u notes/eda/39_population_detector.py
"""
from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notes" / "eda"))

from evaluator.local_evaluator import (  # noqa: E402
    behavior_for, catalog_index, coarse_category, customer_reply, evaluate,
    initial_message, intent_card, load_jsonl,
)
from submission.agent import Agent  # noqa: E402

_p30 = __import__("30_robustness_benchmark")
mint, SEEDS = _p30.mint, _p30.SEEDS

PROBE_TURNS = 4        # how many turns of evidence the detector is allowed to see


# ======================================================= (1) score-aligned bandit
class AdaptivePriorV2(Agent):
    """Explore-then-commit on OBSERVED SCORE rather than on turns.

    Uses only the number of respond() calls a session consumed -- never any target's
    identity. See the module docstring for why that quantity is the honest one.
    """

    ARMS = (0.25, 0.0)
    EXPLORE = 60
    MIN_PER_ARM = 20
    LONG_SESSION_VALUE = 0.30      # constant for any session that used all 10 calls

    def _ctl(self):
        c = getattr(self, "_ctl_state", None)
        if c is None:
            c = {"n": 0, "val": {a: [] for a in self.ARMS}, "cur": None,
                 "cur_arm": None, "committed": None}
            self._ctl_state = c
        return c

    @classmethod
    def _session_value(cls, calls: int) -> float:
        if calls < 10:                       # hit at turn `calls`, rank 1 under width-1
            return 0.50 + 0.30 + 0.20 * (11 - calls) / 10.0
        return cls.LONG_SESSION_VALUE

    def _close_previous(self):
        c = self._ctl()
        if c["cur"] is not None and c["cur_arm"] is not None:
            c["val"][c["cur_arm"]].append(self._session_value(c["cur"]))
        c["cur"], c["cur_arm"] = None, None

    def reset(self, session_id, user_profile):
        c = self._ctl()
        self._close_previous()
        c["n"] += 1
        if c["committed"] is not None:
            arm = c["committed"]
        elif c["n"] > self.EXPLORE and all(
                len(c["val"][a]) >= self.MIN_PER_ARM for a in self.ARMS):
            means = {a: statistics.fmean(c["val"][a]) for a in self.ARMS}
            arm = max(means, key=lambda a: means[a])          # higher value is better
            c["committed"] = arm
        else:
            arm = self.ARMS[c["n"] % len(self.ARMS)]
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
        return {"committed_to": c["committed"],
                "mean_value": {str(a): (round(statistics.fmean(v), 4) if v else None)
                               for a, v in c["val"].items()},
                "n_per_arm": {str(a): len(v) for a, v in c["val"].items()}}


# ======================================================= (2) label-free observables
def observe(agent, sample, prods, cats):
    """Replay a session's first PROBE_TURNS turns and record LABEL-FREE statistics.

    The target asin is needed to drive the simulator (the harness does the same), but
    nothing about it is recorded -- only properties of the messages and of what OUR OWN
    retrieval returned for them.
    """
    tgt = str(sample["ground_truth"]["parent_asin"])
    card = intent_card(prods[tgt])
    rng = random.Random(f"{sample['sample_id']}\0{sample['scenario_type']}")
    eff = {**sample, "intent_card": card,
           "behavior": behavior_for(str(sample["scenario_type"]), card, rng)}
    ov = eff.get("behavior", {}).get("override") or {}
    sid = f"det_{sample['sample_id']}"
    agent.reset(sid, sample["user_profile"])
    st = agent.sessions[sid]
    disclosed, bu = set(), False
    applied = sample["scenario_type"] != "intent_override"
    msg = initial_message(eff, coarse_category(cats.get(tgt, [])), disclosed)

    pool = []
    for turn in range(1, PROBE_TURNS + 1):
        st.turn += 1
        agent._observe(st, msg)
        pool = agent._candidates(st, msg)
        probe = agent._next_probe(st)
        st.asked.append(probe)
        if not applied and turn + 1 == int(ov.get("turn", 3)):
            applied = True
            nv = str(ov.get("new_value", ""))
            if nv:
                disclosed.add(nv)
            msg = str(ov.get("message", ""))
        else:
            msg, bu = customer_reply(eff, probe, disclosed, bu)

    dfs = [df for _p, (df, _t) in st.evidence.items()]
    pops = [agent.ix.pop.get(a, 0.0) for a in pool[:100]]
    out = {
        "pool_size": float(len(pool)),
        "mean_df": float(statistics.fmean(dfs)) if dfs else 0.0,
        "n_evidence": float(len(st.evidence)),
        "pool_pop": float(statistics.fmean(pops)) if pops else 0.0,
    }
    agent.sessions.pop(sid, None)
    return out


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

    OUT: dict = {}

    # ---------------------------------------------------------- detector separability
    print("=" * 88)
    print(f"LABEL-FREE OBSERVABLES -- do they separate the populations? "
          f"(first {PROBE_TURNS} turns, 250 sessions each)")
    print("=" * 88)
    KEYS = ["pool_size", "mean_df", "n_evidence", "pool_pop"]
    stats: dict[str, dict] = {}
    for pname, sub in pops.items():
        agent = share()
        rows = [observe(agent, s, prods, cats) for s in sub[:250]]
        stats[pname] = {k: {"mean": statistics.fmean(r[k] for r in rows),
                            "median": statistics.median(r[k] for r in rows),
                            "sd": statistics.pstdev([r[k] for r in rows])} for k in KEYS}
    print(f"{'observable':<14}" + "".join(f"{p:>17}" for p in pops))
    print("-" * (14 + 17 * len(pops)))
    for k in KEYS:
        print(f"{k:<14}" + "".join(f"{stats[p][k]['mean']:>17.2f}" for p in pops))
    OUT["detector_stats"] = stats

    print(f"\n  separation between real-pop and uniform-pop, in pooled SDs (Cohen's d):")
    usable = []
    for k in KEYS:
        a, b = stats["real-pop"][k], stats["uniform-pop"][k]
        sd = ((a["sd"] ** 2 + b["sd"] ** 2) / 2) ** 0.5
        d = abs(a["mean"] - b["mean"]) / sd if sd > 1e-9 else 0.0
        verdict = ("USABLE" if d > 0.8 else "weak" if d > 0.3 else "no signal")
        if d > 0.8:
            usable.append(k)
        print(f"    {k:<14} d = {d:>6.2f}   {verdict}")
    OUT["usable_observables"] = usable
    print(f"\n  -> {'detector is viable on: ' + ', '.join(usable) if usable else 'NO observable separates the populations; a detector is not available'}")

    # ---------------------------------------------------------- fixed-reward bandit
    print("\n" + "=" * 88)
    print("SCORE-ALIGNED BANDIT (pass 38's reward bug fixed)")
    print("=" * 88)
    truth = {"public200": 0.25, "real-pop": 0.25, "uniform-pop": 0.0, "inverse-pop": 0.0}
    print(f"{'configuration':<28}" + "".join(f"{p:>13}" for p in pops) + f"{'WORST':>10}")
    print("-" * (28 + 13 * len(pops) + 10))
    for name, w in (("static .25 (shipped)", 0.25), ("static 0 (no prior)", 0.0)):
        row = {p: evaluate(share(W_POP=w), sub, cid, cats, prods)[
            "recommended_technical_score"] for p, sub in pops.items()}
        OUT[name] = {"scores": row, "worst": min(row.values())}
        print(f"{name:<28}" + "".join(f"{row[p]:>13.5f}" for p in pops)
              + f"{min(row.values()):>10.5f}")

    for explore in (60, 120):
        row, rep = {}, {}
        for p, sub in pops.items():
            ag = share(AdaptivePriorV2, EXPLORE=explore)
            row[p] = evaluate(ag, sub, cid, cats, prods)["recommended_technical_score"]
            rep[p] = ag.report()
        name = f"adaptive v2 (explore {explore})"
        OUT[name] = {"scores": row, "worst": min(row.values()), "control": rep}
        print(f"{name:<28}" + "".join(f"{row[p]:>13.5f}" for p in pops)
              + f"{min(row.values()):>10.5f}")
        for p in pops:
            r = rep[p]
            ok = "OK" if r["committed_to"] == truth[p] else "WRONG"
            print(f"      {p:<13} chose {str(r['committed_to']):<5} (correct {truth[p]})"
                  f"  {ok:<5} values {r['mean_value']}")

    (ROOT / "notes" / "eda" / "out_39.json").write_text(
        json.dumps(OUT, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n[saved] notes/eda/out_39.json   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
