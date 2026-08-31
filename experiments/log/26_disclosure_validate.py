"""Experiment 26: validate disclosure schedules tune/held-out, and test an ADAPTIVE policy.

Pass 25 solved a schedule by backward induction and pass 25b measured several through the
official evaluator, but on all 200 sessions -- which is exactly the protocol violation
every other pass avoids. This splits them.

It also tests the policy the scoring function actually implies. The optimal width depends
on the CONTINUATION value: how much better the ranking will get if we wait. That is a
function of STATE, not of the turn number:

  * probes still paying out  -> more evidence is coming -> ranking will improve -> stay narrow
  * evidence exhausted       -> waiting gains nothing   -> widen NOW, take the hit

A turn-indexed schedule only approximates that. An adaptive rule reads it directly, and it
is what the brief's Pillar II describes: "trigger an immediate retrieval cutoff when facing
Over-Generality (candidate pool overload)".

Run:  PYTHONIOENCODING=utf-8 python -u experiments/log/26_disclosure_validate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent, SessionState  # noqa: E402

SCHEDULES = {
    "current: 10 every turn":      (10,) * 10,
    "widen 2,3,4..10":             (2, 3, 4, 5, 6, 7, 8, 9, 10, 10),
    "widen 1,2,3..10":             (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    "widen 1,1,2,3,4,5,6,8,9,10":  (1, 1, 2, 3, 4, 5, 6, 8, 9, 10),
    "slow 1,1,1,2,2,3,4,6,8,10":   (1, 1, 1, 2, 2, 3, 4, 6, 8, 10),
    "slower 1,1,1,1,2,2,3,5,8,10": (1, 1, 1, 1, 2, 2, 3, 5, 8, 10),
    "narrow 1x9 then 10":          (1,) * 9 + (10,),
}


class Adaptive(Agent):
    """Width driven by whether evidence is still arriving, not by turn index.

    While a probe is still yielding new constraints the ranking will sharpen next turn, so
    the continuation value is high and we stay narrow. Once the evidence stops growing,
    waiting buys nothing and we widen -- fast, because MTTC is still ticking.
    """
    STALL_WIDTHS = (1, 2, 4, 7, 10)     # width after N consecutive no-gain turns
    CONFIDENT = 1                        # width while evidence is still arriving

    def respond(self, session_id, user_message, turn, top_k):
        st = self.sessions.get(session_id)
        before = len(st.evidence) if st else 0
        r = super().respond(session_id, user_message, turn, top_k)
        st = self.sessions.get(session_id)
        if st is None:
            return r
        gained = len(st.evidence) - before
        stall = getattr(st, "_stall", 0)
        stall = 0 if gained > 0 else stall + 1
        try:
            st._stall = stall
        except AttributeError:                     # __slots__ -- track on the agent
            self._stalls = getattr(self, "_stalls", {})
            self._stalls[session_id] = stall
            stall = self._stalls[session_id]
        w = (self.CONFIDENT if stall == 0
             else self.STALL_WIDTHS[min(stall, len(self.STALL_WIDTHS)) - 1])
        # never let the last turns pass without a full list -- protects HitRate
        if turn >= 9:
            w = 10
        r["recommendations"] = r["recommendations"][:max(1, min(w, top_k))]
        return r


def main() -> None:
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    TUNE = [s for i, s in enumerate(samples) if i % 2 == 0]
    HOLD = [s for i, s in enumerate(samples) if i % 2 == 1]
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")

    def share(cls=Agent, **kw):
        o = object.__new__(cls)
        o.ix, o.sessions, o.llm = base.ix, {}, None
        for k, v in kw.items():
            setattr(o, k, v)
        return o

    def run(ag, subset):
        r = evaluate(ag, subset, cid, cats, prods)
        return {"score": r["recommended_technical_score"], "hr": r["hit_rate_at_10"],
                "mrr": r["mrr"], "mttc": r["mttc"]}

    print(f"{'policy':<32}{'tune':>9}{'hold':>9}{'delta':>9}{'HR':>7}{'MRR':>8}{'MTTC':>7}")
    print("-" * 82)
    OUT = {}
    base_t = run(share(DISCLOSURE=(10,) * 10), TUNE)
    base_h = run(share(DISCLOSURE=(10,) * 10), HOLD)
    for name, sched in SCHEDULES.items():
        t = run(share(DISCLOSURE=sched), TUNE)
        h = run(share(DISCLOSURE=sched), HOLD)
        OUT[name] = {"tune": t, "hold": h, "schedule": list(sched)}
        print(f"{name:<32}{t['score']:>9.5f}{h['score']:>9.5f}"
              f"{h['score']-base_h['score']:>+9.5f}{h['hr']:>7.1%}{h['mrr']:>8.4f}{h['mttc']:>7.2f}")

    t = run(share(Adaptive, DISCLOSURE=(10,) * 10), TUNE)
    h = run(share(Adaptive, DISCLOSURE=(10,) * 10), HOLD)
    OUT["ADAPTIVE (evidence-stall)"] = {"tune": t, "hold": h}
    print(f"{'ADAPTIVE (evidence-stall)':<32}{t['score']:>9.5f}{h['score']:>9.5f}"
          f"{h['score']-base_h['score']:>+9.5f}{h['hr']:>7.1%}{h['mrr']:>8.4f}{h['mttc']:>7.2f}")

    best = max(OUT, key=lambda k: OUT[k]["hold"]["score"])
    bt, bh = OUT[best]["tune"]["score"], OUT[best]["hold"]["score"]
    print(f"\n  best on HELD-OUT: {best}")
    print(f"    tune {bt:.5f}  hold {bh:.5f}  "
          f"(both above baseline {base_t['score']:.5f}/{base_h['score']:.5f}? "
          f"{'YES' if bt > base_t['score'] and bh > base_h['score'] else 'NO'})")
    print("  a real effect must win on BOTH halves -- that is the whole point of the split")

    (ROOT / "experiments" / "results" / "out_26.json").write_text(
        json.dumps(OUT, indent=2) + "\n", encoding="utf-8")
    print("\n[saved] experiments/results/out_26.json")


if __name__ == "__main__":
    main()
