"""Experiment 28: autopsy of the remaining HitRate misses.

HR@10 is 99.0% -- two sessions out of 200 never surface the target. This pass answers,
for each of them, WHICH LAYER failed, because the fix is completely different per layer:

    retrieval   the target never entered the candidate pool  -> Layer 4 problem
    ranking     it entered the pool but never reached the disclosed width -> Layer 5
    disclosure  it was inside the top-10 but outside the width we showed -> policy
    gating      intent_override withheld hits until too late  -> unfixable by us

It also re-measures HR under the full-10 policy on the SAME sessions, which is the direct
test of whether sequential disclosure costs any recall at all.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/scripts/28_miss_autopsy.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    behavior_for, catalog_index, coarse_category, customer_reply, evaluate,
    initial_message, intent_card, load_jsonl,
)
from submission.agent import Agent  # noqa: E402

MAXT = 10


def trace(agent: Agent, sample: dict, prods: dict) -> dict:
    """Replay one session with UNBOUNDED ranking depth, recording where the target sits."""
    tgt = str(sample["ground_truth"]["parent_asin"])
    card = intent_card(prods[tgt])
    rng = random.Random(f"{sample['sample_id']}\0{sample['scenario_type']}")
    eff = {**sample, "intent_card": card,
           "behavior": behavior_for(str(sample["scenario_type"]), card, rng)}
    ov = eff.get("behavior", {}).get("override") or {}
    gate = int(ov.get("turn", 3)) if sample["scenario_type"] == "intent_override" else 0

    sid = f"autopsy_{sample['sample_id']}"
    agent.reset(sid, sample["user_profile"])
    st = agent.sessions[sid]
    disclosed: set[str] = set()
    bu = False
    applied = sample["scenario_type"] != "intent_override"
    msg = initial_message(eff, coarse_category(
        [str(v) for v in (prods[tgt].get("categories") or [])]), disclosed)

    turns = []
    for turn in range(1, MAXT + 1):
        st.turn += 1
        agent._observe(st, msg)
        pool = agent._candidates(st, msg)
        full = agent._rank(st, pool, len(pool)) if pool else []
        turns.append({
            "turn": turn,
            "msg": msg[:150],
            "in_pool": tgt in pool,
            "pool_size": len(pool),
            "rank": (full.index(tgt) + 1) if tgt in full else None,
            "n_evidence": len(st.evidence),
        })
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
    ev = dict(st.evidence)
    agent.sessions.pop(sid, None)
    return {"target": tgt, "gate": gate, "turns": turns,
            "evidence": [(p, d, t) for p, (d, t) in ev.items()],
            "covered": {p: agent.ix.covers(tgt, p) for p in ev}}


def main() -> None:
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    agent = Agent(ROOT / "data" / "catalog.jsonl")

    print("=== shipped policy (width 1x9 then 10) ===")
    shipped = evaluate(agent, samples, cid, cats, prods)
    print(f"  HR {shipped['hit_rate_at_10']:.3f}  MRR {shipped['mrr']:.4f}  "
          f"MTTC {shipped['mttc']:.3f}  score {shipped['recommended_technical_score']:.5f}")
    miss_ship = [s for s in shipped["sessions"] if not s["hit"]]

    print("\n=== full-10 every turn, SAME sessions (does width-1 cost recall?) ===")
    full = Agent.__new__(Agent)
    full.ix, full.sessions, full.llm = agent.ix, {}, None
    full.DISCLOSURE = (10,) * 10
    r10 = evaluate(full, samples, cid, cats, prods)
    print(f"  HR {r10['hit_rate_at_10']:.3f}  MRR {r10['mrr']:.4f}  "
          f"MTTC {r10['mttc']:.3f}  score {r10['recommended_technical_score']:.5f}")
    miss_full = {s["sample_id"] for s in r10["sessions"] if not s["hit"]}
    ship_ids = {s["sample_id"] for s in miss_ship}
    print(f"\n  misses under width-1 : {sorted(ship_ids)}")
    print(f"  misses under full-10 : {sorted(miss_full)}")
    print(f"  caused BY width-1    : {sorted(ship_ids - miss_full)}  "
          f"(empty => disclosure costs zero recall)")

    by_id = {s["sample_id"]: s for s in samples}
    OUT = {"misses": []}
    for m in miss_ship:
        s = by_id[m["sample_id"]]
        tr = trace(agent, s, prods)
        best = [t["rank"] for t in tr["turns"] if t["rank"]]
        print(f"\n--- MISS {m['sample_id']}  ({m['scenario_type']}) "
              f"target {tr['target']} " + "-" * 20)
        print(f"  target reachable at all? in_pool on "
              f"{sum(t['in_pool'] for t in tr['turns'])}/10 turns")
        print(f"  best unbounded rank ever: "
              f"{min(best) if best else 'NEVER RANKED'}   (gate turn {tr['gate']})")
        for t in tr["turns"]:
            print(f"    t{t['turn']:>2} pool={t['pool_size']:>4} ev={t['n_evidence']:>2} "
                  f"rank={str(t['rank']):>6}  | {t['msg'][:88]}")
        cov = tr["covered"]
        nc = [p for p, c in cov.items() if not c]
        print(f"  evidence phrases {len(cov)}; target FAILS to contain {len(nc)}:")
        for p in nc[:12]:
            df = dict((x[0], x[1]) for x in tr["evidence"]).get(p)
            tier = dict((x[0], x[2]) for x in tr["evidence"]).get(p)
            print(f"      [{tier}] df={df:<6} {p!r}")
        OUT["misses"].append({
            "sample_id": m["sample_id"], "scenario": m["scenario_type"],
            "target": tr["target"], "gate": tr["gate"],
            "best_rank": min(best) if best else None,
            "turns": tr["turns"],
            "uncovered": [[p, dict((x[0], x[1]) for x in tr["evidence"]).get(p),
                           dict((x[0], x[2]) for x in tr["evidence"]).get(p)] for p in nc],
        })

    OUT["shipped"] = {k: shipped[k] for k in
                      ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score")}
    OUT["full10"] = {k: r10[k] for k in
                     ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score")}
    OUT["width1_caused"] = sorted(ship_ids - miss_full)
    (ROOT / "experiments" / "results" / "out_28.json").write_text(
        json.dumps(OUT, indent=2) + "\n", encoding="utf-8")
    print("\n[saved] experiments/results/out_28.json")


if __name__ == "__main__":
    main()
