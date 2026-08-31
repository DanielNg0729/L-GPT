"""Experiment 20: can the LLM actually break these ties? Measured directly.

End-to-end TechnicalScore is a noisy instrument for judging a tie-breaker: it moves only
when a swap happens to land on the exact turn a session converts, and everything else in
the pipeline adds variance. This pass measures the tie-break in isolation.

For every coverage-tie group that CONTAINS the target, we ask:

    where does the deterministic order (popularity) put the target inside the group?
    where does the LLM order put it?

That yields a clean, low-variance comparison of two tie-breaking policies on exactly the
decisions they are asked to make -- and a random baseline for scale, since a group of size
k gives 1/k by chance.

Runs mostly from cache once pass 19 has populated it.

Usage:
  PYTHONIOENCODING=utf-8 LLM_RERANK=1 python -u experiments/scripts/20_tiebreak_accuracy.py --half tune
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    behavior_for, catalog_index, coarse_category, customer_reply, initial_message,
    intent_card, load_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--dataset", default="data/public_set.jsonl")
    ap.add_argument("--half", choices=("tune", "hold", "all"), default="tune")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.model:
        os.environ["GROQ_MODEL"] = args.model
    os.environ.setdefault("LLM_RERANK", "1")

    from submission.agent import Agent, CONSTRAINT, CAT  # noqa: E402

    samples = load_jsonl(args.dataset)
    if args.half == "tune":
        samples = [s for i, s in enumerate(samples) if i % 2 == 0]
    elif args.half == "hold":
        samples = [s for i, s in enumerate(samples) if i % 2 == 1]
    if args.limit:
        samples = samples[:args.limit]

    cid, cats, prods = catalog_index(args.catalog)
    ag = Agent(args.catalog)
    if not (ag.llm and ag.llm.enabled):
        print("LLM disabled -- set GROQ_API_KEY and LLM_RERANK=1")
        return
    print(f"model={ag.llm.model}  half={args.half}  sessions={len(samples)}  "
          f"cache={len(ag.llm.cache)}")

    rng = random.Random(0)
    det_pos, llm_pos, rnd_pos, sizes = [], [], [], []
    groups = with_target = 0

    for s in samples:
        tgt = str(s["ground_truth"]["parent_asin"])
        card = intent_card(prods[tgt])
        eff = {**s, "intent_card": card,
               "behavior": behavior_for(str(s["scenario_type"]), card,
                                        random.Random(f"{s['sample_id']}\0{s['scenario_type']}"))}
        disclosed, bu = set(), False
        sid = s["sample_id"]
        ag.reset(sid, s["user_profile"])
        st = ag.sessions[sid]
        msg = initial_message(eff, coarse_category(
            [str(v) for v in (prods[tgt].get("categories") or [])]), disclosed)
        applied = s["scenario_type"] != "intent_override"

        for turn in range(1, 11):
            st.turn += 1
            try:
                ag._observe(st, msg)
                pool = ag._candidates(st, msg)
                ranked = ag._rank(st, pool, 10) if pool else []
            except Exception:
                ranked = []
            if len(ranked) >= 2 and st.evidence:
                wmap = {p: ag._weight(p, df, t) for p, (df, t) in st.evidence.items()}

                def cov(a):
                    return sum(w for p, w in wmap.items() if ag.ix.covers(a, p))

                i = 0
                while i < len(ranked):
                    v = cov(ranked[i])
                    j = i + 1
                    while j < len(ranked) and abs(cov(ranked[j]) - v) < 1e-12:
                        j += 1
                    if j - i >= 2 and i < ag.LLM_TIE_DEPTH:
                        groups += 1
                        grp = ranked[i:j][:ag.llm.MAX_CANDIDATES]
                        if tgt in grp:
                            with_target += 1
                            sizes.append(len(grp))
                            det_pos.append(grp.index(tgt))
                            rnd_pos.append(rng.randrange(len(grp)))
                            reqs = [p for p, (_, t) in st.evidence.items()
                                    if t in (CONSTRAINT, CAT)]
                            out = ag.llm.rerank(reqs, grp,
                                                [ag.ix.doc.get(a, "") for a in grp])
                            llm_pos.append(out.index(tgt) if out else grp.index(tgt))
                    i = j
            if applied and tgt in ranked:
                break
            ov = eff.get("behavior", {}).get("override") or {}
            if not applied and turn + 1 == int(ov.get("turn", 3)):
                applied = True
                nv = str(ov.get("new_value", ""))
                if nv:
                    disclosed.add(nv)
                msg = str(ov.get("message", ""))
            else:
                msg, bu = customer_reply(eff, ag._next_probe(st), disclosed, bu)

    if not det_pos:
        print("no tie groups containing the target were observed")
        return

    def rep(name, pos):
        first = sum(1 for p in pos if p == 0) / len(pos)
        mrr = statistics.fmean(1.0 / (p + 1) for p in pos)
        print(f"  {name:<26} target-first {first:>6.1%}   mean pos {statistics.fmean(pos)+1:>5.2f}"
              f"   within-group MRR {mrr:.4f}")
        return {"target_first": first, "mean_pos": statistics.fmean(pos) + 1, "mrr": mrr}

    print(f"\ntie groups sent to the LLM: {groups}   of which contain the target: {with_target}")
    print(f"mean group size: {statistics.fmean(sizes):.2f}   "
          f"(random baseline = {statistics.fmean(1/ s for s in sizes):.1%} target-first)\n")
    out = {"groups": groups, "with_target": with_target,
           "mean_group_size": statistics.fmean(sizes), "model": ag.llm.model,
           "deterministic": rep("deterministic (popularity)", det_pos),
           "llm": rep("LLM tie-break", llm_pos),
           "random": rep("random shuffle", rnd_pos),
           "llm_stats": ag.llm.stats()}
    better = sum(1 for d, l in zip(det_pos, llm_pos) if l < d)
    worse = sum(1 for d, l in zip(det_pos, llm_pos) if l > d)
    print(f"\n  LLM moved the target UP in {better} groups, DOWN in {worse}, "
          f"unchanged in {len(det_pos)-better-worse}")
    out["moved_up"], out["moved_down"] = better, worse
    print(f"  api_calls={out['llm_stats']['api_calls']} "
          f"cache_hits={out['llm_stats']['cache_hits']} "
          f"failures={out['llm_stats']['failures']}")

    p = ROOT / "experiments" / "scripts" / f"out_20_{args.half}.json"
    p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {p}")


if __name__ == "__main__":
    main()
