"""Experiment 19: does an LLM break the ties the evidence cannot?

The irreducibility diagnostic found that 59% of rank>1 hits are cases where every product
ranked above the target covers EXACTLY the same evidence. Nothing in the disclosed text
separates them, so every deterministic feature we tried (popularity, neural relevance,
generative verification) failed on them by construction. Breaking those ties needs a
signal orthogonal to the constraints -- which is the one thing an LLM plausibly has.

Scope, deliberately narrow:
  * ties are computed on phrase COVERAGE only, excluding the popularity prior
  * only the tie group occupying the #1 slot is sent (LLM_TIE_DEPTH = 1) -- the only
    position where a swap converts rank>1 into rank 1
  * the model may only PERMUTE the candidates given; any other output is discarded
  * ~219 calls per full public evaluation, ~110 per half -- inside Groq's free tier

Protocol matches every other pass: tune half first, held-out half adjudicates.

Usage:
  python experiments/log/19_llm_rerank.py --half tune  --online
  python experiments/log/19_llm_rerank.py --half hold  --online
  python experiments/log/19_llm_rerank.py --half tune  --online --model llama-3.3-70b-versatile
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402

KEEP = ("sample_count", "hit_rate_at_10", "mrr", "mttc", "efficiency",
        "recommended_technical_score")


def summary(r: dict) -> dict:
    return {k: r[k] for k in KEEP}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--dataset", default="data/public_set.jsonl")
    ap.add_argument("--half", choices=("tune", "hold", "all"), default="tune")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--online", action="store_true", help="allow Groq calls")
    ap.add_argument("--model", default=None, help="override GROQ_MODEL")
    ap.add_argument("--depth", type=int, default=1, help="tie groups starting below this rank")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.model:
        os.environ["GROQ_MODEL"] = args.model

    samples = load_jsonl(args.dataset)
    if args.half == "tune":
        samples = [s for i, s in enumerate(samples) if i % 2 == 0]
    elif args.half == "hold":
        samples = [s for i, s in enumerate(samples) if i % 2 == 1]
    if args.limit:
        samples = samples[:args.limit]

    catalog_ids, categories, products = catalog_index(args.catalog)
    print(f"half={args.half}  sessions={len(samples)}  depth={args.depth}")

    # --- deterministic arm -------------------------------------------------
    os.environ["LLM_RERANK"] = "0"
    from submission.agent import Agent  # imported AFTER the gate is set
    t0 = time.time()
    base_agent = Agent(args.catalog)
    base_agent.LLM_TIE_DEPTH = args.depth
    base = evaluate(base_agent, samples, catalog_ids, categories, products)
    print(f"  deterministic : SCORE {base['recommended_technical_score']:.5f}  "
          f"HR@10 {base['hit_rate_at_10']:.1%}  MRR {base['mrr']:.4f}  "
          f"MTTC {base['mttc']:.2f}   [{time.time()-t0:.0f}s]")

    if not args.online:
        print(json.dumps({"baseline": summary(base), "online": False}, indent=2))
        return

    # --- LLM arm -----------------------------------------------------------
    os.environ["LLM_RERANK"] = "1"
    import importlib
    import submission.llm_rerank as lr
    importlib.reload(lr)              # re-read the gate and GROQ_MODEL
    import submission.agent as sa
    importlib.reload(sa)

    t0 = time.time()
    llm_agent = sa.Agent(args.catalog)
    llm_agent.LLM_TIE_DEPTH = args.depth
    if not (llm_agent.llm and llm_agent.llm.enabled):
        print("  LLM arm DISABLED -- set GROQ_API_KEY (and LLM_RERANK=1) and retry.")
        return
    print(f"  model={llm_agent.llm.model}  cache={len(llm_agent.llm.cache)} entries")
    llm = evaluate(llm_agent, samples, catalog_ids, categories, products)
    stats = llm_agent.llm.stats()
    print(f"  llm arm       : SCORE {llm['recommended_technical_score']:.5f}  "
          f"HR@10 {llm['hit_rate_at_10']:.1%}  MRR {llm['mrr']:.4f}  "
          f"MTTC {llm['mttc']:.2f}   [{time.time()-t0:.0f}s]")

    delta = llm["recommended_technical_score"] - base["recommended_technical_score"]
    verdict = ("ADOPT" if delta > 0.005 else
               "inside noise" if delta > -0.005 else "REJECT")
    print(f"\n  DELTA {delta:+.5f}  -> {verdict}")
    print(f"  api_calls={stats['api_calls']}  cache_hits={stats['cache_hits']}  "
          f"failures={stats['failures']}  cached={stats['cached_entries']}")
    if stats["api_calls"]:
        print(f"  failure rate: {stats['failures']/max(1,stats['api_calls']+stats['cache_hits']):.1%}")

    out = Path(args.output or f"experiments/results/out_19_{args.half}.json")
    out.write_text(json.dumps({
        "half": args.half, "depth": args.depth, "model": llm_agent.llm.model,
        "baseline": summary(base), "llm": summary(llm),
        "delta": {k: round(llm[k] - base[k], 6)
                  for k in ("hit_rate_at_10", "mrr", "efficiency",
                            "recommended_technical_score")},
        "llm_stats": stats, "verdict": verdict,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
