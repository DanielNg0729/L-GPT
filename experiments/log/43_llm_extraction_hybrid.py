"""Experiment 43: measure the three-way hybrid -- templates + mining + gated LLM extraction.

TWO QUESTIONS, IN THIS ORDER. The second only matters if the first passes.

  1. IS THE CLEAN SCORE UNTOUCHED?  The requirement was "never degrades at zero
     paraphrase", and the recognition gate is supposed to make that structural rather than
     statistical. So the control run must reproduce 0.96960 EXACTLY, with the extractor's
     own counters confirming it was never called. Anything else and the gate leaks and the
     layer must not ship.

  2. HOW MUCH DOES IT RECOVER UNDER PARAPHRASE?  Deterministic baselines to beat, from
     pass 31/42, are the numbers the agent already achieves with templates dead and mining
     carrying alone:

         T1 scaffolding reworded    0.84670
         T2 scaffolding stripped    0.86930
         T5 reworded + filler       0.83800

     Context for how much room there is: with mining ALSO removed these collapse to 0.2173
     and 0.1637, and the clean ceiling is 0.96960.

WHY THE DETERMINISTIC FLOOR CANNOT BE PUSHED FURTHER (pass 42). Mining's two governing
constants were swept, and the floor is already at its optimum:

    minn 2   T1 0.88317 (+0.036) but CLEAN 0.95340 (-0.016) and T5 0.74310 (-0.095)
    minn 4   clean held, T1 0.76290 (-0.084)
    maxn 6/12  neutral everywhere

So paraphrase robustness beyond today's floor genuinely requires a new channel; it cannot
be tuned out of the existing one. That is the case for this layer.

COST. Pass 41 counted the messages that reach the gate: ~1,500 unique calls per 800
sessions after message-level caching (48% hit rate). Groq's free tier is ~1,000 requests
per day, so a full private run needs either a paid tier or span-level caching.

Requires LLM_EXTRACT=1 and GROQ_API_KEY. Without them the extractor reports disabled and
this pass measures the deterministic baseline only -- which is itself the proof that the
agent is byte-identical offline.

Run:  PYTHONIOENCODING=utf-8 LLM_EXTRACT=1 python -u experiments/log/43_llm_extraction_hybrid.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "log"))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent, recognised  # noqa: E402
from submission.llm_extract import LLMExtractor  # noqa: E402

_p31 = __import__("31_paraphrase_stress")
evaluate_transformed, TRANSFORMS = _p31.evaluate_transformed, _p31.TRANSFORMS

CONDITIONS = ["T0 identity (control)", "T1 scaffold reworded", "T2 scaffold stripped",
              "T5 realistic (T1+T3)"]


def main() -> None:
    ap = argparse.ArgumentParser()
    # Groq's free tier allows ~1,000 requests/day and a full sweep needs ~1,100 unique
    # calls, so validate on a subset before spending the quota on the whole set.
    ap.add_argument("--limit", type=int, default=0, help="use only the first N sessions")
    args = ap.parse_args()
    t0 = time.time()
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    if args.limit:
        samples = samples[:args.limit]
        print(f"SUBSET: first {len(samples)} sessions "
              f"(scores are not comparable to full-set numbers)")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")

    probe = LLMExtractor()
    print(f"extractor enabled: {probe.enabled}   model: {probe.model}")
    print(f"cache: {probe.cache_path.name} ({len(probe.cache)} entries)")
    if not probe.enabled:
        print("  -> set LLM_EXTRACT=1 and GROQ_API_KEY to exercise the LLM arm.")
        print("     The deterministic column below is what ships when they are unset.\n")

    def share(with_llm: bool):
        o = object.__new__(Agent)
        o.ix, o.sessions, o.llm = base.ix, {}, None
        o.llm_extract = LLMExtractor() if with_llm else None
        return o

    OUT: dict = {}
    print(f"{'condition':<28}{'deterministic':>15}{'+ LLM extract':>15}{'delta':>10}"
          f"{'calls':>8}{'hits':>7}{'fails':>7}")
    print("-" * 90)

    for cond in CONDITIONS:
        fn = TRANSFORMS[cond]
        det_agent = share(False)
        if cond.startswith("T0"):
            det = evaluate(det_agent, samples, cid, cats, prods)[
                "recommended_technical_score"]
        else:
            det = evaluate_transformed(det_agent, samples, cid, cats, prods, fn)[
                "recommended_technical_score"]

        llm_agent = share(True)
        if cond.startswith("T0"):
            llm = evaluate(llm_agent, samples, cid, cats, prods)[
                "recommended_technical_score"]
        else:
            llm = evaluate_transformed(llm_agent, samples, cid, cats, prods, fn)[
                "recommended_technical_score"]
        st = llm_agent.llm_extract.stats() if llm_agent.llm_extract else {}
        OUT[cond] = {"deterministic": det, "llm": llm, "delta": llm - det,
                     "extractor": st, "gate": llm_agent.paraphrase_rate()}
        print(f"{cond:<28}{det:>15.5f}{llm:>15.5f}{llm-det:>+10.5f}"
              f"{st.get('api_calls', 0):>8}{st.get('cache_hits', 0):>7}"
              f"{st.get('failures', 0):>7}")

    ctl = OUT["T0 identity (control)"]
    print("\n" + "=" * 74)
    print("GUARANTEE CHECK -- the clean path must be untouched")
    print("=" * 74)
    same = abs(ctl["llm"] - ctl["deterministic"]) < 1e-12
    calls = ctl["extractor"].get("api_calls", 0)
    rate = ctl["gate"]
    print(f"  clean score  deterministic {ctl['deterministic']:.5f}  "
          f"with LLM {ctl['llm']:.5f}   identical: {'YES' if same else 'NO'}")
    print(f"  LLM API calls on the clean run: {calls}  (must be 0)")
    print(f"  gate: {rate['unrecognised']}/{rate['messages']} messages unrecognised "
          f"= {rate['rate']:.2%}")
    ok = same and calls == 0 and rate["unrecognised"] == 0
    print(f"\n  VERDICT: {'PASS -- the gate holds; the layer cannot affect a clean run' if ok else 'FAIL -- the gate leaked; do not ship this layer'}")

    print("\n  paraphrase recovery (vs the deterministic floor):")
    for cond in CONDITIONS[1:]:
        v = OUT[cond]
        head = 0.96960 - v["deterministic"]
        got = v["delta"]
        print(f"    {cond:<26}{v['deterministic']:.5f} -> {v['llm']:.5f}  "
              f"({got:+.5f}, {got/head:>6.1%} of the {head:.3f} gap to clean)")

    tot = OUT[CONDITIONS[-1]]["extractor"]
    if tot.get("enabled"):
        print(f"\n  cost: {tot.get('prompt_tokens',0):,} prompt + "
              f"{tot.get('completion_tokens',0):,} completion tokens, "
              f"{tot.get('cached_entries',0):,} cached spans")

    (ROOT / "experiments" / "results" / "out_43.json").write_text(
        json.dumps(OUT, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n[saved] experiments/results/out_43.json   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
