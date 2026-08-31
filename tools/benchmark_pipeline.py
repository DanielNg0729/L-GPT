"""Runtime cost of each shipped configuration, on traffic that does and does not reach the models.

WHY BOTH CONDITIONS
-------------------
Measuring latency on the official public set alone would be misleading in the flattering
direction. The recognition gate matches 463 of 463 clean messages, so the learned layers
are never constructed there -- no checkpoint is read, torch is never imported, and the
three configurations execute identical code. A clean-traffic benchmark would report that
the models are free, which is true and also not the whole answer.

The honest question is what they cost WHEN THEY RUN. So each configuration is measured
twice: on the released public set, where the gate holds them back, and on reworded wrappers,
where every message reaches them.

CONFIGURATIONS

  deterministic   exact catalogue machinery only: FTS5 retrieval, template extraction,
                  exact span recovery, n-gram mining. No model of any kind.
  + local models  adds the two DistilBERT components -- the dialogue-act router and the
                  scaffolding tagger. Still no network.
  all shipped     adds the hosted deparaphraser. Inert without GROQ_API_KEY, which is
                  reported rather than assumed.

WHAT IS REPORTED. Wall time per session and per turn, the one-time checkpoint load, and the
inference counts that explain the difference. The catalogue index is built ONCE and shared,
because it is a fixed startup cost identical across configurations and including it would
bury the thing being measured.

MTTC IS NOT LATENCY. The scored Efficiency term is `clip((11 - MTTC) / 10, 0, 1)`, where
MTTC counts TURNS, not seconds. Nothing here can move it. This measures deployability --
whether the architecture holds under real conditions and whether its resource use is
proportionate -- not score.

Run:  PYTHONIOENCODING=utf-8 python -u tools/benchmark_pipeline.py
"""
from __future__ import annotations

import importlib.util as ilu
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments" / "results" / "out_71_pipeline_runtime.json"

_s = ilu.spec_from_file_location("_stress", ROOT / "experiments" / "log"
                                 / "31_paraphrase_stress.py")
_stress = ilu.module_from_spec(_s)
_s.loader.exec_module(_stress)
_t = ilu.spec_from_file_location("_tmpl", ROOT / "experiments" / "studies"
                                 / "run_official_template_paraphrase.py")
_tm = ilu.module_from_spec(_t)
_t.loader.exec_module(_tm)


def main() -> None:
    os.environ["LLM_RERANK"] = "0"
    os.environ["LLM_EXTRACT"] = "0"
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    from submission.agent import Agent
    from submission.bert_extract import ScaffoldingTagger
    from submission.llm_resolve import LLMResolver
    from submission.route_node import StrictGatedRouteNode

    print("building the catalogue index (one-time, shared across configurations)")
    t0 = time.perf_counter()
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    index_seconds = time.perf_counter() - t0
    print(f"  {index_seconds:.2f}s\n")

    public = load_jsonl(ROOT / "data" / "public_set.jsonl")
    review = load_jsonl(ROOT / "experiments" / "datasets" / "open_vocabulary"
                        / "review800_canonical_replay.jsonl")[:200]
    transform = _tm.transform(_tm.bank())
    conditions = {"official200 (clean)": (public, False),
                  "reworded wrappers (200)": (review, True)}

    CONFIGS = (("deterministic", False, False, False),
               ("+ local models", True, True, False),
               ("all shipped", True, True, True))

    def make(tagger_on, route_on, llm_on):
        a = object.__new__(Agent)
        a.ix, a.sessions = base.ix, {}
        a.llm = a.llm_extract = None
        os.environ["BERT_EXTRACT"] = "1" if tagger_on else "0"
        a.tagger = ScaffoldingTagger() if tagger_on else None
        a.span_node = base.span_node
        os.environ["V2_ROUTE"] = "1" if route_on else "0"
        a.route_node = StrictGatedRouteNode() if route_on else None
        os.environ["LLM_RESOLVE"] = "1" if llm_on else "0"
        a.resolver = (LLMResolver().bind(base.ix.df) if llm_on else None)
        return a

    # REPEATS, because a single pass is not measurable here. The first version of this
    # benchmark reported "all shipped" as FASTER than "+ local models" on identical work --
    # same 1,215 inferences, same score -- and deterministic as slower than all-shipped on
    # clean traffic where the two execute the same code. That is run-to-run variance of
    # roughly 15%, and no conclusion survives it. Median of REPEATS is reported with the
    # observed spread, so the reader can see when a difference is inside the noise.
    REPEATS = int(os.environ.get("BENCH_REPEATS", "5"))
    rows = {}
    print(f"median of {REPEATS} runs per cell")
    print()
    print(f"{'configuration':<18}{'condition':<26}{'ms/session':>12}{'spread':>10}"
          f"{'ms/turn':>10}{'infer':>8}")
    print("-" * 84)
    for label, tg, rt, llm in CONFIGS:
        for cname, (samples, warp) in conditions.items():
            timings = []
            for _ in range(REPEATS):
                agent = make(tg, rt, llm)
                started = time.perf_counter()
                if warp:
                    r = _stress.evaluate_transformed(agent, samples, cid, cats, prods,
                                                     transform)
                else:
                    r = evaluate(agent, samples, cid, cats, prods)
                timings.append(time.perf_counter() - started)
            elapsed = statistics.median(timings)
            spread = max(timings) - min(timings)
            turns = len(samples) * max(r["mttc"], 1.0)
            rn = agent.route_node.stats() if agent.route_node else {}
            tg_s = agent.tagger.stats() if agent.tagger else {}
            res = agent.resolver.stats() if agent.resolver else {}
            infer = int(rn.get("inferences", 0)) + int(tg_s.get("calls", 0))
            rows[f"{label} | {cname}"] = {
                "seconds": round(elapsed, 3),
                "ms_per_session": round(1000 * elapsed / len(samples), 2),
                "ms_per_turn": round(1000 * elapsed / max(turns, 1), 2),
                "score": round(r["recommended_technical_score"], 6),
                "mttc": round(r["mttc"], 3),
                "route_loads": int(rn.get("model_loads", 0)),
                "route_inferences": int(rn.get("inferences", 0)),
                "tagger_calls": int(tg_s.get("calls", 0)),
                "llm_calls": int(res.get("calls", 0)),
                "llm_enabled": bool(res.get("enabled", False)),
            }
            rows[f"{label} | {cname}"]["spread_ms"] = round(1000 * spread / len(samples), 2)
            rows[f"{label} | {cname}"]["repeats"] = REPEATS
            print(f"{label:<18}{cname:<26}{1000*elapsed/len(samples):>12.2f}"
                  f"{1000*spread/len(samples):>10.2f}"
                  f"{1000*elapsed/max(turns,1):>10.2f}{infer:>8}", flush=True)

    print(f"\nwhat the models actually did")
    print(f"{'configuration | condition':<46}{'route inf':>11}{'tagger':>9}"
          f"{'llm':>7}{'score':>11}")
    print("-" * 84)
    for k, v in rows.items():
        print(f"{k:<46}{v['route_inferences']:>11}{v['tagger_calls']:>9}"
              f"{v['llm_calls']:>7}{v['score']:>11.6f}")

    det_clean = rows["deterministic | official200 (clean)"]["ms_per_session"]
    all_clean = rows["all shipped | official200 (clean)"]["ms_per_session"]
    det_para = rows["deterministic | reworded wrappers (200)"]["ms_per_session"]
    all_para = rows["all shipped | reworded wrappers (200)"]["ms_per_session"]
    print(f"\n  clean traffic      deterministic {det_clean:.1f} ms/session -> "
          f"all shipped {all_clean:.1f} ms/session   ({all_clean-det_clean:+.1f})")
    print(f"  reworded wrappers  deterministic {det_para:.1f} ms/session -> "
          f"all shipped {all_para:.1f} ms/session   ({all_para-det_para:+.1f})")
    print(f"  one-time index build: {index_seconds:.2f}s, shared and excluded above")
    print(f"\n  A near-zero clean-traffic difference is the recognition gate doing its job:")
    print(f"  the models are not merely unused there, they are never constructed. The")
    print(f"  reworded column is what they cost when every message reaches them.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "runtime cost by configuration",
        "index_build_seconds": round(index_seconds, 3),
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
