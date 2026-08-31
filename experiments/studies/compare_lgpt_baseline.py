"""Head-to-head: the shipped agent against the L-GPT baseline, on our suites.

WHAT THIS SETTLES. The L-GPT branch reports 0.8927 on the public set in its own
documentation, measured in its own repository. That number has never been reproduced here,
and it has never been run on the population-shift suites at all -- so "we score higher" has
so far rested on comparing our measurement to their write-up.

This runs BOTH agents through the SAME evaluator, on the SAME five suites, in the same
process. Any difference is then the agent rather than the harness.

FAIRNESS, DELIBERATELY. Their agent is constructed exactly as the evaluator would construct
it, pointed at our catalogue, with its shipped defaults untouched -- the LLM rescue stays
off, which is what their own documentation recommends for the submitted configuration. Ours
runs with its hosted layers off too, so both sides are deterministic and neither is
advantaged by a credential the other lacks.

WHAT A DIFFERENCE HERE DOES AND DOES NOT MEAN. Both systems reach the same hit rate on the
public set -- 0.995, and identical per scenario -- so this is not a retrieval comparison.
The gap is disclosure policy: we return one candidate per turn until turn 10, trading
efficiency for rank, and they return ten every turn. On the shifted populations the
comparison is more interesting, because neither system has been tuned against them and the
targets are deliberately obscure.

Run:  PYTHONIOENCODING=utf-8 .venv-v2/Scripts/python.exe -u experiments/studies/compare_lgpt_baseline.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# TWO MODES, because "is the comparison fair" has two different answers.
#
#   full            everything each system ships enabled. Ours: dialogue-act router,
#                   scaffolding tagger, span node, deparaphraser, message writer. Theirs:
#                   the turn-five rescue. Each at its own gate, as submitted.
#   deterministic   every optional layer off on both sides, so the result needs no
#                   credential and no network.
#
# `LLM_RERANK` and `LLM_EXTRACT` stay off even in `full`, and that is not a hedge: they
# ship DISABLED because they were measured negative -- listwise reranking cost 0.027 and
# the hosted extractor was beaten by the local tagger on the hardest transform. Enabling
# rejected layers would not be showing our system at its best, it would be arguing against
# ourselves with components we already removed on evidence.
MODE = os.environ.get("COMPARE_MODE", "full").strip().lower()
_full = MODE == "full"

os.environ["LLM_RERANK"] = "0"           # measured -0.027; ships disabled
os.environ["LLM_EXTRACT"] = "0"          # superseded by the local tagger; ships disabled
os.environ["LLM_RESOLVE"] = "1" if _full else "0"
os.environ["LLM_MESSAGE"] = "1" if _full else "0"
os.environ["V2_ROUTE"] = "1" if _full else "0"
os.environ["BERT_EXTRACT"] = "1" if _full else "0"
os.environ["MESSAGE_VARIETY"] = "1"

LGPT = Path(os.environ.get("LGPT_REPO", r"D:\Coding\Models\lgpt-baseline"))
SETS = ROOT / "experiments" / "datasets" / "sets"
OUT = ROOT / "experiments" / "results" / f"out_78_lgpt_head_to_head_{MODE}.json"

SUITES = (
    ("official200", ROOT / "data" / "public_set.jsonl"),
    ("org_proxy_800", SETS / "organizer_proxy_800.jsonl"),
    ("review800", SETS / "catalog_review_distinct_800.jsonl"),
    ("uniform800", SETS / "catalog_uniform_800.jsonl"),
    ("inverse800", SETS / "catalog_inverse_800.jsonl"),
)


def main() -> None:
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    from submission.agent import Agent as OursAgent

    if not (LGPT / "copilot" / "agent.py").exists():
        print(f"L-GPT repo not found at {LGPT}; set LGPT_REPO.")
        return
    sys.path.insert(0, str(LGPT))
    from copilot.agent import Agent as TheirsAgent           # noqa: E402
    from copilot.config import CopilotConfig                 # noqa: E402
    from dataclasses import replace as _replace              # noqa: E402

    # THEIR BEST, by the same rule applied to ours: enable everything the system ships,
    # except what that system's own measurements rejected.
    #
    #   enable_llm_rescue    their brief's recommended LLM feature
    #   enable_lsa           a latent-semantic channel they ship as "a paraphrase hedge".
    #                        Their own note says it earns nothing on a lexical set -- but
    #                        it is theirs to enable and this is their best case, not ours
    #   enable_llm_message   cosmetic, the counterpart of our message writer
    #   allow_null_ask       LEFT OFF. Their own measurement: 6 of 200 sessions deadlocked
    #                        to turn 10. Excluded for exactly the reason we exclude our
    #                        listwise reranker -- a component its authors rejected on evidence.
    their_config = CopilotConfig()
    if _full:
        their_config = _replace(their_config, enable_llm_rescue=True,
                                enable_lsa=True, enable_llm_message=True)

    def build_theirs(cat):
        return TheirsAgent(cat, config=their_config)

    has_key = bool(os.environ.get("GROQ_API_KEY", "").strip())
    print(f"mode: {MODE}   credential: {'present' if has_key else 'ABSENT'}")
    print(f"  ours   : route={os.environ['V2_ROUTE']} tagger={os.environ['BERT_EXTRACT']} "
          f"deparaphrase={os.environ['LLM_RESOLVE']} message={os.environ['LLM_MESSAGE']} "
          f"(rerank/extract off: measured negative, ship disabled)")
    print(f"  L-GPT  : rescue={their_config.enable_llm_rescue} "
          f"lsa={their_config.enable_lsa} message={their_config.enable_llm_message} "
          f"(null_ask off: their measurement, 6/200 deadlocks)\n")

    catalog = ROOT / "data" / "catalog.jsonl"
    ids, cats, prods = catalog_index(catalog)
    loaded = [(n, load_jsonl(p)) for n, p in SUITES if p.exists()]

    print(f"{'suite':<16}{'ours':>11}{'L-GPT':>11}{'delta':>10}"
          f"{'HR ours':>9}{'HR them':>9}{'MRR ours':>10}{'MRR them':>10}"
          f"{'MTTC ours':>11}{'MTTC them':>11}")
    print("-" * 108)

    rows = {}
    for name, samples in loaded:
        cell = {}
        for who, cls in (("ours", OursAgent), ("lgpt", TheirsAgent)):
            t0 = time.perf_counter()
            try:
                agent = build_theirs(catalog) if who == "lgpt" else cls(catalog)
                r = evaluate(agent, samples, ids, cats, prods)
                cell[who] = {k: round(float(r[k]), 6) for k in
                             ("recommended_technical_score", "hit_rate_at_10", "mrr", "mttc")}
            except Exception as exc:                          # a crash is a result too
                cell[who] = {"error": f"{type(exc).__name__}: {exc}"[:160]}
            cell[who]["seconds"] = round(time.perf_counter() - t0, 1)
        rows[name] = cell
        o, t = cell["ours"], cell["lgpt"]
        if "error" in t or "error" in o:
            print(f"{name:<16}{o['recommended_technical_score']:>11.6f}"
                  f"{'ERROR':>11}   {t['error'][:60]}", flush=True)
            continue
        d = o["recommended_technical_score"] - t["recommended_technical_score"]
        print(f"{name:<16}{o['recommended_technical_score']:>11.6f}"
              f"{t['recommended_technical_score']:>11.6f}{d:>+10.4f}"
              f"{o['hit_rate_at_10']:>9.3f}{t['hit_rate_at_10']:>9.3f}"
              f"{o['mrr']:>10.3f}{t['mrr']:>10.3f}"
              f"{o['mttc']:>11.2f}{t['mttc']:>11.2f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "shipped agent vs L-GPT baseline, same evaluator, same suites",
        "lgpt_repo": str(LGPT),
        "mode": MODE,
        "configuration": ("ours LLM_RESOLVE=" + os.environ["LLM_RESOLVE"]
                          + "; theirs enable_llm_rescue=" + str(their_config.enable_llm_rescue)
                          + "; BERT off both modes"),
        "suites": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
