"""V2.61: the arms against the hostile condition -- no free channels left.

WHAT CHANGED FROM `both`
------------------------
  category            neutralised. Every target is given an EMPTY category list, so
                      `coarse_category` returns its constant fallback "clothing item" for
                      every session: present, uniform, zero discriminative signal. This is
                      not a handicap invented for the test -- the category in a normal
                      `both` run is a deterministic slice of the product's own taxonomy
                      field, so exact lookup hits it by identity rather than by capability.
  canonical values    dropped. 1,916 of them, which the span node was recovering for free
                      because the perturbation only rewrote values that had an accepted
                      paraphrase.

What remains is 710 sessions and 1,280 values, every one of them paraphrased, wrapped in
reworded templates, with no category to lean on. This is the honest measurement of what the
template machinery recovers when nothing is handed to it.

THE CONTROL IS THE SAME SESSIONS WITH THE SAME VALUE SLOTS, canonical rather than
paraphrased. Dropping values changes how much evidence a session carries at all, so
comparing against the full-card ceiling would confound paraphrasing with card size.

EXPECT THE SPAN NODE TO SCORE NEAR ZERO HERE, and that is the point rather than a
disappointment. It is exact lookup against catalogue vocabulary; every value it could match
has been reworded out of that vocabulary. If it still scores, the gain is coming from
somewhere that needs explaining.

Run:  PYTHONIOENCODING=utf-8 python -u robustness/v2/evaluate_hostile.py
"""
from __future__ import annotations

import importlib.util as ilu
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "robustness" / "v2"
OV = V2 / "open_vocabulary"
OUT = V2 / "results" / "hostile_v2_61.json"

_s = ilu.spec_from_file_location("_stress", ROOT / "experiments" / "scripts"
                                 / "31_paraphrase_stress.py")
_stress = ilu.module_from_spec(_s)
_s.loader.exec_module(_stress)

_t = ilu.spec_from_file_location("_tmpl", V2 / "run_official_template_paraphrase.py")
_tm = ilu.module_from_spec(_t)
_t.loader.exec_module(_tm)


def main() -> None:
    os.environ["LLM_RERANK"] = "0"
    os.environ["LLM_EXTRACT"] = "0"
    os.environ["LLM_RESOLVE"] = "0"
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    from submission.agent import Agent
    from submission.bert_extract import ScaffoldingTagger

    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    transform = _tm.transform(_tm.bank())
    canon = load_jsonl(OV / "review800_hostile_canonical.jsonl")
    para = load_jsonl(OV / "review800_hostile_paraphrase.jsonl")
    man = json.loads((OV / "hostile_manifest.json").read_text(encoding="utf-8"))
    print(f"hostile suite: {man['sessions']} sessions, {man['values_kept']} values "
          f"(all paraphrased), {man['values_dropped_as_canonical']} canonical values "
          f"dropped\n")

    # CATEGORY NEUTRALISATION. `initial_message` interpolates
    # `coarse_category(categories.get(target, []))`; an empty list makes that return the
    # constant "clothing item", so every session names the same non-informative category.
    blank_cats = {k: [] for k in cats}

    span, route = base.span_node, base.route_node
    ARMS = (("V1 baseline", False, False, False),
            ("+BERT", True, False, False),
            ("+SPAN", False, True, False),
            ("+ROUTE+SPAN", False, True, True),
            ("V2 full", True, True, True))

    def make(bert, use_span, use_route):
        a = object.__new__(Agent)
        a.ix, a.sessions = base.ix, {}
        a.llm = a.llm_extract = a.resolver = None
        os.environ["BERT_EXTRACT"] = "1" if bert else "0"
        a.tagger = ScaffoldingTagger() if bert else None
        a.span_node = span if use_span else None
        a.route_node = route if use_route else None
        return a

    def run(agent, samples, warp, categories):
        if warp:
            r = _stress.evaluate_transformed(agent, samples, cid, categories, prods,
                                             transform)
        else:
            r = evaluate(agent, samples, cid, categories, prods)
        return r

    t0 = time.time()
    # Ceiling: same sessions, canonical values, SAME neutralised category and SAME reworded
    # wrappers -- so the only difference from the floor is the wording of the values.
    ceil_r = run(make(False, True, True), canon, True, blank_cats)
    ceiling = round(ceil_r["recommended_technical_score"], 6)
    print(f"ceiling (canonical values, same hostile framing): {ceiling:.6f} "
          f"HR@10 {ceil_r['hit_rate_at_10']:.4f}\n")

    print(f"{'arm':<16}{'score':>11}{'HR@10':>9}{'MRR':>9}{'MTTC':>8}{'% of gap':>11}")
    print("-" * 64)
    table, floor = {}, None
    for label, bert, us, ur in ARMS:
        r = run(make(bert, us, ur), para, True, blank_cats)
        s = round(r["recommended_technical_score"], 6)
        if floor is None:
            floor = s
        gap = ceiling - floor
        table[label] = {"score": s, "hr10": round(r["hit_rate_at_10"], 4),
                        "mrr": round(r["mrr"], 4), "mttc": round(r["mttc"], 3)}
        pct = (s - floor) / gap if abs(gap) > 1e-9 else float("nan")
        print(f"{label:<16}{s:>11.6f}{r['hit_rate_at_10']:>9.4f}{r['mrr']:>9.4f}"
              f"{r['mttc']:>8.3f}{pct:>11.1%}", flush=True)

    gap = ceiling - floor
    print(f"\n  gap to close on the hostile condition: {gap:.6f}")
    best = max(table, key=lambda k: table[k]["score"])
    print(f"  best arm: {best} at {table[best]['score']:.6f} "
          f"({(table[best]['score']-floor)/gap:.1%} of gap)" if abs(gap) > 1e-9 else "")
    print(f"\n  For scale, on the flattering `both` condition the span node was worth")
    print(f"  +0.2001. Whatever it is worth here is what it recovers when the category is")
    print(f"  gone and every value has been reworded out of catalogue vocabulary.")
    print(f"\n  {time.time()-t0:.0f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "V2.61 hostile condition",
        "sessions": man["sessions"], "values": man["values_kept"],
        "category": "neutralised", "ceiling": ceiling, "floor": floor,
        "gap": round(gap, 6), "arms": table,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
