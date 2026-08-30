"""V2.65: where does paraphrase damage actually land, by scenario?

THE HYPOTHESIS, AND WHY IT IS NOT OBVIOUS
------------------------------------------
The three scenarios carry different evidence by construction:

    browsing          a CATEGORY only. "I'm looking for X, but I'm still exploring."
    buying            a category AND a constraint value
    intent_override   a category, an old value, then a replacement value

On clean traffic browsing is the STRONGEST scenario (official200: 0.980000, HR@10 1.0000),
which is already informative -- a category alone is enough to win, because the hostile
experiment showed the category carries roughly two thirds of the achievable score.

That suggests a prediction worth testing rather than assuming. Paraphrase perturbs
different channels in different conditions:

    template     the WRAPPER is reworded. The category survives as a string, and the span
                 node recovers it by exact taxonomy match, so browsing should hold up.
    attribute    the VALUES are reworded and wrappers are intact. Browsing HAS no value to
                 rewrite, so browsing should be almost untouched while buying takes the
                 whole hit.
    both         both channels reworded at once.

If that holds, the damage is concentrated in the scenarios that depend on a CONSTRAINT
surviving, and every paraphrase experiment in this project has been measuring the smaller
half of the traffic. It also says where any remaining effort belongs.

WHY SLICE BY SCENARIO RATHER THAN BY EVIDENCE TYPE. The scenario label is what the released
simulator actually branches on, so it is the slice the organizer's own generator produces.
Slicing by "what evidence did the agent end up with" would be slicing by our own behaviour,
which confounds the measurement with the thing being measured.

Offline and deterministic; no LLM, no network.

Run:  PYTHONIOENCODING=utf-8 python -u robustness/v2/audit_scenario_under_paraphrase.py
"""
from __future__ import annotations

import importlib.util as ilu
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "robustness" / "v2"
OV = V2 / "open_vocabulary"
OUT = V2 / "results" / "scenario_under_paraphrase_v2_65.json"

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
    os.environ["BERT_EXTRACT"] = "0"
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    from submission.agent import Agent

    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    transform = _tm.transform(_tm.bank())

    conditions = {
        "unseen800 (clean)": (load_jsonl(OV / "review800_canonical_replay.jsonl"), False),
        "template": (load_jsonl(OV / "review800_canonical_replay.jsonl"), True),
        "attribute": (load_jsonl(OV / "review800_open_vocab_paraphrase.jsonl"), False),
        "both": (load_jsonl(OV / "review800_open_vocab_paraphrase.jsonl"), True),
    }

    def make():
        a = object.__new__(Agent)
        a.ix, a.sessions = base.ix, {}
        a.llm = a.llm_extract = a.tagger = a.resolver = None
        a.span_node, a.route_node = base.span_node, base.route_node
        return a

    t0, table = time.time(), defaultdict(dict)
    scenarios = ("browsing", "buying", "intent_override", "boundary")
    for cname, (samples, warp) in conditions.items():
        by = defaultdict(list)
        for s in samples:
            by[s["scenario_type"]].append(s)
        for sc in scenarios:
            sub = by.get(sc) or []
            if not sub:
                continue
            if warp:
                r = _stress.evaluate_transformed(make(), sub, cid, cats, prods, transform)
            else:
                r = evaluate(make(), sub, cid, cats, prods)
            table[cname][sc] = {"n": len(sub),
                                "score": round(r["recommended_technical_score"], 6),
                                "hr10": round(r["hit_rate_at_10"], 4),
                                "mttc": round(r["mttc"], 3)}
        print(f"{cname} done ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'scenario':<18}" + "".join(f"{c:>21}" for c in conditions))
    print("-" * (18 + 21 * len(conditions)))
    for sc in scenarios:
        if not any(sc in table[c] for c in conditions):
            continue
        cells = []
        for c in conditions:
            d = table[c].get(sc)
            cells.append(f"{d['score']:>12.6f}/{d['hr10']:<8.3f}" if d else " " * 21)
        print(f"{sc:<18}" + "".join(cells))

    clean = "unseen800 (clean)"
    print(f"\ndamage vs clean, by scenario  (score delta)")
    print(f"{'scenario':<18}" + "".join(f"{c:>14}" for c in conditions if c != clean))
    print("-" * (18 + 14 * (len(conditions) - 1)))
    for sc in scenarios:
        if sc not in table[clean]:
            continue
        base_s = table[clean][sc]["score"]
        row = "".join(f"{table[c][sc]['score'] - base_s:>+14.6f}"
                      for c in conditions if c != clean and sc in table[c])
        print(f"{sc:<18}{row}")

    print(f"\n  THE PREDICTION UNDER TEST: browsing carries a CATEGORY and no constraint,")
    print(f"  so `attribute` paraphrase -- which rewrites values and leaves wrappers")
    print(f"  intact -- should barely touch it, while buying takes the whole hit. If that")
    print(f"  holds, paraphrase damage is concentrated in the scenarios that need a")
    print(f"  CONSTRAINT to survive, and the category channel is doing the heavy lifting")
    print(f"  everywhere it is available.")
    print(f"\n  {time.time()-t0:.0f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "V2.65 paraphrase damage by scenario",
        "conditions": {k: dict(v) for k, v in table.items()},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
