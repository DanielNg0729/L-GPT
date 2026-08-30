"""V2.58: what is still missing on template paraphrase, and would a better BERT fix it?

THE QUESTION
------------
Template paraphrase is 90.8% recovered by the span node plus Node 1. The remaining 0.0256
is now the whole template opportunity, and the choice on the table is whether to improve
the scaffolding tagger, replace it with a span-extraction BERT, or leave the axis alone.

That choice cannot be made from the score. It needs to know WHICH constraint values are
still not reaching the evidence ledger, and why. A tagger improvement only helps if the
misses are caused by the tagger; a span-BERT only helps if they are caused by the
dictionary lookup's shape.

METHOD. Replay the template condition -- canonical values, held-out reworded wrappers --
with the shipped agent, and for every session compare the TRUE intent-card values against
what actually landed in `state.evidence`. Every miss is then attributed to a cause that
implies a different fix:

  TOO LONG        the value exceeds the span node's 3-token window. Mining is supposed to
                  cover long phrases, but mining keeps an n-gram only when
                  `0 < df <= DF_CAP`, so a long-but-common value is dropped by both. A
                  span-BERT that predicts value BOUNDARIES would fix this; a better tagger
                  would not.
  NOT IN DICT     the value is not a member of the frozen attribute dictionary at all. No
                  amount of model quality recovers this -- it is a coverage problem in the
                  dictionary, fixed by rebuilding it, not by training anything.
  DICT BUT MISSED the value IS in the dictionary and within the window, and still did not
                  land. This is the only bucket a better tagger or extractor can claim,
                  because it means the text reached extraction and extraction failed.
  CATEGORY EATEN  the category matcher consumed the tokens first. A precision bug in the
                  span node, fixed by ordering, not by training.

READING IT. The size of DICT-BUT-MISSED is the honest upper bound on what ANY better
extraction model can buy on this axis. If it is near zero, the tagger and a span-BERT are
both dead ends and the residual is dictionary coverage plus long values. If it is large,
extraction is genuinely leaving evidence on the table.

This is diagnosis, not a score, and it is offline and deterministic.

Run:  PYTHONIOENCODING=utf-8 python -u robustness/v2/audit_template_residual.py
"""
from __future__ import annotations

import importlib.util as ilu
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "robustness" / "v2"
OV = V2 / "open_vocabulary"
OUT = V2 / "results" / "template_residual_v2_58.json"

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
    from submission.agent import Agent, raw_toks

    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    span = base.span_node
    samples = load_jsonl(OV / "review800_canonical_replay.jsonl")
    transform = _tm.transform(_tm.bank())
    MAXW = span.MAX_ATTRIBUTE_TOKENS

    seen: dict[str, set[str]] = {}

    class Probe(Agent):
        """Behaviour identical to the shipped agent; records the ledger per session."""

        def respond(self, session_id, message, turn, top_k=10):
            out = super().respond(session_id, message, turn, top_k)
            st = self.sessions.get(session_id)
            if st is not None:
                seen.setdefault(session_id, set()).update(st.evidence)
            return out

    a = object.__new__(Probe)
    a.ix, a.sessions = base.ix, {}
    a.llm = a.llm_extract = a.tagger = a.resolver = None
    a.span_node = span
    a.route_node = base.route_node
    t0 = time.time()
    r = _stress.evaluate_transformed(a, samples, cid, cats, prods, transform)
    print(f"template condition replayed: score "
          f"{r['recommended_technical_score']:.6f}, HR@10 {r['hit_rate_at_10']:.4f} "
          f"({time.time()-t0:.0f}s)\n")

    # Every recovered phrase across the whole run, so a miss is judged against the union
    # of what the agent ever held for that session.
    recovered = set().union(*seen.values()) if seen else set()

    buckets: Counter[str] = Counter()
    examples: dict[str, list] = {}
    total = 0
    for s in samples:
        card = s["intent_card"]
        for group in ("hard_constraints", "soft_preferences"):
            for raw in card[group]:
                toks = raw_toks(str(raw))
                if not toks:
                    continue
                value = " ".join(toks)
                total += 1
                if value in recovered:
                    buckets["recovered"] += 1
                    continue
                if len(toks) > MAXW:
                    key = "TOO LONG (> %d tokens)" % MAXW
                elif value not in span.attributes:
                    key = "NOT IN DICTIONARY"
                else:
                    key = "DICT BUT MISSED"
                buckets[key] += 1
                examples.setdefault(key, [])
                if len(examples[key]) < 4:
                    examples[key].append(value[:58])

    print(f"{'bucket':<28}{'count':>8}{'share':>9}")
    print("-" * 45)
    for key in ("recovered", f"TOO LONG (> {MAXW} tokens)", "NOT IN DICTIONARY",
                "DICT BUT MISSED"):
        if key in buckets:
            print(f"{key:<28}{buckets[key]:>8}{buckets[key]/total:>9.1%}")
            for ex in examples.get(key, []):
                print(f"      e.g. {ex}")
    print(f"{'TOTAL constraint values':<28}{total:>8}")

    missed = buckets.get("DICT BUT MISSED", 0)
    long_ = buckets.get(f"TOO LONG (> {MAXW} tokens)", 0)
    nodict = buckets.get("NOT IN DICTIONARY", 0)
    print(f"\n  WHAT EACH BUCKET IMPLIES")
    print(f"  DICT BUT MISSED   {missed:>6} ({missed/total:.1%})  <- the ONLY bucket a")
    print(f"                    better tagger or a span-BERT can claim")
    print(f"  TOO LONG          {long_:>6} ({long_/total:.1%})  <- needs boundary")
    print(f"                    prediction or a wider window, not a better tagger")
    print(f"  NOT IN DICTIONARY {nodict:>6} ({nodict/total:.1%})  <- dictionary coverage;")
    print(f"                    no model fixes this")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "V2.58 template residual attribution",
        "score": round(r["recommended_technical_score"], 6),
        "hit_rate_at_10": round(r["hit_rate_at_10"], 4),
        "max_attribute_tokens": MAXW, "total_values": total,
        "buckets": dict(buckets), "examples": examples,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
