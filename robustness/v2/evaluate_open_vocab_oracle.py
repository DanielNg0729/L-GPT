"""V2.56: what is the CEILING on open-vocabulary attribute paraphrase?

THE QUESTION THIS ANSWERS BEFORE ANY MORE EFFORT IS SPENT
----------------------------------------------------------
Template paraphrase is now 90.8% recovered by the span node plus Node 1. Attribute
paraphrase is not: every arm scores 0.847103 there, because both of those layers are gated
behind `not recognised(msg)` and attribute paraphrase leaves the WRAPPER intact. The
message is recognised, the template regex hands over the value cleanly, `_resolve` finds
`df == 0`, and suppression drops it. On this axis the only live mechanism is suppression
plus the optional deparaphraser, which is worth +0.0169.

Whether +0.0169 is good or bad depends entirely on a number nobody has measured: how much
of this gap is recoverable AT ALL. V2.43 measured that on the OLD 27-phrase suite and found
a perfect resolver recovers only 50.7% -- paraphrasing destroys information that no
resolution can rebuild. That figure cannot be carried over: this suite has different
targets, 204 independently generated paraphrases, and the high-frequency atoms removed.

ARMS. Every one is deterministic and offline; the resolver is not consulted.

  0 canonical replay   values unchanged. The ceiling, and the control -- the same session
                       base materialised the same way, so the gap is not confounded with
                       materialisation.
  1 suppression        the shipped agent. The floor.
  2 ORACLE-RESOLVE     every paraphrased value replaced by the TRUE atom it was generated
                       from, injected as ordinary constraint evidence. A PERFECT resolver.
                       Uses ground truth, is a diagnostic only, and can never ship.
  3 ORACLE-DROP        every paraphrased value deleted instead of resolved. Separates two
                       diagnoses: if dropping recovers most of the gap then the paraphrased
                       text is POLLUTING rather than merely uninformative, and the fix is
                       suppression rather than resolution.
  4 ORACLE-FAMILY      the true atom's coarse family is revealed but not the value, by
                       injecting only the atom's LAST token. This prices Node 3 (family
                       routing) separately from Node 4: if arm 4 captures most of arm 2,
                       knowing WHICH attribute is being described is worth nearly as much
                       as knowing its value, and a family classifier is the cheaper target.

READING IT
    arm2 - arm1   the total value of perfect attribute resolution. THE CEILING.
    arm3 - arm1   how much is pollution rather than missing evidence
    arm4 - arm1   how much comes from family alone
    0.0169        what the LLM deparaphraser currently captures, for scale

If arm2 - arm1 is small, this axis is nearly closed and further work is misallocated. If it
is large, the deparaphraser's 17.2% is a poor showing and the headroom is real.

Run:  PYTHONIOENCODING=utf-8 python -u robustness/v2/evaluate_open_vocab_oracle.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "robustness" / "v2"
OV = V2 / "open_vocabulary"
OUT = V2 / "results" / "open_vocab_oracle_v2_56.json"

SEM = "sem"


def main() -> None:
    os.environ["LLM_RERANK"] = "0"
    os.environ["LLM_EXTRACT"] = "0"
    os.environ["LLM_RESOLVE"] = "0"
    os.environ["BERT_EXTRACT"] = "0"
    os.environ["V2_ROUTE"] = "0"          # attribute paraphrase never reaches it anyway
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    from submission.agent import CONSTRAINT, Agent, raw_toks

    # paraphrase -> true atom, recovered from the generation record the suite was built
    # from. This is ground truth and is why these arms are diagnostics, not candidates.
    rows = [json.loads(l) for l in
            (V2 / "open_vocabulary_paraphrases.jsonl").open(encoding="utf-8") if l.strip()]
    pmap = {}
    for r in rows:
        para = " ".join(raw_toks(str(r.get("paraphrase", ""))))
        atom = " ".join(raw_toks(str(r.get("atom", ""))))
        if para and atom and para != "skip":
            pmap[para] = atom
    print(f"{len(pmap)} paraphrase -> atom mappings")

    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    canon_s = load_jsonl(OV / "review800_canonical_replay.jsonl")
    para_s = load_jsonl(OV / "review800_open_vocab_paraphrase.jsonl")

    def arm(mode: str):
        class Arm(Agent):
            def _observe(self, st, msg):
                if mode == "drop":
                    for text, tier in super()._extract_templated(msg):
                        if tier == CONSTRAINT and " ".join(raw_toks(text)) in pmap:
                            msg = msg.replace(text, "")
                    return super()._observe(st, msg)
                super()._observe(st, msg)
                if mode == "baseline":
                    return
                for text, tier in super()._extract_templated(msg):
                    if tier != CONSTRAINT:
                        continue
                    atom = pmap.get(" ".join(raw_toks(text)))
                    if not atom:
                        continue
                    # `family` reveals only the head noun of the true atom, never the atom.
                    phrase = atom if mode == "oracle" else raw_toks(atom)[-1]
                    df = self.ix.df(phrase)
                    if df > 0 and phrase not in st.evidence:
                        st.evidence[phrase] = (df, CONSTRAINT)
        return Arm

    def run(cls, samples):
        a = object.__new__(cls)
        a.ix, a.sessions = base.ix, {}
        a.llm = a.llm_extract = a.tagger = a.resolver = a.route_node = None
        a.span_node = None
        return round(evaluate(a, samples, cid, cats, prods)[
            "recommended_technical_score"], 6)

    t0 = time.time()
    ceiling = run(Agent, canon_s)
    floor = run(arm("baseline"), para_s)
    oracle = run(arm("oracle"), para_s)
    drop = run(arm("drop"), para_s)
    family = run(arm("family"), para_s)
    gap = ceiling - floor

    def frac(x):
        return (x - floor) / gap if abs(gap) > 1e-9 else float("nan")

    print(f"\n{'arm':<34}{'score':>11}{'vs floor':>11}{'% of gap':>11}")
    print("-" * 67)
    for label, v in (("0 canonical replay (ceiling)", ceiling),
                     ("1 suppression (floor)", floor),
                     ("2 ORACLE-RESOLVE (perfect)", oracle),
                     ("3 ORACLE-DROP (delete value)", drop),
                     ("4 ORACLE-FAMILY (head noun only)", family)):
        print(f"{label:<34}{v:>11.6f}{v - floor:>+11.6f}{frac(v):>11.1%}")
    print(f"{'-- LLM deparaphraser, measured --':<34}{floor + 0.016866:>11.6f}"
          f"{0.016866:>+11.6f}{frac(floor + 0.016866):>11.1%}")

    print(f"\n  gap to close                     {gap:.6f}")
    print(f"  a PERFECT resolver would recover  {oracle-floor:+.6f}  "
          f"({frac(oracle):.1%})  <- THE CEILING")
    print(f"  deleting the value instead        {drop-floor:+.6f}  ({frac(drop):.1%})")
    print(f"  family (head noun) alone          {family-floor:+.6f}  ({frac(family):.1%})")
    if oracle - floor > 1e-9:
        print(f"\n  the deparaphraser captures {0.016866/(oracle-floor):.1%} of what is "
              f"achievable, not {frac(floor+0.016866):.1%} of the gap.")
        print(f"  headroom still on the table: {(oracle-floor)-0.016866:+.6f}")
    print(f"\n  {time.time()-t0:.0f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "V2.56 open-vocabulary attribute ceiling",
        "mappings": len(pmap), "ceiling": ceiling, "floor": floor, "gap": round(gap, 6),
        "oracle_resolve": oracle, "oracle_drop": drop, "oracle_family": family,
        "llm_measured": 0.016866,
        "fraction_of_gap": {"oracle": round(frac(oracle), 4),
                            "drop": round(frac(drop), 4),
                            "family": round(frac(family), 4),
                            "llm": round(frac(floor + 0.016866), 4)},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
