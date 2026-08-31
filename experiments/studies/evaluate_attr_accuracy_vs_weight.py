"""V2.57: is the deparaphraser's shortfall ACCURACY or INTEGRATION?

THE CONFOUND THIS RESOLVES
--------------------------
V2.56 put the open-vocabulary attribute ceiling at 95.1% of the gap and the deparaphraser
at 17.2%, i.e. 18.1% of what is achievable. That comparison is not clean, and the flaw is
mine: the oracle arms inject at CONSTRAINT weight while the deparaphraser injects at the
attenuated SEM weight of 0.15. So the shortfall mixes two very different causes.

Worse, V2.56's ORACLE-FAMILY arm injects a SINGLE correct token from the true atom -- often
not even the informative one ("hand wash only" -> "only") -- and still recovers 54.6%. One
token at full weight beating the entire LLM layer threefold is not a plausible statement
about knowledge. It is a statement about weight.

WHAT IS MEASURED, separating the two axes explicitly:

  ACCURACY   the resolver's proposals scored against the TRUE atom each paraphrase was
             generated from, at three strengths: exact match, token-overlap (does the
             proposal contain a token of the atom, which is what the ranker actually
             rewards), and abstention rate.
  INTEGRATION the same proposals injected end to end at SEM 0.15, at SEM 0.45, and at full
             CONSTRAINT weight. Plus ORACLE at the SEM weight, which is the missing cell:
             it says what a PERFECT resolver would score if it were attenuated the way the
             real one is.

READING IT
    oracle@CONSTRAINT - oracle@SEM     the cost of attenuation alone, knowledge held perfect
    llm@CONSTRAINT - llm@SEM           the same cost paid by the real resolver
    oracle@SEM - llm@SEM               the genuine ACCURACY shortfall, weight held equal

If oracle@SEM is close to llm@SEM, the resolver is nearly as good as perfect knowledge AT
THAT WEIGHT and the whole shortfall is attenuation. If oracle@SEM stays high, attenuation is
cheap and the resolver is simply wrong too often.

WHY THE ANSWER IS NOT OBVIOUS. On the OLD 27-phrase suite attenuation HELPED -- full
CONSTRAINT weight scored 81.5% of oracle against ~96% attenuated -- because a wrong
proposal at full weight outranks the correct evidence beside it. That was measured where
the resolver was accurate. Where it is less accurate, attenuation should help MORE, not
less. If it does not, the failure is elsewhere.

Nothing here is tuned: the two SEM weights are the ones already measured on the old suite,
and CONSTRAINT is the untenuated bound. The suite stays characterisation, so none of this
can license a change to the decision criteria.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/studies/evaluate_attr_accuracy_vs_weight.py
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
V2 = ROOT / "experiments" / "studies"
OV = V2 / "open_vocabulary"
OUT = V2 / "results" / "attr_accuracy_vs_weight_v2_57.json"

SEM = "sem"

_e = ilu.spec_from_file_location("_v2_46", V2 / "evaluate_llm_resolver_end_to_end.py")
_em = ilu.module_from_spec(_e)
_e.loader.exec_module(_em)


def main() -> None:
    os.environ["LLM_RERANK"] = "0"
    os.environ["LLM_EXTRACT"] = "0"
    os.environ["BERT_EXTRACT"] = "0"
    os.environ["V2_ROUTE"] = "0"
    os.environ.setdefault("LLM_RESOLVE_CACHE",
                          str(OV / ".resolver_cache_open_vocab.json"))
    _em.CACHE = OV / ".resolver_cache_open_vocab.json"
    from submission.llm_rerank import ENDPOINT, _load_project_env
    _em.ENDPOINT = ENDPOINT
    _load_project_env()
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print("GROQ_API_KEY not set."); return
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    from submission.agent import CONSTRAINT, Agent, raw_toks

    rows = [json.loads(l) for l in
            (V2 / "open_vocabulary_paraphrases.jsonl").open(encoding="utf-8") if l.strip()]
    pmap = {}
    for r in rows:
        para = " ".join(raw_toks(str(r.get("paraphrase", ""))))
        atom = " ".join(raw_toks(str(r.get("atom", ""))))
        if para and atom and para != "skip":
            pmap[para] = atom

    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    canon_s = load_jsonl(OV / "review800_canonical_replay.jsonl")
    para_s = load_jsonl(OV / "review800_open_vocab_paraphrase.jsonl")
    res = _em.Resolver(base.ix, model, key)
    print(f"{len(pmap)} paraphrase -> atom mappings; model={model}\n")

    # ---- ACCURACY, independent of how the result is integrated -----------------------
    t0 = time.time()
    exact = overlap = abstain = wrong = 0
    detail = []
    for para, atom in sorted(pmap.items()):
        prop = res.resolve(para)
        if not prop:
            abstain += 1
            detail.append({"paraphrase": para, "atom": atom, "proposal": None})
            continue
        ptoks, atoks = set(raw_toks(prop)), set(raw_toks(atom))
        is_exact = prop == atom
        is_overlap = bool(ptoks & atoks)
        exact += is_exact
        overlap += is_overlap
        wrong += (not is_overlap)
        detail.append({"paraphrase": para, "atom": atom, "proposal": prop,
                       "exact": is_exact, "overlap": is_overlap})
    res.flush()
    n = len(pmap)
    print(f"{'ACCURACY of the resolver against the true atom':<50}")
    print("-" * 50)
    print(f"  exact match                {exact:>4}/{n}  {exact/n:>6.1%}")
    print(f"  shares >=1 token           {overlap:>4}/{n}  {overlap/n:>6.1%}"
          f"   <- what the ranker rewards")
    print(f"  abstained                  {abstain:>4}/{n}  {abstain/n:>6.1%}")
    print(f"  answered but NO overlap    {wrong:>4}/{n}  {wrong/n:>6.1%}   <- actively wrong")
    print(f"  resolver calls={res.calls} accepted={res.accepted} "
          f"abstained={res.abstained} unattested={res.unattested} "
          f"failed={res.failed}")
    print()

    # ---- INTEGRATION -----------------------------------------------------------------
    def arm(source: str, weight):
        class Arm(Agent):
            def _observe(self, st, msg):
                super()._observe(st, msg)
                for text, tier in super()._extract_templated(msg):
                    if tier != CONSTRAINT:
                        continue
                    key_ = " ".join(raw_toks(text))
                    if key_ not in pmap:
                        continue
                    ph = pmap[key_] if source == "oracle" else res.resolve(key_)
                    if not ph:
                        continue
                    df = self.ix.df(ph)
                    if df > 0 and ph not in st.evidence:
                        st.evidence[ph] = (df, CONSTRAINT if weight is None else SEM)

            def _weight(self, phrase, df, tier):
                if tier == SEM and weight is not None:
                    return weight / (1.0 + df) ** self.IDF_POW
                return super()._weight(phrase, df, tier)
        return Arm

    def run(cls, samples):
        a = object.__new__(cls)
        a.ix, a.sessions = base.ix, {}
        a.llm = a.llm_extract = a.tagger = a.resolver = a.route_node = None
        a.span_node = None
        return round(evaluate(a, samples, cid, cats, prods)[
            "recommended_technical_score"], 6)

    ceiling = run(Agent, canon_s)
    floor = run(Agent, para_s)
    gap = ceiling - floor
    cells = {}
    for source in ("oracle", "llm"):
        for label, w in (("SEM 0.15", 0.15), ("SEM 0.45", 0.45), ("CONSTRAINT", None)):
            cells[(source, label)] = run(arm(source, w), para_s)

    def frac(v):
        return (v - floor) / gap if abs(gap) > 1e-9 else float("nan")

    print(f"{'INTEGRATION':<22}{'oracle':>22}{'llm':>22}")
    print(f"{'weight':<22}{'score':>11}{'% gap':>11}{'score':>11}{'% gap':>11}")
    print("-" * 66)
    for label in ("SEM 0.15", "SEM 0.45", "CONSTRAINT"):
        o, l = cells[("oracle", label)], cells[("llm", label)]
        print(f"{label:<22}{o:>11.6f}{frac(o):>11.1%}{l:>11.6f}{frac(l):>11.1%}")
    print(f"{'(floor)':<22}{floor:>11.6f}{0.0:>11.1%}")
    print(f"{'(ceiling)':<22}{ceiling:>11.6f}{1.0:>11.1%}")

    o_c, o_s = cells[("oracle", "CONSTRAINT")], cells[("oracle", "SEM 0.15")]
    l_c, l_s = cells[("llm", "CONSTRAINT")], cells[("llm", "SEM 0.15")]
    print(f"\n  cost of attenuation, knowledge PERFECT   {o_s - o_c:+.6f}")
    print(f"  cost of attenuation, real resolver       {l_s - l_c:+.6f}")
    print(f"  ACCURACY shortfall at equal weight       {l_s - o_s:+.6f}  (SEM 0.15)")
    print(f"                                           {l_c - o_c:+.6f}  (CONSTRAINT)")
    print(f"\n  If the two shortfalls are similar, the gap is knowledge, not integration.")
    print(f"  {time.time()-t0:.0f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "V2.57 attribute resolver: accuracy vs integration",
        "n": n, "exact": exact, "overlap": overlap, "abstain": abstain, "wrong": wrong,
        "ceiling": ceiling, "floor": floor, "gap": round(gap, 6),
        "cells": {f"{s}@{w}": v for (s, w), v in cells.items()},
        "resolver": {"calls": res.calls, "accepted": res.accepted,
                     "abstained": res.abstained,
                     "unattested": res.unattested, "failed": res.failed},
        "detail": detail,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
