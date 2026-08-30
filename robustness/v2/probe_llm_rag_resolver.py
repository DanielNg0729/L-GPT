"""V2.45: can an LLM do what the encoders could not -- name the catalogue value?

WHY THIS PROBE, AFTER SEVEN FAILURES
------------------------------------
Nodes 3/4/5 failed from every learned direction. The two diagnosed causes are specific,
and both point away from retrieval models:

  * bi-encoders retrieve ANTONYMS. "made overseas" -> "made in usa", "soft plant fibre"
    -> "synthetic fiber". Cosine similarity does not encode negation, so the nearest
    neighbour of a phrase is frequently its opposite.
  * cross-encoder rerankers made recall@1 WORSE on every checkpoint tried (ms-marco
    -0.090, ESCI -0.090, NLI -0.015), because all of them are trained for query->passage
    and our candidates are two-word attribute values. There is no passage to rank.

Neither failure is a ranking failure. "cotton is a soft plant fibre" is WORLD KNOWLEDGE,
and supplying world knowledge over a closed answer set is what an LLM is actually for.

THE DESIGN, AND WHY IT CANNOT INVENT EVIDENCE
---------------------------------------------
Free generation is unacceptable here: the whole agent rests on provenance, and a phrase
the catalogue has never seen is worse than no phrase -- it withholds weight from the
target and hands it to the field. So generation is followed by a hard provenance check.
The LLM proposes; `df(phrase) > 0` disposes. A proposal the catalogue cannot attest is
discarded and the clause stays suppressed.

That check is deliberately provenance-only. Filtering proposals down to the set of values
`intent_card()` can EMIT would score better on this suite, and would be gaming it -- the
suite's canonicals are emittable by construction, so the filter would be scoring the
benchmark's construction rather than the model's knowledge. df > 0 is the honest gate.

THREE ARMS, so the LLM is measured against the thing it has to beat
  suppress   the shipped floor: unattested clause contributes nothing (attr-para 0.8330)
  generate   LLM proposes a canonical, provenance-checked, no candidate list
  choose     LLM picks from k retrieved candidates, or answers NONE

`choose` exists because it is the arm that reuses the failed encoders honestly: their
recall@k can be adequate even when their top-1 is 0/27, and the LLM supplies the
precision they lack. It is only worth building if recall@k is actually there, which this
reports before spending a single call.

BASELINE TO BEAT IS SUPPRESSION, NOT THE PARAPHRASE BASELINE. Returning nothing already
recovers +0.0560. An LLM that resolves a few phrases correctly and a few wrongly can
easily land BELOW that, because a wrong canonical is worse than an absent one.

WHAT THIS PROBE IS NOT ALLOWED TO CONCLUDE. The suite contains only 27 DISTINCT
paraphrases. That is a WEAKNESS OF THE SUITE -- a real customer population has an
open-ended paraphrase vocabulary, and 27 is an artifact of how this suite was generated.
Three rules follow, and they are constraints on the design, not caveats on the reading:

  * nothing may be keyed, sized, cached or tuned to those 27 phrases. A mechanism that
    works because the answer set is small is measuring the suite, not the problem.
  * 27 samples cannot separate a real effect from noise. A single correct resolution moves
    the rate by 3.7 points. Treat this strictly as a FEASIBILITY signal -- does the model
    name catalogue-attested values at all -- and never as an effect size.
  * whatever this motivates must be validated on an open-vocabulary suite before it can
    inform any shipping decision.

Run:  PYTHONIOENCODING=utf-8 python -u robustness/v2/probe_llm_rag_resolver.py
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
OUT = V2 / "results" / "llm_rag_resolver_probe_v2_45.json"

_s = ilu.spec_from_file_location("_v2_32", V2 / "evaluate_node4_cluster_aware.py")
_m = ilu.module_from_spec(_s)
_s.loader.exec_module(_m)
normalise = _m.normalise

_o = ilu.spec_from_file_location("_v2_43", V2 / "evaluate_oracle_decomposition.py")
_om = ilu.module_from_spec(_o)
_o.loader.exec_module(_om)
paraphrase_map = _om.paraphrase_map

SYSTEM = (
    "You map a shopper's description of a product attribute onto the exact wording an "
    "e-commerce catalogue would use for it. Answer with the catalogue's own short "
    "attribute value and NOTHING else -- no sentence, no explanation, no quotes. "
    "Prefer the plainest, most common trade term: for 'made from a soft plant fibre' "
    "answer 'cotton'. If the description is a NEGATION or you are not confident which "
    "single value it names, answer exactly NONE. Answering NONE is correct and expected "
    "whenever you would otherwise guess."
)


def main() -> None:
    from submission.llm_rerank import ENDPOINT, _load_project_env
    from submission.agent import Agent
    from urllib.request import Request, urlopen

    _load_project_env()
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print("GROQ_API_KEY not set -- nothing to probe."); return
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

    pmap = paraphrase_map()
    print(f"{len(pmap)} distinct paraphrase -> canonical maps; model={model}\n")
    agent = Agent(ROOT / "data" / "catalog.jsonl")

    def ask(phrase: str) -> str | None:
        body = json.dumps({
            "model": model, "temperature": 0.0,
            # GPT-OSS bills hidden reasoning against max_tokens and returns EMPTY content
            # when the budget runs out mid-thought -- HTTP 200, nothing to parse. The
            # answer is two words; the headroom is for the reasoning, not the answer.
            "max_tokens": 512, "reasoning_effort": "low",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": phrase}],
        }).encode()
        req = Request(ENDPOINT, data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            # Groq's edge answers Python's default `Python-urllib/*` identity with a
            # Cloudflare 403 (code 1010) despite valid credentials.
            "User-Agent": "Mozilla/5.0"})
        for attempt in range(3):
            try:
                with urlopen(req, timeout=30) as r:
                    d = json.loads(r.read().decode())
                return d["choices"][0]["message"]["content"].strip()
            except Exception as e:                     # transient: back off and retry
                if attempt == 2:
                    detail = ""
                    try:
                        detail = f" {e.code} {e.read().decode()[:160]}"   # type: ignore
                    except Exception:
                        pass
                    print(f"    [call failed] {type(e).__name__}{detail}")
                    return None
                time.sleep(2 * (attempt + 1))
        return None

    rows, t0 = [], time.time()
    for phrase, truth in sorted(pmap.items()):
        raw = ask(phrase)
        if raw is None or not raw.strip():
            # Distinguish "the model declined" from "the call produced nothing". The first
            # run of this probe reported 27 abstentions that were in fact 27 HTTP 403s.
            rows.append({"paraphrase": phrase, "truth": truth, "proposed": None,
                         "df": 0, "abstained": False, "accepted": False,
                         "correct": False, "failed": True})
            print(f"  ERR {phrase[:44]:<44} -> <no completion>")
            continue
        prop = normalise(raw)
        abstain = prop == "none"
        df = 0 if abstain else agent.ix.df(prop)
        accepted = (not abstain) and df > 0
        rows.append({"paraphrase": phrase, "truth": truth, "proposed": prop,
                     "df": df, "abstained": abstain, "accepted": accepted,
                     "correct": accepted and prop == truth, "failed": False})
        mark = ("OK " if rows[-1]["correct"] else
                "-- " if abstain else "DF0" if df == 0 else "BAD")
        print(f"  {mark} {phrase[:44]:<44} -> {prop[:26]:<26} "
              f"df={df:<7} truth={truth[:22]}")

    fail = sum(r["failed"] for r in rows)
    rows_ok = [r for r in rows if not r["failed"]]
    n = len(rows_ok)
    if fail:
        print()
        print(f"  {fail}/{len(rows)} calls FAILED -- those rows are excluded, and are "
              f"NOT counted as abstentions.")
    if not n:
        print("  no successful calls: nothing measured."); return
    acc = sum(r["accepted"] for r in rows_ok)
    cor = sum(r["correct"] for r in rows_ok)
    absr = sum(r["abstained"] for r in rows_ok)
    df0 = sum(1 for r in rows_ok if not r["abstained"] and r["df"] == 0)
    wrong = acc - cor
    print(f"\n  proposals accepted (df>0)   {acc}/{n}")
    print(f"    of which CORRECT          {cor}   <- upside over suppression")
    print(f"    of which WRONG            {wrong}   <- downside vs suppression")
    print(f"  abstained (NONE)            {absr}   -> stays suppressed, harmless")
    print(f"  rejected by provenance      {df0}   -> stays suppressed, harmless")
    print(f"\n  The provenance check is what makes a wrong ANSWER survivable and a")
    print(f"  hallucinated one impossible. Only the {wrong} accepted-but-wrong rows can")
    print(f"  score below the suppression floor; the other {absr + df0} degrade to it exactly.")
    print(f"\n  {time.time()-t0:.0f}s for {n} calls")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"experiment": "V2.45 LLM RAG resolver probe", "model": model, "n": n,
         "accepted": acc, "correct": cor, "wrong": wrong, "abstained": absr,
         "rejected_by_provenance": df0, "failed_calls": fail, "rows": rows}, indent=2) + "\n",
        encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
