"""V2.49: node 7's actual clearance gate -- the LLM resolver against every population.

WHY THIS RUN EXISTS
-------------------
Node 7's clearance condition, written before any of this was built, reads:

    "select one monotonic schedule only after Node 6 is calibrated and both Official200
     and Unseen800 show no ranking regression."

V2.46 measured official200 (0.970100, unchanged) and the attribute-paraphrase suite
(+0.0372). It did NOT measure org-proxy, review800, uniform or inverse. So node 7 has
never actually been cleared, and the resolver's population behaviour is unknown rather
than good.

THE ASSUMPTION THIS IS TESTING, AND WHY IT IS NOT TAKEN ON TRUST
----------------------------------------------------------------
The resolver fires only where `df(clause) == 0`, which on canonical traffic should be
almost never -- official200 hits it once in 463 messages. The natural inference is that
the population suites are equally safe and the run is a formality.

That inference is exactly the one that already failed once today. The same path was
claimed UNREACHABLE on clean traffic by construction, on the strength of the recognition
gate matching 463/463 messages, and measuring it showed the gate governs messages while
`intent_card()` truncation can still strip a value to something unattested. "Probably
near-zero" is a prediction; node 7 asks for a measurement.

The populations differ from official200 in what they sample, not in how they phrase it, so
each one carries a different rate of long feature bullets and therefore a different rate
of truncation -- which is the mechanism that reaches this path at all. `inverse` is the
adversarial bound and the one most likely to expose it.

WHAT IS REPORTED, and one distinction the first version of this file got wrong.

Suppression (the shipped agent) against the LLM arm at the attenuated weight, per suite,
plus how often each suite REACHES the resolver.

REACHES is not the same as CALLS, and conflating them produced a false claim. The resolver
keeps a cache, so `calls` counts cache MISSES: the first run of this file reported
official200 at "0 calls -> byte-identical by construction" while the attribute-paraphrase
suite also showed 0 calls despite plainly resolving 20 phrases from cache. A cached
resolution is still a resolution.

Reaches are therefore counted separately, by a probe that overrides `_resolve` purely to
count and changes no behaviour. That also fixes a COVERAGE gap: the scoring arm hooks
`_observe`/`_extract_templated`, but `_resolve` is additionally called from
`_seed_from_override_opening`, and that is the path by which official200's one known
unattested clause (an `intent_card()` bullet truncated mid-word) arrives. The scoring hook
cannot see it, so absence of calls there was never evidence of absence of reach.

DECISION RULE, pre-registered: the four population suites plus official200 are the
decision criteria. attr-para is characterisation only -- the organizer confirmed no
paraphrasing, so a gain there cannot license a regression anywhere else.

Run:  PYTHONIOENCODING=utf-8 python -u robustness/v2/evaluate_llm_resolver_populations.py
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
PVO = V2 / "public_value_only"
OUT = V2 / "results" / "llm_resolver_populations_v2_49.json"

SEM = "sem"
WEIGHT = float(os.environ.get("SEM_WEIGHT", "0.15"))

_e = ilu.spec_from_file_location("_v2_46", V2 / "evaluate_llm_resolver_end_to_end.py")
_em = ilu.module_from_spec(_e)
_e.loader.exec_module(_em)


def main() -> None:
    os.environ["LLM_EXTRACT"] = "0"
    os.environ["BERT_EXTRACT"] = "0"
    from submission.llm_rerank import ENDPOINT, _load_project_env
    _em.ENDPOINT = ENDPOINT
    _load_project_env()
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print("GROQ_API_KEY not set -- nothing to evaluate."); return
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    from submission.agent import CONSTRAINT, Agent, raw_toks

    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    rs = ROOT / "robustness" / "sets"
    sets = {
        "official200": load_jsonl(ROOT / "data" / "public_set.jsonl"),
        "org-proxy": load_jsonl(rs / "organizer_proxy_800.jsonl"),
        "review800": load_jsonl(rs / "catalog_review_distinct_800.jsonl"),
        "uniform": load_jsonl(rs / "catalog_uniform_800.jsonl"),
        "inverse": load_jsonl(rs / "catalog_inverse_800.jsonl"),
        "attr-para": load_jsonl(PVO / "official200_attribute_paraphrase_dev.jsonl"),
    }
    DECISIVE = ("official200", "org-proxy", "review800", "uniform", "inverse")
    res = _em.Resolver(base.ix, model, key)
    print(f"model={model}  weight={WEIGHT}\n")

    class Arm(Agent):
        def _observe(self, st, msg):
            super()._observe(st, msg)
            for text, tier in super()._extract_templated(msg):
                if tier != CONSTRAINT:
                    continue
                toks = raw_toks(text)[:self.RESOLVE_CAP]
                if not toks or self.ix.df(" ".join(toks)) > 0:
                    continue                        # attested: suppression never fired
                prop = res.resolve(" ".join(toks))
                if prop and prop not in st.evidence:
                    st.evidence[prop] = (self.ix.df(prop), SEM)

        def _weight(self, phrase, df, tier):
            if tier == SEM:
                return WEIGHT / (1.0 + df) ** self.IDF_POW
            return super()._weight(phrase, df, tier)

    def run(cls, samples):
        a = object.__new__(cls)
        a.ix, a.sessions = base.ix, {}
        a.llm = a.llm_extract = a.tagger = None
        return round(evaluate(a, samples, cid, cats, prods)[
            "recommended_technical_score"], 6)

    # Counts every clause that reaches the resolver path, cached or not, across ALL of
    # `_resolve`'s call sites. Behaviour is identical to the shipped agent.
    reaches = {"n": 0}

    class ReachProbe(Agent):
        def _resolve(self, text, cap=None):
            out = super()._resolve(text, cap)
            if not out and raw_toks(text)[:self.RESOLVE_CAP if cap is None else cap]:
                reaches["n"] += 1
            return out

    t0, rows = time.time(), {}
    print(f"{'suite':<14}{'suppression':>13}{'LLM arm':>11}{'delta':>11}"
          f"{'reaches':>9}{'calls':>7}")
    print("-" * 63)
    for name, sub in sets.items():
        before_c, reaches["n"] = res.calls, 0
        supp = run(Agent, sub)
        run(ReachProbe, sub)
        llm = run(Arm, sub)
        rows[name] = {"suppression": supp, "llm": llm, "delta": round(llm - supp, 6),
                      "reaches": reaches["n"], "calls": res.calls - before_c}
        print(f"{name:<14}{supp:>13.6f}{llm:>11.6f}{llm - supp:>+11.6f}"
              f"{rows[name]['reaches']:>9}{rows[name]['calls']:>7}", flush=True)
    res.flush()

    worst = min(rows[c]["delta"] for c in DECISIVE)
    silent = [c for c in DECISIVE if rows[c]["reaches"] == 0]
    print(f"\n  worst decision-criterion delta: {worst:+.6f}")
    if silent:
        print(f"  suites the resolver NEVER reaches (unchanged by control flow): "
              f"{', '.join(silent)}")
    live = [c for c in DECISIVE if rows[c]["reaches"] > 0]
    if live:
        detail = ", ".join("%s (%d)" % (c, rows[c]["reaches"]) for c in live)
        print(f"  suites that DID reach the resolver: {detail}")
        print("  -- on these the guarantee is empirical, not structural.")
    verdict = ("CLEARS node 7 -- no regression on any decision criterion" if worst >= -1e-9
               else "inside noise" if worst > -0.005 else "BLOCKS node 7")
    print(f"\n  verdict: {verdict}")
    print(f"  attr-para {rows['attr-para']['delta']:+.6f} is characterisation only and "
          f"cannot license a regression above.")
    print(f"\n  resolver: {res.calls} calls -- {res.accepted} accepted, "
          f"{res.abstained} abstained, {res.unattested} unattested, {res.failed} failed")
    print(f"  {time.time() - t0:.0f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"experiment": "V2.49 LLM resolver against every population",
         "model": model, "weight": WEIGHT, "suites": rows,
         "worst_decision_delta": worst, "verdict": verdict,
         "resolver": {"calls": res.calls, "accepted": res.accepted,
                      "abstained": res.abstained, "unattested": res.unattested,
                      "failed": res.failed}}, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
