"""Does the transcript rescue do better at full strength or attenuated?

THE QUESTION. The rescue enters its recovered requirements at CONSTRAINT weight -- 1.0, the
same as something the customer literally said -- replicating L-GPT's own policy. Our
deparaphraser answers the same shape of question and does NOT: it enters proposals at
W_SEM, a fraction of constraint strength, because a model's inference about what was meant
is not the same kind of evidence as a phrase the customer spoke.

That decision was measured on our layer and it was the largest single one there:

    proposals at CONSTRAINT weight    81.5% of a perfect resolver
    proposals at attenuated weight    ~96%

Same knowledge in both. The difference is entirely what a WRONG proposal costs. The rescue
is in the regime where that matters -- on the wrapper suite it accepted 43 requirements and
had 41 rejected as unattested, so roughly half of what it proposes is already known to be
wrong before weighting is considered.

This has been RECOMMENDED but never MEASURED, which is the reason for this script. A
recommendation carried on someone else's numbers is a hypothesis.

ONE SET OF CALLS, TWO WEIGHTS. The model output does not depend on how the ledger stores
it, so both arms replay a single shared cache. The first arm pays for the calls; the second
is free. That matters because the provider reserves `prompt + max_tokens` against a daily
allowance and this experiment's cache was previously lost.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/studies/evaluate_rescue_weight.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

L_GPT = ROOT / ".review-l-gpt-shopping-copilot"
CACHE = ROOT / "experiments" / "datasets" / "prompt_arm_caches" / ".rescue_weight_cache.json"
OUT = ROOT / "experiments" / "results" / "out_76_rescue_weight.json"


def main() -> None:
    from submission.llm_rerank import _load_project_env
    _load_project_env()
    if not os.environ.get("GROQ_API_KEY", "").strip():
        print("GROQ_API_KEY not set -- nothing to evaluate.")
        return

    # Keep the comparison to the rescue alone: every other optional layer off, so a
    # difference between the two arms can only be the weight.
    os.environ["LLM_RERANK"] = "0"
    os.environ["LLM_EXTRACT"] = "0"
    os.environ["LLM_RESOLVE"] = "0"
    os.environ["BERT_EXTRACT"] = "1"

    from evaluator.local_evaluator import catalog_index, load_jsonl
    from submission.agent import CAT, CONSTRAINT, SEM, Agent, raw_toks
    from experiments.studies.run_official_template_paraphrase import bank, transform

    spec = importlib.util.spec_from_file_location(
        "stress", ROOT / "experiments" / "log" / "31_paraphrase_stress.py")
    stress = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stress)

    rspec = importlib.util.spec_from_file_location(
        "lgpt_rescue", L_GPT / "copilot" / "llm_rescue.py")
    llm_rescue = importlib.util.module_from_spec(rspec)
    rspec.loader.exec_module(llm_rescue)

    # max_tokens IS UNRESOLVED, and two attempts to resolve it here were both invalid.
    #
    # A local simulation sized the prompt at ~504 tokens and the structured output at 152 at
    # p95, which suggested 3072 was mostly reasoning headroom. A run at 1024 then failed 14
    # times in 17, and that was read as evidence the headroom is needed. It was not: the
    # failures were HTTP 429 on a tokens-per-day limit (199,841 of 200,000 used), and a
    # rerun at the published 3072 failed 16 of 17 for the same reason.
    #
    # So nothing here has yet measured what this parameter costs or needs. Both the token
    # arithmetic and the failure counts are consistent with any answer.
    #
    # The check that missed it is worth naming: a liveness probe sent max_tokens=16, which
    # fits in a nearly exhausted daily allowance and returns 200. Capacity for a cheap call
    # is not capacity for an expensive one. Probe with the size you intend to send.
    max_tokens = int(os.environ.get("RESCUE_MAX_TOKENS", "3072"))
    config = SimpleNamespace(enable_llm_rescue=True, llm_rescue_turn=5,
                             llm_provider="groq", llm_model="openai/gpt-oss-20b",
                             llm_max_tokens=max_tokens, rescue_fn=None)
    model = llm_rescue.build_model(config)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception:
        cache = {}

    rows = load_jsonl(ROOT / "data" / "public_set.jsonl")
    ids, cats, products = catalog_index(ROOT / "data" / "catalog.jsonl")
    warp = transform(bank())

    def make(tier):
        stats = {"reaches": 0, "calls": 0, "cache_hits": 0, "usable": 0,
                 "accepted": 0, "dropped_unattested": 0, "failures": 0}

        class Rescued(Agent):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self._hist, self._done = {}, set()

            def _observe(self, st, msg):
                sid = st.sid or ""
                hist = self._hist.setdefault(sid, [])
                hist.append({"turn": st.turn, "user_message": msg})
                super()._observe(st, msg)
                if st.turn < config.llm_rescue_turn or sid in self._done:
                    return
                self._done.add(sid)
                stats["reaches"] += 1
                graph = {"turns": hist, "asked": list(st.asked),
                         "exhausted_attributes": [], "product_nodes": {}}
                # KEY THE CACHE BY max_tokens. It is a request parameter that decides
                # whether the call succeeds at all -- at 1024 this experiment measured 14
                # failures in 17 calls -- so a 1024-era entry must not answer a 3072 run.
                key = hashlib.sha256(
                    f"{max_tokens}\n{llm_rescue.transcript(graph)}".encode("utf-8")
                ).hexdigest()
                if key in cache:
                    stats["cache_hits"] += 1
                    rescued = cache[key]
                else:
                    stats["calls"] += 1
                    print(f"    live call {stats['calls']}", flush=True)
                    rescued = llm_rescue.rescue(graph, {}, config, model=model)
                    # NEVER CACHE A FAILURE. The upstream harness caches whatever comes
                    # back, so one truncated response or transient error becomes a
                    # permanent absence of evidence on every later run -- indistinguishable
                    # from the model having nothing to say. Our own resolver learned this
                    # already; the rule belongs here too.
                    if isinstance(rescued, dict):
                        cache[key] = rescued
                        CACHE.write_text(
                            json.dumps(cache, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
                if not isinstance(rescued, dict):
                    stats["failures"] += 1
                    return
                stats["usable"] += 1
                values = list(rescued.get("requirements") or [])
                for field in ("color", "material"):
                    if rescued.get(field):
                        values.append(str(rescued[field]))
                category = str(rescued.get("category") or "").strip()
                if category:
                    for phrase in self._resolve(category):
                        if phrase not in st.evidence:
                            st.evidence[phrase] = (self.ix.df(phrase), CAT)
                    for token in raw_toks(category):
                        if self.ix.df(token) > 0 and token not in st.evidence:
                            st.evidence[token] = (self.ix.df(token), CAT)
                for value in values:
                    recovered = self._resolve(str(value).strip())
                    if not recovered:
                        stats["dropped_unattested"] += 1
                        continue
                    for phrase in recovered:
                        if phrase not in st.evidence:
                            # THE ONLY DIFFERENCE BETWEEN THE TWO ARMS.
                            st.evidence[phrase] = (self.ix.df(phrase), tier)
                            stats["accepted"] += 1
        return Rescued, stats

    t0 = time.time()
    print("baseline (no rescue)")
    base = stress.evaluate_transformed(Agent(ROOT / "data" / "catalog.jsonl"),
                                       rows, ids, cats, products, warp)
    results, allstats = {}, {}
    for label, tier in (("full strength (CONSTRAINT)", CONSTRAINT),
                        ("attenuated (SEM)", SEM)):
        print(f"\n{label}")
        cls, stats = make(tier)
        results[label] = stress.evaluate_transformed(
            cls(ROOT / "data" / "catalog.jsonl"), rows, ids, cats, products, warp)
        allstats[label] = stats

    def m(r):
        return {k: round(float(r[k]), 6) for k in
                ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score")}

    print(f"\n{'arm':<30}{'HR@10':>9}{'MRR':>9}{'MTTC':>9}{'score':>11}{'delta':>11}")
    print("-" * 79)
    b = m(base)
    print(f"{'baseline (no rescue)':<30}{b['hit_rate_at_10']:>9.4f}{b['mrr']:>9.4f}"
          f"{b['mttc']:>9.3f}{b['recommended_technical_score']:>11.6f}{'':>11}")
    for label, r in results.items():
        v = m(r)
        d = v["recommended_technical_score"] - b["recommended_technical_score"]
        print(f"{label:<30}{v['hit_rate_at_10']:>9.4f}{v['mrr']:>9.4f}"
              f"{v['mttc']:>9.3f}{v['recommended_technical_score']:>11.6f}{d:>+11.6f}")

    OUT.write_text(json.dumps({
        "experiment": "transcript rescue: full-strength vs attenuated evidence weight",
        "suite": "public_set with held-out wrapper paraphrase",
        "shared_calls": "one cache serves both arms; only the ledger tier differs",
        "max_tokens": max_tokens,
        "baseline": b,
        "arms": {k: m(v) for k, v in results.items()},
        "stats": allstats,
        "seconds": round(time.time() - t0, 2),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
