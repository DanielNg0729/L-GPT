"""Evaluate L-GPT's turn-five whole-chat rescue on the actual shipped V2 agent.

The evaluator does not modify submission code.  It imports the linked branch's
``copilot.llm_rescue`` module for the prompt, schema, model construction and model call.
Our frozen V2 still owns state, retrieval, ranking and question selection.

Comparison: actual V2 Route + Span versus the same agent plus one L-GPT rescue at turn 5,
on held-out TemplateParaphrase9600-Test wrappers.  Attribute values remain canonical.
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
sys.path.insert(0, str(L_GPT))
V2 = ROOT / "experiments" / "studies"
OUT = V2 / "results" / "lgpt_rescue_on_shipped_v2_template9600.json"
CACHE = V2 / ".lgpt_rescue_template9600_cache.json"


def main() -> None:
    from submission.llm_rerank import _load_project_env
    _load_project_env()
    if not os.environ.get("GROQ_API_KEY", "").strip():
        print("GROQ_API_KEY not set -- nothing to evaluate.")
        return

    # Keep this experiment to L-GPT's rescue only, as published.  The submitted V2's
    # reranker, extractor and attribute deparaphraser are disabled to avoid attribution
    # ambiguity on a wrapper-only suite.
    os.environ["LLM_RERANK"] = "0"
    os.environ["LLM_EXTRACT"] = "0"
    os.environ["LLM_RESOLVE"] = "0"
    os.environ["BERT_EXTRACT"] = "1"

    from evaluator.local_evaluator import catalog_index, load_jsonl
    from submission.agent import Agent, CAT, CONSTRAINT, raw_toks
    from experiments.studies.run_official_template_paraphrase import bank, transform

    stress_spec = importlib.util.spec_from_file_location(
        "lgpt_template_stress", ROOT / "experiments" / "log" / "31_paraphrase_stress.py")
    assert stress_spec and stress_spec.loader
    paraphrase_stress = importlib.util.module_from_spec(stress_spec)
    stress_spec.loader.exec_module(paraphrase_stress)

    rescue_spec = importlib.util.spec_from_file_location(
        "linked_lgpt_rescue", L_GPT / "copilot" / "llm_rescue.py")
    assert rescue_spec and rescue_spec.loader
    llm_rescue = importlib.util.module_from_spec(rescue_spec)
    rescue_spec.loader.exec_module(llm_rescue)

    # L-GPT's published configuration, verbatim.
    config = SimpleNamespace(enable_llm_rescue=True, llm_rescue_turn=5,
                             llm_provider="groq", llm_model="openai/gpt-oss-20b",
                             llm_max_tokens=3072, rescue_fn=None)
    model = llm_rescue.build_model(config)

    try:
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception:
        cache = {}
    stats = {"reaches": 0, "calls": 0, "cache_hits": 0, "usable": 0,
             "accepted_requirements": 0, "dropped_unattested": 0, "failures": 0}

    class LGPTRescueOnV2(Agent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._lgpt_histories = {}
            self._lgpt_done = set()

        def _observe(self, st, msg):
            sid = st.sid or ""
            history = self._lgpt_histories.setdefault(sid, [])
            history.append({"turn": st.turn, "user_message": msg})
            super()._observe(st, msg)
            if st.turn < config.llm_rescue_turn or sid in self._lgpt_done:
                return
            self._lgpt_done.add(sid)
            stats["reaches"] += 1
            graph = {"turns": history, "asked": list(st.asked),
                     "exhausted_attributes": [], "product_nodes": {}}
            transcript = llm_rescue.transcript(graph)
            key = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
            if key in cache:
                stats["cache_hits"] += 1
                rescued = cache[key]
                print(f"[L-GPT rescue] turn {st.turn}: cache hit", flush=True)
            else:
                stats["calls"] += 1
                print(f"[L-GPT rescue] turn {st.turn}: live call {stats['calls']}", flush=True)
                rescued = llm_rescue.rescue(graph, {}, config, model=model)
                cache[key] = rescued
                CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
            if not isinstance(rescued, dict):
                stats["failures"] += 1
                return
            stats["usable"] += 1

            # L-GPT applies recovered requirements with weight 1.0.  We replicate that
            # policy but retain our catalogue attestation before adding an evidence key:
            # an unattested phrase has no posting list in either system and must not be
            # allowed to masquerade as target evidence in V2's ledger.
            values = list(rescued.get("requirements") or [])
            for field in ("color", "material"):
                value = rescued.get(field)
                if value:
                    values.append(str(value))
            category = str(rescued.get("category") or "").strip()
            if category:
                for phrase in self._resolve(category):
                    if phrase not in st.evidence:
                        st.evidence[phrase] = (self.ix.df(phrase), CAT)
                for token in raw_toks(category):
                    if self.ix.df(token) > 0 and token not in st.evidence:
                        st.evidence[token] = (self.ix.df(token), CAT)
            for value in values:
                text = str(value).strip()
                recovered = self._resolve(text)
                if not recovered:
                    stats["dropped_unattested"] += 1
                    continue
                for phrase in recovered:
                    if phrase not in st.evidence:
                        st.evidence[phrase] = (self.ix.df(phrase), CONSTRAINT)
                        stats["accepted_requirements"] += 1

    rows = load_jsonl(ROOT / "data" / "public_set.jsonl")
    ids, cats, products = catalog_index(ROOT / "data" / "catalog.jsonl")
    warp = transform(bank())
    t0 = time.time()

    baseline = Agent(ROOT / "data" / "catalog.jsonl")
    rescued = LGPTRescueOnV2(ROOT / "data" / "catalog.jsonl")
    base_result = paraphrase_stress.evaluate_transformed(baseline, rows, ids, cats, products, warp)
    rescue_result = paraphrase_stress.evaluate_transformed(rescued, rows, ids, cats, products, warp)

    def metric(result):
        return {key: round(float(result[key]), 6) for key in
                ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score")}

    report = {
        "experiment": "L-GPT turn-five rescue on actual shipped V2",
        "suite": "TemplateParaphrase9600-Test, held-out wrapper families, Official200 sessions",
        "integration": "L-GPT prompt/model/schema; submitted V2 retrieval/ranking; full-strength recovered requirements",
        "lgpt_config": {"model": config.llm_model, "provider": config.llm_provider,
                        "reasoning_effort": "low", "max_tokens": config.llm_max_tokens,
                        "turn": config.llm_rescue_turn},
        "baseline": metric(base_result),
        "lgpt_rescue": metric(rescue_result),
        "delta": {key: round(float(rescue_result[key]) - float(base_result[key]), 6) for key in
                  ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score")},
        "rescue_stats": stats,
        "seconds": round(time.time() - t0, 2),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
