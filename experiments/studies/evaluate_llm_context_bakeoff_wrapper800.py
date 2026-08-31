"""Target-disjoint Wrapper800 bakeoff: message extraction versus transcript rescue.

All LLM arms share the same label-free gate: at least one unfamiliar wrapper in the
session, four observed rejected recommendations, and no previous rescue.  The frozen V2
agent supplies every non-LLM operation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2, OV = ROOT / "experiments" / "studies", ROOT / "experiments" / "datasets" / "open_vocabulary"
L_GPT = ROOT / ".review-l-gpt-shopping-copilot"
OUT = V2 / "results" / "llm_context_bakeoff_wrapper800.json"
CACHE = V2 / ".llm_context_bakeoff_wrapper800_cache.json"


def main() -> None:
    from submission.llm_rerank import _load_project_env
    _load_project_env()
    if not os.environ.get("GROQ_API_KEY", "").strip():
        print("GROQ_API_KEY not set -- nothing to evaluate."); return
    os.environ.update({"LLM_RERANK": "0", "LLM_EXTRACT": "0", "LLM_RESOLVE": "0", "BERT_EXTRACT": "1"})

    from langchain_groq import ChatGroq
    from evaluator.local_evaluator import catalog_index, load_jsonl
    from submission.agent import Agent, CAT, CONSTRAINT, LLM, raw_toks, recognised
    from submission.llm_extract import SYSTEM as MESSAGE_SYSTEM
    stress_spec = importlib.util.spec_from_file_location("wrapper800_stress", ROOT / "experiments" / "log" / "31_paraphrase_stress.py")
    assert stress_spec and stress_spec.loader
    stress = importlib.util.module_from_spec(stress_spec); stress_spec.loader.exec_module(stress)
    template_spec = importlib.util.spec_from_file_location("wrapper800_template", V2 / "run_official_template_paraphrase.py")
    assert template_spec and template_spec.loader
    template = importlib.util.module_from_spec(template_spec); template_spec.loader.exec_module(template)
    rescue_spec = importlib.util.spec_from_file_location("linked_lgpt_rescue", L_GPT / "copilot" / "llm_rescue.py")
    assert rescue_spec and rescue_spec.loader
    lgpt = importlib.util.module_from_spec(rescue_spec); rescue_spec.loader.exec_module(lgpt)

    cfg = SimpleNamespace(llm_provider="groq", llm_model="openai/gpt-oss-20b", llm_max_tokens=3072, rescue_fn=None)
    message_model = ChatGroq(model=cfg.llm_model, temperature=0.0, max_tokens=cfg.llm_max_tokens, reasoning_effort="low")
    transcript_model = lgpt.build_model(cfg)
    try: cache = json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception: cache = {}
    stats = {name: {"reaches": 0, "calls": 0, "cache_hits": 0, "usable": 0,
                    "accepted": 0, "dropped": 0, "failures": 0}
             for name in ("message", "transcript", "both_message", "both_transcript")}

    def cache_get(kind, source, ask):
        key = hashlib.sha256((kind + "\0" + source).encode()).hexdigest()
        if key in cache:
            stats[kind]["cache_hits"] += 1; return cache[key]
        stats[kind]["calls"] += 1
        try: value = ask()
        except Exception: value = None; stats[kind]["failures"] += 1
        cache[key] = value
        CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return value

    def message_spans(msg, stat_name):
        raw = cache_get(stat_name, msg, lambda: message_model.invoke([
            ("system", MESSAGE_SYSTEM), ("human", f"Customer message:\n{msg}\n\nSpans:")]).content)
        if not isinstance(raw, str): return []
        hay = " ".join(msg.lower().split()); spans = []
        for line in raw.splitlines():
            span = re.sub(r"^\d+[.)]\s*", "", line.strip().strip("-*• \t"))
            if span and span.upper() != "NONE" and 2 <= len(span) <= 120 and " ".join(span.lower().split()) in hay and span not in spans:
                spans.append(span)
            if len(spans) == 6: break
        if spans: stats[stat_name]["usable"] += 1
        return spans

    def transcript_payload(history, asked, stat_name):
        graph = {"turns": history, "asked": asked, "exhausted_attributes": [], "product_nodes": {}}
        text = lgpt.transcript(graph)
        got = cache_get(stat_name, text, lambda: lgpt.rescue(graph, {}, cfg, model=transcript_model))
        if isinstance(got, dict): stats[stat_name]["usable"] += 1
        return got if isinstance(got, dict) else {}

    def add_exact(agent, st, values, tier, stat_name):
        added = 0
        for value in values:
            got = agent._resolve(str(value).strip())
            if not got:
                stats[stat_name]["dropped"] += 1; continue
            for phrase in got:
                if phrase not in st.evidence:
                    st.evidence[phrase] = (agent.ix.df(phrase), tier)
                    stats[stat_name]["accepted"] += 1; added += 1
        return added

    def arm(mode):
        class ContextArm(Agent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs); self.histories = {}; self.unknown = {}; self.done = set()
            def _observe(self, st, msg):
                sid = st.sid or ""; history = self.histories.setdefault(sid, [])
                history.append({"turn": st.turn, "user_message": msg})
                self.unknown[sid] = self.unknown.get(sid, False) or not recognised(msg)
                super()._observe(st, msg)
                # Four rejected rank-one recommendations are observed failure evidence.
                if sid in self.done or not self.unknown[sid] or len(st.rejected) < 4:
                    return
                self.done.add(sid)
                if mode in ("message", "both"):
                    name = "message" if mode == "message" else "both_message"; stats[name]["reaches"] += 1
                    n = add_exact(self, st, message_spans(msg, name), LLM, name)
                    if mode == "message" or n:
                        return
                name = "transcript" if mode == "transcript" else "both_transcript"; stats[name]["reaches"] += 1
                payload = transcript_payload(history, list(st.asked), name)
                values = list(payload.get("requirements") or []) + [payload[x] for x in ("color", "material") if payload.get(x)]
                add_exact(self, st, values, CONSTRAINT, name)
                category = str(payload.get("category") or "").strip()
                for phrase in self._resolve(category):
                    if phrase not in st.evidence: st.evidence[phrase] = (self.ix.df(phrase), CAT)
        return ContextArm

    rows = load_jsonl(OV / "review800_canonical_replay.jsonl")
    ids, cats, products = catalog_index(ROOT / "data" / "catalog.jsonl")
    warp, t0 = template.transform(template.bank()), time.time()
    def run(cls): return stress.evaluate_transformed(cls(ROOT / "data" / "catalog.jsonl"), rows, ids, cats, products, warp)
    arms = {"V2 baseline": run(Agent), "message-only": run(arm("message")),
            "transcript": run(arm("transcript")), "both": run(arm("both"))}
    fields = ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score")
    base = arms["V2 baseline"]
    report = {"experiment": "Target-disjoint Wrapper800 LLM context bakeoff",
              "suite": "review800 canonical replay with held-out wrapper paraphrase",
              "gate": "unrecognised wrapper seen AND at least four prior rejected recommendations AND one use per session",
              "model": {"name": cfg.llm_model, "reasoning_effort": "low", "max_tokens": cfg.llm_max_tokens},
              "arms": {name: {k: round(float(value[k]), 6) for k in fields} for name, value in arms.items()},
              "deltas_vs_v2": {name: {k: round(float(value[k]) - float(base[k]), 6) for k in fields} for name, value in arms.items() if name != "V2 baseline"},
              "stats": stats, "seconds": round(time.time() - t0, 2)}
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

if __name__ == "__main__": main()
