"""New arm only: immediate, gated full-transcript rebuild on actual shipped V2.

The LLM receives the entire conversation and rebuilds the current requirements whenever
an unfamiliar message contributes no new non-category catalogue evidence.  There is no
minimum turn.  Existing baseline and late-rescue scores are read from prior results rather
than rerun.
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
V2, OV = ROOT / "experiments" / "studies", ROOT / "experiments" / "studies" / "open_vocabulary"
L_GPT = ROOT / ".review-l-gpt-shopping-copilot"
OUT = V2 / "results" / "early_transcript_rebuild_v2_71.json"
CACHE = V2 / ".early_transcript_rebuild_cache.json"

# Results already measured with the same actual V2 configuration and source suites.
REFERENCES = {
    "TemplateParaphrase9600-Test": {"v2": 0.943450, "late_transcript": 0.952650},
    "Wrapper800": {"v2": 0.918948, "late_transcript": 0.921623,
                    "late_message_then_transcript": 0.923823},
}


def main() -> None:
    from submission.llm_rerank import _load_project_env
    _load_project_env()
    if not os.environ.get("GROQ_API_KEY", "").strip():
        print("GROQ_API_KEY not set -- nothing to evaluate."); return
    os.environ.update({"LLM_RERANK": "0", "LLM_EXTRACT": "0", "LLM_RESOLVE": "0", "BERT_EXTRACT": "1"})
    from evaluator.local_evaluator import catalog_index, load_jsonl
    from submission.agent import Agent, CAT, CONSTRAINT, raw_toks, recognised

    stress_spec = importlib.util.spec_from_file_location("early_stress", ROOT / "experiments" / "log" / "31_paraphrase_stress.py")
    assert stress_spec and stress_spec.loader
    stress = importlib.util.module_from_spec(stress_spec); stress_spec.loader.exec_module(stress)
    tmpl_spec = importlib.util.spec_from_file_location("early_template", V2 / "run_official_template_paraphrase.py")
    assert tmpl_spec and tmpl_spec.loader
    tmpl = importlib.util.module_from_spec(tmpl_spec); tmpl_spec.loader.exec_module(tmpl)
    rescue_spec = importlib.util.spec_from_file_location("linked_lgpt_rescue", L_GPT / "copilot" / "llm_rescue.py")
    assert rescue_spec and rescue_spec.loader
    lgpt = importlib.util.module_from_spec(rescue_spec); rescue_spec.loader.exec_module(lgpt)

    cfg = SimpleNamespace(llm_provider="groq", llm_model="openai/gpt-oss-20b", llm_max_tokens=3072, rescue_fn=None)
    model = lgpt.build_model(cfg)
    try: cache = json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception: cache = {}
    stats = {"gate_opens": 0, "calls": 0, "cache_hits": 0, "usable": 0,
             "accepted": 0, "dropped_unattested": 0, "failures": 0}

    class EarlyTranscriptRebuild(Agent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs); self.histories = {}; self.done_messages = set()
        def _observe(self, st, msg):
            sid = st.sid or ""; history = self.histories.setdefault(sid, [])
            history.append({"turn": st.turn, "user_message": msg})
            unfamiliar = not recognised(msg)
            before = set(st.evidence)
            super()._observe(st, msg)
            # Gate: the message is outside the literal organizer grammar and normal V2
            # recovered no new non-category evidence from it.  The transcript may then
            # reconstruct the whole conversation, including evidence from earlier turns.
            gained = any(tier != CAT for phrase, (_, tier) in st.evidence.items() if phrase not in before)
            message_key = hashlib.sha256((sid + "\0" + msg).encode("utf-8")).hexdigest()
            if not unfamiliar or gained or message_key in self.done_messages:
                return
            self.done_messages.add(message_key); stats["gate_opens"] += 1
            graph = {"turns": history, "asked": list(st.asked), "exhausted_attributes": [], "product_nodes": {}}
            transcript = lgpt.transcript(graph); key = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
            if key in cache:
                stats["cache_hits"] += 1; payload = cache[key]
            else:
                stats["calls"] += 1; print(f"[early transcript] call {stats['calls']} turn={st.turn}", flush=True)
                try: payload = lgpt.rescue(graph, {}, cfg, model=model)
                except Exception: payload = None; stats["failures"] += 1
                cache[key] = payload
                CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if not isinstance(payload, dict):
                stats["failures"] += 1; return
            stats["usable"] += 1
            values = list(payload.get("requirements") or []) + [payload[x] for x in ("color", "material") if payload.get(x)]
            for value in values:
                recovered = self._resolve(str(value).strip())
                if not recovered:
                    stats["dropped_unattested"] += 1; continue
                for phrase in recovered:
                    if phrase not in st.evidence:
                        st.evidence[phrase] = (self.ix.df(phrase), CONSTRAINT); stats["accepted"] += 1
            category = str(payload.get("category") or "").strip()
            for phrase in self._resolve(category):
                if phrase not in st.evidence: st.evidence[phrase] = (self.ix.df(phrase), CAT)
            for token in raw_toks(category):
                if self.ix.df(token) > 0 and token not in st.evidence:
                    st.evidence[token] = (self.ix.df(token), CAT)

    ids, cats, products = catalog_index(ROOT / "data" / "catalog.jsonl")
    cases = {
        "TemplateParaphrase9600-Test": (load_jsonl(ROOT / "data" / "public_set.jsonl"), tmpl.transform(tmpl.bank())),
        "Wrapper800": (load_jsonl(OV / "review800_canonical_replay.jsonl"), tmpl.transform(tmpl.bank())),
    }
    fields, report, started = ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score"), {}, time.time()
    for name, (rows, warp) in cases.items():
        before = dict(stats)
        result = stress.evaluate_transformed(EarlyTranscriptRebuild(ROOT / "data" / "catalog.jsonl"), rows, ids, cats, products, warp)
        delta_stats = {k: stats[k] - before[k] for k in stats}
        report[name] = {"new_arm": {k: round(float(result[k]), 6) for k in fields},
                        "known_references": REFERENCES[name], "run_stats": delta_stats}
        print(f"[{name}] score={result['recommended_technical_score']:.6f} stats={delta_stats}", flush=True)
    payload = {"experiment": "V2.71 immediate gated transcript rebuild", "gate": "unfamiliar message AND normal V2 added no new non-category evidence; once per message", "model": {"name": cfg.llm_model, "reasoning_effort": "low", "max_tokens": cfg.llm_max_tokens}, "results": report, "seconds": round(time.time() - started, 2)}
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)

if __name__ == "__main__": main()
