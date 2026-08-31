"""Local audit: does actual V2 recover the disclosed target category on turn one?

This is a coverage diagnostic only.  It makes no LLM calls and produces no ranking
recommendations.  The two inputs are the same wrapper-transformed suites used in the
LLM rescue comparison.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2, OV = ROOT / "experiments" / "studies", ROOT / "experiments" / "datasets" / "open_vocabulary"
OUT = V2 / "results" / "turn1_category_coverage_v2_72.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    # Force the exact local V2 path.  No external model may participate in this audit.
    os.environ.update({"LLM_RERANK": "0", "LLM_EXTRACT": "0", "LLM_RESOLVE": "0", "BERT_EXTRACT": "1"})
    from evaluator.local_evaluator import (catalog_index, coarse_category, initial_message,
                                           load_jsonl, materialize_hidden_fields)
    from submission.agent import Agent, CAT, raw_toks
    tmpl = load_module("category_audit_template", V2 / "run_official_template_paraphrase.py")
    _, categories, products = catalog_index(ROOT / "data" / "catalog.jsonl")
    warp = tmpl.transform(tmpl.bank())
    suites = {
        "TemplateParaphrase9600-Test": load_jsonl(ROOT / "data" / "public_set.jsonl"),
        "Wrapper800": load_jsonl(OV / "review800_canonical_replay.jsonl"),
    }
    report: dict[str, object] = {}
    for name, rows in suites.items():
        agent = Agent(ROOT / "data" / "catalog.jsonl")
        failures, details = [], []
        for n, sample in enumerate(rows):
            target = str(sample["ground_truth"]["parent_asin"])
            expected = coarse_category(categories.get(target, []))
            card, behavior = materialize_hidden_fields(sample, products)
            effective = {**sample, "intent_card": card, "behavior": behavior}
            message = warp(initial_message(effective, expected, set()))
            sid = f"category-audit-{name}-{n}"
            agent.reset(sid, sample["user_profile"])
            state = agent.sessions[sid]
            agent._observe(state, message)
            category_phrases = sorted(phrase for phrase, (_, tier) in state.evidence.items() if tier == CAT)
            # Exact normalized phrase is the relevant criterion: V2 ranks with
            # phrase-level FTS evidence.  Punctuation is deliberately not relevant
            # because the index normalizes ``Tops & Tees`` to ``tops tees``.
            expected_normalized = " ".join(raw_toks(expected))
            recovered = expected_normalized in {" ".join(raw_toks(phrase)) for phrase in category_phrases}
            if not recovered:
                failures.append(sample["sample_id"])
                if len(details) < 20:
                    details.append({"sample_id": sample["sample_id"], "expected_category": expected,
                                    "transformed_turn_1": message, "category_evidence": category_phrases,
                                    "expected_tokens": raw_toks(expected)})
        report[name] = {"sessions": len(rows), "exact_category_recovered": len(rows) - len(failures),
                        "exact_category_failure": len(failures),
                        "failure_rate": round(len(failures) / len(rows), 6),
                        "failure_examples": details}
        print(f"[{name}] recovered={len(rows)-len(failures)}/{len(rows)} failures={len(failures)}", flush=True)
    output = {"experiment": "V2.72 turn-one category coverage audit", "definition": "The exact normalized coarse target category disclosed by the organizer is present as CAT evidence after actual V2 local extraction.", "llm_calls": 0, "results": report}
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
