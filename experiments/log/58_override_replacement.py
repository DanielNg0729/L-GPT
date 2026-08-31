"""Experiment 58: test evidence replacement for intent overrides.

The shipped agent clears rejected recommendations when it sees an override, but retains
positive evidence. This comparison keeps the released agent unchanged and evaluates two
counterfactual policies that retain category evidence while dropping prior non-category
evidence on an override. A broader cue is included only as a diagnostic for reworded
override language; the hand-authored examples are not an organizer-score estimate.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["LLM_EXTRACT"] = "0"
os.environ["LLM_RERANK"] = "0"

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent, CAT, PAT_OVERRIDE, PAT_OVERRIDE_CUE, recognised  # noqa: E402

OUT = ROOT / "experiments" / "results" / "out_58_override_replacement.json"
UNSEEN = ROOT / "robustness" / "optuna_v2_sets" / "population_shift_01_800.jsonl"

# Deliberately permissive only for the evidence-reset candidate. A false positive drops
# useful evidence, so the official score is the guard against adopting it.
BROAD_OVERRIDE_CUE = re.compile(
    r"\b(?:ignore|scratch|forget|instead|actually|change(?:d)?\s+(?:my\s+)?mind|"
    r"make\s+(?:them|it)|rather|switch)\b",
    re.I,
)


def current_override_cue(message: str) -> bool:
    return bool(PAT_OVERRIDE.search(message) or PAT_OVERRIDE_CUE.search(message))


def broad_override_cue(message: str) -> bool:
    return current_override_cue(message) or bool(BROAD_OVERRIDE_CUE.search(message))


class CategoryResetCurrentCue(Agent):
    """Counterfactual: discard previous constraints only on the released cue."""

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if current_override_cue(user_message):
            state = self.sessions.get(session_id)
            if state is not None:
                state.evidence = {phrase: record for phrase, record in state.evidence.items()
                                  if record[1] == CAT}
                state.rejected.clear()
        return super().respond(session_id, user_message, turn, top_k)


class CategoryResetBroadCue(Agent):
    """Counterfactual: reset on broader English replacement language."""

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if broad_override_cue(user_message):
            state = self.sessions.get(session_id)
            if state is not None:
                state.evidence = {phrase: record for phrase, record in state.evidence.items()
                                  if record[1] == CAT}
                state.rejected.clear()
        return super().respond(session_id, user_message, turn, top_k)


def shared_agent(cls: type[Agent], base: Agent) -> Agent:
    """Create an isolated agent state while reusing the immutable 50k-product index."""
    agent = object.__new__(cls)
    agent.ix = base.ix
    agent.sessions = {}
    agent.llm = None
    agent.llm_extract = None
    agent.tagger = None
    return agent


def compact(result: dict) -> dict:
    return {
        "technical_score": result["recommended_technical_score"],
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "intent_override": result["scenario_metrics"].get("intent_override", {}),
    }


def main() -> None:
    catalog = ROOT / "data" / "catalog.jsonl"
    public = load_jsonl(ROOT / "data" / "public_set.jsonl")
    unseen = load_jsonl(UNSEEN)
    ids, categories, products = catalog_index(catalog)
    base = Agent(catalog)
    variants: dict[str, type[Agent]] = {
        "shipped_accumulate": Agent,
        "category_reset_current_cue": CategoryResetCurrentCue,
        "category_reset_broad_cue": CategoryResetBroadCue,
    }
    result: dict = {
        "purpose": "isolated override replacement comparison; no shipped-agent change",
        "datasets": {
            "Official200": "data/public_set.jsonl",
            "Unseen800": str(UNSEEN.relative_to(ROOT)),
        },
        "manual_cue_diagnostic": [],
        "variants": {},
    }
    examples = [
        ("official", "Actually, ignore my earlier preference. What I need is: cotton."),
        ("slide_example", "Actually, make them casual white sneakers."),
        ("reworded", "Hmm, scratch all that. What I actually need is cotton."),
        ("reworded", "Forget the earlier style. I would rather have white casual sneakers."),
        ("reworded", "I changed my mind. Please switch to white casual sneakers."),
        ("non_override", "Actually, I need cotton."),
        ("non_override", "Can you show white casual sneakers?"),
    ]
    for label, text in examples:
        result["manual_cue_diagnostic"].append({
            "label": label,
            "message": text,
            "recognised_official_form": recognised(text),
            "current_cue": current_override_cue(text),
            "broad_cue": broad_override_cue(text),
        })

    for name, cls in variants.items():
        rows = {}
        for label, dataset in (("Official200", public), ("Unseen800", unseen)):
            started = time.perf_counter()
            measured = evaluate(shared_agent(cls, base), dataset, ids, categories, products)
            rows[label] = {**compact(measured), "wall_seconds": time.perf_counter() - started}
        result["variants"][name] = rows

    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, rows in result["variants"].items():
        print(name)
        for label, row in rows.items():
            override = row["intent_override"]
            print(f"  {label:<11} score={row['technical_score']:.6f} "
                  f"override_hr={override.get('hit_rate_at_10', 0):.3f} "
                  f"override_mrr={override.get('mrr', 0):.3f}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
