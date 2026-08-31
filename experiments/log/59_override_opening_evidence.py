"""Experiment 59: recover target evidence from intent-override opening messages.

Organizer source establishes that an intent-override opening is:
    I'm looking for {category}. {old_value}
where old_value is soft_preferences[-1] from the hidden target's intent card.  The
submitted template extractor currently parses only category from that message.  This
analysis compares the shipped behaviour with two target-grounded additions:

* full: old_value receives ordinary constraint weight;
* weak: old_value receives the existing lower mined-evidence weight.

No submitted source file is modified.  Both variants retain the released override
semantics and rejection-state handling.
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
from submission.agent import Agent, CONSTRAINT, MINED, raw_toks  # noqa: E402

OUT = ROOT / "experiments" / "results" / "out_59_override_opening_evidence.json"
UNSEEN = ROOT / "robustness" / "optuna_v2_sets" / "population_shift_01_800.jsonl"

# This is the exact remaining organizer opening form after the category prefix.  Buying
# openings have an explicit requirement label and are deliberately excluded.
PAT_OVERRIDE_OPENING = re.compile(
    r"^I'm looking for .+?\. (?!A key requirement is:)(?P<old_value>.+)$", re.I
)


class OverrideOpeningEvidence(Agent):
    """Attach the source-confirmed old_value as target-grounded session evidence."""

    OPENING_TIER = CONSTRAINT

    def _observe(self, state, message: str) -> None:
        super()._observe(state, message)
        match = PAT_OVERRIDE_OPENING.match(message.strip())
        if not match:
            return
        old_value = match.group("old_value").strip()
        # Defend against an accidental match with a malformed no-information message.
        if not old_value or not raw_toks(old_value):
            return
        for phrase in self._resolve(old_value):
            if not phrase or phrase in state.evidence:
                continue
            df = self.ix.df(phrase)
            if df > 0:
                state.evidence[phrase] = (df, self.OPENING_TIER)


class OpeningAsFullConstraint(OverrideOpeningEvidence):
    OPENING_TIER = CONSTRAINT


class OpeningAsWeakEvidence(OverrideOpeningEvidence):
    OPENING_TIER = MINED


def shared_agent(cls: type[Agent], base: Agent) -> Agent:
    """Reuse immutable 50k index while isolating every evaluation's session state."""
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
    datasets = {
        "Official200": load_jsonl(ROOT / "data" / "public_set.jsonl"),
        "Unseen800": load_jsonl(UNSEEN),
    }
    ids, categories, products = catalog_index(catalog)
    base = Agent(catalog)
    variants: dict[str, type[Agent]] = {
        "shipped_no_opening_value": Agent,
        "opening_as_full_constraint": OpeningAsFullConstraint,
        "opening_as_weak_grounded_evidence": OpeningAsWeakEvidence,
    }
    result: dict = {
        "purpose": "recover source-confirmed target evidence from intent-override opening slot",
        "scope": "analysis only; submitted agent unchanged",
        "source_fact": "old_value = soft_preferences[-1] from target intent card",
        "datasets": {"Official200": "data/public_set.jsonl", "Unseen800": str(UNSEEN.relative_to(ROOT))},
        "variants": {},
    }
    for name, cls in variants.items():
        by_set = {}
        for label, rows in datasets.items():
            started = time.perf_counter()
            measured = evaluate(shared_agent(cls, base), rows, ids, categories, products)
            by_set[label] = {**compact(measured), "wall_seconds": time.perf_counter() - started}
        result["variants"][name] = by_set
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, by_set in result["variants"].items():
        print(name)
        for label, row in by_set.items():
            override = row["intent_override"]
            print(f"  {label:<11} score={row['technical_score']:.6f} "
                  f"HR={row['hit_rate_at_10']:.3f} MRR={row['mrr']:.3f} MTTC={row['mttc']:.3f} "
                  f"override_HR={override.get('hit_rate_at_10', 0):.3f}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
