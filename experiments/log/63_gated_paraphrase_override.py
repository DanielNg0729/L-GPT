"""Experiment 63: aggressive override replacement behind a zero-public-reach gate.

The controller is unreachable for every recognised released simulator message. For an
unrecognised message it requires both a strong replacement cue and an explicit new-value
clause, then clears prior non-category evidence and parses the new value as a constraint.
It has no external model or API dependency.
"""
from __future__ import annotations

import copy
import importlib
import json
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "log"))
os.environ["LLM_EXTRACT"] = "0"
os.environ["LLM_RERANK"] = "0"

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS, TOP_K, catalog_index, coarse_category, customer_reply, evaluate,
    initial_message, load_jsonl, materialize_hidden_fields, metric_summary,
    normalize_recommendations,
)
from submission.agent import Agent, CAT, CONSTRAINT, recognised  # noqa: E402

probe = importlib.import_module("60_override_focus_contradiction")
DATASET = probe.DATASET
OUT = ROOT / "experiments" / "results" / "out_63_gated_paraphrase_override.json"

# Require the customer to signal replacement and explicitly state a new value. This is
# intentionally narrower than conversational-language coverage: false activation would
# throw away useful history on an unfamiliar organizer format.
PAT_GATED_REPLACEMENT = re.compile(
    r"\b(?:ignore(?:\s+my)?\s+(?:earlier|previous)|changed\s+my\s+mind|"
    r"scratch\s+(?:that|all\s+that)|instead|rather)\b.*?"
    r"\b(?:i\s+)?(?:need|want|prefer)\s+(?P<value>.+?)[.!]?$",
    re.I,
)


class GatedParaphraseReplacement(Agent):
    """Replacement logic that cannot run on recognised released messages."""

    def __init__(self, catalog_path):
        super().__init__(catalog_path)
        self.override_gate = {"seen": 0, "recognised_blocked": 0, "triggered": 0,
                              "no_explicit_value": 0}

    def _gated_value(self, message: str) -> str | None:
        self.override_gate["seen"] += 1
        if recognised(message):
            self.override_gate["recognised_blocked"] += 1
            return None
        match = PAT_GATED_REPLACEMENT.search(message)
        if not match:
            self.override_gate["no_explicit_value"] += 1
            return None
        value = match.group("value").strip()
        return value or None

    def _extract_templated(self, message: str):
        extracted = super()._extract_templated(message)
        value = getattr(self, "_current_gated_value", None)
        if value:
            extracted.append((value, CONSTRAINT))
        return extracted

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        value = self._gated_value(user_message or "")
        self._current_gated_value = value
        if value:
            self.override_gate["triggered"] += 1
            state = self.sessions.get(session_id)
            if state is not None:
                state.evidence = {phrase: item for phrase, item in state.evidence.items()
                                  if item[1] == CAT}
                state.rejected.clear()
        try:
            return super().respond(session_id, user_message, turn, top_k)
        finally:
            self._current_gated_value = None


def shared_agent(cls: type[Agent], base: Agent) -> Agent:
    if cls is Agent:
        agent = object.__new__(Agent)
        agent.ix, agent.sessions, agent.llm, agent.llm_extract, agent.tagger = base.ix, {}, None, None, None
        return agent
    agent = object.__new__(GatedParaphraseReplacement)
    agent.ix, agent.sessions, agent.llm, agent.llm_extract, agent.tagger = base.ix, {}, None, None, None
    agent.override_gate = {"seen": 0, "recognised_blocked": 0, "triggered": 0, "no_explicit_value": 0}
    agent._current_gated_value = None
    return agent


def evaluate_paraphrased_override(agent: Agent, samples: list[dict], catalog_ids: set[str],
                                  categories: dict[str, list[str]], products: dict[str, dict],
                                  contradiction: bool) -> dict:
    sessions = []
    for sample in samples:
        session_id = f"gated_override_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        behavior = copy.deepcopy(behavior)
        original_old = str(behavior["override"]["old_value"])
        if contradiction:
            behavior["override"]["old_value"] = probe.counterfactual_old_value(products[target], target)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        # Under a true replacement, the old value is not repeated after the opening.
        disclosed: set[str] = {original_old} if contradiction else set()
        boundary_used = False
        override_applied = False
        user_message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)
        hit_turn = best_rank = None
        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            if not isinstance(response, dict):
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                best_rank, hit_turn = ranked.index(target) + 1, turn
                break
            if turn == MAX_TURNS:
                break
            override = effective["behavior"].get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                # Same timing and content as the organizer message, but an unseen wording.
                user_message = f"I changed my mind. Instead, I need {new_value}."
            else:
                user_message, boundary_used = customer_reply(
                    effective, response.get("ask_attribute"), disclosed, boundary_used
                )
        sessions.append({"sample_id": sample["sample_id"], "scenario_type": "intent_override",
                         "hit": hit_turn is not None, "first_hit_turn": hit_turn,
                         "best_rank": best_rank,
                         "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank})
    overall = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    score = 0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency
    return {**overall, "efficiency": round(efficiency, 6), "recommended_technical_score": round(score, 6)}


def compact(result: dict) -> dict:
    return {key: result[key] for key in (
        "sample_count", "hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score"
    )}


def main() -> None:
    rows = load_jsonl(DATASET)
    catalog = ROOT / "data" / "catalog.jsonl"
    ids, categories, products = catalog_index(catalog)
    base = Agent(catalog)
    variants = {"shipped": Agent, "gated_paraphrase_replacement": GatedParaphraseReplacement}
    result = {
        "purpose": "aggressive replacement branch structurally unreachable on released message forms",
        "gate": "not recognised(message) AND explicit replacement cue AND explicit need/want/prefer value",
        "paraphrased_override_template": "I changed my mind. Instead, I need {new_value}.",
        "variants": {},
    }
    for name, cls in variants.items():
        compatible_agent = shared_agent(cls, base)
        started = time.perf_counter()
        compatible = evaluate_paraphrased_override(compatible_agent, rows, ids, categories, products, contradiction=False)
        conflict_agent = shared_agent(cls, base)
        started_conflict = time.perf_counter()
        conflict = evaluate_paraphrased_override(conflict_agent, rows, ids, categories, products, contradiction=True)
        public_agent = shared_agent(cls, base)
        public = evaluate(public_agent, load_jsonl(ROOT / "data" / "public_set.jsonl"), ids, categories, products)
        item = {
            "paraphrased_compatible_override_800": {**compact(compatible), "wall_seconds": time.perf_counter() - started},
            "paraphrased_contradictory_override_800": {**compact(conflict), "wall_seconds": time.perf_counter() - started_conflict},
            "official200": compact(public),
        }
        if isinstance(compatible_agent, GatedParaphraseReplacement):
            item["compatible_gate"] = compatible_agent.override_gate
            item["contradictory_gate"] = conflict_agent.override_gate
            item["official_gate"] = public_agent.override_gate
        result["variants"][name] = item
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, item in result["variants"].items():
        print(name)
        for label in ("paraphrased_compatible_override_800", "paraphrased_contradictory_override_800", "official200"):
            metrics = item[label]
            print(f"  {label:<43} score={metrics['recommended_technical_score']:.6f} HR={metrics['hit_rate_at_10']:.3f}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
