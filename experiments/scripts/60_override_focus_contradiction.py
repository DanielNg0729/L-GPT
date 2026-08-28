"""Experiment 60: all-override stress and true-contradiction recovery.

The released generator's old override value is target-derived and compatible with the
final target. This experiment first tests that source-faithful condition on 800 all-
override sessions. It then creates a clearly labelled counterfactual: only the initial
old-value slot becomes a catalogue-attested material absent from the target and is not
re-disclosed later. The final target-derived new intent and official timing are retained.

The counterfactual is a robustness probe, not a claim about organizer-private language.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["LLM_EXTRACT"] = "0"
os.environ["LLM_RERANK"] = "0"

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    evaluate,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    metric_summary,
    normalize_recommendations,
    searchable_text,
)
from submission.agent import (  # noqa: E402
    Agent,
    CONSTRAINT,
    PAT_MATTERS,
    PAT_OVERRIDE,
    PAT_OVERRIDE_OPENING,
    raw_toks,
)

DATASET = ROOT / "robustness" / "override_focus" / "override_focus_800.jsonl"
OUT = ROOT / "experiments" / "results" / "out_60_override_focus_contradiction.json"
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon")


class NoOpeningEvidence(Agent):
    """Reconstruct the pre-pass-59 behavior for an explicit control."""

    def _extract_templated(self, message: str):
        extracted = super()._extract_templated(message)
        opening = PAT_OVERRIDE_OPENING.match(message.strip())
        if not opening:
            return extracted
        old_value = opening.group(1).strip()
        removed = False
        kept = []
        for phrase, tier in extracted:
            if not removed and tier == CONSTRAINT and phrase == old_value:
                removed = True
                continue
            kept.append((phrase, tier))
        return kept


class NeutraliseUnconfirmedOpening(Agent):
    """Clear only an opening value that was never confirmed before the override.

    This is a neutral dialogue-policy repair: source-faithful opening evidence remains
    available when the simulator later confirms it, while a genuinely superseded initial
    value does not survive into final-intent ranking.
    """

    def reset(self, session_id: str, user_profile: dict) -> None:
        super().reset(session_id, user_profile)
        if not hasattr(self, "_opening_state"):
            self._opening_state = {}
        self._opening_state[session_id] = {"opening": set(), "confirmed": set()}

    def _observe(self, state, message: str) -> None:
        opening = PAT_OVERRIDE_OPENING.match(message.strip())
        opening_phrases = self._resolve(opening.group(1)) if opening else []
        super()._observe(state, message)
        record = self._opening_state.setdefault(state.sid, {"opening": set(), "confirmed": set()})
        if opening_phrases:
            record["opening"].update(opening_phrases)
        matters = PAT_MATTERS.search(message)
        if matters:
            confirmed = set()
            for value in matters.group(1).split(";"):
                confirmed.update(self._resolve(value.strip()))
            record["confirmed"].update(record["opening"] & confirmed)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if PAT_OVERRIDE.search(user_message):
            record = getattr(self, "_opening_state", {}).get(session_id)
            state = self.sessions.get(session_id)
            if record is not None and state is not None:
                for phrase in record["opening"] - record["confirmed"]:
                    state.evidence.pop(phrase, None)
        return super().respond(session_id, user_message, turn, top_k)


def shared_agent(cls: type[Agent], base: Agent) -> Agent:
    agent = object.__new__(cls)
    agent.ix = base.ix
    agent.sessions = {}
    agent.llm = None
    agent.llm_extract = None
    agent.tagger = None
    if cls is NeutraliseUnconfirmedOpening:
        agent._opening_state = {}
    return agent


def counterfactual_old_value(product: dict, target: str) -> str:
    """A catalogue-attested material absent from the target's searchable document."""
    text = set(raw_toks(searchable_text(product)))
    offset = sum(ord(char) for char in target) % len(MATERIALS)
    for index in range(len(MATERIALS)):
        candidate = MATERIALS[(offset + index) % len(MATERIALS)]
        if candidate not in text:
            return candidate
    raise AssertionError(f"no absent conflict material for {target}")


def evaluate_contradictory_opening(agent: Agent, samples: list[dict], catalog_ids: set[str],
                                   categories: dict[str, list[str]], products: dict[str, dict]) -> dict:
    """Official loop with a single, labelled change to old-value disclosure."""
    sessions = []
    conflicts = defaultdict(int)
    for sample in samples:
        session_id = f"contradiction_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        original_old = str(behavior["override"]["old_value"])
        conflict = counterfactual_old_value(products[target], target)
        behavior = copy.deepcopy(behavior)
        behavior["override"]["old_value"] = conflict
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = {original_old}  # do not re-disclose the superseded source value
        boundary_used = False
        override_applied = False
        user_message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)
        hit_turn = best_rank = None
        conflicts[conflict] += 1
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
                user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                user_message, boundary_used = customer_reply(
                    effective, response.get("ask_attribute"), disclosed, boundary_used
                )
        sessions.append({
            "sample_id": sample["sample_id"], "scenario_type": "intent_override",
            "hit": hit_turn is not None, "first_hit_turn": hit_turn, "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })
    overall = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    score = 0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(score, 6),
        "conflicting_old_value_counts": dict(sorted(conflicts.items())),
    }


def compact(result: dict) -> dict:
    return {key: result[key] for key in (
        "sample_count", "hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score"
    )}


def main() -> None:
    rows = load_jsonl(DATASET)
    if len(rows) != 800 or any(row["scenario_type"] != "intent_override" for row in rows):
        raise ValueError("OverrideFocus800 invariant failed")
    catalog = ROOT / "data" / "catalog.jsonl"
    ids, categories, products = catalog_index(catalog)
    base = Agent(catalog)
    variants = {
        "shipped_full_opening_evidence": Agent,
        "no_opening_evidence_control": NoOpeningEvidence,
        "neutralise_unconfirmed_opening": NeutraliseUnconfirmedOpening,
    }
    result = {
        "purpose": "all-override source-faithful comparison and counterfactual contradiction probe",
        "source_faithful_set": str(DATASET.relative_to(ROOT)),
        "counterfactual_contract": (
            "replace only old_value with a catalogue-attested material absent from target; "
            "suppress later re-disclosure of the old value; preserve target-derived new_value and timing"
        ),
        "variants": {},
    }
    for name, cls in variants.items():
        compatible_start = time.perf_counter()
        compatible = evaluate(shared_agent(cls, base), rows, ids, categories, products)
        conflict_start = time.perf_counter()
        conflict = evaluate_contradictory_opening(shared_agent(cls, base), rows, ids, categories, products)
        result["variants"][name] = {
            "source_faithful_override_focus_800": {**compact(compatible), "wall_seconds": time.perf_counter() - compatible_start},
            "contradictory_opening_probe_800": {**compact(conflict), "wall_seconds": time.perf_counter() - conflict_start},
        }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, row in result["variants"].items():
        print(name)
        for label, metrics in row.items():
            print(f"  {label:<42} score={metrics['recommended_technical_score']:.6f} "
                  f"HR={metrics['hit_rate_at_10']:.3f} MRR={metrics['mrr']:.3f} MTTC={metrics['mttc']:.3f}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
