"""Experiment 62: replace unconfirmed same-family opening evidence on override.

Unlike experiment 61, this rule does not ask whether two values can co-occur somewhere
in the catalogue. A later explicit override wins when it and the unconfirmed opening
value are both recognisably material, colour, or closure evidence. Generic feature prose
is deliberately left untouched.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "log"))
os.environ["LLM_EXTRACT"] = "0"
os.environ["LLM_RERANK"] = "0"

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent, PAT_MATTERS, PAT_OVERRIDE, PAT_OVERRIDE_OPENING  # noqa: E402

guard = importlib.import_module("61_catalog_conflict_guard")
probe = importlib.import_module("60_override_focus_contradiction")
DATASET = probe.DATASET
OUT = ROOT / "experiments" / "results" / "out_62_same_family_override_replace.json"


class SameFamilyOpeningReplace(Agent):
    """Later explicit value replaces only an unconfirmed earlier value of its family."""

    def reset(self, session_id: str, user_profile: dict) -> None:
        super().reset(session_id, user_profile)
        if not hasattr(self, "_override_state"):
            self._override_state = {}
            self.replace_stats = {"opening_values": 0, "confirmed_openings": 0,
                                  "family_comparable": 0, "replaced": 0, "abstained": 0}
        self._override_state[session_id] = {"opening": {}, "confirmed": set()}

    def _observe(self, state, message: str) -> None:
        opening = PAT_OVERRIDE_OPENING.match(message.strip())
        old_value = opening.group(1).strip() if opening else ""
        old_resolved = self._resolve(old_value) if old_value else []
        super()._observe(state, message)
        record = self._override_state.setdefault(state.sid, {"opening": {}, "confirmed": set()})
        if old_value:
            self.replace_stats["opening_values"] += 1
            for phrase in old_resolved:
                record["opening"][phrase] = old_value
        matters = PAT_MATTERS.search(message)
        if matters:
            resolved = set()
            for value in matters.group(1).split(";"):
                resolved.update(self._resolve(value.strip()))
            new_confirmed = set(record["opening"]) & resolved - record["confirmed"]
            self.replace_stats["confirmed_openings"] += len(new_confirmed)
            record["confirmed"].update(resolved)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        override = PAT_OVERRIDE.search(user_message)
        state = self.sessions.get(session_id)
        record = getattr(self, "_override_state", {}).get(session_id)
        if override and state is not None and record is not None:
            new_family = guard.family(override.group(1).strip())
            for old_phrase, old_value in list(record["opening"].items()):
                if old_phrase in record["confirmed"]:
                    continue
                old_family = guard.family(old_value)
                if old_family and old_family == new_family:
                    state.evidence.pop(old_phrase, None)
                    self.replace_stats["family_comparable"] += 1
                    self.replace_stats["replaced"] += 1
                else:
                    self.replace_stats["abstained"] += 1
        return super().respond(session_id, user_message, turn, top_k)


def shared_agent(cls: type[Agent], base: Agent) -> Agent:
    agent = object.__new__(cls)
    agent.ix = base.ix
    agent.sessions = {}
    agent.llm = None
    agent.llm_extract = None
    agent.tagger = None
    if cls is SameFamilyOpeningReplace:
        agent._override_state = {}
        agent.replace_stats = {"opening_values": 0, "confirmed_openings": 0,
                               "family_comparable": 0, "replaced": 0, "abstained": 0}
    return agent


def compact(result: dict) -> dict:
    return {key: result[key] for key in (
        "sample_count", "hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score"
    )}


def main() -> None:
    rows = load_jsonl(DATASET)
    catalog = ROOT / "data" / "catalog.jsonl"
    ids, categories, products = catalog_index(catalog)
    base = Agent(catalog)
    variants = {"shipped_full_opening_evidence": Agent, "same_family_opening_replace": SameFamilyOpeningReplace}
    result = {
        "purpose": "selective replacement of unconfirmed same-family opening evidence",
        "rule": "explicit later override wins only for same high-confidence material, colour, or closure family",
        "source_faithful_set": str(DATASET.relative_to(ROOT)),
        "variants": {},
    }
    for name, cls in variants.items():
        source_agent = shared_agent(cls, base)
        started = time.perf_counter()
        source = evaluate(source_agent, rows, ids, categories, products)
        conflict_agent = shared_agent(cls, base)
        started_conflict = time.perf_counter()
        conflict = probe.evaluate_contradictory_opening(conflict_agent, rows, ids, categories, products)
        item = {
            "source_faithful_override_focus_800": {**compact(source), "wall_seconds": time.perf_counter() - started},
            "contradictory_opening_probe_800": {**compact(conflict), "wall_seconds": time.perf_counter() - started_conflict},
        }
        if isinstance(source_agent, SameFamilyOpeningReplace):
            item["source_faithful_detector"] = source_agent.replace_stats
            item["contradictory_detector"] = conflict_agent.replace_stats
        result["variants"][name] = item
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, item in result["variants"].items():
        print(name)
        for label in ("source_faithful_override_focus_800", "contradictory_opening_probe_800"):
            metrics = item[label]
            print(f"  {label:<42} score={metrics['recommended_technical_score']:.6f} "
                  f"HR={metrics['hit_rate_at_10']:.3f} MRR={metrics['mrr']:.3f} MTTC={metrics['mttc']:.3f}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
