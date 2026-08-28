"""Measure one cold index build and representative released public sessions.

This utility is intentionally a timing sample, not a full 200-session evaluation. It
selects the first released session from each scenario type and runs the exact evaluator
conversation loop against one shared offline Agent instance.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Measure the submitted offline path, regardless of developer-shell environment variables.
os.environ["LLM_EXTRACT"] = "0"
os.environ["LLM_RERANK"] = "0"

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from submission.agent import Agent  # noqa: E402


def run_session(agent: Agent, sample: dict, catalog_ids: set[str], categories: dict,
                products: dict, ordinal: int) -> dict:
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    session_id = f"runtime_{sample['sample_id']}_{ordinal}"
    agent.reset(session_id, sample["user_profile"])
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    customer = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)
    turn_seconds: list[float] = []
    hit_turn = None
    best_rank = None
    started = time.perf_counter()

    for turn in range(1, MAX_TURNS + 1):
        turn_started = time.perf_counter()
        response = agent.respond(session_id, customer, turn, TOP_K)
        turn_seconds.append(time.perf_counter() - turn_started)
        ranked = normalize_recommendations(response["recommendations"], catalog_ids)
        if override_applied and target in ranked:
            hit_turn = turn
            best_rank = ranked.index(target) + 1
            break
        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            customer = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            customer, boundary_used = customer_reply(
                effective, response["ask_attribute"], disclosed, boundary_used
            )

    return {
        "sample_id": sample["sample_id"],
        "scenario": sample["scenario_type"],
        "turns": len(turn_seconds),
        "hit_turn": hit_turn,
        "target_rank": best_rank,
        "session_seconds": time.perf_counter() - started,
        "agent_response_seconds": sum(turn_seconds),
        "mean_response_ms": 1000 * sum(turn_seconds) / len(turn_seconds),
        "max_response_ms": 1000 * max(turn_seconds),
        "turn_response_ms": [round(1000 * value, 3) for value in turn_seconds],
    }


def main() -> None:
    catalog = ROOT / "data" / "catalog.jsonl"
    public = load_jsonl(ROOT / "data" / "public_set.jsonl")
    # Build the agent first so evaluator convenience structures cannot pre-warm the raw
    # catalogue file before the index measurement.
    cold_started = time.perf_counter()
    agent = Agent(catalog)
    index_seconds = time.perf_counter() - cold_started
    catalog_ids, categories, products = catalog_index(catalog)
    selected = []
    for scenario in ("buying", "browsing", "intent_override", "boundary"):
        selected.append(next(row for row in public if row["scenario_type"] == scenario))
    rows = [run_session(agent, sample, catalog_ids, categories, products, i)
            for i, sample in enumerate(selected, start=1)]
    total_turns = sum(row["turns"] for row in rows)
    total_agent_seconds = sum(row["agent_response_seconds"] for row in rows)
    print(json.dumps({
        "configuration": "submitted offline path, optional Groq features forced off",
        "catalog_products": len(catalog_ids),
        "cold_agent_and_index_seconds": index_seconds,
        "representative_sessions": rows,
        "aggregate": {
            "sessions": len(rows),
            "turns": total_turns,
            "mean_session_seconds": sum(row["session_seconds"] for row in rows) / len(rows),
            "mean_agent_response_ms_per_turn": 1000 * total_agent_seconds / total_turns,
        },
    }, indent=2))


if __name__ == "__main__":
    main()
