"""Emit an exact layer-by-layer trace for one released public session.

This is a diagnostic utility. It invokes the submitted Agent exactly once per customer
turn and records the internal outputs that lead to its public response. It never reads
organizer-private data or changes the agent's ranking behavior.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (
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
from submission.agent import Agent, PAT_OVERRIDE, PAT_OVERRIDE_CUE, recognised


def json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(json_safe(item) for item in value)
    return value


def trace_turn(agent: Agent, session_id: str, customer: str, turn: int,
               catalog_ids: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call respond once while observing the exact internal method outputs."""
    state = agent.sessions[session_id]
    captured: dict[str, Any] = {
        "input": {
            "turn": turn,
            "customer_message": customer,
            "recognised_official_form": recognised(customer),
            "override_cue": bool(PAT_OVERRIDE.search(customer) or PAT_OVERRIDE_CUE.search(customer)),
            "rejected_before_respond": sorted(state.rejected),
            "evidence_before_respond": dict(state.evidence),
        }
    }

    original_observe: Callable[..., Any] = agent._observe
    original_candidates: Callable[..., Any] = agent._candidates
    original_rank: Callable[..., Any] = agent._rank
    original_rerank: Callable[..., Any] = agent._rerank_exact_ties

    def observe_wrapper(st: Any, message: str) -> None:
        raw = agent._extract_templated(message)
        resolved = [
            {"input": phrase, "tier": tier, "resolved": agent._resolve(phrase)}
            for phrase, tier in raw
        ]
        before = dict(st.evidence)
        original_observe(st, message)
        captured["extraction"] = {
            "template_matches": raw,
            "template_resolution": resolved,
            "evidence_added": {
                phrase: record for phrase, record in st.evidence.items()
                if phrase not in before
            },
            "evidence_after_extraction": dict(st.evidence),
            "rejected_entering_extraction": sorted(st.rejected),
        }

    def candidates_wrapper(st: Any, message: str) -> list[str]:
        pool = original_candidates(st, message)
        captured["retrieval"] = {
            "candidate_count": len(pool),
            "fts5_order_first_10": pool[:10],
        }
        return pool

    def rank_wrapper(st: Any, pool: list[str], top_k: int) -> list[str]:
        ranked = original_rank(st, pool, top_k)
        captured["ranking"] = {"coverage_ranked_top_10": ranked}
        return ranked

    def rerank_wrapper(st: Any, ranked: list[str]) -> list[str]:
        output = original_rerank(st, ranked)
        captured.setdefault("ranking", {})["after_optional_llm_tie_rerank"] = output
        return output

    agent._observe = observe_wrapper  # type: ignore[method-assign]
    agent._candidates = candidates_wrapper  # type: ignore[method-assign]
    agent._rank = rank_wrapper  # type: ignore[method-assign]
    agent._rerank_exact_ties = rerank_wrapper  # type: ignore[method-assign]
    try:
        response = agent.respond(session_id, customer, turn, TOP_K)
    finally:
        agent._observe = original_observe  # type: ignore[method-assign]
        agent._candidates = original_candidates  # type: ignore[method-assign]
        agent._rank = original_rank  # type: ignore[method-assign]
        agent._rerank_exact_ties = original_rerank  # type: ignore[method-assign]

    captured["policy"] = {
        "asked_attribute": response["ask_attribute"],
        "recommendation_width": len(response["recommendations"]),
    }
    captured["output"] = {
        "response": response,
        "normalised_recommendations": normalize_recommendations(
            response["recommendations"], catalog_ids
        ),
        "rejected_after_respond": sorted(state.rejected),
        "evidence_after_respond": dict(state.evidence),
    }
    return json_safe(captured), response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", default="public_0002")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--public-set", type=Path, default=Path("data/public_set.jsonl"))
    args = parser.parse_args()

    samples = load_jsonl(args.public_set)
    sample = next(row for row in samples if row["sample_id"] == args.sample_id)
    catalog_ids, categories, products = catalog_index(args.catalog)
    target = str(sample["ground_truth"]["parent_asin"])
    intent_card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": intent_card, "behavior": behavior}

    agent = Agent(args.catalog)
    session_id = f"trace_{args.sample_id}"
    agent.reset(session_id, sample["user_profile"])
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    customer = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

    print(json.dumps({
        "sample_id": args.sample_id,
        "scenario_type": sample["scenario_type"],
        "public_target": target,
        "user_profile": sample["user_profile"],
    }, indent=2))
    for turn in range(1, MAX_TURNS + 1):
        trace, response = trace_turn(agent, session_id, customer, turn, catalog_ids)
        ranked = trace["output"]["normalised_recommendations"]
        trace["outcome"] = {
            "target_eligible": override_applied,
            "target_rank": ranked.index(target) + 1 if override_applied and target in ranked else None,
        }
        print(json.dumps(trace, indent=2))
        if trace["outcome"]["target_rank"] is not None:
            return
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


if __name__ == "__main__":
    main()
