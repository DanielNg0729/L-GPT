"""Run one participant-visible multi-turn session through the submitted agent.

This demonstration uses only the released public set and the official simulator functions.
It does not read organizer-private labels or state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from submission.agent import Agent


def demonstrate(sample_id: str, catalog_path: Path, public_path: Path) -> None:
    samples = load_jsonl(public_path)
    try:
        sample = next(row for row in samples if row["sample_id"] == sample_id)
    except StopIteration as exc:
        raise ValueError(f"unknown public sample: {sample_id}") from exc

    catalog_ids, categories, products = catalog_index(catalog_path)
    target = str(sample["ground_truth"]["parent_asin"])
    intent_card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": intent_card, "behavior": behavior}

    agent = Agent(catalog_path)
    session_id = f"demo_{sample_id}"
    agent.reset(session_id, sample["user_profile"])

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    customer = initial_message(
        effective, coarse_category(categories.get(target, [])), disclosed
    )

    print(json.dumps({
        "sample_id": sample_id,
        "scenario_type": sample["scenario_type"],
        "public_target": target,
    }))

    for turn in range(1, MAX_TURNS + 1):
        response = agent.respond(session_id, customer, turn, TOP_K)
        ranked = normalize_recommendations(response["recommendations"], catalog_ids)
        hit = override_applied and target in ranked
        print(json.dumps({
            "turn": turn,
            "customer": customer,
            "agent_message": response["message"],
            "ask_attribute": response["ask_attribute"],
            "recommendations": ranked,
            "usage": response["usage"],
            "hit": hit,
            "target_rank": ranked.index(target) + 1 if hit else None,
        }))
        if hit or turn == MAX_TURNS:
            return

        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            customer = str(
                override.get("message", "Actually, please ignore my earlier preference.")
            )
        else:
            customer, boundary_used = customer_reply(
                effective, response["ask_attribute"], disclosed, boundary_used
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", default="public_0002")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--public-set", type=Path, default=Path("data/public_set.jsonl"))
    args = parser.parse_args()
    demonstrate(args.sample_id, args.catalog, args.public_set)


if __name__ == "__main__":
    main()
