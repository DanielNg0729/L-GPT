"""V2.02: verify route and dialogue-act recognition before semantic extraction.

The semantic benchmark preserves organizer wrappers. This audit replays the development
sessions and checks that every customer message is recognised as its generated dialogue
act. It neither resolves attributes nor changes the submitted agent.
"""
from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from pathlib import Path

from evaluator.local_evaluator import MAX_TURNS, TOP_K, catalog_index, load_jsonl, normalize_recommendations
from submission.agent import Agent
from experiments.studies.run_semantic_attribute import v2_behavior, v2_customer_reply, v2_initial_message

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "experiments" / "studies" / "sets" / "semantic_attribute_development_200.jsonl"
OUT = ROOT / "experiments" / "studies" / "results" / "route_node_development.json"

SHAPES = (
    ("buying_open", re.compile(r"^I'm looking for .+\. A key requirement is: .+\.$")),
    ("browsing_open", re.compile(r"^I'm looking for .+, but I'm still exploring\.$")),
    ("reply_matters", re.compile(r"^For that, what matters is: .+\.$")),
    ("reply_none", re.compile(r"^I don't have an additional preference for [a-z_]+\.$")),
    ("boundary", re.compile(r"^I don't have a preference for [a-z_]+; please use your judgment\.$")),
    ("nudge", re.compile(r"^Those options are not quite right yet\. Ask me about one specific attribute\.$")),
    ("override", re.compile(r"^Actually, ignore my earlier preference\. What I need is: .+\.$")),
    ("override_dflt", re.compile(r"^Actually, please ignore my earlier preference\.$")),
    ("override_open", re.compile(r"^I'm looking for .+\. .+$")),
)


def route(message: str) -> str | None:
    return next((name for name, pattern in SHAPES if pattern.match(message.strip())), None)


def main() -> None:
    catalog = ROOT / "data" / "catalog.jsonl"
    ids, categories, _ = catalog_index(catalog)
    agent = Agent(catalog)
    messages: list[dict] = []
    for sample in load_jsonl(DATASET):
        session = f"route_{uuid.uuid4().hex}"
        agent.reset(session, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = v2_initial_message(sample, str(sample["category"]), disclosed, "paraphrase")
        expected = "buying_open" if sample["scenario_type"] == "buying" else ("override_open" if sample["scenario_type"] == "intent_override" else "browsing_open")
        behavior = v2_behavior(sample, "paraphrase")
        for turn in range(1, MAX_TURNS + 1):
            observed = route(user_message)
            messages.append({"sample_id": sample["sample_id"], "turn": turn, "expected": expected, "observed": observed})
            response = agent.respond(session, user_message, turn, TOP_K)
            ranked = normalize_recommendations(response.get("recommendations"), ids)
            if override_applied and target in ranked or turn == MAX_TURNS:
                break
            override = behavior.get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
                expected = "override" if user_message.startswith("Actually, ignore") else "override_dflt"
            else:
                user_message, boundary_used = v2_customer_reply(sample, response.get("ask_attribute"), disclosed, boundary_used, "paraphrase")
                expected = "boundary" if user_message.startswith("I don't have a preference") else ("reply_none" if user_message.startswith("I don't have an additional") else ("nudge" if user_message.startswith("Those options") else "reply_matters"))
    total = len(messages)
    matched = sum(row["expected"] == row["observed"] for row in messages)
    result = {
        "experiment": "V2.02 route and dialogue-act recognition",
        "dataset": str(DATASET),
        "messages": total,
        "recognised": sum(row["observed"] is not None for row in messages),
        "exact_dialogue_act_accuracy": round(matched / total, 6),
        "unmatched": [row for row in messages if row["observed"] is None],
        "confusions": dict(Counter(f"{row['expected']}->{row['observed']}" for row in messages if row["expected"] != row["observed"])),
        "decision": "Pass only at complete recognition and complete dialogue-act agreement; otherwise isolate route repair before extraction.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"unmatched", "confusions"}}, indent=2))


if __name__ == "__main__":
    main()
