"""Build a fresh final test for the turn-gated route-classifier experiment.

The earlier V2.15 final set is consumed and becomes development-only evidence for
this experiment.  This file creates newly authored, full-wrapper-disjoint final
templates while retaining the same canonical catalogue values.
"""
from __future__ import annotations

import json
from pathlib import Path

from evaluator.local_evaluator import load_jsonl


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "datasets" / "sets" / "semantic_attribute_development_200.jsonl"
OUT = ROOT / "experiments" / "studies" / "turn_gated_bank"

TEMPLATES = {
    "buying_opening": [
        "Please find {category} that offers {a}.",
        "I want to purchase {category} made from {a}.",
        "Find me {category}; {a} is required.",
        "Bring up {category} with a focus on {a}.",
        "Search for {category} meeting {a}.",
        "I am after {category} and require {a}.",
        "Recommend {category} where {a} is present.",
        "Show {category} satisfying {a}.",
    ],
    "plain_opening": [
        "Could we look through {category} without deciding criteria?",
        "Start by showing {category}; I am only exploring.",
        "I am casually reviewing {category} right now.",
        "Help me survey {category} before specifying needs.",
        "Display {category}; I have not chosen requirements.",
        "Take me through {category} while I remain undecided.",
        "Give me an overview of {category}, with no criteria yet.",
        "Let me inspect {category} before committing.",
    ],
    "override_opening": [
        "Before deciding, I considered {category}, preferring {b}.",
        "My first idea was {category}, especially {b}.",
        "I began with {category} because {b} appealed.",
        "I previously looked at {category} with {b} in view.",
        "Initially, {category} interested me for {b}.",
        "Earlier I was considering {category} and favored {b}.",
        "My starting option was {category}, with {b} preferred.",
        "I originally examined {category}, thinking about {b}.",
    ],
    "constraint_update": [
        "Please ensure {a} and {b} are both covered.",
        "One thing I need is {a}, together with {b}.",
        "Include {a} as well as {b} in the selection.",
        "The selection should satisfy {a} and {b}.",
        "I want both {a} and {b} considered.",
        "Keep {a} plus {b} in mind.",
        "My requirements now include {a} and {b}.",
        "Account for {a} alongside {b}.",
    ],
    "no_evidence": [
        "Attribute {attribute} does not affect my choice.",
        "I have nothing to add about {attribute}.",
        "{attribute} is not important to me.",
        "No condition is needed for {attribute}.",
        "Please decide {attribute} yourself.",
        "I am indifferent about {attribute}.",
        "Do not filter on {attribute} for me.",
        "Leave {attribute} unspecified.",
    ],
    "override_update": [
        "Revise the request so that {a} is required.",
        "I am replacing the previous choice with {a}.",
        "My new priority is {a}.",
        "Please treat {a} as the new requirement.",
        "Set {a} as the requirement from now on.",
        "The earlier option should be changed to {a}.",
        "Use {a} in place of the old preference.",
        "I now need {a}, not the earlier choice.",
    ],
}


def main():
    rows = []
    for source in load_jsonl(SOURCE):
        atoms = [atom for group in source["semantic_card"].values() for atom in group]
        slots = {"category": source["category"], "a": str(atoms[0]["canonical"]), "b": str(atoms[1]["canonical"]), "attribute": "material"}
        for action, templates in TEMPLATES.items():
            for template in templates:
                rows.append({"sample_id": source["sample_id"], "action": action, "message": template.format(**slots), "template": template, "slots": slots, "source": "v2_18_final"})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "final_test.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = {"templates_per_action": 8, "total_templates": 48, "rows": len(rows), "invariant": "Only the conversational wrapper changes; category and attribute values remain verbatim.", "role": "untouched final test for the turn-gated route-classifier experiment"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
