"""Create fixed Node 1 format-shift cases without paraphrasing attribute values."""
from __future__ import annotations

import json
from pathlib import Path

from evaluator.local_evaluator import load_jsonl

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "datasets" / "sets" / "semantic_attribute_development_200.jsonl"
OUT = ROOT / "experiments" / "studies" / "route_shift"

TEMPLATES = {
    "development": {
        "opening": "I need {category} with {a}.",
        "constraint_reply": "Please prioritize {a} and {b}.",
        "no_preference": "No other preference for {attribute}.",
        "boundary": "Choose freely on {attribute}; I have no preference.",
        "nudge": "These do not suit me. Ask one product property.",
        "override": "Please replace my earlier requirement: {a}.",
    },
    "holdout": {
        "opening": "Seeking {category}; it must have {a}.",
        "constraint_reply": "The important details are {a}; {b}.",
        "no_preference": "Nothing more matters for {attribute}.",
        "boundary": "I do not care about {attribute}; use your judgment.",
        "nudge": "Not right yet. Ask about one attribute at a time.",
        "override": "Actually, change the previous choice. I now want {a}.",
    },
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, templates in TEMPLATES.items():
        output = []
        for sample in load_jsonl(SOURCE):
            atoms = [atom for group in sample["semantic_card"].values() for atom in group]
            a, b = str(atoms[0]["canonical"]), str(atoms[1]["canonical"])
            for route, template in templates.items():
                output.append({
                    "sample_id": sample["sample_id"], "route": route,
                    "message": template.format(category=sample["category"], a=a, b=b, attribute="material"),
                    "value_preservation": [a, b],
                })
        path = OUT / f"route_shift_{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in output), encoding="utf-8")
        rows.append({"split": split, "path": path.name, "rows": len(output), "templates": templates})
    (OUT / "manifest.json").write_text(json.dumps({"schema": 1, "source": str(SOURCE), "value_change": "none", "splits": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
