"""V2.03: measure exact template span extraction on semantic-value traffic.

Format is held constant in this V2 pass. Therefore the valid Node 2 baseline is the
submitted template parser, not the BERT fallback that is deliberately inactive on known
templates. The audit checks that every value placed in an exposed constraint slot is
returned verbatim by ``Agent._extract_templated``.
"""
from __future__ import annotations

import json
from pathlib import Path

from evaluator.local_evaluator import load_jsonl
from submission.agent import Agent, CONSTRAINT
from experiments.studies.run_semantic_attribute import v2_behavior

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "experiments" / "studies" / "sets" / "semantic_attribute_development_200.jsonl"
OUT = ROOT / "experiments" / "studies" / "results" / "span_node_development.json"


def expected_values(sample: dict) -> list[str]:
    return [str(atom["paraphrase"]) for group in sample["semantic_card"].values() for atom in group]


def main() -> None:
    agent = Agent(ROOT / "data" / "catalog.jsonl")
    cases: list[dict] = []
    for sample in load_jsonl(DATASET):
        values = expected_values(sample)
        hard = [str(atom["paraphrase"]) for atom in sample["semantic_card"].get("hard_constraints", [])]
        soft = [str(atom["paraphrase"]) for atom in sample["semantic_card"].get("soft_preferences", [])]
        if hard:
            cases.append((sample["sample_id"], "buying", f"I'm looking for {sample['category']}. A key requirement is: {hard[0]}.", [hard[0]]))
        cases.append((sample["sample_id"], "matters", "For that, what matters is: " + "; ".join(values) + ".", values))
        behavior = v2_behavior(sample, "paraphrase")
        override = str((behavior.get("override") or {}).get("message", ""))
        new_value = str((behavior.get("override") or {}).get("new_value", ""))
        if override and new_value:
            cases.append((sample["sample_id"], "override", override, [new_value]))
        # The initial override opening has no labelled constraint slot. Its value is
        # deliberately audited separately as source-confirmed opening evidence.
        if sample["scenario_type"] == "intent_override" and soft:
            opening = f"I'm looking for {sample['category']}. {soft[-1]}"
            cases.append((sample["sample_id"], "override_opening", opening, []))
    failures = []
    for sample_id, kind, message, expected in cases:
        observed = [text for text, tier in agent._extract_templated(message) if tier == CONSTRAINT]
        if observed != expected:
            failures.append({"sample_id": sample_id, "kind": kind, "expected": expected, "observed": observed, "message": message})
    result = {
        "experiment": "V2.03 exact template span extraction",
        "dataset": str(DATASET),
        "cases": len(cases),
        "exact_span_recovery": round((len(cases) - len(failures)) / max(1, len(cases)), 6),
        "failures": failures,
        "scope_note": "The BERT scaffolding tagger is intentionally excluded: all messages use recognised official-format wrappers.",
        "decision": "Pass only at exact recovery; route or template parsing must be corrected before semantic family or attribute resolution.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "failures"}, indent=2))


if __name__ == "__main__":
    main()
