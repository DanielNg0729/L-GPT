"""Publish an aligned Node 2 comparison for TemplateParaphrase9600-Test.

This script performs no inference.  It normalizes the already-completed V2.24 and
V2.25 measurements, which used the same frozen 9,600-message split and exact same
V1 BERT model, into one explicit metric table.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "experiments" / "studies" / "results" / "v2_24_node2_scaffolding_final_test.json"
NEW = ROOT / "experiments" / "studies" / "results" / "v2_25_node2_short_span_dictionary_final_test.json"
OUT = ROOT / "experiments" / "studies" / "results" / "template_paraphrase9600_node2_comparison.json"


def main() -> None:
    old = json.loads(OLD.read_text(encoding="utf-8"))
    new = json.loads(NEW.read_text(encoding="utf-8"))
    result = {
        "dataset_name": "TemplateParaphrase9600",
        "split": "Test",
        "definition": "Fixed 9,600-message, template-disjoint unfamiliar-wrapper test. Category and attribute values remain verbatim catalogue text.",
        "scope": "Node 2 extraction only. These are not whole-session Hit Rate or MTTC metrics.",
        "methods": {
            "raw_v1_mining": {
                "constraint_catalogue_recovery": old["constraint_mining_recovery"]["raw"],
                "category_catalogue_recovery": old["category_mining_recovery"]["raw"],
            },
            "existing_bert_then_v1_mining": {
                "constraint_slot_retention": old["constraint_slot_retention"],
                "constraint_catalogue_recovery": old["constraint_mining_recovery"]["bert"],
                "category_slot_retention": old["category_slot_retention"],
                "category_catalogue_recovery": old["category_mining_recovery"]["bert"],
            },
            "existing_bert_then_short_dictionary": {
                "constraint_slot_retention": old["constraint_slot_retention"],
                "constraint_catalogue_recovery": new["constraint_recall"],
                "category_slot_retention": old["category_slot_retention"],
                "category_catalogue_recovery": old["category_mining_recovery"]["bert"],
                "mean_extra_constraint_candidates": new["mean_extra_candidates_per_constraint_message"],
            },
            "existing_bert_then_category_masked_short_dictionary": {
                "constraint_slot_retention": old["constraint_slot_retention"],
                "constraint_catalogue_recovery": new["category_masked"]["constraint_recall"],
                "category_slot_retention": old["category_slot_retention"],
                "category_catalogue_recovery": new["category_exact_catalogue_recovery"],
                "mean_extra_constraint_candidates": new["category_masked"]["mean_extra_candidates_per_constraint_message"],
            },
            "category_masked_maximal_dictionary": {
                "constraint_catalogue_recovery": new["category_masked_maximal"]["constraint_recall"],
                "mean_extra_constraint_candidates": new["category_masked_maximal"]["mean_extra_candidates_per_constraint_message"],
            },
        },
        "six_metric_interpretation": {
            "old_constraint_slot_retention": old["constraint_slot_retention"],
            "old_raw_constraint_mining_recovery": old["constraint_mining_recovery"]["raw"],
            "old_bert_constraint_mining_recovery": old["constraint_mining_recovery"]["bert"],
            "old_category_slot_retention": old["category_slot_retention"],
            "old_raw_category_mining_recovery": old["category_mining_recovery"]["raw"],
            "old_bert_category_mining_recovery": old["category_mining_recovery"]["bert"],
        },
        "decision": "The category-masked non-maximal dictionary path is the Node 2 integration candidate. It recovers categories through longest exact visible coarse-category matching and short constraints through exact dictionary matching. Maximal pruning is rejected because it reduces constraint recovery.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
