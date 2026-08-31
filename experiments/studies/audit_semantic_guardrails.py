"""Exercise the V2 guardrail contract on canonical traffic and representative inputs."""
from __future__ import annotations

import json
import argparse
from collections import Counter
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from experiments.studies.semantic_guardrails import AttributePrediction, FamilyPrediction, GuardrailConfig, evaluate as guard, load_dictionary
from experiments.studies.semantic_evidence_tier import SemanticEvidence, SemanticIntegrationPolicy, apply_policy
from submission.agent import Agent, CONSTRAINT

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "studies" / "results" / "semantic_guardrail_audit.json"
# This configuration is deliberately audit-only.  It verifies lexical provenance and
# disabled Node 7 integration while Nodes 4 and 5 have not supplied real predictions.
SCAFFOLD_CONFIG = GuardrailConfig(min_verifier_score=0.0, max_retrieval_rank=0)


class GuardrailAuditAgent(Agent):
    def __init__(self, catalog: Path) -> None:
        super().__init__(catalog)
        self.dictionary = load_dictionary()
        self.counts: Counter[str] = Counter()
        self.nonzero: list[dict] = []

    def _observe(self, st, msg: str) -> None:
        super()._observe(st, msg)
        for text, tier in self._extract_templated(msg):
            if tier != CONSTRAINT:
                continue
            decision = guard(
                self, text,
                # Deliberately a fixed control prediction.  This audit only verifies
                # provenance and zero-weight non-interference; it does not claim that
                # a resolver predicted leather for the observed text.
                AttributePrediction("leather", "material", 0.90, 0.10),
                FamilyPrediction("material", 0.90), self.dictionary, SCAFFOLD_CONFIG,
            )
            self.counts["seen"] += 1
            self.counts["routed"] += int(decision.routed)
            self.counts["allowed"] += int(decision.allowed)
            self.counts[decision.reason] += 1
            if decision.allowed:
                self.nonzero.append({
                    "text": text,
                    "weight": decision.semantic_confidence,
                    "provenance_confidence": decision.provenance.confidence,
                    "full_attested": decision.provenance.full_attested,
                    "source_continuation": decision.provenance.source_continuation,
                })


def run_suite(path: Path, ids, categories, products, catalog: Path) -> dict:
    agent = GuardrailAuditAgent(catalog)
    evaluate(agent, load_jsonl(path), ids, categories, products)
    return {"counts": dict(agent.counts), "nonzero_cases": agent.nonzero}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-only", action="store_true")
    args = parser.parse_args()
    catalog = ROOT / "data" / "catalog.jsonl"
    ids, categories, products = catalog_index(catalog)
    dictionary = load_dictionary()
    agent = Agent(catalog)
    semantic_example = guard(
        agent, "made from animal hide",
        AttributePrediction("leather", "material", 0.90, 0.10),
        FamilyPrediction("material", 0.90), dictionary, SCAFFOLD_CONFIG,
    )
    literal_example = guard(
        agent, "leather",
        AttributePrediction("leather", "material", 0.90, 0.10),
        FamilyPrediction("material", 0.90), dictionary, SCAFFOLD_CONFIG,
    )
    strict_config = GuardrailConfig(min_verifier_score=0.80, max_retrieval_rank=5)
    strict_example = guard(
        agent, "made from animal hide",
        AttributePrediction("leather", "material", 0.90, 0.10, verifier_score=0.90, retrieval_rank=3),
        FamilyPrediction("material", 0.90), dictionary, strict_config,
    )
    scores = {"p1": 1.0, "p2": 0.5}
    identity_scores = apply_policy(
        scores,
        SemanticEvidence("leather", strict_example.semantic_confidence, frozenset({"p2"})),
    )
    result = {
        "contract_examples": {
            "semantic_attribute": {"routed": semantic_example.routed, "allowed": semantic_example.allowed, "weight": semantic_example.semantic_confidence, "reason": semantic_example.reason, "trace": semantic_example.trace},
            "literal_attribute": {"routed": literal_example.routed, "allowed": literal_example.allowed, "weight": literal_example.semantic_confidence, "reason": literal_example.reason, "trace": literal_example.trace},
            "strict_resolver_contract": {"routed": strict_example.routed, "accepted": strict_example.allowed, "confidence": strict_example.semantic_confidence, "reason": strict_example.reason, "trace": strict_example.trace},
            "node7_default_identity": {"input_scores": scores, "output_scores": identity_scores, "is_identity": scores == identity_scores, "policy": {"maximum_weight": 0.0, "minimum_confidence": 1.0}},
        },
        "legacy_semantic_shift": {
            "development200": run_suite(ROOT / "experiments" / "studies" / "sets" / "semantic_attribute_development_200.jsonl", ids, categories, products, catalog),
            "holdout800": run_suite(ROOT / "experiments" / "studies" / "sets" / "semantic_attribute_holdout_800.jsonl", ids, categories, products, catalog),
        },
    }
    if not args.legacy_only:
        result["canonical_traffic"] = {
            "official200": run_suite(ROOT / "experiments" / "studies" / "public_value_only" / "official200_canonical_replay.jsonl", ids, categories, products, catalog),
            "unseen800": run_suite(ROOT / "robustness" / "optuna_v2_sets" / "population_shift_01_800.jsonl", ids, categories, products, catalog),
        }
    output = OUT if not args.legacy_only else OUT.with_name("semantic_guardrail_legacy_audit.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
