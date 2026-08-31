"""Materialise a value-only semantic perturbation of Official200.

The evaluator normally derives an intent card from each public target at runtime.  This builder
persists the same deterministic cards, then substitutes only supported attribute values with a
fixed paraphrase.  It preserves target, profile, scenario, category, evaluator wrapper, turn
seed, ranking constants, and population prior.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from evaluator.local_evaluator import behavior_for, catalog_index, intent_card, load_jsonl
from experiments.studies.build_semantic_attribute_sets import RULES


ROOT = Path(__file__).resolve().parents[2]


def rewrite(value: str, family: str) -> tuple[str, str | None]:
    """Replace one known attribute atom, preserving all unrecognised card values verbatim."""
    import re
    if family == "canonical":
        return value, None
    for name, pattern, development, holdout in RULES:
        if re.search(pattern, value, flags=re.I):
            return (development if family == "development" else holdout), name
    return value, None


def materialise(samples: list[dict], products: dict[str, dict], family: str) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    counts: dict[str, int] = {"rewritten_atoms": 0, "sessions_with_rewrite": 0}
    rules: dict[str, int] = {}
    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        card = intent_card(products[target])
        rewritten = False
        transformed: dict[str, list[str]] = {}
        for group in ("hard_constraints", "soft_preferences"):
            values: list[str] = []
            for raw in card[group]:
                value, rule = rewrite(str(raw), family)
                values.append(value)
                if rule:
                    rewritten = True
                    counts["rewritten_atoms"] += 1
                    rules[rule] = rules.get(rule, 0) + 1
            transformed[group] = values
        seed = f"{sample.get('sample_id', '')}\0{sample.get('scenario_type', '')}"
        rows.append({
            **sample,
            "intent_card": transformed,
            "behavior": behavior_for(str(sample["scenario_type"]), transformed, random.Random(seed)),
            "semantic_value_family": family,
        })
        if rewritten:
            counts["sessions_with_rewrite"] += 1
    counts["rules"] = dict(sorted(rules.items()))
    return rows, counts


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    out = ROOT / "experiments" / "studies" / "public_value_only"
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    _, _, products = catalog_index(ROOT / "data" / "catalog.jsonl")
    canonical, canonical_counts = materialise(samples, products, "canonical")
    development, development_counts = materialise(samples, products, "development")
    # Canonical materialisation keeps all original constraint strings unchanged.
    for row in canonical:
        row.pop("semantic_value_family", None)
    canonical_path = out / "official200_canonical_replay.jsonl"
    development_path = out / "official200_attribute_paraphrase_dev.jsonl"
    manifest = {
        "schema_version": 1,
        "truth_status": "public-target value-only semantic perturbation; not private evaluation data",
        "invariants": [
            "same 200 released targets", "same user profiles", "same scenario labels",
            "same evaluator templates and turn seed", "same V1 prior and ranking",
        ],
        "canonical_replay": {"path": canonical_path.name, "sha256": write_jsonl(canonical_path, canonical)},
        "attribute_paraphrase_development": {
            "path": development_path.name,
            "sha256": write_jsonl(development_path, development),
            **development_counts,
        },
        "canonical_materialisation": canonical_counts,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
