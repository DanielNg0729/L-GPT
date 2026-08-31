"""Audit generated synonym pairs before they are eligible for encoder training."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "datasets" / "catalogue_synonym_training.jsonl"
OUT = ROOT / "experiments" / "studies" / "results" / "synonym_corpus_audit.json"


def key(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs = [(str(row["canonical"]), str(value)) for row in rows for value in row.get("synonyms", [])]
    candidate_to_canonical: dict[str, set[str]] = defaultdict(set)
    for canonical, candidate in pairs:
        candidate_to_canonical[key(candidate)].add(canonical)
    ambiguous = {candidate: sorted(values) for candidate, values in candidate_to_canonical.items() if len(values) > 1}
    result = {
        "processed_attributes": len(rows),
        "accepted_pairs": len(pairs),
        "unique_surface_forms": len(candidate_to_canonical),
        "ambiguous_surface_forms": len(ambiguous),
        "ambiguous_training_pairs": sum(len(values) for values in ambiguous.values()),
        "ambiguous_examples": dict(list(ambiguous.items())[:20]),
        "training_policy": "Exclude ambiguous surface forms from contrastive encoder training; retain them for verifier and ambiguity analysis.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
