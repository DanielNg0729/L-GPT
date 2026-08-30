"""Import proposed equivalence links between exact frozen-catalogue phrases.

Unlike external paraphrase training, both endpoints must already be visible dictionary
phrases.  The import is catalogue-grounded but not automatically semantically verified;
the output is an auditable Node 5 candidate bank, never direct V1 evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from robustness.v2.generate_catalogue_synonyms import norm

ROOT = Path(__file__).resolve().parents[2]
DICTIONARY = ROOT / "robustness" / "v2" / "catalogue_attribute_dictionary.jsonl"
OUT = ROOT / "robustness" / "v2" / "catalogue_equivalence_pairs_proposed.jsonl"


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="JSONL: {canonical: str, equivalents: [exact catalogue phrase, ...]}")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    dictionary = {norm(json.loads(line)["canonical"]): json.loads(line)["canonical"]
                  for line in DICTIONARY.read_text(encoding="utf-8").splitlines() if line.strip()}
    pairs: set[tuple[str, str]] = set()
    rejected: list[dict] = []
    for line_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        left = row.get("canonical")
        values = row.get("equivalents")
        if not isinstance(left, str) or not isinstance(values, list) or norm(left) not in dictionary:
            rejected.append({"line": line_number, "reason": "invalid_canonical_or_equivalents"})
            continue
        left = dictionary[norm(left)]
        for right in values:
            if not isinstance(right, str) or norm(right) not in dictionary:
                rejected.append({"line": line_number, "canonical": left, "candidate": right, "reason": "candidate_not_in_catalogue"})
                continue
            right = dictionary[norm(right)]
            if norm(left) == norm(right):
                rejected.append({"line": line_number, "canonical": left, "candidate": right, "reason": "self_link"})
                continue
            pairs.add(tuple(sorted((left, right))))
    ordered = sorted(pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps({"left": left, "right": right, "label": "proposed_equivalent_catalogue_alias"}) + "\n" for left, right in ordered), encoding="utf-8")
    uf = UnionFind()
    for left, right in ordered:
        uf.union(left, right)
    clusters: dict[str, list[str]] = {}
    for value in uf.parent:
        clusters.setdefault(uf.find(value), []).append(value)
    cluster_output = args.output.with_name("catalogue_equivalence_clusters_proposed.jsonl")
    cluster_output.write_text("".join(json.dumps({"members": sorted(values), "size": len(values), "status": "proposed"}) + "\n" for values in sorted(clusters.values(), key=lambda values: (values[0], len(values)))) , encoding="utf-8")
    report = {"input": str(args.input), "pairs": len(ordered), "clusters": len(clusters), "linked_phrases": len(uf.parent), "rejected": len(rejected), "rejected_examples": rejected[:50], "status": "catalogue_grounded_proposals_require_verification_before_runtime_use"}
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
