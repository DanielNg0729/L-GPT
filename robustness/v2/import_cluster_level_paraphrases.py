"""Validate new paraphrases generated for known catalogue-equivalence clusters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from robustness.v2.generate_catalogue_synonyms import banned_phrases, norm, valid

ROOT = Path(__file__).resolve().parents[2]
CLUSTERS = ROOT / "robustness" / "v2" / "cluster_level_paraphrase_generation_input.jsonl"
INVENTORY = ROOT / "robustness" / "v2" / "external_train_only_canonicals.jsonl"
OUT = ROOT / "robustness" / "v2" / "cluster_level_paraphrases_train_only.jsonl"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="JSONL: {cluster_id, representative, candidates}")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    clusters = {int(row["cluster_id"]): row for row in rows(CLUSTERS)}
    inventory = {norm(row["canonical"]) for row in rows(INVENTORY)}
    forbidden = banned_phrases()
    accepted: dict[int, set[str]] = {}
    rejected: list[dict] = []
    for line_number, row in enumerate(rows(args.input), 1):
        cluster_id, representative, candidates = row.get("cluster_id"), row.get("representative"), row.get("candidates")
        cluster = clusters.get(cluster_id) if isinstance(cluster_id, int) else None
        if not cluster or representative != cluster["representative"] or not isinstance(candidates, list):
            rejected.append({"line": line_number, "reason": "unknown_cluster_or_schema_mismatch"})
            continue
        for candidate in candidates:
            candidate_norm = norm(str(candidate)) if isinstance(candidate, str) else ""
            if candidate_norm in inventory:
                rejected.append({"line": line_number, "cluster_id": cluster_id, "candidate": candidate, "reason": "already_catalogue_attested"})
            elif not isinstance(candidate, str) or not valid(candidate, representative, forbidden):
                rejected.append({"line": line_number, "cluster_id": cluster_id, "candidate": candidate, "reason": "invalid_or_benchmark_overlap"})
            else:
                accepted.setdefault(cluster_id, set()).add(candidate_norm)
    output = []
    for cluster_id, candidates in sorted(accepted.items()):
        cluster = clusters[cluster_id]
        output.append({**cluster, "paraphrases": sorted(candidates), "source": "external_cluster_generation"})
    args.output.write_text("".join(json.dumps(row) + "\n" for row in output), encoding="utf-8")
    report = {"input": str(args.input), "accepted_clusters": len(output), "accepted_paraphrases": sum(len(row["paraphrases"]) for row in output), "rejected": len(rejected), "rejected_examples": rejected[:50], "policy": "Candidates must be absent from the full catalogue inventory and the frozen semantic benchmark wording."}
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
