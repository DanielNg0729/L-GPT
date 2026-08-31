"""Create a high-precision, cluster-level request for new non-catalogue paraphrases."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DICTIONARY = ROOT / "experiments" / "datasets" / "catalogue_attribute_dictionary.jsonl"
CLUSTERS = ROOT / "experiments" / "datasets" / "catalogue_equivalence_clusters_measurement_checked.jsonl"
OUT = ROOT / "experiments" / "datasets" / "cluster_level_paraphrase_generation_input.jsonl"


def main() -> None:
    frequency = {row["canonical"]: int(row["document_frequency"])
                 for row in (json.loads(line) for line in DICTIONARY.read_text(encoding="utf-8").splitlines() if line.strip())}
    rows = []
    for cluster_id, row in enumerate(json.loads(line) for line in CLUSTERS.read_text(encoding="utf-8").splitlines() if line.strip()):
        members = sorted(row["members"])
        representative = min(members, key=lambda value: (-frequency.get(value, 0), len(value.split()), value))
        rows.append({"cluster_id": cluster_id, "representative": representative, "catalogue_equivalents": members})
    OUT.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "clusters": len(rows), "policy": "Representative maximises catalogue frequency, then prefers shorter wording. All members remain equivalent targets."}, indent=2))


if __name__ == "__main__":
    main()
