"""Apply deterministic measurement consistency checks to proposed catalogue aliases."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "robustness" / "v2" / "catalogue_equivalence_pairs_proposed.jsonl"
DEFAULT_OUTPUT = ROOT / "robustness" / "v2" / "catalogue_equivalence_pairs_measurement_checked.jsonl"
MEASURE = re.compile(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*(meters?|\bm\b|feet|foot|ft|inches|inch|in)(?![a-z])", re.I)
TO_METRES = {"m": 1.0, "meter": 1.0, "meters": 1.0, "ft": 0.3048, "foot": 0.3048, "feet": 0.3048, "in": 0.0254, "inch": 0.0254, "inches": 0.0254}


def measurements(text: str) -> list[float]:
    return [float(value) * TO_METRES[unit.lower()] for value, unit in MEASURE.findall(text)]


def numerically_compatible(left: str, right: str) -> bool:
    first, second = measurements(left), measurements(right)
    # No comparable physical measure means this deterministic guard has no opinion.
    if not first or not second:
        return True
    # Any representation must have a close equivalent on the other side. The tolerance
    # covers rounded catalogue conversions such as 30 m versus 99 ft, but rejects 33 ft.
    return all(any(abs(a - b) <= max(0.30, 0.03 * max(a, b)) for b in second) for a in first) and all(any(abs(a - b) <= max(0.30, 0.03 * max(a, b)) for a in first) for b in second)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    accepted, rejected = [], []
    for row in rows:
        if numerically_compatible(row["left"], row["right"]):
            accepted.append({**row, "measurement_check": "passed_or_not_applicable"})
        else:
            rejected.append({**row, "measurement_check": "incompatible_measurement"})
    args.output.write_text("".join(json.dumps(row) + "\n" for row in accepted), encoding="utf-8")
    parent: dict[str, str] = {}
    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]
    for row in accepted:
        left, right = find(row["left"]), find(row["right"])
        if left != right:
            parent[right] = left
    clusters: dict[str, list[str]] = {}
    for value in parent:
        clusters.setdefault(find(value), []).append(value)
    cluster_output = args.output.with_name("catalogue_equivalence_clusters_measurement_checked.jsonl")
    cluster_output.write_text("".join(json.dumps({"members": sorted(values), "size": len(values), "status": "measurement_checked_proposed"}) + "\n" for values in sorted(clusters.values(), key=lambda values: (values[0], len(values)))), encoding="utf-8")
    report = {"input_pairs": len(rows), "accepted_pairs": len(accepted), "clusters": len(clusters), "cluster_output": str(cluster_output), "rejected_incompatible_measurements": len(rejected), "rejected": rejected, "status": "measurement_consistent_proposals_still_require_semantic_verification"}
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
