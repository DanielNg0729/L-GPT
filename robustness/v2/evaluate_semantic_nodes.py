"""Score V2 semantic nodes from versioned prediction files.

This program deliberately evaluates resolver outputs before they are permitted to change
ranking.  A prediction row is keyed by ``sample_id`` and may contain ``spans``,
``family`` and an ordered ``candidates`` list.  This simple JSONL boundary keeps model
experiments interchangeable and makes their decisions reproducible.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gold(row: dict) -> tuple[set[str], set[str]]:
    atoms = [atom for group in row["semantic_card"].values() for atom in group]
    return ({str(atom["canonical"]).lower() for atom in atoms}, {str(atom["attribute"]) for atom in atoms})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate isolated V2 semantic nodes")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dataset = {row["sample_id"]: row for row in read(args.dataset)}
    predictions = {row["sample_id"]: row for row in read(args.predictions)}
    counts: Counter[str] = Counter(samples=len(dataset))
    canonical_hits = family_hits = 0
    for sample_id, row in dataset.items():
        expected_values, expected_families = gold(row)
        prediction = predictions.get(sample_id, {})
        candidates = [str(value).lower() for value in prediction.get("candidates", [])][:args.k]
        predicted_family = str(prediction.get("family", ""))
        canonical_hits += int(bool(expected_values & set(candidates)))
        family_hits += int(predicted_family in expected_families)
        counts["missing_predictions"] += int(sample_id not in predictions)
    total = max(1, len(dataset))
    result = {
        "dataset": str(args.dataset), "predictions": str(args.predictions), "k": args.k,
        "samples": len(dataset), "missing_predictions": counts["missing_predictions"],
        "canonical_recall_at_k": round(canonical_hits / total, 6),
        "family_accuracy": round(family_hits / total, 6),
        "protocol": "Report this result before any end-to-end semantic ranking evaluation.",
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
