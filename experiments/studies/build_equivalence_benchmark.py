"""Freeze a verifier-only split from independently accepted synonym pairs.

The selected canonicals are excluded from future attribute-encoder training. Positives are
accepted synonym pairs; negatives come from the deterministic canonical negative bank.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYNONYMS = ROOT / "experiments" / "datasets" / "catalogue_synonym_training.jsonl"
NEGATIVES = ROOT / "experiments" / "datasets" / "sets" / "canonical_verification_negatives.jsonl"
OUT = ROOT / "experiments" / "datasets" / "sets" / "frozen_equivalence_verification.jsonl"


def selected(canonical: str) -> bool:
    return int(hashlib.sha256(canonical.encode("utf-8")).hexdigest(), 16) % 5 == 0


def main() -> None:
    positives = [json.loads(line) for line in SYNONYMS.read_text(encoding="utf-8").splitlines() if line.strip()]
    anchors = {str(row["canonical"]) for row in positives if row.get("synonyms") and selected(str(row["canonical"]))}
    rows = []
    for row in positives:
        canonical = str(row["canonical"])
        if canonical in anchors:
            rows.extend({"canonical": canonical, "candidate": str(synonym), "label": 1, "source": "verified_synonym"} for synonym in row["synonyms"])
    for line in NEGATIVES.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if str(row["canonical"]) in anchors:
            rows.append({"canonical": row["canonical"], "candidate": row["candidate"], "label": 0, "source": row["negative_type"]})
    OUT.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"path": str(OUT), "anchors": len(anchors), "positives": sum(r["label"] for r in rows), "negatives": sum(not r["label"] for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
