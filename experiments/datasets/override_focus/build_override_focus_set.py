"""Build an 800-session, all-intent-override diagnostic set.

Targets and synthetic profiles are copied from the fixed held-out Unseen800 population
fold. Only the scenario label and sample identifier change, so the set measures dialogue
policy under a constant target population rather than a new private-set assumption.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=ROOT / "robustness" / "optuna_v2_sets" / "population_shift_01_800.jsonl")
    parser.add_argument("--public", type=Path, default=ROOT / "data" / "public_set.jsonl")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "robustness" / "override_focus" / "override_focus_800.jsonl")
    args = parser.parse_args()

    source = read_jsonl(args.source)
    public_targets = {str(row["ground_truth"]["parent_asin"]) for row in read_jsonl(args.public)}
    targets = [str(row["ground_truth"]["parent_asin"]) for row in source]
    if len(source) != 800 or len(set(targets)) != 800:
        raise ValueError("source must contain 800 distinct targets")
    if set(targets) & public_targets:
        raise ValueError("source overlaps released public targets")

    rows = []
    for index, row in enumerate(source, start=1):
        rows.append({
            **row,
            "sample_id": f"override_focus_800_{index:04d}",
            "scenario_type": "intent_override",
            "difficulty_bucket": "all_intent_override_from_unseen800",
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "truth_status": "internal dialogue-policy stress set; not organizer private data",
        "source": str(args.source.relative_to(ROOT)),
        "source_sha256": digest(args.source),
        "rows": len(rows),
        "scenario_type": "intent_override",
        "distinct_targets": len(set(targets)),
        "public_target_overlap": 0,
        "output": args.out.name,
        "output_sha256": digest(args.out),
    }
    (args.out.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
