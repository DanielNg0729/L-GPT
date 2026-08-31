"""Export the only catalogue material permitted to an external augmentation generator."""
from __future__ import annotations

import json
import re
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "datasets" / "catalogue_attribute_dictionary.jsonl"
OUTPUT = ROOT / "experiments" / "datasets" / "external_train_only_canonicals.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-feature-subset", action="store_true", help="Feature-source, df>=5, <=5 non-numeric tokens")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.semantic_feature_subset:
        rows = [row for row in rows if "feature" in row["sources"] and row["document_frequency"] >= 5
                and len(row["canonical"].split()) <= 5 and not re.search(r"\d", row["canonical"])]
    canonicals = sorted({row["canonical"] for row in rows})
    args.output.write_text("".join(json.dumps({"canonical": canonical}) + "\n" for canonical in canonicals), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "canonical_count": len(canonicals), "semantic_feature_subset": args.semantic_feature_subset}))


if __name__ == "__main__":
    main()
