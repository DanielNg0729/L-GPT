"""Create the full product-level canonical attribute dictionary for V2 training."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from experiments.studies.measure_attribute_inventory import CATALOG, DETAIL_KEY_EXCLUSIONS, usable

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "datasets" / "catalogue_attribute_dictionary.jsonl"


def main() -> None:
    document_frequency: Counter[str] = Counter()
    sources: dict[str, set[str]] = defaultdict(set)
    with CATALOG.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            atoms: dict[str, set[str]] = defaultdict(set)
            for value in product.get("features") or []:
                if phrase := usable(value):
                    atoms[phrase].add("feature")
            for key, value in (product.get("details") or {}).items():
                if not DETAIL_KEY_EXCLUSIONS.search(str(key)) and (phrase := usable(value)):
                    atoms[phrase].add("detail")
            for phrase, atom_sources in atoms.items():
                document_frequency[phrase] += 1
                sources[phrase].update(atom_sources)

    rows = [
        {"canonical": phrase, "document_frequency": df, "sources": sorted(sources[phrase])}
        for phrase, df in document_frequency.items() if df >= 2
    ]
    rows.sort(key=lambda row: (-row["document_frequency"], row["canonical"]))
    OUT.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "attributes": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
