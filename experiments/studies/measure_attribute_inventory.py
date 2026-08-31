"""Measure the catalogue-derived canonical attribute inventory before V2 training.

This deliberately counts source units rather than arbitrary n-grams.  A free-text
product sentence is not an attribute class; short, repeated visible feature/detail values
are the conservative candidates for the semantic attribute dictionary.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data" / "catalog.jsonl"
OUT = ROOT / "experiments" / "results" / "catalogue_attribute_inventory.json"

TOKEN = re.compile(r"[a-z0-9]+")
DETAIL_KEY_EXCLUSIONS = re.compile(
    r"date|dimension|weight|model|asin|manufacturer|package|item number|product number",
    re.I,
)


def normalise(value: object) -> str:
    return " ".join(TOKEN.findall(str(value).lower()))


def usable(value: object) -> str | None:
    text = normalise(value)
    tokens = text.split()
    if not 1 <= len(tokens) <= 8:
        return None
    if len(text) < 3 or not any(char.isalpha() for char in text):
        return None
    return text


def add(counter: set[str], value: object) -> None:
    if atom := usable(value):
        counter.add(atom)


def main() -> None:
    feature, detail, category = Counter(), Counter(), Counter()
    products = 0
    with CATALOG.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            products += 1
            feature_atoms, detail_atoms, category_atoms = set(), set(), set()
            for value in product.get("features") or []:
                add(feature_atoms, value)
            for key, value in (product.get("details") or {}).items():
                if not DETAIL_KEY_EXCLUSIONS.search(str(key)):
                    add(detail_atoms, value)
            for value in (product.get("categories") or [])[2:]:
                add(category_atoms, value)
            feature.update(feature_atoms)
            detail.update(detail_atoms)
            category.update(category_atoms)

            # Training eligibility is product-level, even if a phrase is repeated in
            # multiple visible fields of one item.
            if products == 1:
                attribute, merged = Counter(), Counter()
            attribute.update(feature_atoms | detail_atoms)
            merged.update(feature_atoms | detail_atoms | category_atoms)

    # `attribute` and `merged` were intentionally constructed with per-product unions.

    def summary(counter: Counter[str]) -> dict[str, int]:
        return {
            "source_units": sum(counter.values()),
            "unique_values": len(counter),
            "seen_once": sum(count == 1 for count in counter.values()),
            "repeated_at_least_2": sum(count >= 2 for count in counter.values()),
            "repeated_at_least_5": sum(count >= 5 for count in counter.values()),
            "repeated_at_least_20": sum(count >= 20 for count in counter.values()),
        }

    result = {
        "method": {
            "included": "normalised feature lines, non-identifier detail values, and leaf categories; 1 to 8 tokens",
            "excluded": "titles, descriptions, dates, identifiers, dimensions, weights, arbitrary n-grams, and units longer than 8 tokens",
            "trainable_attribute_inventory_definition": "feature/detail source units occurring in at least two distinct products",
            "category_inventory_definition": "leaf category routes occurring in at least two distinct products, measured separately",
        },
        "products": products,
        "by_source": {"feature": summary(feature), "detail": summary(detail), "category": summary(category)},
        "attribute_only": summary(attribute),
        "merged": summary(merged),
        "trainable_attribute_inventory_count": sum(count >= 2 for count in attribute.values()),
        "trainable_inventory_including_categories_count": sum(count >= 2 for count in merged.values()),
        "sample_high_frequency": merged.most_common(20),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
