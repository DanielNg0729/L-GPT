"""Build deterministic, catalogue-grounded negative pairs for V2 verification.

The output contains no generated paraphrases and no target labels.  It is a reusable
bank of plausible but non-equivalent canonical attributes for measuring a later
equivalence verifier.  Positives must come from a separate, audited synonym corpus.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DICTIONARY = ROOT / "robustness" / "v2" / "catalogue_attribute_dictionary.jsonl"
DEFAULT_OUT = ROOT / "robustness" / "v2" / "sets" / "canonical_verification_negatives.jsonl"


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def family(text: str) -> str:
    """Coarse, auditable buckets used only for hard-negative selection."""
    words = set(tokens(text))
    if words & {"black", "white", "blue", "red", "green", "pink", "brown", "beige", "gray", "grey", "purple"}:
        return "color"
    if words & {"cotton", "leather", "polyester", "nylon", "rubber", "suede", "canvas", "wool", "fabric", "faux"}:
        return "material"
    if words & {"closure", "zipper", "buckle", "button", "lace", "fastener", "pull"}:
        return "closure"
    if words & {"water", "wash", "breathable", "lightweight", "resistant", "proof", "elastic", "adjustable"}:
        return "performance"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical negative verification bank")
    parser.add_argument("--dictionary", type=Path, default=DICTIONARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--per-canonical", type=int, default=3)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.dictionary.read_text(encoding="utf-8").splitlines() if line.strip()]
    values = [str(row["canonical"]) for row in rows]
    by_family: dict[str, list[str]] = defaultdict(list)
    by_head: dict[str, list[str]] = defaultdict(list)
    for value in values:
        by_family[family(value)].append(value)
        words = tokens(value)
        if words:
            by_head[words[-1]].append(value)

    rng = random.Random(args.seed)
    output: list[dict] = []
    for canonical in values:
        candidates: list[tuple[str, str]] = []
        head_matches = [value for value in by_head[tokens(canonical)[-1]] if value != canonical]
        if head_matches:
            candidates.append((rng.choice(head_matches), "shared_head"))
        family_matches = [value for value in by_family[family(canonical)] if value != canonical and value not in {x[0] for x in candidates}]
        if family_matches:
            candidates.append((rng.choice(family_matches), "same_family"))
        broad = [value for value in values if value != canonical and value not in {x[0] for x in candidates}]
        if broad:
            candidates.append((rng.choice(broad), "unrelated"))
        for negative, negative_type in candidates[:args.per_canonical]:
            output.append({
                "canonical": canonical,
                "canonical_family": family(canonical),
                "candidate": negative,
                "candidate_family": family(negative),
                "label": "not_equivalent",
                "negative_type": negative_type,
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({"rows": len(output), "canonicals": len(values), "seed": args.seed, "path": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
