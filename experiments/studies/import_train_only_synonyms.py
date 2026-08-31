"""Validate and import externally generated train-only semantic attribute pairs.

The generator is intentionally separated from this script.  It receives only canonical
visible-catalogue phrases.  This importer is the trusted boundary: it checks canonical
membership, rejects known semantic benchmark wording, and writes a reproducible accepted
or rejected record.  It never alters the frozen evaluation sets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.studies.generate_catalogue_synonyms import banned_phrases, norm, shares_ngram, valid

ROOT = Path(__file__).resolve().parents[2]
DICTIONARY = ROOT / "experiments" / "datasets" / "catalogue_attribute_dictionary.jsonl"
DEFAULT_OUTPUT = ROOT / "experiments" / "datasets" / "catalogue_synonym_external_train_only.jsonl"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def valid_overlap_augmentation(candidate: str, canonical: str, forbidden: set[str]) -> bool:
    """Allow lexical overlap for representation training, never benchmark overlap.

    This tier is useful for learning compositional variants such as ``pure cotton``. It
    is not proof that the semantic fallback should open, so the strict unknown-wording
    evaluator remains unchanged.
    """
    text, label = norm(candidate), norm(canonical)
    return bool(text and text != label and len(text.split()) <= 8 and text not in forbidden
                and not any(shares_ngram(text, phrase) for phrase in forbidden))


def main() -> None:
    parser = argparse.ArgumentParser(description="Import externally authored train-only catalogue synonym pairs")
    parser.add_argument("input", type=Path, help="JSONL: {canonical: str, candidates: [str, ...]}")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--allow-lexical-overlap", action="store_true",
                        help="Accept non-identical paraphrases sharing canonical words; benchmark overlap stays forbidden.")
    args = parser.parse_args()

    canonical_by_norm = {norm(row["canonical"]): row["canonical"] for row in rows(DICTIONARY)}
    forbidden = banned_phrases()
    accepted: dict[str, set[str]] = {}
    rejected: list[dict] = []
    for line_number, row in enumerate(rows(args.input), 1):
        canonical = row.get("canonical")
        candidates = row.get("candidates")
        if not isinstance(canonical, str) or norm(canonical) not in canonical_by_norm:
            rejected.append({"line": line_number, "reason": "canonical_not_in_visible_dictionary"})
            continue
        if not isinstance(candidates, list):
            rejected.append({"line": line_number, "canonical": canonical, "reason": "candidates_not_list"})
            continue
        canonical = canonical_by_norm[norm(canonical)]
        for candidate in candidates:
            valid_candidate = (valid_overlap_augmentation(candidate, canonical, forbidden)
                               if args.allow_lexical_overlap and isinstance(candidate, str)
                               else isinstance(candidate, str) and valid(candidate, canonical, forbidden))
            if not valid_candidate:
                reason = "invalid_or_benchmark_overlap" if not args.allow_lexical_overlap else "invalid_or_benchmark_overlap_or_identity"
                rejected.append({"line": line_number, "canonical": canonical, "candidate": candidate, "reason": reason})
                continue
            accepted.setdefault(canonical, set()).add(norm(candidate))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for canonical in sorted(accepted):
            handle.write(json.dumps({"canonical": canonical, "synonyms": sorted(accepted[canonical]), "source": "external_overlap_augmentation" if args.allow_lexical_overlap else "external_train_only"}) + "\n")
    report = {
        "input": str(args.input), "output": str(args.output),
        "accepted_canonicals": len(accepted),
        "accepted_pairs": sum(len(values) for values in accepted.values()),
        "rejected_rows_or_candidates": len(rejected),
        "rejected_examples": rejected[:50],
        "policy": "Only visible-catalogue canonicals are accepted. Existing semantic development and holdout surface forms are forbidden. Lexical overlap is " + ("allowed for representation-only augmentation." if args.allow_lexical_overlap else "not allowed in strict unfamiliar-wording training."),
    }
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
