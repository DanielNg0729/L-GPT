"""Merge validated semantic augmentation sources into a reproducible train/eval corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from robustness.v2.generate_catalogue_synonyms import banned_phrases, norm, valid
from robustness.v2.import_train_only_synonyms import valid_overlap_augmentation

ROOT = Path(__file__).resolve().parents[2]
GROQ = ROOT / "robustness" / "v2" / "catalogue_synonym_training.jsonl"
EXTERNAL = ROOT / "robustness" / "v2" / "catalogue_synonym_external_train_only.jsonl"
BROAD = ROOT / "robustness" / "v2" / "catalogue_synonym_broad_semantic_train_only.jsonl"
OVERLAP = ROOT / "robustness" / "v2" / "catalogue_synonym_broad_overlap_train_only.jsonl"
OUT = ROOT / "robustness" / "v2" / "catalogue_synonym_train_only_merged.jsonl"
DICTIONARY = ROOT / "robustness" / "v2" / "catalogue_attribute_dictionary.jsonl"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split(canonical: str) -> str:
    """Canonical-disjoint internal split. This is not the final semantic holdout."""
    return "evaluation" if int(hashlib.sha256(canonical.encode("utf-8")).hexdigest(), 16) % 10 == 0 else "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groq", type=Path, default=GROQ)
    parser.add_argument("--external", type=Path, default=EXTERNAL)
    parser.add_argument("--broad", type=Path, default=BROAD)
    parser.add_argument("--overlap", type=Path, default=OVERLAP)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--allow-missing-external", action="store_true")
    args = parser.parse_args()
    if not args.external.exists() and not args.allow_missing_external:
        raise RuntimeError(f"External train-only import is not available: {args.external}")

    forbidden = banned_phrases()
    catalogue_surface_forms = {norm(row["canonical"]): row["canonical"] for row in load(DICTIONARY)}
    collected: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    alias_pairs: list[dict] = []
    rejected = 0
    for source, path, allow_overlap in (("groq_verified", args.groq, False), ("external_train_only", args.external, False), ("broad_semantic", args.broad, False), ("overlap_augmentation", args.overlap, True)):
        if not path.exists():
            continue
        for row in load(path):
            canonical = str(row.get("canonical", ""))
            for candidate in row.get("synonyms", []):
                candidate_ok = (isinstance(candidate, str) and
                                (valid_overlap_augmentation(candidate, canonical, forbidden) if allow_overlap else valid(candidate, canonical, forbidden)))
                if not candidate_ok:
                    rejected += 1
                    continue
                # The phrase may itself be another released dictionary entry, for
                # example ``gray`` versus ``grey``.  It is a valid equivalence pair but
                # not a unique retrieval label: a one-to-one encoder would receive
                # contradictory supervision. Preserve it for Node 5 and exclude it from
                # Node 4 contrastive training.
                candidate_canonical = catalogue_surface_forms.get(norm(candidate))
                if candidate_canonical and norm(candidate_canonical) != norm(canonical):
                    alias_pairs.append({"canonical": canonical, "candidate": candidate, "source": source, "label": "equivalent_catalogue_alias"})
                    continue
                collected[canonical][norm(candidate)].add(source)

    # A surface form that names multiple canonicals has no unambiguous contrastive label.
    owners: dict[str, set[str]] = defaultdict(set)
    for canonical, candidates in collected.items():
        for candidate in candidates:
            owners[candidate].add(canonical)
    ambiguous = {candidate for candidate, canonicals in owners.items() if len(canonicals) > 1}
    records = []
    for canonical in sorted(collected):
        synonyms = [candidate for candidate in sorted(collected[canonical]) if candidate not in ambiguous]
        if synonyms:
            records.append({
                "canonical": canonical,
                "synonyms": synonyms,
                "sources": {candidate: sorted(collected[canonical][candidate]) for candidate in synonyms},
                "split": split(canonical),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    alias_output = args.output.with_name("catalogue_alias_equivalence_train_only.jsonl")
    alias_output.write_text("".join(json.dumps(row) + "\n" for row in alias_pairs), encoding="utf-8")
    report = {
        "output": str(args.output),
        "accepted_canonicals": len(records),
        "accepted_pairs": sum(len(row["synonyms"]) for row in records),
        "train_pairs": sum(len(row["synonyms"]) for row in records if row["split"] == "train"),
        "evaluation_pairs": sum(len(row["synonyms"]) for row in records if row["split"] == "evaluation"),
        "ambiguous_surface_forms_excluded": len(ambiguous),
        "catalogue_alias_pairs_reserved_for_verifier": len(alias_pairs),
        "catalogue_alias_output": str(alias_output),
        "invalid_or_benchmark_overlapping_candidates_rejected": rejected,
        "external_source_present": args.external.exists(),
        "broad_semantic_source_present": args.broad.exists(),
        "overlap_augmentation_source_present": args.overlap.exists(),
        "policy": "The internal evaluation split is canonical-disjoint from training. The existing semantic holdout remains unopened and is not included here.",
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
