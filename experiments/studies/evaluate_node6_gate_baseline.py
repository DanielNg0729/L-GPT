"""Pre-augmentation feasibility audit for Node 6 semantic acceptance.

This does not select a production threshold and it does not modify the submission.  It
asks a narrower question: can frozen retrieval similarity, margin, and coarse family
agreement isolate correct top-one canonical attributes on a deterministic split of the
existing semantic development data?  If not, Nodes 3 to 5 need better resolver evidence
before Node 6 can usefully open.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.studies.build_semantic_attribute_sets import classify
from experiments.studies.pretrained_attribute_baseline import DEV, DICTIONARY, MODEL, load_model, normalise, read_jsonl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "results" / "node6_preaugmentation_gate_baseline.json"


def split(canonical: str) -> str:
    """Keep all variants of one canonical phrase in one split."""
    return "calibration" if int(hashlib.sha256(canonical.encode()).hexdigest(), 16) % 5 else "evaluation"


def summary(rows: list[dict], similarity: float, margin: float, family_required: bool) -> dict:
    accepted = [row for row in rows if row["top1_similarity"] >= similarity and row["margin"] >= margin
                and (not family_required or row["family_top2"])]
    correct = sum(row["top1_correct"] for row in accepted)
    return {
        "min_similarity": similarity,
        "min_margin": margin,
        "family_top2_required": family_required,
        "accepted": len(accepted),
        "coverage": round(len(accepted) / len(rows), 6),
        "correct": correct,
        "precision": round(correct / len(accepted), 6) if accepted else None,
        "false_positives": len(accepted) - correct,
    }


def main() -> None:
    dictionary = read_jsonl(DICTIONARY)
    canonicals = [str(row["canonical"]) for row in dictionary]
    canonical_index = {normalise(value): index for index, value in enumerate(canonicals)}
    families = [classify(value) for value in canonicals]
    family_indices = {family: np.asarray([i for i, value in enumerate(families) if value == family]) for family in sorted(set(families))}
    examples = [
        {"query": str(atom["paraphrase"]), "canonical": str(atom["canonical"]), "expected_family": str(atom["attribute"])}
        for sample in read_jsonl(DEV)
        for group in sample["semantic_card"].values()
        for atom in group
    ]
    model = load_model(MODEL)
    matrix = np.asarray(model.encode(canonicals, batch_size=256, normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)
    queries = np.asarray(model.encode([row["query"] for row in examples], batch_size=256, normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)
    rows: list[dict] = []
    for example, scores in zip(examples, queries @ matrix.T):
        order = np.argsort(-scores)
        top1, top2 = int(order[0]), int(order[1])
        family_scores = {name: float(scores[indices].max()) for name, indices in family_indices.items()}
        ranked_families = sorted(family_scores, key=family_scores.get, reverse=True)
        expected_index = canonical_index[normalise(example["canonical"])]
        rows.append({
            **example,
            "split": split(example["canonical"]),
            "top1": canonicals[top1],
            "top1_correct": top1 == expected_index,
            "top1_similarity": round(float(scores[top1]), 8),
            "margin": round(float(scores[top1] - scores[top2]), 8),
            "predicted_family": ranked_families[0],
            "family_top2": example["expected_family"] in ranked_families[:2],
        })
    calibration = [row for row in rows if row["split"] == "calibration"]
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    grid = [summary(calibration, similarity, margin, family) for similarity in (0.45, 0.55, 0.65, 0.75, 0.85)
            for margin in (0.00, 0.03, 0.08, 0.15) for family in (False, True)]
    # No selection rule is applied.  The evaluation split is retained untouched for the
    # later chosen configuration, after a trained resolver exists.
    result = {
        "experiment": "V2.29 Node 6 pre-augmentation gate feasibility baseline",
        "model": MODEL,
        "device": str(model.device),
        "dictionary_size": len(canonicals),
        "rows": len(rows),
        "calibration_rows": len(calibration),
        "evaluation_rows": len(evaluation),
        "calibration_grid": grid,
        "evaluation_locked": True,
        "interpretation": "This is not a final threshold search. It only determines whether the frozen resolver produces a precision-coverage region worth calibrating before the augmentation corpus is ready.",
        "rows_detail": rows,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows_detail"}, indent=2))


if __name__ == "__main__":
    main()
