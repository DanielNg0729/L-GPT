"""V2.04: frozen pretrained family routing before any supervised fine-tuning."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.studies.build_semantic_attribute_sets import classify
from experiments.studies.pretrained_attribute_baseline import (
    DEV, DICTIONARY, MODEL, load_model, read_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "studies" / "results" / "pretrained_family_baseline_dev.json"


def main() -> None:
    canonicals = [str(row["canonical"]) for row in read_jsonl(DICTIONARY)]
    families = [classify(value) for value in canonicals]
    family_names = sorted(set(families))
    indices = {name: np.asarray([index for index, family in enumerate(families) if family == name]) for name in family_names}
    examples = [
        {"query": str(atom["paraphrase"]), "expected": str(atom["attribute"])}
        for row in read_jsonl(DEV)
        for group in row["semantic_card"].values()
        for atom in group
    ]
    model = load_model()
    matrix = np.asarray(model.encode(canonicals, batch_size=128, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    queries = np.asarray(model.encode([row["query"] for row in examples], batch_size=128, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    top1 = top2 = 0
    detail = []
    for example, scores in zip(examples, queries @ matrix.T):
        aggregate = {name: float(scores[index].max()) for name, index in indices.items()}
        ranked = sorted(aggregate, key=aggregate.get, reverse=True)
        top1 += ranked[0] == example["expected"]
        top2 += example["expected"] in ranked[:2]
        detail.append({**example, "ranked_families": ranked, "scores": aggregate})
    count = len(examples)
    result = {
        "experiment": "V2.04 frozen pretrained family route baseline",
        "model": MODEL,
        "dictionary_size": len(canonicals),
        "examples": count,
        "family_accuracy": round(top1 / count, 6),
        "family_top2_recall": round(top2 / count, 6),
        "details": detail,
        "decision_rule": "A fine-tuned family router must beat this matched baseline and must not block global retrieval on a wrong route.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
