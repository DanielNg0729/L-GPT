"""Measure a frozen pretrained encoder on catalogue attribute retrieval only.

This is the V2.01 baseline. It does not use generated training pairs, modify the submitted
agent, or inspect the target-disjoint holdout. Queries are the declared semantic values in
the development card; candidates are only the full visible canonical attribute dictionary.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = ROOT / ".v2_model_cache"
DICTIONARY = ROOT / "experiments" / "datasets" / "catalogue_attribute_dictionary.jsonl"
DEV = ROOT / "experiments" / "studies" / "sets" / "semantic_attribute_development_200.jsonl"
OUT = ROOT / "experiments" / "studies" / "results" / "pretrained_attribute_baseline_dev.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def load_model(model_name: str = MODEL, device: str | None = None):
    # A baseline must be reproducible offline and must not silently download a different
    # checkpoint during evaluation.
    os.environ.setdefault("HF_HOME", str(CACHE / "huggingface"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from sentence_transformers import SentenceTransformer
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[attribute-encoder] device={selected_device}", flush=True)
    return SentenceTransformer(model_name, cache_folder=str(CACHE), device=selected_device, local_files_only=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="V2.01 frozen pretrained attribute retrieval baseline")
    parser.add_argument("--dataset", type=Path, default=DEV)
    parser.add_argument("--dictionary", type=Path, default=DICTIONARY)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model", default=MODEL, help="Frozen base model or local fine-tuned model directory")
    args = parser.parse_args()

    canonicals = [str(row["canonical"]) for row in read_jsonl(args.dictionary)]
    canonical_index = {normalise(value): index for index, value in enumerate(canonicals)}
    examples = []
    for row in read_jsonl(args.dataset):
        for group in row["semantic_card"].values():
            for atom in group:
                examples.append({
                    "sample_id": row["sample_id"],
                    "query": str(atom["paraphrase"]),
                    "canonical": str(atom["canonical"]),
                    "family": str(atom["attribute"]),
                })
    model = load_model(args.model)
    matrix = np.asarray(model.encode(canonicals, batch_size=args.batch_size, normalize_embeddings=True,
                                     show_progress_bar=False), dtype=np.float32)
    queries = np.asarray(model.encode([row["query"] for row in examples], batch_size=args.batch_size,
                                      normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    scores = queries @ matrix.T
    ranks: list[int] = []
    rows: list[dict] = []
    for example, row_scores in zip(examples, scores):
        order = np.argsort(-row_scores)
        expected_index = canonical_index.get(normalise(example["canonical"]))
        if expected_index is None:
            raise RuntimeError(f"benchmark canonical absent from dictionary: {example['canonical']!r}")
        rank = int(np.where(order == expected_index)[0][0]) + 1
        ranks.append(rank)
        rows.append({**example, "rank": rank, "top5": [canonicals[int(index)] for index in order[:5]]})
    count = len(ranks)
    result = {
        "experiment": "V2.01 frozen pretrained bi-encoder baseline",
        "model": args.model,
        "candidate_dictionary_size": len(canonicals),
        "dataset": str(args.dataset),
        "examples": count,
        "recall_at_1": round(sum(rank <= 1 for rank in ranks) / count, 6),
        "recall_at_3": round(sum(rank <= 3 for rank in ranks) / count, 6),
        "recall_at_5": round(sum(rank <= 5 for rank in ranks) / count, 6),
        "recall_at_10": round(sum(rank <= 10 for rank in ranks) / count, 6),
        "mrr": round(sum(1.0 / rank for rank in ranks) / count, 6),
        "examples_detail": rows,
        "decision_rule": "Do not integrate or open holdout unless retrieval substantially exceeds the literal baseline and passes independent equivalence verification.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "examples_detail"}, indent=2))


if __name__ == "__main__":
    main()
