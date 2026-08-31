"""V2.05: frozen pretrained synonym-equivalence verifier baseline."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np

from experiments.studies.pretrained_attribute_baseline import MODEL, load_model, read_jsonl

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments" / "studies" / "sets" / "frozen_equivalence_verification.jsonl"
OUT = ROOT / "experiments" / "studies" / "results" / "pretrained_equivalence_baseline.json"


def auc(labels: list[int], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    return sum((positive > negative) + 0.5 * (positive == negative) for positive in positives for negative in negatives) / (len(positives) * len(negatives))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen or fine-tuned equivalence encoder")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    rows = read_jsonl(DATA)
    model = load_model(args.model)
    anchors = model.encode([row["canonical"] for row in rows], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    candidates = model.encode([row["candidate"] for row in rows], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    scores = [float(np.dot(anchor, candidate)) for anchor, candidate in zip(anchors, candidates)]
    labels = [int(row["label"]) for row in rows]
    positive = [score for label, score in zip(labels, scores) if label]
    negative = [score for label, score in zip(labels, scores) if not label]
    result = {
        "experiment": "V2.05 frozen pretrained equivalence verifier baseline",
        "model": args.model,
        "rows": len(rows), "positives": len(positive), "negatives": len(negative),
        "auroc": round(auc(labels, scores), 6),
        "mean_positive_similarity": round(float(np.mean(positive)), 6),
        "mean_negative_similarity": round(float(np.mean(negative)), 6),
        "decision_rule": "A trained verifier must beat this held-out canonical split and be evaluated separately from retrieval recall.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
