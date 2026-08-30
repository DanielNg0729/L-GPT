"""Inspect phase-aware classifier failures on its eval split only."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from evaluator.local_evaluator import load_jsonl
from transformers import AutoTokenizer

from robustness.v2.train_phase_aware_shared_route_classifier import (
    BASE, EVAL_SPLIT, LABELS, MODEL_OUT, PhaseAwareRouteModel, phase_for, valid_mask,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "robustness" / "v2" / "results" / "phase_aware_shared_sixway_eval_failure_analysis.json"


def main():
    rows = load_jsonl(EVAL_SPLIT)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    model = PhaseAwareRouteModel().to(device).eval()
    model.load_state_dict(torch.load(MODEL_OUT, map_location=device, weights_only=True)["state_dict"])
    tokens = tokenizer([row["message"] for row in rows], padding=True, truncation=True, max_length=80, return_tensors="pt")
    phase = torch.tensor([phase_for(row) for row in rows])
    data = TensorDataset(tokens["input_ids"], tokens["attention_mask"], phase)
    predicted, confidences = [], []
    with torch.no_grad():
        for ids, mask, phases in DataLoader(data, batch_size=128):
            phases = phases.to(device)
            logits = model(ids.to(device), mask.to(device), phases)
            masked = logits.masked_fill(~valid_mask(phases, device), float("-inf"))
            probability = masked.softmax(-1)
            score, index = probability.max(-1)
            predicted.extend(LABELS[value] for value in index.cpu().tolist())
            confidences.extend(score.cpu().tolist())
    confusion = {label: Counter() for label in LABELS}
    templates = defaultdict(lambda: {"correct": 0, "predictions": Counter(), "examples": []})
    for row, route, confidence in zip(rows, predicted, confidences):
        actual = row["action"]
        confusion[actual][route] += 1
        item = templates[row["template"]]
        item["correct"] += int(actual == route)
        item["predictions"][route] += 1
        if actual != route and len(item["examples"]) < 4:
            item["examples"].append({"message": row["message"], "predicted": route, "confidence": round(confidence, 6)})
    report = {
        "split": "eval only",
        "confusion": {label: dict(confusion[label]) for label in LABELS},
        "template_summary": {
            template: {"accuracy": round(value["correct"] / 200, 6), "predictions": dict(value["predictions"]), "examples": value["examples"]}
            for template, value in sorted(templates.items())
        },
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
