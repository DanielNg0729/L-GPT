"""Diagnostic only: inspect the consumed V2.15 route-classifier test predictions."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from evaluator.local_evaluator import load_jsonl
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments" / "studies" / "route_template_bank" / "test.jsonl"
MODEL = ROOT / ".v2_model_cache" / "v1_route_template_bank_classifier_cuda"
OUT = ROOT / "experiments" / "results" / "v1_route_template_bank_failure_analysis.json"


def main():
    device = torch.device("cuda:0")
    rows = load_jsonl(DATA)
    labels = sorted({row["action"] for row in rows})
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, local_files_only=True).to(device).eval()
    encoded = tokenizer([row["message"] for row in rows], padding=True, truncation=True, max_length=80, return_tensors="pt")
    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"])
    predictions: list[int] = []
    confidence: list[float] = []
    gated_predictions: list[int] = []
    opening = {"buying_opening", "plain_opening", "override_opening"}
    with torch.no_grad():
        cursor = 0
        for ids, mask in DataLoader(dataset, batch_size=128):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            probability = logits.softmax(-1)
            values, indices = probability.max(-1)
            predictions.extend(indices.cpu().tolist())
            confidence.extend(values.cpu().tolist())
            batch_rows = rows[cursor:cursor + len(ids)]
            cursor += len(ids)
            for row, row_logits in zip(batch_rows, logits):
                permitted = opening if row["action"] in opening else set(labels) - opening
                blocked = row_logits.clone()
                for index, label in enumerate(labels):
                    if label not in permitted:
                        blocked[index] = float("-inf")
                gated_predictions.append(int(blocked.argmax().item()))

    confusion = {label: Counter() for label in labels}
    by_template: dict[str, list[dict]] = defaultdict(list)
    examples = defaultdict(list)
    for row, prediction, score in zip(rows, predictions, confidence):
        predicted = labels[prediction]
        actual = row["action"]
        confusion[actual][predicted] += 1
        by_template[row["template"]].append({"correct": predicted == actual, "predicted": predicted})
        if predicted != actual and len(examples[(actual, predicted)]) < 12:
            examples[(actual, predicted)].append({"message": row["message"], "confidence": round(score, 6)})

    template_summary = {}
    gated_template_summary = {}
    for template, outcomes in sorted(by_template.items()):
        correct = sum(item["correct"] for item in outcomes)
        predicted_counts = Counter(item["predicted"] for item in outcomes)
        template_summary[template] = {
            "rows": len(outcomes), "accuracy": round(correct / len(outcomes), 6),
            "predictions": dict(predicted_counts),
        }
    gated_by_template = defaultdict(list)
    for row, prediction in zip(rows, gated_predictions):
        gated_by_template[row["template"]].append(labels[prediction] == row["action"])
    for template, outcomes in sorted(gated_by_template.items()):
        gated_template_summary[template] = {
            "accuracy": round(sum(outcomes) / len(outcomes), 6),
        }
    report = {
        "purpose": "diagnostic only; no model selection or retraining",
        "labels": labels,
        "confusion": {actual: dict(confusion[actual]) for actual in labels},
        "template_summary": template_summary,
        "turn_gated_template_summary": gated_template_summary,
        "failure_examples": {f"{actual} -> {predicted}": value for (actual, predicted), value in examples.items()},
        "turn_gated_diagnostic": {
            "definition": "For each already-labelled test row, mask actions impossible at its known runtime phase before argmax. This is post-hoc diagnosis only, not a selection result.",
            "accuracy": round(sum(labels[prediction] == row["action"] for row, prediction in zip(rows, gated_predictions)) / len(rows), 6),
            "per_action_accuracy": {
                label: round(
                    sum(labels[prediction] == row["action"] for row, prediction in zip(rows, gated_predictions) if row["action"] == label) /
                    sum(row["action"] == label for row in rows), 6
                ) for label in labels
            },
        },
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
