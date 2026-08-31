"""Post-hoc comparison of frozen six-way V2.16 model on consumed V2.18 test."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from evaluator.local_evaluator import load_jsonl
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "experiments" / "studies" / "turn_gated_bank" / "final_test.jsonl"
MODEL = ROOT / ".v2_model_cache" / "v1_route_template_bank_classifier_cuda"
OUT = ROOT / "experiments" / "results" / "v2_18_sixway_reference_diagnostic.json"
LABELS = ["buying_opening", "constraint_update", "no_evidence", "override_opening", "override_update", "plain_opening"]
OPENING = {"buying_opening", "plain_opening", "override_opening"}


def metrics(rows, predictions):
    return {
        "overall": round(sum(prediction == row["action"] for row, prediction in zip(rows, predictions)) / len(rows), 6),
        "per_action": {
            label: round(
                sum(prediction == label for row, prediction in zip(rows, predictions) if row["action"] == label) /
                sum(row["action"] == label for row in rows), 6
            ) for label in LABELS
        },
        "opening": round(
            sum(prediction == row["action"] for row, prediction in zip(rows, predictions) if row["action"] in OPENING) /
            sum(row["action"] in OPENING for row in rows), 6
        ),
        "followup": round(
            sum(prediction == row["action"] for row, prediction in zip(rows, predictions) if row["action"] not in OPENING) /
            sum(row["action"] not in OPENING for row in rows), 6
        ),
    }


def main():
    rows = load_jsonl(TEST)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, local_files_only=True).to(device).eval()
    encoded = tokenizer([row["message"] for row in rows], padding=True, truncation=True, max_length=80, return_tensors="pt")
    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"])
    raw, masked = [], []
    with torch.no_grad():
        cursor = 0
        for ids, mask in DataLoader(dataset, batch_size=128):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            raw.extend(LABELS[index] for index in logits.argmax(-1).cpu().tolist())
            for row, vector in zip(rows[cursor:cursor + len(ids)], logits):
                permitted = OPENING if row["action"] in OPENING else set(LABELS) - OPENING
                constrained = vector.clone()
                for index, label in enumerate(LABELS):
                    if label not in permitted:
                        constrained[index] = float("-inf")
                masked.append(LABELS[int(constrained.argmax().item())])
            cursor += len(ids)
    report = {
        "purpose": "post-hoc diagnostic only; V2.18 test was already consumed",
        "frozen_model": str(MODEL),
        "raw_sixway": metrics(rows, raw),
        "turn_masked_sixway": metrics(rows, masked),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
