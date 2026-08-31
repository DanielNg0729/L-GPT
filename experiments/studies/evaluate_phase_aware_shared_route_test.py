"""Score the fixed phase-aware shared model on the already-consumed V2.18 test."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from evaluator.local_evaluator import load_jsonl
from transformers import AutoTokenizer

from experiments.studies.train_phase_aware_shared_route_classifier import (
    BASE, LABELS, MODEL_OUT, PhaseAwareRouteModel, phase_for, valid_mask,
)


ROOT = Path(__file__).resolve().parents[2]
TEST_SPLIT = ROOT / "experiments" / "studies" / "v1_turn_gated_bank" / "final_test.jsonl"
OUT = ROOT / "experiments" / "studies" / "results" / "phase_aware_shared_sixway_test.json"

def status(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def summarize(rows, predictions, label_ids):
    return {
        "accuracy": round(sum(prediction == label_ids[row["action"]] for row, prediction in zip(rows, predictions)) / len(rows), 6),
        "per_action_accuracy": {
            label: round(
                sum(prediction == label_ids[label] for row, prediction in zip(rows, predictions) if row["action"] == label) /
                sum(row["action"] == label for row in rows), 6
            ) for label in LABELS
        },
    }


def main():
    status("loading V2.18 test split")
    rows = load_jsonl(TEST_SPLIT)
    device = torch.device("cuda:0")
    status(f"loaded {len(rows)} test rows; loading tokenizer and fixed epoch-5 checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    model = PhaseAwareRouteModel().to(device).eval()
    model.load_state_dict(torch.load(MODEL_OUT, map_location=device, weights_only=True)["state_dict"])
    status("checkpoint loaded; tokenizing test rows")
    tokens = tokenizer([row["message"] for row in rows], padding=True, truncation=True, max_length=80, return_tensors="pt")
    phases = torch.tensor([phase_for(row) for row in rows])
    raw, masked = [], []
    status("running raw and phase-masked predictions")
    with torch.no_grad():
        for ids, mask, phase in DataLoader(TensorDataset(tokens["input_ids"], tokens["attention_mask"], phases), batch_size=128):
            phase = phase.to(device)
            logits = model(ids.to(device), mask.to(device), phase)
            raw.extend(logits.argmax(-1).cpu().tolist())
            masked.extend(logits.masked_fill(~valid_mask(phase, device), float("-inf")).argmax(-1).cpu().tolist())
    label_ids = {label: index for index, label in enumerate(LABELS)}
    result = {
        "experiment": "V2.21 phase-aware shared route test",
        "checkpoint": str(MODEL_OUT),
        "test_rows": len(rows),
        "selection": "fixed epoch 5 selected on eval before this test scoring",
        "raw_sixway": summarize(rows, raw, label_ids),
        "turn_masked": summarize(rows, masked, label_ids),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    status(f"test complete: masked accuracy={result['turn_masked']['accuracy']:.6f}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
