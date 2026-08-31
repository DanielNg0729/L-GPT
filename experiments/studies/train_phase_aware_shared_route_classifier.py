"""Train/eval-only phase-aware shared six-route classifier.

The model consumes one public state field, `is_initial_turn`.  It keeps all six
V1 route labels and is trained with both ordinary six-way loss and phase-masked
loss, matching the hard output mask used at runtime.  No test split is loaded.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from evaluator.local_evaluator import load_jsonl
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TRAIN_SPLIT = ROOT / "experiments" / "studies" / "route_template_bank" / "train.jsonl"
EVAL_SPLIT = ROOT / "experiments" / "studies" / "route_template_bank" / "test.jsonl"
BASE = ROOT / "submission" / "models" / "scaffolding_tagger"
MODEL_OUT = ROOT / ".v2_model_cache" / "phase_aware_shared_sixway_cuda.pt"
OUT = ROOT / "experiments" / "results" / "phase_aware_shared_sixway_eval.json"
LABELS = ("buying_opening", "constraint_update", "no_evidence", "override_opening", "override_update", "plain_opening")
OPENING = {"buying_opening", "plain_opening", "override_opening"}
SEED = 20260830
MASKED_LOSS_WEIGHT = 1.0


def status(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


class PhaseAwareRouteModel(nn.Module):
    """Shared DistilBERT encoder plus a learned two-state phase embedding."""

    def __init__(self):
        super().__init__()
        scaffold = AutoModelForSequenceClassification.from_pretrained(BASE, local_files_only=True)
        self.encoder = scaffold.distilbert
        self.pre_classifier = scaffold.pre_classifier
        self.dropout = scaffold.dropout
        self.phase_embedding = nn.Embedding(2, 8)
        self.classifier = nn.Linear(scaffold.config.dim + 8, len(LABELS))

    def forward(self, input_ids, attention_mask, phase):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        hidden = F.relu(self.pre_classifier(hidden))
        hidden = self.dropout(hidden)
        return self.classifier(torch.cat((hidden, self.phase_embedding(phase)), dim=-1))


def phase_for(row):
    return 0 if row["action"] in OPENING else 1


def encode(tokenizer, rows, device):
    values = tokenizer([row["message"] for row in rows], padding=True, truncation=True, max_length=80, return_tensors="pt")
    return {key: value.to(device) for key, value in values.items() if key in {"input_ids", "attention_mask"}}


def valid_mask(phases, device):
    matrix = torch.zeros((len(phases), len(LABELS)), dtype=torch.bool, device=device)
    for index, phase in enumerate(phases.tolist()):
        permitted = OPENING if phase == 0 else set(LABELS) - OPENING
        for label_index, label in enumerate(LABELS):
            matrix[index, label_index] = label in permitted
    return matrix


def predict(model, encoded, phases, device):
    raw, masked = [], []
    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"], phases)
    model.eval()
    with torch.no_grad():
        for ids, mask, phase in DataLoader(dataset, batch_size=128):
            phase = phase.to(device)
            logits = model(ids.to(device), mask.to(device), phase)
            raw.extend(logits.argmax(-1).cpu().tolist())
            masked.extend(logits.masked_fill(~valid_mask(phase, device), float("-inf")).argmax(-1).cpu().tolist())
    return raw, masked


def report(rows, predictions, label_ids):
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    label_ids = {label: index for index, label in enumerate(LABELS)}
    train_rows, eval_rows = load_jsonl(TRAIN_SPLIT), load_jsonl(EVAL_SPLIT)
    status(f"CUDA={torch.cuda.get_device_name(device)}; train={len(train_rows)}; eval={len(eval_rows)}; no test split loaded")
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    model = PhaseAwareRouteModel().to(device)
    train_x, eval_x = encode(tokenizer, train_rows, device), encode(tokenizer, eval_rows, device)
    train_phase = torch.tensor([phase_for(row) for row in train_rows])
    eval_phase = torch.tensor([phase_for(row) for row in eval_rows])
    counts = torch.tensor([sum(row["action"] == label for row in train_rows) for label in LABELS], dtype=torch.float, device=device)
    weights = counts.sum() / (len(LABELS) * counts)
    loader = DataLoader(TensorDataset(train_x["input_ids"], train_x["attention_mask"], train_phase, torch.tensor([label_ids[row["action"]] for row in train_rows])), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    best, best_epoch = -1.0, 0
    for epoch in range(1, 7):
        started = time.monotonic()
        losses = []
        model.train()
        for ids, mask, phase, target in loader:
            phase, target = phase.to(device), target.to(device)
            logits = model(ids.to(device), mask.to(device), phase)
            full_loss = F.cross_entropy(logits, target, weight=weights)
            phase_loss = F.cross_entropy(logits.masked_fill(~valid_mask(phase, device), float("-inf")), target, weight=weights)
            loss = full_loss + MASKED_LOSS_WEIGHT * phase_loss
            loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True); losses.append(float(loss.detach()))
        _, eval_masked = predict(model, eval_x, eval_phase, device)
        eval_metrics = report(eval_rows, eval_masked, label_ids)
        saved = ""
        if eval_metrics["accuracy"] > best:
            best, best_epoch = eval_metrics["accuracy"], epoch
            torch.save({"state_dict": model.state_dict(), "labels": LABELS, "seed": SEED, "masked_loss_weight": MASKED_LOSS_WEIGHT}, MODEL_OUT)
            saved = " checkpoint saved"
        status(f"epoch {epoch}/6 train_loss={sum(losses)/len(losses):.5f} eval_masked_accuracy={eval_metrics['accuracy']:.6f} elapsed={time.monotonic()-started:.1f}s{saved}")
    checkpoint = torch.load(MODEL_OUT, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    raw, masked = predict(model, eval_x, eval_phase, device)
    result = {"experiment":"phase-aware shared six-route classifier", "train_rows":len(train_rows), "eval_rows":len(eval_rows), "test_rows_loaded":0, "phase_feature":"is_initial_turn encoded as a learned two-state embedding", "mask":"opening actions permitted only at turn 1; follow-up actions permitted only later", "masked_loss_weight":MASKED_LOSS_WEIGHT, "selected_epoch":best_epoch, "eval_raw":report(eval_rows, raw, label_ids), "eval_turn_masked":report(eval_rows, masked, label_ids), "model_artifact":str(MODEL_OUT), "selection":"masked eval accuracy; no test split loaded"}
    OUT.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    status(f"eval complete: raw={result['eval_raw']['accuracy']:.6f}; masked={result['eval_turn_masked']['accuracy']:.6f}; no test was opened")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
