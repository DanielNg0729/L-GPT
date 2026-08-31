"""Post-hoc diagnostic: test the follow-up classifier's last epoch on V2.18.

Terminology:
* train split: parameter fitting;
* eval split: epoch observation only;
* test split: V2.18, loaded only after epoch six.

The test has already been consumed by V2.18. This is not a new final model-selection
result and must not be used to tune a successor.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader, TensorDataset
from evaluator.local_evaluator import load_jsonl
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TRAIN_SPLIT = ROOT / "experiments" / "studies" / "route_template_bank" / "train.jsonl"
EVAL_SPLIT = ROOT / "experiments" / "studies" / "route_template_bank" / "test.jsonl"
TEST_SPLIT = ROOT / "experiments" / "studies" / "turn_gated_bank" / "final_test.jsonl"
BASE = ROOT / "submission" / "models" / "scaffolding_tagger"
MODEL_OUT = ROOT / ".v2_model_cache" / "v1_turn_gated_followup_last_epoch_cuda"
OUT = ROOT / "experiments" / "results" / "v1_turn_gated_followup_last_epoch_diagnostic.json"
LABELS = ("constraint_update", "no_evidence", "override_update")
SEED = 20260829


def status(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def encode(tokenizer, rows, device):
    values = tokenizer([row["message"] for row in rows], padding=True, truncation=True, max_length=80, return_tensors="pt")
    return {key: value.to(device) for key, value in values.items() if key in {"input_ids", "attention_mask"}}


def predict(model, encoded, device):
    values = []
    model.eval()
    with torch.no_grad():
        for ids, mask in DataLoader(TensorDataset(encoded["input_ids"], encoded["attention_mask"]), batch_size=128):
            values.extend(model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.argmax(-1).cpu().tolist())
    return values


def score(rows, predictions, label_ids):
    overall = sum(prediction == label_ids[row["action"]] for row, prediction in zip(rows, predictions)) / len(rows)
    per_action = {}
    for label in LABELS:
        matched = [(row, prediction) for row, prediction in zip(rows, predictions) if row["action"] == label]
        per_action[label] = round(sum(prediction == label_ids[label] for _, prediction in matched) / len(matched), 6)
    return round(overall, 6), per_action


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    label_ids = {label: index for index, label in enumerate(LABELS)}
    train_rows = [row for row in load_jsonl(TRAIN_SPLIT) if row["action"] in LABELS]
    eval_rows = [row for row in load_jsonl(EVAL_SPLIT) if row["action"] in LABELS]
    status(f"CUDA={torch.cuda.get_device_name(device)}; train={len(train_rows)}; eval={len(eval_rows)}; test is not yet loaded")
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=3, ignore_mismatched_sizes=True, local_files_only=True).to(device)
    train_x, eval_x = encode(tokenizer, train_rows, device), encode(tokenizer, eval_rows, device)
    counts = torch.tensor([sum(row["action"] == label for row in train_rows) for label in LABELS], dtype=torch.float, device=device)
    weights = counts.sum() / (len(LABELS) * counts)
    loader = DataLoader(TensorDataset(train_x["input_ids"], train_x["attention_mask"], torch.tensor([label_ids[row["action"]] for row in train_rows])), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    for epoch in range(1, 7):
        losses = []
        model.train()
        for ids, mask, target in loader:
            loss = torch.nn.functional.cross_entropy(model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits, target.to(device), weight=weights)
            loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True); losses.append(float(loss.detach()))
        eval_accuracy, _ = score(eval_rows, predict(model, eval_x, device), label_ids)
        status(f"epoch {epoch}/6 complete: train_loss={sum(losses)/len(losses):.5f}; eval_accuracy={eval_accuracy:.6f}")
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_OUT); tokenizer.save_pretrained(MODEL_OUT)
    status("epoch 6 saved; loading already-consumed test split for diagnostic")
    test_rows = [row for row in load_jsonl(TEST_SPLIT) if row["action"] in LABELS]
    test_accuracy, per_action = score(test_rows, predict(model, encode(tokenizer, test_rows, device), device), label_ids)
    result = {"experiment":"V2.18 post-hoc last-epoch follow-up diagnostic","selection":"none: epoch 6 chosen by user after V2.18 test was consumed","train_rows":len(train_rows),"eval_rows":len(eval_rows),"test_rows":len(test_rows),"test_accuracy":test_accuracy,"per_action_test_accuracy":per_action,"model_artifact":str(MODEL_OUT)}
    OUT.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    status(f"diagnostic test complete: accuracy={test_accuracy:.6f}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
