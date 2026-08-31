"""Train turn-gated opening and follow-up V1 route classifiers on CUDA.

The prior V2.15 test is consumed and is used only as development data here.  The
new V2.18 final bank remains unread until both three-class checkpoints are selected.
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
TRAIN = ROOT / "experiments" / "studies" / "v1_route_template_bank" / "train.jsonl"
DEVELOPMENT = ROOT / "experiments" / "studies" / "v1_route_template_bank" / "test.jsonl"
FINAL = ROOT / "experiments" / "studies" / "v1_turn_gated_bank" / "final_test.jsonl"
BASE = ROOT / "submission" / "models" / "scaffolding_tagger"
MODEL_ROOT = ROOT / ".v2_model_cache"
OUT = ROOT / "experiments" / "studies" / "results" / "v1_turn_gated_classifiers_cuda.json"
SEED = 20260829
GROUPS = {
    "opening": ("buying_opening", "plain_opening", "override_opening"),
    "followup": ("constraint_update", "no_evidence", "override_update"),
}


def status(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def encode(tokenizer, rows, device):
    encoded = tokenizer([row["message"] for row in rows], padding=True, truncation=True, max_length=80, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}


def predict(model, encoded, device, batch_size=128):
    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"])
    values = []
    model.eval()
    with torch.no_grad():
        for ids, mask in DataLoader(dataset, batch_size=batch_size):
            values.extend(model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.argmax(-1).cpu().tolist())
    return values


def accuracy(rows, predictions, ids):
    return sum(prediction == ids[row["action"]] for row, prediction in zip(rows, predictions)) / len(rows)


def train_group(name, labels, train_rows, development_rows, tokenizer, device):
    status(f"{name}: preparing {len(train_rows)} fit and {len(development_rows)} consumed-development rows")
    label_ids = {label: index for index, label in enumerate(labels)}
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(labels), ignore_mismatched_sizes=True, local_files_only=True,
    ).to(device)
    fit_x, dev_x = encode(tokenizer, train_rows, device), encode(tokenizer, development_rows, device)
    counts = torch.tensor([sum(row["action"] == label for row in train_rows) for label in labels], dtype=torch.float, device=device)
    weights = counts.sum() / (len(labels) * counts)
    dataset = TensorDataset(
        fit_x["input_ids"], fit_x["attention_mask"],
        torch.tensor([label_ids[row["action"]] for row in train_rows]),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model_dir = MODEL_ROOT / f"v1_turn_gated_{name}_classifier_cuda"
    best_accuracy, best_epoch = -1.0, 0
    for epoch in range(1, 7):
        epoch_started = time.monotonic()
        model.train()
        losses = []
        for ids, mask, target in loader:
            target = target.to(device)
            loss = torch.nn.functional.cross_entropy(model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits, target, weight=weights)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
        development_accuracy = accuracy(development_rows, predict(model, dev_x, device), label_ids)
        saved = ""
        if development_accuracy > best_accuracy:
            best_accuracy, best_epoch = development_accuracy, epoch
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            saved = " checkpoint saved"
        status(f"{name}: epoch {epoch}/6 loss={sum(losses)/len(losses):.5f} development_accuracy={development_accuracy:.6f} elapsed={time.monotonic()-epoch_started:.1f}s{saved}")
    return {"labels": labels, "label_ids": label_ids, "model_dir": model_dir, "selected_epoch": best_epoch, "development_accuracy": best_accuracy}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment.")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    status(f"using {torch.cuda.get_device_name(device)} with CUDA {torch.version.cuda}")
    all_train = load_jsonl(TRAIN)
    all_dev = load_jsonl(DEVELOPMENT)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    selected = {}
    for name, labels in GROUPS.items():
        selected[name] = train_group(
            name, labels,
            [row for row in all_train if row["action"] in labels],
            [row for row in all_dev if row["action"] in labels],
            tokenizer, device,
        )

    status("both checkpoints selected; opening new V2.18 final test")
    final_rows = load_jsonl(FINAL)
    report = {"experiment": "V2.18 turn-gated route classifiers", "seed": SEED, "device": torch.cuda.get_device_name(device), "groups": {}}
    for name, spec in selected.items():
        labels, ids = spec["labels"], spec["label_ids"]
        rows = [row for row in final_rows if row["action"] in labels]
        encoded = encode(tokenizer, rows, device)
        model = AutoModelForSequenceClassification.from_pretrained(spec["model_dir"], local_files_only=True).to(device)
        predictions = predict(model, encoded, device)
        per_action = {label: round(accuracy([row for row in rows if row["action"] == label], [prediction for row, prediction in zip(rows, predictions) if row["action"] == label], ids), 6) for label in labels}
        report["groups"][name] = {
            "fit_rows": sum(row["action"] in labels for row in all_train),
            "development_rows": sum(row["action"] in labels for row in all_dev),
            "final_rows": len(rows),
            "labels": labels,
            "selected_epoch": spec["selected_epoch"],
            "development_accuracy": round(spec["development_accuracy"], 6),
            "final_accuracy": round(accuracy(rows, predictions, ids), 6),
            "per_action_final_accuracy": per_action,
            "model_artifact": str(spec["model_dir"]),
        }
    total_correct = sum(group["final_accuracy"] * group["final_rows"] for group in report["groups"].values())
    report["combined_final_accuracy"] = round(total_correct / len(final_rows), 6)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    status(f"final test complete: combined_accuracy={report['combined_final_accuracy']:.6f}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
