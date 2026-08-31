"""Train the exact V1 fallback-action classifier on the template-disjoint bank.

Checkpoint selection uses only three paraphrased template families per action held out
from the training bank.  The independently authored `test.jsonl` file is opened once,
only after the selected checkpoint is saved.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

# Required before PyTorch initializes CUDA when strict reproducibility is enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader, TensorDataset
from evaluator.local_evaluator import load_jsonl
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments" / "studies" / "v1_route_template_bank"
BASE = ROOT / "submission" / "models" / "scaffolding_tagger"
OUT = ROOT / "experiments" / "studies" / "results" / "v1_route_template_bank_classifier_cuda.json"
MODEL_OUT = ROOT / ".v2_model_cache" / "v1_route_template_bank_classifier_cuda"
SEED = 20260829


def status(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def encode(tokenizer, rows):
    return tokenizer(
        [row["message"] for row in rows], padding=True, truncation=True,
        max_length=80, return_tensors="pt",
    )


def predict(model, encoded):
    model.eval()
    with torch.no_grad():
        return model(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
        ).logits.argmax(-1).tolist()


def accuracy_by_action(rows, predictions, label_ids, labels):
    per = {label: [0, 0] for label in labels}
    for row, prediction in zip(rows, predictions):
        label = row["action"]
        per[label][1] += 1
        per[label][0] += int(prediction == label_ids[label])
    return {label: round(correct / total, 6) for label, (correct, total) in per.items()}


def main():
    started = time.monotonic()
    status("starting V2.15 template-disjoint route-classifier training")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run but is not available in the selected environment.")
    device = torch.device("cuda:0")
    status(f"using {torch.cuda.get_device_name(device)} with CUDA {torch.version.cuda}")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)

    all_train = load_jsonl(DATA / "train.jsonl")
    status(f"loaded training bank: {len(all_train)} rows")
    labels = sorted({row["action"] for row in all_train})
    label_ids = {label: index for index, label in enumerate(labels)}

    # Select three entire paraphrased wrapper families per action for development.
    development_templates = set()
    for label in labels:
        templates = sorted({
            row["template"] for row in all_train
            if row["source"] == "train" and row["action"] == label
        })
        assert len(templates) == 12, (label, len(templates))
        development_templates.update(templates[::4])
    development = [
        row for row in all_train
        if row["source"] == "train" and row["template"] in development_templates
    ]
    development_keys = {(row["sample_id"], row["action"], row["template"]) for row in development}
    fit = [
        row for row in all_train
        if (row["sample_id"], row["action"], row["template"]) not in development_keys
    ]
    assert len(development_templates) == 18
    assert not {row["template"] for row in fit if row["source"] == "train"} & development_templates
    status(f"split complete: fit={len(fit)}, development={len(development)}, development_templates=18")

    status("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    status("loading six-class DistilBERT model")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(labels), ignore_mismatched_sizes=True, local_files_only=True,
    ).to(device)
    status("encoding fit and development text")
    fit_x, development_x = (
        {key: value.to(device) for key, value in encode(tokenizer, rows).items()}
        for rows in (fit, development)
    )
    status("fit and development encoding complete")
    status("computing class weights")
    counts = torch.tensor(
        [sum(row["action"] == label for row in fit) for label in labels], dtype=torch.float
    )
    weights = counts.sum() / (len(labels) * counts)
    status("creating tensor dataset")
    dataset = TensorDataset(
        fit_x["input_ids"], fit_x["attention_mask"],
        torch.tensor([label_ids[row["action"]] for row in fit]),
    )
    status("creating deterministic data loader")
    loader = DataLoader(
        dataset, batch_size=32, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    status("creating optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    development_gold = [label_ids[row["action"]] for row in development]
    best_accuracy, best_epoch = -1.0, 0

    for epoch in range(1, 7):
        status(f"starting epoch {epoch}/6")
        epoch_started = time.monotonic()
        model.train()
        losses = []
        for ids, mask, target in loader:
            ids, mask, target = ids.to(device), mask.to(device), target.to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = torch.nn.functional.cross_entropy(logits, target, weight=weights.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
        dev_predictions = predict(model, development_x)
        dev_accuracy = sum(prediction == gold for prediction, gold in zip(dev_predictions, development_gold)) / len(development_gold)
        checkpoint = ""
        if dev_accuracy > best_accuracy:
            best_accuracy, best_epoch = dev_accuracy, epoch
            MODEL_OUT.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_OUT)
            tokenizer.save_pretrained(MODEL_OUT)
            checkpoint = " checkpoint saved"
        status(
            f"epoch {epoch}/6 complete: mean_loss={sum(losses)/len(losses):.5f}, "
            f"development_accuracy={dev_accuracy:.6f}, elapsed={time.monotonic()-epoch_started:.1f}s{checkpoint}"
        )

    status(f"checkpoint selection complete: epoch {best_epoch}, development_accuracy={best_accuracy:.6f}")
    status("opening untouched final test split")
    test = load_jsonl(DATA / "test.jsonl")
    test_x = {key: value.to(device) for key, value in encode(tokenizer, test).items()}
    selected = AutoModelForSequenceClassification.from_pretrained(MODEL_OUT, local_files_only=True).to(device)
    test_predictions = predict(selected, test_x)
    test_gold = [label_ids[row["action"]] for row in test]
    result = {
        "experiment": "V2.15 template-disjoint V1-action route classifier",
        "seed": SEED,
        "epochs": 6,
        "fit_rows": len(fit),
        "development_rows": len(development),
        "development_templates": len(development_templates),
        "test_rows": len(test),
        "selected_epoch": best_epoch,
        "development_accuracy": round(best_accuracy, 6),
        "test_accuracy": round(sum(p == g for p, g in zip(test_predictions, test_gold)) / len(test_gold), 6),
        "per_action_test_accuracy": accuracy_by_action(test, test_predictions, label_ids, labels),
        "model_artifact": str(MODEL_OUT),
        "selection": "best accuracy on template-disjoint internal development families; independently authored test opened once",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    status(f"final test complete: accuracy={result['test_accuracy']:.6f}; total_elapsed={time.monotonic()-started:.1f}s")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
