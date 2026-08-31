"""V2.63: train the value-span extractor.

Token classification over BIO tags for VALUE and CATEGORY, on data whose value vocabulary
was deliberately randomised (V2.62) so the model must learn slot POSITION rather than
attribute-shaped words. The point of the extractor is to find values it has never seen; a
model that recognises `Cotton` is useless for `made from a soft plant fibre`.

WHAT IS REPORTED, AND WHY NOT ACCURACY. Token accuracy is dominated by `O` -- 78,600 of
149,739 tags -- so a model predicting nothing scores well on it. The reported metric is
VALUE-span exact match plus token-level precision/recall for the VALUE class, and both are
compared against two trivial baselines that any useful model must beat:

    predict-nothing        the degenerate optimum for token accuracy
    predict-everything     every token is a VALUE, the recall-maximal degenerate

DEV IS FROM THE SAME TEMPLATES AS TRAIN and therefore proves nothing about transfer. It is
used only to pick the epoch. The real measurement is the held-out template bank with
PARAPHRASED values substituted, which is a separate script -- run that before believing
anything here.

Artifacts follow the model policy: a run directory with the base reference, seeds,
hyperparameters, and the selected checkpoint, never overwriting a prior run.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/studies/train_span_bert.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "experiments" / "studies"
DATA = V2 / "span_data_valrand"
CACHE = ROOT / ".v2_model_cache"
RUN = CACHE / os.environ.get("SPAN_RUN", "span_extractor_valrand")
OUT = V2 / "results" / "span_bert_training_v2_63.json"

BASE = "distilbert-base-uncased"
LABELS = ("O", "B-VALUE", "I-VALUE", "B-CATEGORY", "I-CATEGORY")
L2I = {l: i for i, l in enumerate(LABELS)}
SEED = 20260830           # literal; hash() is per-process randomised and must never seed
EPOCHS = int(os.environ.get("SPAN_EPOCHS", "4"))
BATCH = 32
LR = 5e-5
MAX_LEN = 96


def load(path):
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def spans_of(labels):
    """BIO -> set of (start, end) VALUE spans, half-open."""
    out, start = set(), None
    for i, l in enumerate(list(labels) + ["O"]):
        if l == "B-VALUE":
            if start is not None:
                out.add((start, i))
            start = i
        elif l == "I-VALUE":
            if start is None:
                start = i
        else:
            if start is not None:
                out.add((start, i))
                start = None
    return out


def main() -> None:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train, dev = load(DATA / "train.jsonl"), load(DATA / "dev.jsonl")
    print(f"train {len(train)}  dev {len(dev)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE, cache_dir=str(CACHE))
    model = AutoModelForTokenClassification.from_pretrained(
        BASE, num_labels=len(LABELS), cache_dir=str(CACHE)).to(device)

    def collate(batch):
        enc = tok([b["tokens"] for b in batch], is_split_into_words=True, padding=True,
                  truncation=True, max_length=MAX_LEN, return_tensors="pt")
        lab = torch.full(enc["input_ids"].shape, -100, dtype=torch.long)
        for i, b in enumerate(batch):
            prev = None
            for pos, wid in enumerate(enc.word_ids(i)):
                if wid is None or wid == prev:
                    continue              # label the FIRST subword of each word only
                prev = wid
                if wid < len(b["labels"]):
                    lab[i, pos] = L2I[b["labels"][wid]]
        enc["labels"] = lab
        return enc

    dl = DataLoader(train, batch_size=BATCH, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps = EPOCHS * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps,
                                                pct_start=0.1)

    def predict(rows, bs=64):
        model.eval()
        out = []
        for i in range(0, len(rows), bs):
            batch = rows[i:i + bs]
            enc = tok([b["tokens"] for b in batch], is_split_into_words=True,
                      padding=True, truncation=True, max_length=MAX_LEN,
                      return_tensors="pt").to(device)
            with torch.no_grad():
                pred = model(**{k: v for k, v in enc.items()
                                if k in ("input_ids", "attention_mask")}).logits.argmax(-1)
            for j, b in enumerate(batch):
                lab, prev = ["O"] * len(b["tokens"]), None
                for pos, wid in enumerate(enc.word_ids(j)):
                    if wid is None or wid == prev:
                        continue
                    prev = wid
                    if wid < len(lab):
                        lab[wid] = LABELS[int(pred[j, pos])]
                out.append(lab)
        return out

    def score(rows, preds):
        tp = fp = fn = 0
        exact = total = 0
        for r, p in zip(rows, preds):
            g, h = spans_of(r["labels"]), spans_of(p)
            total += 1
            exact += int(g == h)
            gt = {i for s, e in g for i in range(s, e)}
            ht = {i for s, e in h for i in range(s, e)}
            tp += len(gt & ht); fp += len(ht - gt); fn += len(gt - ht)
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        return {"span_exact": exact / max(total, 1), "token_p": p_, "token_r": r_,
                "token_f1": f}

    # Trivial baselines, so "the model learned something" is a comparison rather than a claim.
    nothing = [["O"] * len(r["tokens"]) for r in dev]
    everything = [["B-VALUE"] + ["I-VALUE"] * (len(r["tokens"]) - 1) for r in dev]
    print(f"\nbaseline predict-nothing    {json.dumps(score(dev, nothing))}")
    print(f"baseline predict-everything {json.dumps(score(dev, everything))}\n")

    t0, history, best = time.time(), [], None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            total += float(loss)
        m = score(dev, predict(dev))
        m["epoch"], m["loss"] = epoch, total / max(len(dl), 1)
        history.append(m)
        print(f"epoch {epoch}  loss {m['loss']:.4f}  span-exact {m['span_exact']:.4f}  "
              f"VALUE P {m['token_p']:.4f} R {m['token_r']:.4f} F1 {m['token_f1']:.4f}",
              flush=True)
        if best is None or m["span_exact"] > best["span_exact"]:
            best = m
            RUN.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(RUN); tok.save_pretrained(RUN)
            (RUN / "labels.json").write_text(json.dumps(list(LABELS)), encoding="utf-8")

    print(f"\nselected epoch {best['epoch']} on dev span-exact {best['span_exact']:.4f}")
    print(f"  DEV SHARES TEMPLATES WITH TRAIN, so this is epoch selection and nothing more.")
    print(f"  The held-out measurement is the test bank with PARAPHRASED values.")
    print(f"  {time.time()-t0:.0f}s on {device}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "V2.63 value-span extractor training",
        "base": BASE, "seed": SEED, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
        "train": len(train), "dev": len(dev), "labels": list(LABELS),
        "run_dir": str(RUN), "selected": best, "history": history,
        "baselines": {"predict_nothing": score(dev, nothing),
                      "predict_everything": score(dev, everything)},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
