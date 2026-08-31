"""Experiment 50: train the scaffolding tagger and SAVE it as a shippable asset.

Pass 49 established the result; this produces the artifact the agent loads.

    held-out-transform lift (the measurement that killed the linear model in pass 47)
        T2 scaffolding stripped   linear -0.435  ->  BERT +0.100
        T5 realistic              linear +0.108  ->  BERT +0.251
        T4 case/punct churn       linear +0.196  ->  BERT +0.423

    end-to-end   clean 0.96960 (unchanged)   unseen-800 0.95722 (unchanged)
                 T1 +0.0407                  T5 +0.0498

MLM adaptation on the 50,000-product catalogue was measured and is NOT used: it scored
fractionally worse than plain pretrained distilbert (T1 0.88820 vs 0.89300). The tagger's
job is separating product words from FILLER, and catalogue listings contain no filler to
contrast against, so the discriminative knowledge has to come from general pretraining.
That is a real finding about where the value sits, and it saves ~6 minutes of adaptation.

The model is saved to `submission/models/scaffolding_tagger/`. It is a genuine dependency
cost -- see `bert_extract.py` for how that is contained.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/log/50_train_and_save_tagger.py
"""
from __future__ import annotations

# torch FIRST -- importing sklearn first breaks c10.dll on this Windows build (WinError 1114).
import torch  # noqa: E402

import argparse  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "log"))

from evaluator.local_evaluator import catalog_index, load_jsonl  # noqa: E402

_p49 = __import__("49_bert_scaffolding_tagger")

OUT_DIR = ROOT / "submission" / "models" / "scaffolding_tagger"
DEV = _p49.DEV


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--sessions", type=int, default=1600)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    t0 = time.time()

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    _cid, _cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    pub_t = {str(s["ground_truth"]["parent_asin"]) for s in samples}

    pool = [a for a in prods if a not in pub_t]
    random.Random(3).shuffle(pool)
    train_asins = pool[:args.sessions]

    tok = AutoTokenizer.from_pretrained(_p49.MODEL)
    rows = _p49.gen(prods, train_asins, _p49.TRAIN_TF, seed=1)
    print(f"training messages {len(rows):,}  device {DEV}  [{time.time()-t0:.0f}s]")

    model = AutoModelForTokenClassification.from_pretrained(
        _p49.MODEL, num_labels=2,
        id2label={0: "SCAFFOLD", 1: "CONTENT"},
        label2id={"SCAFFOLD": 0, "CONTENT": 1},
    ).to(DEV)
    enc, labels = _p49.encode(tok, rows)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    n = len(rows)
    order = list(range(n))
    for ep in range(args.epochs):
        model.train()
        random.Random(ep).shuffle(order)
        tot = 0.0
        for k in range(0, n - args.batch, args.batch):
            idx = order[k:k + args.batch]
            out = model(input_ids=enc["input_ids"][idx].to(DEV),
                        attention_mask=enc["attention_mask"][idx].to(DEV),
                        labels=labels[idx].to(DEV))
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            tot += out.loss.item()
        print(f"  epoch {ep+1}/{args.epochs} loss {tot/max(1,n//args.batch):.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    print("\nheld-out-transform check before saving:")
    te_asins = pool[args.sessions:args.sessions + 300]
    ok = True
    for tf in _p49.HELDOUT_TF:
        te = _p49.gen(prods, te_asins, [tf], seed=2)
        acc, majority = _p49.evaluate_tagger(model, tok, te)
        lift = acc - majority
        print(f"  {tf:<28} acc {acc:.3f}  majority {majority:.3f}  lift {lift:+.3f}")
        if lift <= 0.02:
            ok = False
    if not ok:
        print("\n  REFUSING TO SAVE: a transform shows no lift over majority class, which is")
        print("  the pass-47 failure mode. Do not ship a tagger that has not cleared it.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    size = sum(f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"\n[saved] {OUT_DIR}  ({size:.0f} MB)   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
