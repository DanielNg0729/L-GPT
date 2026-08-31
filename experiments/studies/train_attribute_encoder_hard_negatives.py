"""V2.33: actually fine-tune the Node 4 attribute encoder, and measure it honestly.

WHAT WENT WRONG BEFORE
----------------------
The recorded V2.30/V2.31 conclusion is "fine-tuning did not improve Recall@5". That
conclusion is not supported by what was run:

    verified_pairs 158-178, batch_size 16, epochs 3   ->  ~30 optimizer steps total

Thirty steps at a standard encoder learning rate barely perturbs the weights, and the
checkpoints confirm it: `max|delta|` on `embeddings.word_embeddings.weight` against the
frozen model is 2.3e-4 (a real fine-tune moves 1e-2 to 1e-1). That is also why two
*differently trained* checkpoints produced byte-identical Recall@1/3/5/10 equal to the
frozen baseline. Fine-tuning was never tested; it was merely attempted.

WHAT THIS CHANGES
-----------------
1. ENOUGH STEPS. Epochs are set so training runs for hundreds of steps rather than ~30.

2. IN-BATCH NEGATIVES instead of one hard negative. With ~180 pairs the binding constraint
   is negatives-per-anchor, not epochs. A MultipleNegativesRankingLoss-style objective
   (softmax over the in-batch similarity matrix) gives batch_size-1 negatives per anchor
   for free -- 15 at batch 16, against the 1 the previous objective used. Implemented
   directly in torch because `sentence_transformers` is not installed in this environment.

3. HONEST EVALUATION. Scored with the V2.32 cluster-aware metric on the SAME frozen
   benchmark, and the frozen baseline is re-run in the same process so the comparison
   cannot drift. Both are reported; the fine-tune is only interesting if it beats frozen.

CAVEAT STATED UP FRONT, NOT AFTERWARDS. The benchmark has 712 rows but only **67 distinct
concepts**, so its resolution is coarse and any single-digit movement is noise. This pass
answers "does proper training change anything at all", not "which configuration is best".
Configuration selection needs the larger corpus and the concept-split benchmark first.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/studies/train_attribute_encoder_hard_negatives.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "experiments" / "studies"
MERGED = V2 / "catalogue_synonym_train_only_merged.jsonl"
CLUSTER_PARA = V2 / "cluster_level_paraphrases_train_only.jsonl"
OUT = V2 / "results" / "node4_finetune_v2_33.json"

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_v2_32", V2 / "evaluate_node4_cluster_aware.py")
_v232 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v232)
normalise, surface_key = _v232.normalise, _v232.surface_key


def snapshot() -> str:
    snaps = glob.glob(str(ROOT / ".v2_model_cache" /
                          "models--sentence-transformers--all-MiniLM-L6-v2" /
                          "snapshots" / "*"))
    if not snaps:
        raise SystemExit("frozen MiniLM snapshot not found in .v2_model_cache")
    return snaps[0]


def load_pairs() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(unfamiliar phrase, canonical). Split is by CONCEPT, as the data contract requires."""
    train, evaluation = [], []
    for line in MERGED.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        bucket = train if row.get("split") == "train" else evaluation
        for syn in row.get("synonyms", []):
            bucket.append((str(syn), str(row["canonical"])))
    if CLUSTER_PARA.exists():
        for line in CLUSTER_PARA.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            for para in row.get("paraphrases", []):
                train.append((str(para), str(row["representative"])))
    return train, evaluation


def encoder(device):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModel, AutoTokenizer
    path = snapshot()
    return AutoTokenizer.from_pretrained(path), AutoModel.from_pretrained(path).to(device)


def embed(model, tok, texts, device, grad=False, bs=256):
    outs = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    for i in range(0, len(texts), bs):
        batch = tok(texts[i:i + bs], padding=True, truncation=True, max_length=64,
                    return_tensors="pt").to(device)
        with ctx:
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            outs.append(F.normalize(vec, dim=-1) if grad else
                        F.normalize(vec, dim=-1).cpu())
    return torch.cat(outs)


def build_benchmark():
    dev = V2 / "sets" / "semantic_attribute_development_200.jsonl"
    canonicals = [json.loads(l)["canonical"]
                  for l in (V2 / "catalogue_attribute_dictionary.jsonl").open(encoding="utf-8")
                  if l.strip()]
    norm_to_idx = {normalise(c): i for i, c in enumerate(canonicals)}
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(canonicals):
        groups.setdefault(surface_key(c), []).append(i)
    atoms = []
    for line in dev.open(encoding="utf-8"):
        if not line.strip():
            continue
        card = json.loads(line).get("semantic_card") or {}
        for grp in ("hard_constraints", "soft_preferences"):
            for a in card.get(grp, []):
                if a.get("paraphrase") and a.get("canonical"):
                    atoms.append((str(a["paraphrase"]), str(a["canonical"])))
    return canonicals, norm_to_idx, groups, atoms


def score(model, tok, device, canonicals, norm_to_idx, groups, atoms):
    import numpy as np
    matrix = embed(model, tok, canonicals, device).numpy()
    distinct = sorted({a for a in atoms})
    queries = embed(model, tok, [q for q, _c in distinct], device).numpy()
    order = np.argsort(-(queries @ matrix.T), axis=1)
    rank_of = {}
    for (q, canon), row in zip(distinct, order):
        idx = norm_to_idx.get(normalise(canon))
        if idx is None:
            continue
        acc = set(groups.get(surface_key(canonicals[idx]), [idx]))
        rank_of[(q, canon)] = next((p + 1 for p, j in enumerate(row) if int(j) in acc),
                                   len(canonicals))
    out = {}
    for label, population in (("weighted", atoms), ("per_concept", distinct)):
        ranks = [rank_of[a] for a in population if a in rank_of]
        n = max(len(ranks), 1)
        out[label] = {f"R@{k}": round(sum(r <= k for r in ranks) / n, 4)
                      for k in (1, 3, 5, 10)}
        out[label]["MRR"] = round(sum(1.0 / r for r in ranks) / n, 4)
        out[label]["n"] = len(ranks)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--scale", type=float, default=20.0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train, evaluation = load_pairs()
    print(f"train pairs {len(train)}  evaluation pairs {len(evaluation)}  device {device}")
    canonicals, norm_to_idx, groups, atoms = build_benchmark()
    print(f"dictionary {len(canonicals):,}  benchmark atoms {len(atoms)} "
          f"({len(set(atoms))} distinct concepts)\n")

    tok, model = encoder(device)
    frozen = score(model, tok, device, canonicals, norm_to_idx, groups, atoms)
    print(f"FROZEN    per-concept {frozen['per_concept']}")
    print(f"          weighted    {frozen['weighted']}\n")

    steps_per_epoch = max(1, len(train) // args.batch)
    total = steps_per_epoch * args.epochs
    print(f"training: {args.epochs} epochs x {steps_per_epoch} steps = {total} optimizer "
          f"steps (previous runs did ~30)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total, pct_start=0.1)
    rng = random.Random(0)
    model.train()
    step = 0
    for ep in range(args.epochs):
        rng.shuffle(train)
        running = 0.0
        for i in range(0, steps_per_epoch * args.batch, args.batch):
            chunk = train[i:i + args.batch]
            if len(chunk) < 2:
                continue
            q = embed(model, tok, [a for a, _b in chunk], device, grad=True)
            d = embed(model, tok, [b for _a, b in chunk], device, grad=True)
            # In-batch negatives: every other document in the batch is a negative for
            # this anchor. batch-1 negatives per anchor instead of the single hard
            # negative the earlier objective used.
            loss = F.cross_entropy(q @ d.T * args.scale,
                                   torch.arange(len(chunk), device=device))
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running += loss.item()
            step += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1:>3}/{args.epochs}  loss {running/steps_per_epoch:.4f}  "
                  f"step {step}", flush=True)
    model.eval()

    tuned = score(model, tok, device, canonicals, norm_to_idx, groups, atoms)
    print(f"\nFINE-TUNED per-concept {tuned['per_concept']}")
    print(f"           weighted    {tuned['weighted']}")

    base = torch.load if False else None  # placeholder to keep imports tidy
    print(f"\n{'metric':<12}{'frozen':>10}{'tuned':>10}{'delta':>10}")
    print("-" * 42)
    for k in ("R@1", "R@3", "R@5", "R@10", "MRR"):
        f, t = frozen["per_concept"][k], tuned["per_concept"][k]
        print(f"{'per-concept ' + k:<12}{f:>10.4f}{t:>10.4f}{t-f:>+10.4f}")
    print("\n  67 distinct concepts: single-digit movement here is NOISE. This answers")
    print("  'does real training change anything', not 'which configuration wins'.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"experiment": "V2.33 properly-trained Node 4 encoder",
         "optimizer_steps": total, "previous_steps_approx": 30,
         "objective": "in-batch negatives (MNRL-style softmax over batch similarity)",
         "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
         "train_pairs": len(train), "benchmark_atoms": len(atoms),
         "benchmark_distinct_concepts": len(set(atoms)),
         "frozen": frozen, "fine_tuned": tuned}, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
