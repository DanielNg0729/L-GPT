"""V2.36: train and evaluate Nodes 3, 4 and 5 on the concept-expanded corpus.

WHAT CHANGED SINCE THE LAST ATTEMPT, AND WHY THAT MATTERS
---------------------------------------------------------
Four defects invalidated the earlier Node 3/4/5 conclusions. All are fixed here:

  V2.32  the metric was not the cluster-aware one the data contract requires. Exact-string
         scoring counted `light weight` wrong when the target was `lightweight`.
  V2.33  the "fine-tunes" ran ~30 optimizer steps and moved weights by 2.3e-4. Fine-tuning
         was never tested, merely attempted.
  V2.34/35  51% of the candidate index can never be a correct answer. That is applied here
         as a SOFT inference prior with a global fallback, never as a filter.
  --     the training corpus covered ~175 concepts against a 7,922-phrase index (2.2%).

TRAINING DELIBERATELY DOES NOT USE THE EMITTABLE PRIOR. Targets span the FULL inventory,
including the 4,049 phrases the released simulator cannot emit. An encoder trained only on
emittable targets never learns the rest, so if the private simulator emits any of them the
model is blind and nothing would tell us. The prior is an inference-time device, reported
separately so its contribution stays independently measurable and independently removable.

SPLIT DISCIPLINE (data contract): the split key is the CONCEPT, not the pair. 80/10/10 by
stable hash of the canonical, so no concept crosses train, evaluation and test.

PRETRAINED COMPARISON REQUIREMENT: the frozen encoder is evaluated in the same process,
with the same split, index and metric. A fine-tune that cannot beat it is rejected.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/studies/train_nodes_345.py --epochs 40
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util as ilu
import json
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "experiments" / "studies"
OUT = V2 / "results" / "nodes_345_v2_36.json"

_s = ilu.spec_from_file_location("_v2_32", V2 / "evaluate_node4_cluster_aware.py")
_m = ilu.module_from_spec(_s)
_s.loader.exec_module(_m)
normalise, surface_key, load_encoder = _m.normalise, _m.surface_key, _m.load_encoder

_s2 = ilu.spec_from_file_location("_v2_34", V2 / "evaluate_node4_emittable_index.py")
_m2 = ilu.module_from_spec(_s2)
_s2.loader.exec_module(_m2)
emittable_set = _m2.emittable_set

FAMILIES = ("material", "color", "size", "style", "feature", "other")


def split_of(concept: str) -> str:
    h = int(hashlib.sha256(normalise(concept).encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 80 else ("evaluation" if h < 90 else "test")


def family_of(phrase: str) -> str:
    from submission.agent import _sim_constraint_family
    fam = _sim_constraint_family(phrase)
    return fam if fam in FAMILIES else "other"


def load_corpus() -> list[tuple[str, str]]:
    """(unfamiliar phrase, canonical). Generated corpus plus the pre-existing pairs."""
    pairs: list[tuple[str, str]] = []
    gen = V2 / "generated_shopper_paraphrases.jsonl"
    if gen.exists():
        for line in gen.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue                       # a partially-written trailing line
            for p in row.get("paraphrases", []):
                if p and p.strip():
                    pairs.append((p.strip().lower(), str(row["canonical"])))
    merged = V2 / "catalogue_synonym_train_only_merged.jsonl"
    if merged.exists():
        for line in merged.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            for syn in row.get("synonyms", []):
                pairs.append((str(syn).lower(), str(row["canonical"])))
    clus = V2 / "cluster_level_paraphrases_train_only.jsonl"
    if clus.exists():
        for line in clus.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            for p in row.get("paraphrases", []):
                pairs.append((str(p).lower(), str(row["representative"])))
    # A paraphrase identical to its own canonical teaches nothing.
    return [(a, b) for a, b in dict.fromkeys(pairs) if normalise(a) != normalise(b)]


def embed(model, tok, texts, device, grad=False, bs=256):
    outs, ctx = [], (torch.enable_grad() if grad else torch.no_grad())
    for i in range(0, len(texts), bs):
        batch = tok(texts[i:i + bs], padding=True, truncation=True, max_length=64,
                    return_tensors="pt").to(device)
        with ctx:
            h = model(**batch).last_hidden_state
            m = batch["attention_mask"].unsqueeze(-1).float()
            v = F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=-1)
            outs.append(v if grad else v.cpu())
    return torch.cat(outs)


def node4_scores(model, tok, device, canonicals, pairs, emit_flag, prior=0.0):
    """Cluster-aware Recall@k / MRR over the FULL index, optional soft emittable prior."""
    import numpy as np
    n2i = {normalise(c): i for i, c in enumerate(canonicals)}
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(canonicals):
        groups.setdefault(surface_key(c), []).append(i)
    matrix = embed(model, tok, canonicals, device).numpy()
    queries = embed(model, tok, [q for q, _c in pairs], device).numpy()
    sims = queries @ matrix.T
    if prior:
        sims = sims + prior * emit_flag[None, :]
    order = np.argsort(-sims, axis=1)
    ranks = []
    for (_q, canon), row in zip(pairs, order):
        idx = n2i.get(normalise(canon))
        if idx is None:
            continue
        acc = set(groups.get(surface_key(canonicals[idx]), [idx]))
        ranks.append(next((p + 1 for p, j in enumerate(row) if int(j) in acc),
                          len(canonicals)))
    n = max(len(ranks), 1)
    out = {f"R@{k}": round(sum(r <= k for r in ranks) / n, 4) for k in (1, 3, 5, 10)}
    out["MRR"] = round(sum(1.0 / r for r in ranks) / n, 4)
    out["n"] = len(ranks)
    return out


def node3_scores(model, tok, device, pairs):
    """Family routing by nearest family centroid built from TRAIN canonicals only."""
    import numpy as np
    fams = sorted({family_of(c) for _q, c in pairs})
    return fams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--prior", type=float, default=0.20)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import numpy as np
    from transformers import AutoModel, AutoTokenizer

    canonicals = [json.loads(l)["canonical"]
                  for l in (V2 / "catalogue_attribute_dictionary.jsonl").open(encoding="utf-8")
                  if l.strip()]
    emit = emittable_set()
    emit_flag = np.array([1.0 if normalise(c) in emit else 0.0 for c in canonicals],
                         dtype=np.float32)

    pairs = load_corpus()
    by_split: dict[str, list[tuple[str, str]]] = {"train": [], "evaluation": [], "test": []}
    for a, b in pairs:
        by_split[split_of(b)].append((a, b))
    concepts = {s: len({normalise(c) for _q, c in v}) for s, v in by_split.items()}
    print(f"corpus {len(pairs):,} pairs over {len({normalise(c) for _q, c in pairs}):,} concepts")
    for s in ("train", "evaluation", "test"):
        print(f"  {s:<11}{len(by_split[s]):>6} pairs  {concepts[s]:>5} concepts")
    n_ne = len({normalise(c) for _q, c in by_split['train'] if normalise(c) not in emit})
    print(f"  train concepts that are NOT emittable: {n_ne} "
          f"(kept deliberately -- training must not depend on the prior)\n")
    if len(by_split["test"]) < 20:
        print("  test split too small to conclude anything; run after generation finishes.")

    snap = _m.__dict__  # reuse the snapshot resolution from V2.32
    import glob
    path = glob.glob(str(ROOT / ".v2_model_cache" /
                         "models--sentence-transformers--all-MiniLM-L6-v2" /
                         "snapshots" / "*"))[0]
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).to(device)

    test = by_split["test"]
    report = {"experiment": "V2.36 nodes 3-5 on concept-expanded corpus",
              "corpus_pairs": len(pairs),
              "concepts": {s: concepts[s] for s in concepts},
              "train_non_emittable_concepts": n_ne,
              "prior": args.prior, "epochs": args.epochs}

    model.eval()
    frozen = {"no_prior": node4_scores(model, tok, device, canonicals, test, emit_flag, 0.0),
              "with_prior": node4_scores(model, tok, device, canonicals, test, emit_flag,
                                         args.prior)}
    print(f"NODE 4 frozen   no-prior {frozen['no_prior']}")
    print(f"                w/ prior {frozen['with_prior']}\n")

    train = by_split["train"]
    steps = max(1, len(train) // args.batch)
    total = steps * args.epochs
    print(f"training {args.epochs} epochs x {steps} steps = {total} optimizer steps")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total,
                                                pct_start=0.1)
    rng = random.Random(0)
    model.train()
    for ep in range(args.epochs):
        rng.shuffle(train)
        run = 0.0
        for i in range(0, steps * args.batch, args.batch):
            chunk = train[i:i + args.batch]
            if len(chunk) < 2:
                continue
            q = embed(model, tok, [a for a, _b in chunk], device, grad=True)
            d = embed(model, tok, [b for _a, b in chunk], device, grad=True)
            loss = F.cross_entropy(q @ d.T * 20.0,
                                   torch.arange(len(chunk), device=device))
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run += loss.item()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1:>3}/{args.epochs} loss {run/steps:.4f}", flush=True)
    model.eval()

    tuned = {"no_prior": node4_scores(model, tok, device, canonicals, test, emit_flag, 0.0),
             "with_prior": node4_scores(model, tok, device, canonicals, test, emit_flag,
                                        args.prior)}
    print(f"\nNODE 4 tuned    no-prior {tuned['no_prior']}")
    print(f"                w/ prior {tuned['with_prior']}")
    report["node4"] = {"frozen": frozen, "tuned": tuned}

    print(f"\n{'':<14}{'frozen':>9}{'tuned':>9}{'delta':>9}   (held-out concepts, no prior)")
    print("-" * 52)
    for k in ("R@1", "R@3", "R@5", "R@10", "MRR"):
        f, t = frozen["no_prior"][k], tuned["no_prior"][k]
        print(f"{k:<14}{f:>9.4f}{t:>9.4f}{t-f:>+9.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
