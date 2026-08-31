"""V2.44: SANTA -- self-supervised attribute normalisation, followed closely.

Reference: Sabeh, Kassem et al. is NOT this paper; this follows
"SANTA: Scalable Approach for Normalizing E-commerce Text Attributes",
ACL ECNLP 2021 (arXiv:2106.09493).

WHY THIS PAPER, SPECIFICALLY
----------------------------
SANTA diagnoses our exact failure, in its own words:

    "unsupervised embeddings shows relatively inferior performance for attribute
     normalization task, as embeddings are learnt based on contexts in product titles,
     keeping different canonical forms (e.g. 'HD' and 'Ultra HD') close by as they occur
     in similar context."

That is `made in usa` sitting next to `made overseas` in our V2.43 dump. Every encoder we
have tried -- MiniLM, mpnet, BGE, E5 -- is in the family they report as weak (fastText 65.7
against cosine string similarity 76.6).

THE METHOD, AS SPECIFIED IN THE PAPER
--------------------------------------
Triplet generation (Section 3.3.1), fully automated:
    anchor   q  = an attribute value
    positive a+ = the title of the product that HAS that attribute value
    negative a- = a title drawn from the SAME product category (a hard negative; a random
                  cross-category title "may provide limited signal")
    screening   discard the negative if the anchor value appears in the negative title

Representation (Section 3.3.2), fastText-style:
    token embedding  = mean of its character n-gram embeddings
    phrase/title emb = mean of its token embeddings
    n-gram embeddings are SHARED across the twin network

Loss (Equation 2):
    max{0, M - cos(E(q), E(a+)) + cos(E(q), E(a-))}

Hyperparameters (Section 4.3): margin M = 0.4, dimension 200, n-gram size 2 to 4,
Adadelta, 5 epochs.

THE N-GRAMS ARE THE METHOD. Their Table 2 reports SANTA WITHOUT n-grams at 47.4 accuracy,
BELOW the majority-class baseline of 48.5, against 78.4 with them. Any implementation that
keeps the twin network and drops the sub-word representation is worse than guessing, so the
n-gram table is implemented exactly rather than approximated by a word vocabulary.

EXPECTATIONS, SET BEFORE RUNNING. SANTA's own task is 45% syntactic and 17% semantic, and
its gain over cosine string similarity is only +1.8 (76.6 -> 78.4); the +19.3 is over
fastText. Its hardest cases (`HD` vs `720p`) are lexically anchored abbreviations. Ours
("made from a soft plant fibre" -> cotton) have zero lexical overlap, so this is applied
outside the regime the paper validates. The mechanism is right; the difficulty is higher.

Evaluated with the V2.32 cluster-aware metric against the frozen mpnet baseline on both
benchmarks, and with a direct check of the antonym pathology the method is supposed to fix.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/studies/train_santa_attribute_encoder.py
"""
from __future__ import annotations

import argparse
import importlib.util as ilu
import json
import random
import re
import zlib
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "experiments" / "studies"
OUT = V2 / "results" / "santa_v2_44.json"

_s = ilu.spec_from_file_location("_v2_32", V2 / "evaluate_node4_cluster_aware.py")
_m = ilu.module_from_spec(_s)
_s.loader.exec_module(_m)
normalise, surface_key = _m.normalise, _m.surface_key

NGRAM_MIN, NGRAM_MAX = 2, 4          # paper Section 4.3
DIM = 200                             # paper Section 4.3
MARGIN = 0.4                          # paper Section 4.3
BUCKETS = 400_000                     # hashed n-gram table; paper used 0.63M exact n-grams


def ngrams(token: str) -> list[int]:
    """fastText-style character n-grams with word-boundary markers, hashed to buckets."""
    # zlib.crc32, not hash(): Python randomises str hashing per process, so hash() would
    # bucket n-grams differently between the training run and any later inference run.
    t = f"<{token}>"
    out = []
    for n in range(NGRAM_MIN, NGRAM_MAX + 1):
        for i in range(len(t) - n + 1):
            out.append(zlib.crc32(t[i:i + n].encode()) % BUCKETS)
    if not out:
        out = [zlib.crc32(t.encode()) % BUCKETS]
    return out


class SantaEncoder(nn.Module):
    """Token embedding = mean of its n-gram embeddings; text = mean of its tokens."""

    def __init__(self, buckets: int = BUCKETS, dim: int = DIM):
        super().__init__()
        self.emb = nn.EmbeddingBag(buckets, dim, mode="mean")

    def forward(self, flat: torch.Tensor, offsets: torch.Tensor,
                token_counts: torch.Tensor) -> torch.Tensor:
        # one EmbeddingBag row per TOKEN (mean over its n-grams), then mean over tokens
        tok = self.emb(flat, offsets)
        out, i = [], 0
        for c in token_counts.tolist():
            out.append(tok[i:i + c].mean(0) if c else tok.new_zeros(tok.size(1)))
            i += c
        return F.normalize(torch.stack(out), dim=-1)


def encode_batch(texts: list[str], device):
    flat, offsets, counts = [], [], []
    for t in texts:
        toks = re.findall(r"[a-z0-9]+", t.lower())[:32] or ["<empty>"]
        counts.append(len(toks))
        for tok in toks:
            offsets.append(len(flat))
            flat.extend(ngrams(tok))
    return (torch.tensor(flat, dtype=torch.long, device=device),
            torch.tensor(offsets, dtype=torch.long, device=device),
            torch.tensor(counts, dtype=torch.long, device=device))


def build_triplets(seed: int = 0):
    """(attribute value, own title, same-category title) with the paper's screening."""
    from submission.agent import _sim_constraint_values
    rng = random.Random(seed)
    by_cat: dict[str, list[int]] = {}
    titles: list[str] = []
    values: list[list[str]] = []
    with (ROOT / "data" / "catalog.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            title = str(d.get("title") or "")
            if not title:
                continue
            cats = [str(c) for c in (d.get("categories") or [])]
            cat = cats[-1] if cats else "unknown"
            vals = [normalise(v) for v in _sim_constraint_values(d)]
            vals = [v for v in vals if 1 <= len(v.split()) <= 8]
            if not vals:
                continue
            by_cat.setdefault(cat, []).append(len(titles))
            titles.append(title)
            values.append(vals)

    cat_of = {}
    for cat, idxs in by_cat.items():
        for i in idxs:
            cat_of[i] = cat

    triplets = []
    skipped = 0
    for i, vals in enumerate(values):
        pool = by_cat.get(cat_of[i], [])
        if len(pool) < 2:
            continue
        for v in vals:
            # Negative is a title from the SAME product category -- the paper's hard
            # negative; a random cross-category title "may provide limited signal".
            # Screen out negatives whose title contains the anchor value, since those are
            # false negatives.
            placed = False
            for _ in range(6):
                j = pool[rng.randrange(len(pool))]
                if j != i and v not in normalise(titles[j]):
                    triplets.append((v, titles[i], titles[j]))
                    placed = True
                    break
            if not placed:
                skipped += 1
    print(f"  triplets built, {skipped:,} anchors skipped (no clean same-category negative)")
    rng.shuffle(triplets)
    return triplets, titles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)      # paper Section 4.3
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.0)      # Adadelta default
    ap.add_argument("--max-triplets", type=int, default=400_000)
    args = ap.parse_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    triplets, _titles = build_triplets()
    triplets = triplets[:args.max_triplets]
    print(f"triplets {len(triplets):,}  device {device}  [{time.time()-t0:.0f}s]")
    print(f"sample: q={triplets[0][0]!r}\n        a+={triplets[0][1][:60]!r}\n"
          f"        a-={triplets[0][2][:60]!r}\n")

    model = SantaEncoder().to(device)
    opt = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    steps = len(triplets) // args.batch
    for ep in range(args.epochs):
        model.train()
        random.Random(ep).shuffle(triplets)
        run = 0.0
        for s in range(steps):
            chunk = triplets[s * args.batch:(s + 1) * args.batch]
            q = model(*encode_batch([c[0] for c in chunk], device))
            ap_ = model(*encode_batch([c[1] for c in chunk], device))
            an = model(*encode_batch([c[2] for c in chunk], device))
            loss = torch.clamp(MARGIN - (q * ap_).sum(-1) + (q * an).sum(-1), min=0).mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            run += loss.item()
            if s % 400 == 0:
                print(f"  ep{ep+1} {s}/{steps} loss {run/max(s+1,1):.4f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
        print(f"  epoch {ep+1}/{args.epochs} loss {run/max(steps,1):.4f}", flush=True)
    model.eval()

    # ---------------------------------------------------------------- evaluation
    canonicals = [json.loads(l)["canonical"]
                  for l in (V2 / "catalogue_attribute_dictionary.jsonl").open(encoding="utf-8")
                  if l.strip()]
    n2i = {normalise(c): i for i, c in enumerate(canonicals)}
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(canonicals):
        groups.setdefault(surface_key(c), []).append(i)

    def emb(texts, bs=512):
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                outs.append(model(*encode_batch(texts[i:i + bs], device)).cpu())
        return torch.cat(outs).numpy()

    import numpy as np
    matrix = emb(canonicals)

    def load(which):
        out = []
        if which == "dev200":
            for line in (V2 / "sets" / "semantic_attribute_development_200.jsonl").open(encoding="utf-8"):
                if not line.strip():
                    continue
                card = json.loads(line).get("semantic_card") or {}
                for g in ("hard_constraints", "soft_preferences"):
                    for a in card.get(g, []):
                        if a.get("paraphrase") and a.get("canonical"):
                            out.append((str(a["paraphrase"]).lower(), str(a["canonical"])))
        else:
            for line in (V2 / "catalogue_synonym_train_only_merged.jsonl").open(encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                for syn in r.get("synonyms", []):
                    if normalise(syn) != normalise(r["canonical"]):
                        out.append((str(syn).lower(), str(r["canonical"])))
        return sorted(set(out))

    report = {"experiment": "V2.44 SANTA self-supervised attribute encoder",
              "triplets": len(triplets), "dim": DIM, "margin": MARGIN,
              "ngram": [NGRAM_MIN, NGRAM_MAX], "epochs": args.epochs, "sets": {}}
    print(f"\n{'set':<22}{'R@1':>9}{'R@3':>9}{'R@5':>9}{'R@10':>9}{'MRR':>9}")
    print("-" * 67)
    for name in ("dev200", "corpus"):
        pairs = [(q, c) for q, c in load(name) if normalise(c) in n2i]
        order = np.argsort(-(emb([q for q, _ in pairs]) @ matrix.T), axis=1)
        ranks = []
        for (_q, canon), row in zip(pairs, order):
            idx = n2i[normalise(canon)]
            acc = set(groups.get(surface_key(canonicals[idx]), [idx]))
            ranks.append(next((p + 1 for p, j in enumerate(row) if int(j) in acc),
                              len(canonicals)))
        n = max(len(ranks), 1)
        r = {f"R@{k}": round(sum(x <= k for x in ranks) / n, 4) for k in (1, 3, 5, 10)}
        r["MRR"] = round(sum(1.0 / x for x in ranks) / n, 4)
        report["sets"][name] = r
        print(f"{name:<22}{r['R@1']:>9.4f}{r['R@3']:>9.4f}{r['R@5']:>9.4f}"
              f"{r['R@10']:>9.4f}{r['MRR']:>9.4f}")

    print("\n  frozen mpnet reference: dev200 MRR 0.1134 | corpus MRR 0.5363")

    # ---- does it fix the antonym pathology the paper says it should?
    probes = [("made overseas", "imported", "made in usa"),
              ("made from a soft plant fibre", "cotton", "synthetic fiber"),
              ("made from animal fleece fibre", "wool", "polyester fleece")]
    print(f"\n  antonym check (higher cosine to the RIGHT answer is the fix):")
    for q, right, wrong in probes:
        v = emb([q, right, wrong])
        print(f"    {q[:34]:<36} {right:<10} {float(v[0] @ v[1]):+.3f}   "
              f"{wrong:<18} {float(v[0] @ v[2]):+.3f}   "
              f"{'OK' if v[0] @ v[1] > v[0] @ v[2] else 'STILL INVERTED'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
