"""V2.32: re-score Node 4 retrieval with the CLUSTER-AWARE metric the contract requires.

WHY THIS EXISTS
---------------
`docs/V2_NODES_3_TO_5_DATA_CONTRACT.md` specifies, for Node 4:

    "evaluation counts a hit when its top-k contains ANY MEMBER of the correct cluster ...
     This avoids penalising a correct semantic mapping merely because the target catalogue
     product uses a different member spelling."

`pretrained_attribute_baseline.py` does not implement that. It resolves one exact string
(`canonical_index.get(normalise(example["canonical"]))`) and counts every other dictionary
entry as wrong. Inspection of its own saved rows shows what that costs:

    "built to add very little weight" -> want `lightweight`
        rank1 `light weight`   rank2 `lightweight`     scored as a MISS at k=1
    "safe to clean in a washing machine" -> want `Machine Washable`
        rank1 `wash care machine wash`  rank2 `machine washable`   scored as a MISS at k=1

Recall@1 is reported as exactly 0.000 over 712 examples, which is not a statement about
semantics; it is an artifact of a 7,922-entry dictionary holding many surface variants of
the same phrase.

THREE TIERS, REPORTED SEPARATELY, SO THE METRIC CANNOT BE ACCUSED OF FLATTERING US
----------------------------------------------------------------------------------
  exact      the current metric, unchanged, for continuity
  surface    two phrases are the same if they are identical after removing every
             non-alphanumeric character and folding trivial plurals. `light weight` ==
             `lightweight`. This is a STRING fact, not a semantic judgement.
  cluster    surface, plus the measurement-checked alias clusters.

Deliberately NOT merged: anything requiring a semantic decision. `slip on` is not treated
as `Pull On closure`, and `stretch` is not treated as `elastic`, even though a human would
accept both -- those are exactly the judgements Node 4 exists to be measured on, and
folding them into the metric would be marking our own homework.

Run:  PYTHONIOENCODING=utf-8 python -u robustness/v2/evaluate_node4_cluster_aware.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "robustness" / "v2"
DICT = V2 / "catalogue_attribute_dictionary.jsonl"
CLUSTERS = V2 / "catalogue_equivalence_clusters_measurement_checked.jsonl"
DEV = V2 / "sets" / "semantic_attribute_development_200.jsonl"
OUT = V2 / "results" / "node4_cluster_aware_v2_32.json"


def normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def surface_key(text: str) -> str:
    """Identity modulo whitespace, punctuation and trivial plurals. A STRING fact."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    folded = []
    for tok in tokens:
        if len(tok) > 3 and tok.endswith("es") and not tok.endswith("ses"):
            tok = tok[:-2]
        elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        folded.append(tok)
    return "".join(folded)


def load_encoder(device: str):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModel, AutoTokenizer
    snaps = glob.glob(str(ROOT / ".v2_model_cache" /
                          "models--sentence-transformers--all-MiniLM-L6-v2" /
                          "snapshots" / "*"))
    if not snaps:
        raise SystemExit("frozen MiniLM snapshot not found in .v2_model_cache")
    tok = AutoTokenizer.from_pretrained(snaps[0])
    model = AutoModel.from_pretrained(snaps[0]).to(device).eval()

    def encode(texts, bs=256):
        out = []
        for i in range(0, len(texts), bs):
            batch = tok(texts[i:i + bs], padding=True, truncation=True,
                        max_length=64, return_tensors="pt").to(device)
            with torch.no_grad():
                hidden = model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).float()
                vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out.append(torch.nn.functional.normalize(vec, dim=-1).cpu())
        return torch.cat(out).numpy()
    return encode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    import numpy as np
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    canonicals = [json.loads(l)["canonical"]
                  for l in DICT.open(encoding="utf-8") if l.strip()]
    norm_to_idx = {normalise(c): i for i, c in enumerate(canonicals)}

    surface_groups: dict[str, list[int]] = {}
    for i, c in enumerate(canonicals):
        surface_groups.setdefault(surface_key(c), []).append(i)
    collapsed = sum(len(v) - 1 for v in surface_groups.values() if len(v) > 1)
    print(f"dictionary {len(canonicals):,} phrases -> {len(surface_groups):,} surface "
          f"classes ({collapsed:,} entries are surface duplicates of another entry)")

    cluster_members: dict[int, set[int]] = {}
    n_clusters = 0
    if CLUSTERS.exists():
        for line in CLUSTERS.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            members = row.get("members") or row.get("catalogue_equivalents") or []
            idxs = {norm_to_idx[normalise(m)] for m in members
                    if normalise(m) in norm_to_idx}
            if len(idxs) > 1:
                n_clusters += 1
                for i in idxs:
                    cluster_members[i] = idxs
    print(f"verified alias clusters usable against the dictionary: {n_clusters}")

    examples = []
    for line in DEV.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        card = row.get("semantic_card") or {}
        for group in ("hard_constraints", "soft_preferences"):
            for atom in card.get(group, []):
                if atom.get("paraphrase") and atom.get("canonical"):
                    examples.append((str(atom["paraphrase"]), str(atom["canonical"])))
    print(f"benchmark atoms: {len(examples):,}\n")

    encode = load_encoder(device)
    matrix = encode(canonicals)
    queries = encode([q for q, _c in examples])
    order = np.argsort(-(queries @ matrix.T), axis=1)

    def accepted_set(canon: str, tier: str) -> set[int]:
        idx = norm_to_idx.get(normalise(canon))
        if idx is None:
            return set()
        if tier == "exact":
            return {idx}
        acc = set(surface_groups.get(surface_key(canonicals[idx]), [idx]))
        if tier == "cluster":
            for i in list(acc) + [idx]:
                acc |= cluster_members.get(i, set())
            for i in list(acc):
                acc |= set(surface_groups.get(surface_key(canonicals[i]), [i]))
        return acc

    report = {"experiment": "V2.32 cluster-aware Node 4 rescore (frozen MiniLM)",
              "model": "sentence-transformers/all-MiniLM-L6-v2 (frozen)",
              "dictionary_size": len(canonicals), "examples": len(examples),
              "surface_classes": len(surface_groups),
              "surface_duplicate_entries": collapsed,
              "verified_clusters_used": n_clusters, "tiers": {}}

    print(f"{'tier':<10}{'R@1':>9}{'R@3':>9}{'R@5':>9}{'R@10':>9}{'MRR':>9}")
    print("-" * 55)
    for tier in ("exact", "surface", "cluster"):
        ranks = []
        for (_q, canon), row in zip(examples, order):
            acc = accepted_set(canon, tier)
            if not acc:
                continue
            best = next((p + 1 for p, idx in enumerate(row) if int(idx) in acc),
                        len(canonicals))
            ranks.append(best)
        n = max(len(ranks), 1)
        t = {f"recall_at_{k}": round(sum(r <= k for r in ranks) / n, 6)
             for k in (1, 3, 5, 10)}
        t["mrr"] = round(sum(1.0 / r for r in ranks) / n, 6)
        t["scored"] = len(ranks)
        report["tiers"][tier] = t
        print(f"{tier:<10}{t['recall_at_1']:>9.4f}{t['recall_at_3']:>9.4f}"
              f"{t['recall_at_5']:>9.4f}{t['recall_at_10']:>9.4f}{t['mrr']:>9.4f}")

    e, c = report["tiers"]["exact"], report["tiers"]["cluster"]
    print("\n  the contract-specified metric vs the one that was actually run:")
    print(f"    R@1  {e['recall_at_1']:.4f} -> {c['recall_at_1']:.4f}")
    print(f"    R@5  {e['recall_at_5']:.4f} -> {c['recall_at_5']:.4f}")
    print(f"    MRR  {e['mrr']:.4f} -> {c['mrr']:.4f}")
    print("\n  This is the FROZEN encoder. No training was changed.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
