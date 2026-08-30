"""V2.42: cross-encoder reranking for Node 4 -- the last untried retrieval lever.

WHY A CROSS-ENCODER, AND WHY NOW
---------------------------------
Everything tried so far scores the query and the candidate INDEPENDENTLY and compares
vectors. A bi-encoder must compress "made from a soft plant fibre" into one vector before
it has seen "cotton", so it can never condition its representation on the candidate. A
cross-encoder reads both together and can attend across them, which is exactly the regime
where definitional paraphrase might be recoverable.

Three rerankers, all zero-shot over the frozen bi-encoder's top-k:

    ms-marco-MiniLM-L6-v2   general web relevance, 84M downloads
    ESCI e-commerce         trained on Amazon Shopping Queries -- the closest public
                            domain match that exists for this task
    nli-deberta-v3-small    the entailment verifier from V2.37, for continuity

WHAT WOULD MAKE THIS THE ANSWER, AND WHAT WOULD CLOSE THE QUESTION
-------------------------------------------------------------------
V2.41 established the constraint precisely: on ordinary synonymy the bi-encoder reaches
R@1 0.44, but on the definitional riddles the suites actually contain it reaches 0.09, and
adding candidates at 9% precision costs more than it gains. So a reranker has to raise
precision at rank 1 on the RIDDLE set specifically. Raising it on the synonymy set changes
nothing we did not already have.

The retrieval ceiling is not the issue and is reported to keep that visible: the frozen
bi-encoder already puts the right answer in its top 50 for a large share of queries. The
whole question is whether anything can pick it out.

If all three fail, the conclusion is not "we picked bad models" -- it is that resolving a
definitional paraphrase to a catalogue attribute needs world knowledge that no local
ranking model carries, which is a finding rather than a gap.

Run:  PYTHONIOENCODING=utf-8 python -u robustness/v2/evaluate_node4_cross_encoder.py
"""
from __future__ import annotations

import glob
import importlib.util as ilu
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "robustness" / "v2"
CACHE = ROOT / ".v2_model_cache"
OUT = V2 / "results" / "node4_cross_encoder_v2_42.json"

_s = ilu.spec_from_file_location("_v2_32", V2 / "evaluate_node4_cluster_aware.py")
_m = ilu.module_from_spec(_s)
_s.loader.exec_module(_m)
normalise, surface_key = _m.normalise, _m.surface_key

BI = "sentence-transformers/all-mpnet-base-v2"          # V2.41 winner
RERANKERS = [
    ("ms-marco-MiniLM-L6", "cross-encoder/ms-marco-MiniLM-L6-v2", "rank"),
    ("ESCI e-commerce", "2013khansohail/cartographer-ecommerce-reranker-MiniLM-L6-v2", "rank"),
    ("nli-deberta-v3-small", "cross-encoder/nli-deberta-v3-small", "entail"),
]
TOPK = 50


def load_sets():
    dev = []
    for line in (V2 / "sets" / "semantic_attribute_development_200.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        card = json.loads(line).get("semantic_card") or {}
        for grp in ("hard_constraints", "soft_preferences"):
            for a in card.get(grp, []):
                if a.get("paraphrase") and a.get("canonical"):
                    dev.append((str(a["paraphrase"]).lower(), str(a["canonical"])))
    cor = []
    for line in (V2 / "catalogue_synonym_train_only_merged.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        for syn in row.get("synonyms", []):
            if normalise(syn) != normalise(row["canonical"]):
                cor.append((str(syn).lower(), str(row["canonical"])))
    return {"dev200 (riddles)": sorted(set(dev)), "corpus (synonymy)": sorted(set(cor))}


def main() -> None:
    import numpy as np
    import torch
    from transformers import (AutoModel, AutoModelForSequenceClassification,
                              AutoTokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    canonicals = [json.loads(l)["canonical"]
                  for l in (V2 / "catalogue_attribute_dictionary.jsonl").open(encoding="utf-8")
                  if l.strip()]
    n2i = {normalise(c): i for i, c in enumerate(canonicals)}
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(canonicals):
        groups.setdefault(surface_key(c), []).append(i)

    btok = AutoTokenizer.from_pretrained(BI, cache_dir=str(CACHE))
    bmdl = AutoModel.from_pretrained(BI, cache_dir=str(CACHE)).to(device).eval()

    def embed(texts, bs=256):
        outs = []
        for i in range(0, len(texts), bs):
            b = btok(texts[i:i + bs], padding=True, truncation=True, max_length=64,
                     return_tensors="pt").to(device)
            with torch.no_grad():
                h = bmdl(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                outs.append(torch.nn.functional.normalize(v, dim=-1).cpu())
        return torch.cat(outs).numpy()

    matrix = embed(canonicals)
    sets = load_sets()
    report = {"experiment": "V2.42 cross-encoder rerank for Node 4",
              "bi_encoder": BI, "topk": TOPK, "results": {}}

    def metrics(ranks, n):
        r = {f"R@{k}": round(sum(x <= k for x in ranks) / n, 4) for k in (1, 3, 5)}
        r["MRR"] = round(sum(1.0 / x for x in ranks) / n, 4)
        return r

    print(f"bi-encoder: {BI}   rerank depth {TOPK}\n")
    prepared = {}
    for sname, pairs in sets.items():
        pairs = [(q, c) for q, c in pairs if normalise(c) in n2i]
        qv = embed([q for q, _c in pairs])
        order = np.argsort(-(qv @ matrix.T), axis=1)[:, :TOPK]
        accs = []
        for _q, canon in pairs:
            idx = n2i[normalise(canon)]
            accs.append(set(groups.get(surface_key(canonicals[idx]), [idx])))
        base_ranks = []
        for row, acc in zip(order, accs):
            base_ranks.append(next((p + 1 for p, j in enumerate(row) if int(j) in acc),
                                   TOPK + 1))
        prepared[sname] = (pairs, order, accs)
        m = metrics(base_ranks, len(pairs))
        ceiling = sum(1 for r in base_ranks if r <= TOPK) / len(pairs)
        report["results"].setdefault("bi-encoder only", {})[sname] = {**m, "ceiling": round(ceiling, 4)}
        print(f"bi-encoder only      {sname:<20} R@1 {m['R@1']:.4f}  R@5 {m['R@5']:.4f}  "
              f"MRR {m['MRR']:.4f}   ceiling@{TOPK} {ceiling:.4f}")

    for label, repo, mode in RERANKERS:
        try:
            tok = AutoTokenizer.from_pretrained(repo, cache_dir=str(CACHE))
            mdl = AutoModelForSequenceClassification.from_pretrained(
                repo, cache_dir=str(CACHE)).to(device).eval()
            ent_idx = None
            if mode == "entail":
                ent_idx = next(i for i, l in mdl.config.id2label.items()
                               if "entail" in str(l).lower())
            print()
            for sname, (pairs, order, accs) in prepared.items():
                ranks = []
                for (q, _c), row, acc in zip(pairs, order, accs):
                    cand = [canonicals[int(j)] for j in row]
                    b = tok([q] * len(cand), cand, padding=True, truncation=True,
                            max_length=48, return_tensors="pt").to(device)
                    with torch.no_grad():
                        lg = mdl(**b).logits
                        sc = (torch.softmax(lg, -1)[:, ent_idx] if ent_idx is not None
                              else lg.squeeze(-1)).cpu().numpy()
                    ranked = [int(row[i]) for i in np.argsort(-sc)]
                    ranks.append(next((p + 1 for p, j in enumerate(ranked) if j in acc),
                                      TOPK + 1))
                m = metrics(ranks, len(pairs))
                report["results"].setdefault(label, {})[sname] = m
                bi = report["results"]["bi-encoder only"][sname]
                print(f"{label:<20} {sname:<20} R@1 {m['R@1']:.4f} ({m['R@1']-bi['R@1']:+.4f})"
                      f"  R@5 {m['R@5']:.4f}  MRR {m['MRR']:.4f} ({m['MRR']-bi['MRR']:+.4f})")
            del mdl
            torch.cuda.empty_cache()
        except Exception as exc:
            report["results"][label] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"\n{label:<20} FAILED {type(exc).__name__}: {str(exc)[:80]}")

    print("\n  The decisive column is dev200 R@1 -- definitional paraphrase. Gains confined")
    print("  to the synonymy set do not address the measured constraint.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
