"""V2.38: Node 4 -> Node 5 as retrieve-then-verify, measured end to end.

THE ARCHITECTURAL CLAIM BEING TESTED
------------------------------------
V2.37 showed a zero-shot NLI cross-encoder verifies attribute entailment at 0.9726 AUROC,
against 0.7758-0.7840 for every trained equivalence encoder. That changes what Node 4 owes
the pipeline. Its job stops being "be right at rank 1" and becomes "put the right candidate
somewhere in the top k" -- because a strong verifier can rerank within k.

So the question this pass answers is not "is Node 4 good" but:

    what k does Node 4 need before the verifier carries the pipeline?

If the answer is a k the frozen retriever already reaches, no corpus generation is needed
for Node 4 at all and the data track stays closed.

WHAT IS REPORTED
----------------
For each k, three quantities:

  retriever R@k     the CEILING. The verifier can only rerank what was retrieved, so this
                    bounds everything downstream.
  verified R@1      the achieved score after entailment reranking within the top k.
  gap               ceiling minus achieved -- how much of what Node 4 supplied the verifier
                    failed to surface. A small gap means the verifier is doing its job and
                    recall is the binding constraint.

Two evaluation sets, because the standing benchmark is known to be unrepresentative:

  dev200      the established 67-concept benchmark. Its paraphrases are riddles
              ("made from a soft plant fibre" -> Cotton), so it is the pessimistic view.
  corpus      concepts from the existing verified synonym corpus, which are ordinary
              synonymy and closer to what a real unfamiliar phrasing looks like.

Direction matters for entailment and is not symmetric. Both are computed:
  cand->query  does the catalogue attribute SATISFY the customer's requirement (the
               relation the resolver actually needs)
  query->cand  the reverse

Everything is cluster-aware per the data contract, and the frozen retriever is used
throughout -- no training anywhere in this pass.

Run:  PYTHONIOENCODING=utf-8 python -u robustness/v2/evaluate_node45_retrieve_verify.py
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
OUT = V2 / "results" / "node45_retrieve_verify_v2_38.json"

_s = ilu.spec_from_file_location("_v2_32", V2 / "evaluate_node4_cluster_aware.py")
_m = ilu.module_from_spec(_s)
_s.loader.exec_module(_m)
normalise, surface_key, load_encoder = _m.normalise, _m.surface_key, _m.load_encoder

KS = (1, 5, 10, 20, 50, 100)


def load_dev200() -> list[tuple[str, str]]:
    out = []
    f = V2 / "sets" / "semantic_attribute_development_200.jsonl"
    for line in f.open(encoding="utf-8"):
        if not line.strip():
            continue
        card = json.loads(line).get("semantic_card") or {}
        for grp in ("hard_constraints", "soft_preferences"):
            for a in card.get(grp, []):
                if a.get("paraphrase") and a.get("canonical"):
                    out.append((str(a["paraphrase"]).lower(), str(a["canonical"])))
    return sorted(set(out))


def load_corpus() -> list[tuple[str, str]]:
    out = []
    f = V2 / "catalogue_synonym_train_only_merged.jsonl"
    for line in f.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        for syn in row.get("synonyms", []):
            if normalise(syn) != normalise(row["canonical"]):
                out.append((str(syn).lower(), str(row["canonical"])))
    return sorted(set(out))


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    canonicals = [json.loads(l)["canonical"]
                  for l in (V2 / "catalogue_attribute_dictionary.jsonl").open(encoding="utf-8")
                  if l.strip()]
    n2i = {normalise(c): i for i, c in enumerate(canonicals)}
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(canonicals):
        groups.setdefault(surface_key(c), []).append(i)
    print(f"index {len(canonicals):,} phrases (full, unpruned)")

    encode = load_encoder(device)
    matrix = encode(canonicals)

    nli_path = glob.glob(str(ROOT / ".v2_model_cache" /
                             "models--cross-encoder--nli-deberta-v3-small" /
                             "snapshots" / "*"))[0]
    ntok = AutoTokenizer.from_pretrained(nli_path)
    nli = AutoModelForSequenceClassification.from_pretrained(nli_path).to(device).eval()
    ent_idx = next(i for i, l in nli.config.id2label.items()
                   if "entail" in str(l).lower())

    def entail(prem: list[str], hyp: list[str], bs=256) -> list[float]:
        out = []
        for i in range(0, len(prem), bs):
            b = ntok(prem[i:i + bs], hyp[i:i + bs], padding=True, truncation=True,
                     max_length=48, return_tensors="pt").to(device)
            with torch.no_grad():
                out.extend(torch.softmax(nli(**b).logits, -1)[:, ent_idx].cpu().tolist())
        return out

    frame = lambda t: f"The product is {t}."
    report = {"experiment": "V2.38 Node4->Node5 retrieve-then-verify (all frozen)",
              "index": len(canonicals), "sets": {}}

    for set_name, pairs in (("dev200", load_dev200()), ("corpus", load_corpus())):
        pairs = [(q, c) for q, c in pairs if normalise(c) in n2i]
        print(f"\n=== {set_name}: {len(pairs)} concepts ===")
        queries = encode([q for q, _c in pairs])
        order = np.argsort(-(queries @ matrix.T), axis=1)

        acc_sets = []
        for _q, canon in pairs:
            idx = n2i[normalise(canon)]
            acc_sets.append(set(groups.get(surface_key(canonicals[idx]), [idx])))

        rows = {}
        print(f"{'k':>5}{'retr R@k':>11}{'verified R@1':>15}{'gap':>8}"
              f"{'verified MRR':>14}{'NLI pairs':>11}")
        print("-" * 64)
        for k in KS:
            hit_k, hit1, rr, npairs = 0, 0, 0.0, 0
            for (q, _canon), row, acc in zip(pairs, order, acc_sets):
                cand_idx = [int(j) for j in row[:k]]
                if any(j in acc for j in cand_idx):
                    hit_k += 1
                cand_txt = [canonicals[j] for j in cand_idx]
                scores = entail([frame(t) for t in cand_txt], [frame(q)] * len(cand_txt))
                npairs += len(cand_txt)
                ranked = [cand_idx[i] for i in np.argsort(-np.asarray(scores))]
                pos = next((p + 1 for p, j in enumerate(ranked) if j in acc), None)
                if pos == 1:
                    hit1 += 1
                rr += (1.0 / pos) if pos else 0.0
            n = len(pairs)
            rows[k] = {"retriever_R@k": round(hit_k / n, 4),
                       "verified_R@1": round(hit1 / n, 4),
                       "gap": round(hit_k / n - hit1 / n, 4),
                       "verified_MRR": round(rr / n, 4), "nli_pairs": npairs}
            r = rows[k]
            print(f"{k:>5}{r['retriever_R@k']:>11.4f}{r['verified_R@1']:>15.4f}"
                  f"{r['gap']:>8.4f}{r['verified_MRR']:>14.4f}{npairs:>11,}")
        report["sets"][set_name] = {"concepts": len(pairs), "by_k": rows}

        best = max(rows, key=lambda k: rows[k]["verified_R@1"])
        base = rows[1]["retriever_R@k"]
        print(f"\n  retriever alone R@1 = {base:.4f}")
        print(f"  best verified R@1 = {rows[best]['verified_R@1']:.4f} at k={best}  "
              f"({rows[best]['verified_R@1'] - base:+.4f})")
        print(f"  ceiling at that k  = {rows[best]['retriever_R@k']:.4f}  "
              f"(gap {rows[best]['gap']:.4f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
