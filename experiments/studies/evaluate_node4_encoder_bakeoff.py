"""V2.41: does a stronger base encoder move Node 4?

WHY THIS IS WORTH TESTING NOW
------------------------------
Every Node 4 number so far comes from `all-MiniLM-L6-v2` -- 6 layers, 384 dimensions, 2021.
It was chosen as a default, never compared. And its behaviour splits sharply by benchmark:

    dev200 (riddle paraphrases)     R@1 0.0448   R@5 0.1194
    corpus (real synonymy)          R@1 0.3681   R@5 0.5769

On realistic synonymy it is mediocre rather than hopeless, which is exactly the regime where
a better retriever should pay. Four defects have already been removed from the measurement
(V2.32 metric, V2.33 training, V2.34/35 index, V2.40 evidence form); the encoder is the
remaining untested variable.

CANDIDATES, all frozen, no fine-tuning anywhere in this pass:

    all-MiniLM-L6-v2      22M   the incumbent
    bge-small-en-v1.5     33M   MTEB-era retrieval model, 67M downloads
    e5-base-v2           109M   needs "query:"/"passage:" prefixes -- applied
    all-mpnet-base-v2    109M   the strong sentence-transformers baseline

PREFIX CONVENTIONS ARE HONOURED. E5 is trained with asymmetric "query: " / "passage: "
prefixes and scores materially worse without them; BGE recommends a query instruction for
retrieval. Omitting these would understate both models and make the comparison unfair, so
each model is used the way its authors specify.

EVALUATED ON BOTH BENCHMARKS, because they disagree and the disagreement is informative:
dev200 is the pessimistic riddle case, corpus is ordinary synonymy. Cluster-aware metric
throughout (V2.32). Losers are deleted after the run -- see the report line.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/studies/evaluate_node4_encoder_bakeoff.py
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
V2 = ROOT / "experiments" / "studies"
CACHE = ROOT / ".v2_model_cache"
OUT = V2 / "results" / "node4_encoder_bakeoff_v2_41.json"

_s = ilu.spec_from_file_location("_v2_32", V2 / "evaluate_node4_cluster_aware.py")
_m = ilu.module_from_spec(_s)
_s.loader.exec_module(_m)
normalise, surface_key = _m.normalise, _m.surface_key

# (label, repo, query prefix, document prefix) -- prefixes per each model's own card
CANDIDATES = [
    ("MiniLM-L6 (incumbent)", "sentence-transformers/all-MiniLM-L6-v2", "", ""),
    ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5",
     "Represent this sentence for searching relevant passages: ", ""),
    ("e5-base-v2", "intfloat/e5-base-v2", "query: ", "passage: "),
    ("all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2", "", ""),
]


def load_dev200():
    out = []
    for line in (V2 / "sets" / "semantic_attribute_development_200.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        card = json.loads(line).get("semantic_card") or {}
        for grp in ("hard_constraints", "soft_preferences"):
            for a in card.get(grp, []):
                if a.get("paraphrase") and a.get("canonical"):
                    out.append((str(a["paraphrase"]).lower(), str(a["canonical"])))
    return sorted(set(out))


def load_corpus():
    out = []
    for line in (V2 / "catalogue_synonym_train_only_merged.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        for syn in row.get("synonyms", []):
            if normalise(syn) != normalise(row["canonical"]):
                out.append((str(syn).lower(), str(row["canonical"])))
    return sorted(set(out))


def main() -> None:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    canonicals = [json.loads(l)["canonical"]
                  for l in (V2 / "catalogue_attribute_dictionary.jsonl").open(encoding="utf-8")
                  if l.strip()]
    n2i = {normalise(c): i for i, c in enumerate(canonicals)}
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(canonicals):
        groups.setdefault(surface_key(c), []).append(i)
    sets = {"dev200 (riddles)": load_dev200(), "corpus (synonymy)": load_corpus()}
    for k, v in sets.items():
        print(f"{k}: {len(v)} concepts")
    print(f"index: {len(canonicals):,}\n")

    def encoder(repo):
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        tok = AutoTokenizer.from_pretrained(repo, cache_dir=str(CACHE))
        mdl = AutoModel.from_pretrained(repo, cache_dir=str(CACHE)).to(device).eval()

        def enc(texts, prefix="", bs=256):
            outs = []
            for i in range(0, len(texts), bs):
                chunk = [prefix + t for t in texts[i:i + bs]]
                b = tok(chunk, padding=True, truncation=True, max_length=64,
                        return_tensors="pt").to(device)
                with torch.no_grad():
                    h = mdl(**b).last_hidden_state
                    m = b["attention_mask"].unsqueeze(-1).float()
                    v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                    outs.append(torch.nn.functional.normalize(v, dim=-1).cpu())
            return torch.cat(outs).numpy()
        return enc, mdl

    report = {"experiment": "V2.41 frozen encoder bake-off for Node 4",
              "index": len(canonicals), "results": {}}
    header = f"{'encoder':<24}{'set':<20}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}"
    print(header)
    print("-" * len(header))

    for label, repo, qpre, dpre in CANDIDATES:
        try:
            enc, mdl = encoder(repo)
            doc = enc(canonicals, dpre)
            report["results"][label] = {"repo": repo}
            for sname, pairs in sets.items():
                pairs = [(q, c) for q, c in pairs if normalise(c) in n2i]
                qv = enc([q for q, _c in pairs], qpre)
                order = np.argsort(-(qv @ doc.T), axis=1)
                ranks = []
                for (_q, canon), row in zip(pairs, order):
                    idx = n2i[normalise(canon)]
                    acc = set(groups.get(surface_key(canonicals[idx]), [idx]))
                    ranks.append(next((p + 1 for p, j in enumerate(row) if int(j) in acc),
                                      len(canonicals)))
                n = max(len(ranks), 1)
                r = {f"R@{k}": round(sum(x <= k for x in ranks) / n, 4)
                     for k in (1, 3, 5, 10)}
                r["MRR"] = round(sum(1.0 / x for x in ranks) / n, 4)
                report["results"][label][sname] = r
                print(f"{label:<24}{sname:<20}{r['R@1']:>8.4f}{r['R@3']:>8.4f}"
                      f"{r['R@5']:>8.4f}{r['R@10']:>8.4f}{r['MRR']:>8.4f}")
            del mdl
            torch.cuda.empty_cache()
        except Exception as exc:
            report["results"][label] = {"repo": repo, "error": f"{type(exc).__name__}: {exc}"}
            print(f"{label:<24}FAILED {type(exc).__name__}: {str(exc)[:70]}")

    ok = {k: v for k, v in report["results"].items() if "corpus (synonymy)" in v}
    if ok:
        base = ok.get("MiniLM-L6 (incumbent)", {})
        print(f"\n{'encoder':<24}{'corpus MRR':>12}{'vs incumbent':>14}"
              f"{'dev200 MRR':>12}{'vs incumbent':>14}")
        print("-" * 76)
        for k, v in ok.items():
            cm = v["corpus (synonymy)"]["MRR"]
            dm = v["dev200 (riddles)"]["MRR"]
            bc = base.get("corpus (synonymy)", {}).get("MRR", cm)
            bd = base.get("dev200 (riddles)", {}).get("MRR", dm)
            print(f"{k:<24}{cm:>12.4f}{cm-bc:>+14.4f}{dm:>12.4f}{dm-bd:>+14.4f}")
        best = max(ok, key=lambda k: ok[k]["corpus (synonymy)"]["MRR"])
        print(f"\n  best on realistic synonymy: {best}")
        print("  Delete the losing checkpoints from .v2_model_cache after reviewing this.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
