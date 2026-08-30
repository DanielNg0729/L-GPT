"""V2.47: how deep must a candidate list go before it contains the right answer?

THE QUESTION, AND WHY IT DECIDES A DESIGN RATHER THAN RANKS A MODEL
------------------------------------------------------------------
V2.46 measured `generate`: the LLM deparaphrases from parametric knowledge with no
catalogue context, and a `df > 0` provenance gate accepts or discards the proposal. It
reached ~96% of the V2.43 oracle.

The unbuilt arm is `choose`: retrieve k candidates, put them in the prompt, let the model
pick one or answer NONE. Before building it, one number decides whether it CAN work.

    recall@k is a HARD CEILING on `choose`.

If the correct canonical is absent from the k candidates, no amount of LLM skill recovers
it -- the model can only pick from what it is shown. So `choose` at candidate-list-k can
never exceed recall@k, and that is measurable with zero API calls.

WHY THIS IS NOT THE STATISTIC THE ENCODERS WERE RETIRED ON. They were dismissed on top-1
(0/27, and V2.43 measured the encoder-driven resolver at -0.0202, BELOW the paraphrase
baseline). But a candidate list does not need top-1 precision; it needs recall. Retiring
them on top-1 was retiring them on the wrong statistic, and this pass corrects that --
which is why it runs even though the prior on encoders is poor.

V2.41 measured recall only to k=10, where dev200 sat at 0.0896-0.1493. This extends the
curve to the full index (7,922 canonicals) so the shape is visible: a curve that is still
climbing at k=100 says the signal exists and is merely diffuse; a curve that has flattened
says the encoder has no idea and depth will not save it.

TWO FAMILIES OF ENCODER
  incumbents      MiniLM-L6 / bge-small / e5-base / mpnet -- 22M-109M, BERT-era
  LLM-based       decoder-LLM backbones trained for retrieval, a genuinely different class

The second family is the substantive addition. The question "can we use the LLM's own
encoder" has no answer at the API -- `gpt-oss-120b` is decoder-only and Groq serves no
embedding endpoint -- but open-weight LLM embedders are the same idea in a usable form,
and they are the strongest available test of whether retrieval depth is the missing piece.

POOLING IS PER-MODEL AND MUST BE. Sentence-transformers models mean-pool; Qwen3-Embedding
and gte-Qwen2 take the LAST token and expect an instruction-prefixed query. Mean-pooling a
decoder embedder silently produces garbage and would understate it, so each model is used
the way its authors specify -- the same fairness rule V2.41 applied to E5/BGE prefixes.

WHAT THIS MAY NOT BE USED FOR. Choosing k. Recall@k rises monotonically in k, so "tuning"
it has a trivial direction and any argmax picked on a 27-phrase suite is noise. The curve
is reported whole; if `choose` is built, its k is decided end-to-end, not here.

Run:  PYTHONIOENCODING=utf-8 python -u robustness/v2/evaluate_recall_at_k_ceiling.py
"""
from __future__ import annotations

import importlib.util as ilu
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "robustness" / "v2"
CACHE = ROOT / ".v2_model_cache"
OUT = V2 / "results" / "recall_at_k_ceiling_v2_47.json"

_s = ilu.spec_from_file_location("_v2_32", V2 / "evaluate_node4_cluster_aware.py")
_m = ilu.module_from_spec(_s)
_s.loader.exec_module(_m)
normalise, surface_key = _m.normalise, _m.surface_key

_b = ilu.spec_from_file_location("_v2_41", V2 / "evaluate_node4_encoder_bakeoff.py")
_bm = ilu.module_from_spec(_b)
_b.loader.exec_module(_bm)
load_dev200, load_corpus = _bm.load_dev200, _bm.load_corpus

KS = (1, 3, 5, 10, 20, 50, 100, 200, 500, 1000)

# (label, repo, pooling, query prefix, doc prefix)
# `last` pooling + an instruction-prefixed query is what the Qwen embedders are trained
# for; mean-pooling them produces garbage and would understate them.
QWEN_INSTRUCT = ("Instruct: Given a shopper's description of a product attribute, retrieve "
                 "the catalogue's wording for that attribute\nQuery: ")
CANDIDATES = [
    ("MiniLM-L6 (incumbent)", "sentence-transformers/all-MiniLM-L6-v2", "mean", "", ""),
    ("all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2", "mean", "", ""),
    ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", "mean",
     "Represent this sentence for searching relevant passages: ", ""),
    ("e5-base-v2", "intfloat/e5-base-v2", "mean", "query: ", "passage: "),
    ("Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B", "last", QWEN_INSTRUCT, ""),
    ("gte-Qwen2-1.5B-instruct", "Alibaba-NLP/gte-Qwen2-1.5B-instruct", "last",
     QWEN_INSTRUCT, ""),
]


def main() -> None:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    canonicals = [json.loads(l)["canonical"]
                  for l in (V2 / "catalogue_attribute_dictionary.jsonl").open(encoding="utf-8")
                  if l.strip()]
    n2i = {normalise(c): i for i, c in enumerate(canonicals)}
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(canonicals):
        groups.setdefault(surface_key(c), []).append(i)
    sets = {"dev200 (riddles)": load_dev200(), "corpus (synonymy)": load_corpus()}
    for k, v in sets.items():
        print(f"  {k}: {len(v)} pairs")
    print(f"  index: {len(canonicals):,} canonicals\n")

    def build(repo, pooling):
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        tok = AutoTokenizer.from_pretrained(repo, cache_dir=str(CACHE),
                                            trust_remote_code=True)
        mdl = AutoModel.from_pretrained(
            repo, cache_dir=str(CACHE), trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()

        def enc(texts, prefix="", bs=128):
            outs = []
            for i in range(0, len(texts), bs):
                chunk = [prefix + t for t in texts[i:i + bs]]
                b = tok(chunk, padding=True, truncation=True, max_length=64,
                        return_tensors="pt").to(device)
                with torch.no_grad():
                    h = mdl(**b).last_hidden_state
                    if pooling == "last":
                        # left-padded or right-padded: take the last NON-PAD position
                        idx = b["attention_mask"].sum(1) - 1
                        v = h[torch.arange(h.size(0), device=h.device), idx]
                    else:
                        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
                        v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                    outs.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu())
            return torch.cat(outs).numpy()
        return enc, mdl

    report = {"experiment": "V2.47 recall@k ceiling for a `choose` candidate list",
              "index": len(canonicals), "ks": list(KS), "results": {}}
    hdr = f"{'encoder':<26}{'set':<20}" + "".join(f"{'R@'+str(k):>8}" for k in KS)
    print(hdr); print("-" * len(hdr))

    for label, repo, pooling, qpre, dpre in CANDIDATES:
        t0 = time.time()
        try:
            enc, mdl = build(repo, pooling)
            doc = enc(canonicals, dpre)
            report["results"][label] = {"repo": repo, "pooling": pooling}
            for sname, pairs in sets.items():
                pairs = [(q, c) for q, c in pairs if normalise(c) in n2i]
                qv = enc([q for q, _c in pairs], qpre)
                sims = qv @ doc.T
                order = np.argsort(-sims, axis=1)
                ranks = []
                for (_q, canon), row in zip(pairs, order):
                    idx = n2i[normalise(canon)]
                    acc = set(groups.get(surface_key(canonicals[idx]), [idx]))
                    ranks.append(next((p + 1 for p, j in enumerate(row) if int(j) in acc),
                                      len(canonicals)))
                n = max(len(ranks), 1)
                r = {f"R@{k}": round(sum(x <= k for x in ranks) / n, 4) for k in KS}
                r["MRR"] = round(sum(1.0 / x for x in ranks) / n, 4)
                r["median_rank"] = int(sorted(ranks)[len(ranks) // 2])
                report["results"][label][sname] = r
                print(f"{label:<26}{sname:<20}"
                      + "".join(f"{r['R@'+str(k)]:>8.4f}" for k in KS))
            del mdl
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"{'':<26}{'':<20}  ({time.time()-t0:.0f}s)")
        except Exception as exc:
            report["results"][label] = {"repo": repo,
                                        "error": f"{type(exc).__name__}: {exc}"}
            print(f"{label:<26}FAILED {type(exc).__name__}: {str(exc)[:80]}")

    ok = {k: v for k, v in report["results"].items() if "dev200 (riddles)" in v}
    if ok:
        print("\n  CEILING ON `choose`, from dev200 (the attribute-paraphrase case):")
        print(f"  {'encoder':<26}{'R@10':>9}{'R@50':>9}{'R@100':>9}"
              f"{'median rank':>13}{'still climbing?':>17}")
        print("  " + "-" * 83)
        for k, v in ok.items():
            d = v["dev200 (riddles)"]
            climb = d["R@1000"] - d["R@100"]
            print(f"  {k:<26}{d['R@10']:>9.4f}{d['R@50']:>9.4f}{d['R@100']:>9.4f}"
                  f"{d['median_rank']:>13}"
                  f"{('yes +%.3f' % climb) if climb > 0.02 else 'flat':>17}")
        best = max(ok, key=lambda k: ok[k]["dev200 (riddles)"]["R@100"])
        bd = ok[best]["dev200 (riddles)"]
        print(f"\n  best R@100: {best} at {bd['R@100']:.4f}")
        print(f"  `choose` with a 100-candidate list therefore cannot exceed "
              f"{bd['R@100']:.1%} on this suite,")
        print(f"  against `generate` (no candidates at all) already reaching ~96% of the")
        print(f"  V2.43 oracle end to end. Read that as the design verdict, not as a score.")
        print("\n  Delete any checkpoint this run rejects from .v2_model_cache.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
