"""V2.59: can the bi-encoder tell the resolver's good answers from its harmful ones?

THE PROBLEM, STATED AS A SEPARATION TASK
-----------------------------------------
On the open-vocabulary attribute suite the resolver answers 136 of 233 paraphrases. Of
those, ~77 share at least one token with the true atom -- which is what the ranker actually
rewards -- and ~59 share nothing at all. The weight is already optimal (V2.57: raising it
collapses the arm from 17.2% to 5.2% of the gap, because wrong proposals outrank the
correct evidence beside them), so the ONLY remaining lever on this axis is the error rate.

Dropping the 59 while keeping the 77 is Node 5's job. It is not Node 5's mechanism.

WHY NOT THE OLD NODE 5. It produced a continuous entailment score and needed a CALIBRATED
THRESHOLD. Calibrated on synonym pairs, applied to paraphrase->canonical, it discarded 76%
of CORRECT proposals and cut the gain from +0.0169 to +0.0009. The failure was threshold
transfer, so any retry must avoid depending on a new one.

THE SIGNAL: CORROBORATION BY AN INDEPENDENT MECHANISM
-----------------------------------------------------
The bi-encoder was rejected for SELECTION -- its top-1 was 0/27 and the encoder-driven
resolver scored BELOW the paraphrase baseline. But selection is not what it is good at.
Its recall@100 is 0.54-0.94 depending on the surface: the right answer is usually IN its
ranking, it simply cannot pick it out. "Is the LLM's answer somewhere near the top of that
ranking?" is a RECALL question, which is its strength.

Two independent mechanisms agreeing is much stronger evidence than either alone, and rank
is ordinal rather than a score to calibrate.

MEASURED AS A RANK, NOT AS MEMBERSHIP. The naive form -- "is the proposal in the encoder's
top-k dictionary entries" -- is confounded: a proposal that is catalogue-attested but not a
dictionary member can never be in the list, so it would be rejected for the wrong reason.
Instead the proposal is scored directly against the paraphrase and RANKED against the whole
dictionary. Rank 1 means nothing in 7,920 canonical values is a better match.

THE RETRIEVER IS THE ONE SELECTED ON TRAIN-ONLY DATA. e5-base-v2, chosen in V2.47 on the
train-only synonymy corpus (R@100 0.9396, best of six) and NOT on any evaluation surface --
where it is in fact the worst of the six. The handicap is deliberate and is carried here.

WHAT THIS RUN MAY AND MAY NOT CONCLUDE. It may report whether a separation EXISTS, as a
full curve over k plus an AUROC. It may NOT pick a k: that would be selecting an operating
point on the evaluation suite, which is the trap this project has already fallen into once
today. If a separation exists, k is chosen on train-only data in a follow-up and applied
unchanged.

Fully offline and deterministic. No LLM calls: the proposals are replayed from V2.57.

Run:  PYTHONIOENCODING=utf-8 python -u robustness/v2/evaluate_corroboration_filter.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "robustness" / "v2"
CACHE = ROOT / ".v2_model_cache"
SRC = V2 / "results" / "attr_accuracy_vs_weight_v2_57.json"
OUT = V2 / "results" / "corroboration_filter_v2_59.json"

RETRIEVER = "intfloat/e5-base-v2"      # selected on the train-only corpus in V2.47
QPRE, DPRE = "query: ", "passage: "     # E5 is trained with asymmetric prefixes
KS = (1, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)


def auroc(scores, labels):
    """P(a random positive ranks above a random negative), ties averaged."""
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return float("nan")
    total, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1
        total += sum(avg for k in range(i, j + 1) if pairs[k][1] == 1)
        i = j + 1
    return (total - pos * (pos + 1) / 2) / (pos * neg)


def main() -> None:
    os.environ["LLM_RERANK"] = "0"
    os.environ["LLM_EXTRACT"] = "0"
    os.environ["LLM_RESOLVE"] = "0"
    if not SRC.exists():
        print(f"missing {SRC} -- run V2.57 first."); return
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    from submission.agent import raw_toks

    detail = json.loads(SRC.read_text(encoding="utf-8"))["detail"]
    answered = [d for d in detail if d.get("proposal")]
    useful = [d for d in answered if d.get("overlap")]
    harmful = [d for d in answered if not d.get("overlap")]
    print(f"replayed from V2.57: {len(detail)} paraphrases, {len(answered)} answered")
    print(f"  useful  (>=1 token overlap with the true atom): {len(useful)}")
    print(f"  harmful (no overlap at all):                    {len(harmful)}")
    if not useful or not harmful:
        print("one class is empty; nothing to separate."); return

    canonicals = [json.loads(l)["canonical"]
                  for l in (V2 / "catalogue_attribute_dictionary.jsonl").open(encoding="utf-8")
                  if l.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(RETRIEVER, cache_dir=str(CACHE))
    mdl = AutoModel.from_pretrained(RETRIEVER, cache_dir=str(CACHE)).to(device).eval()

    def embed(texts, prefix, bs=256):
        outs = []
        for i in range(0, len(texts), bs):
            b = tok([prefix + t for t in texts[i:i + bs]], padding=True, truncation=True,
                    max_length=64, return_tensors="pt").to(device)
            with torch.no_grad():
                h = b["attention_mask"].unsqueeze(-1).float()
                v = (mdl(**b).last_hidden_state * h).sum(1) / h.sum(1).clamp(min=1e-9)
                outs.append(torch.nn.functional.normalize(v, dim=-1).cpu())
        return torch.cat(outs).numpy()

    t0 = time.time()
    doc = embed(canonicals, DPRE)
    paras = [d["paraphrase"] for d in answered]
    props = [d["proposal"] for d in answered]
    qv = embed(paras, QPRE)
    pv = embed(props, DPRE)          # the proposal is scored as a DOCUMENT, like the dict

    # The rank the proposal would occupy among all 7,920 canonicals for this paraphrase.
    # Rank 1 = nothing in the dictionary matches the paraphrase better than the proposal.
    sims_dict = qv @ doc.T                                   # (n, 7920)
    sims_prop = (qv * pv).sum(axis=1)                        # (n,)
    ranks = 1 + (sims_dict > sims_prop[:, None]).sum(axis=1)

    labels = [1 if d.get("overlap") else 0 for d in answered]
    # Lower rank is better, so negate for AUROC's "higher = positive" convention.
    a = auroc([-int(r) for r in ranks], labels)
    u = [int(r) for r, l in zip(ranks, labels) if l]
    h = [int(r) for r, l in zip(ranks, labels) if not l]
    print(f"\n  encoder: {RETRIEVER} (train-only selected), {len(canonicals):,} canonicals")
    print(f"  AUROC, useful vs harmful, by proposal rank: {a:.4f}")
    print(f"  median rank  useful {sorted(u)[len(u)//2]:>6}   harmful {sorted(h)[len(h)//2]:>6}")
    print(f"  mean   rank  useful {sum(u)/len(u):>6.0f}   harmful {sum(h)/len(h):>6.0f}")

    print(f"\n  {'keep if rank <=':<16}{'useful kept':>14}{'harmful kept':>14}"
          f"{'harmful dropped':>17}")
    print("  " + "-" * 61)
    rows = []
    for k in KS:
        uk = sum(1 for r in u if r <= k)
        hk = sum(1 for r in h if r <= k)
        rows.append({"k": k, "useful_kept": uk, "harmful_kept": hk,
                     "harmful_dropped": len(h) - hk})
        print(f"  {k:<16}{uk:>6}/{len(u):<7}{hk:>6}/{len(h):<7}"
              f"{len(h)-hk:>10}/{len(h)}")

    print(f"\n  A USEFUL FILTER keeps most of the {len(u)} useful and drops most of the")
    print(f"  {len(h)} harmful. If every k trades them off one-for-one, the encoder")
    print(f"  carries no signal about which answers are right and this line is closed.")
    print(f"  AUROC 0.50 means exactly that; 1.00 means perfect separation.")
    print(f"\n  No k is selected here. Choosing one on this suite would be picking an")
    print(f"  operating point on the evaluation data -- the error already made once today")
    print(f"  with the cue regexes. If the separation is real, k comes from train-only data.")
    print(f"\n  {time.time()-t0:.0f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "V2.59 encoder corroboration as a resolver filter",
        "retriever": RETRIEVER, "selected_on": "train-only synonymy corpus (V2.47)",
        "answered": len(answered), "useful": len(u), "harmful": len(h),
        "auroc": round(float(a), 4),
        "median_rank": {"useful": sorted(u)[len(u)//2], "harmful": sorted(h)[len(h)//2]},
        "curve": rows,
        "detail": [{"paraphrase": d["paraphrase"], "atom": d["atom"],
                    "proposal": d["proposal"], "overlap": bool(d.get("overlap")),
                    "rank": int(r)} for d, r in zip(answered, ranks)],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
