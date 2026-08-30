"""V2.37: Node 5 as zero-shot ENTAILMENT rather than trained equivalence.

THE ARGUMENT
------------
Node 5 is currently framed as symmetric equivalence -- "modern" == "contemporary",
"zipper" == "zip" -- and trained on a synonym corpus. Two measurements say that framing is
both too narrow and unnecessary.

TOO NARROW. At runtime the resolver's real question is not "are these the same phrase" but
"does this catalogue attribute SATISFY what the customer asked". That relation is
asymmetric and much larger:

    customer "leather"    <- satisfied by "genuine leather", "leather upper"
    customer "synthetic"  <- satisfied by "polyester", "nylon"
    customer "warm"       <- satisfied by "fleece lined", "insulated"

None of those are equivalences. An equivalence verifier rejects all of them.

And the symmetric alternative is dangerous, not merely weak: "cotton" and "polyester" sit
close in embedding space -- same domain, same contexts -- while being mutually exclusive.
A similarity threshold that accepts synonyms will also accept those. Directional entailment
cannot make that error, because "polyester entails cotton" is false in the direction that
matters.

UNNECESSARY. We measured the synonym surface at 4-8% of catalogue attributes (ConceptNet
coverage 8.1%; generated-corpus yield 3.8%), so a synonym corpus caps out low no matter how
much is generated. Entailment, by contrast, has ~400k human-annotated public pairs
(MultiNLI/SNLI) and off-the-shelf zero-shot cross-encoders. The data problem disappears.

WHAT THIS MEASURES
------------------
`cross-encoder/nli-deberta-v3-small`, zero-shot, against the frozen 134-row verifier test
that the fine-tuned encoders scored 0.7758 / 0.7840 / 0.7782 AUROC on. Same rows, same
metric, no training.

Two scorings are reported because they answer different questions:
  entail_max   max(P(entail A->B), P(entail B->A))  -- "either satisfies the other"
  entail_both  min of the two                        -- strict mutual entailment == synonymy

The frozen test is labelled for SYNONYMY, so `entail_both` is the like-for-like comparison.
`entail_max` is reported because it is the relation the resolver actually needs, and it is
expected to score differently -- that is the point, not a defect.

Run:  PYTHONIOENCODING=utf-8 python -u robustness/v2/evaluate_node5_nli_zeroshot.py
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "robustness" / "v2"
OUT = V2 / "results" / "node5_nli_zeroshot_v2_37.json"
TEST = V2 / "sets" / "frozen_equivalence_verification.jsonl"


def auroc(scores, labels):
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank, i, total = {}, 0, 0.0
    # average ranks for ties
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            total += avg if pairs[k][1] == 1 else 0.0
        i = j + 1
    return (total - pos * (pos + 1) / 2) / (pos * neg)


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    snaps = glob.glob(str(ROOT / ".v2_model_cache" /
                          "models--cross-encoder--nli-deberta-v3-small" / "snapshots" / "*"))
    path = snaps[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    labels = model.config.id2label
    print(f"model labels: {labels}   device {device}")
    ent_idx = next(i for i, l in labels.items() if "entail" in str(l).lower())

    rows = [json.loads(l) for l in TEST.open(encoding="utf-8") if l.strip()]
    y = [int(r["label"]) for r in rows]
    print(f"frozen verifier test: {len(rows)} rows, {sum(y)} positive, {len(y)-sum(y)} negative")

    def entail(prem, hyp, bs=64):
        out = []
        for i in range(0, len(prem), bs):
            b = tok(prem[i:i + bs], hyp[i:i + bs], padding=True, truncation=True,
                    max_length=64, return_tensors="pt").to(device)
            with torch.no_grad():
                p = torch.softmax(model(**b).logits, -1)[:, ent_idx]
            out.extend(p.cpu().tolist())
        return out

    # A bare attribute phrase is not a sentence; give the NLI model a minimal frame so the
    # premise/hypothesis are well-formed English rather than fragments.
    frame = lambda t: f"The product is {t}."
    A = [frame(r["canonical"]) for r in rows]
    B = [frame(r["candidate"]) for r in rows]
    ab, ba = entail(A, B), entail(B, A)
    both = [min(x, z) for x, z in zip(ab, ba)]
    mx = [max(x, z) for x, z in zip(ab, ba)]

    report = {"experiment": "V2.37 zero-shot NLI entailment for Node 5",
              "model": "cross-encoder/nli-deberta-v3-small", "training": "none (zero-shot)",
              "rows": len(rows), "positives": sum(y),
              "prior_finetuned_auroc": {"V2.05_frozen": 0.7758, "V2.30": 0.7840,
                                        "V2.31": 0.7782}}
    for name, s in (("entail_a_to_b", ab), ("entail_b_to_a", ba),
                    ("entail_both_min", both), ("entail_max", mx)):
        report[name] = round(auroc(s, y), 4)

    print(f"\n{'scoring':<20}{'AUROC':>9}")
    print("-" * 29)
    for k in ("entail_a_to_b", "entail_b_to_a", "entail_both_min", "entail_max"):
        print(f"{k:<20}{report[k]:>9.4f}")
    print(f"\n  prior fine-tuned encoders on these same rows: "
          f"0.7758 / 0.7840 / 0.7782")
    best = max(("entail_both_min", "entail_max"), key=lambda k: report[k])
    print(f"  zero-shot best ({best}): {report[best]:.4f}  "
          f"delta vs best fine-tuned {report[best] - 0.7840:+.4f}")
    print("\n  The test is labelled for SYNONYMY, so entail_both_min is the like-for-like")
    print("  comparison. entail_max scores the relation the resolver actually needs and is")
    print("  expected to differ.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
