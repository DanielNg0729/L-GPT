"""V2.64: does the span extractor find values it has never seen, in wrappers it has never seen?

THE MEASUREMENT THAT DECIDES WHETHER THE MODEL IS REAL
------------------------------------------------------
Training randomised the value vocabulary on purpose (V2.62), so the model should have
learned slot POSITION rather than attribute-shaped words. Whether it actually did is not
something the training loss can say, and dev shares templates with train so dev cannot say
it either.

Two conditions over the SAME held-out template bank -- template-disjoint from training,
verified zero shared templates -- differing only in what sits in the value slot:

  canonical     the real catalogue values: `Cotton`, `Leather`, `Imported`
  paraphrased   the open-vocabulary paraphrases: `made from a soft plant fibre`

THE CONTRAST IS THE RESULT. A model that learned position scores similarly on both. A model
that memorised vocabulary scores well on canonical and collapses on paraphrased -- and
would have looked excellent in any evaluation that held values constant, which is the exact
trap this project already fell into when it measured template recovery with canonical
values.

WHY SEMANTICALLY WRONG SUBSTITUTIONS ARE FINE HERE. The paraphrase inserted in a slot need
not be the correct paraphrase OF that value. This measures LOCALISATION -- can the model
mark the span -- not resolution. Any paraphrase-shaped string exercises the same skill, and
using all 204 gives full coverage instead of the handful that happen to match this bank's
values. The 204 were never in training; the training prose pool came from catalogue
`features` text only.

REPORTED AGAINST THE DEGENERATE OPTIMA, because VALUE tokens are a minority and token
accuracy rewards silence:

    predict-nothing        scores well on accuracy, zero on recall
    predict-everything     perfect recall, useless precision

PER-ACTION, and `no_evidence` is the row that matters most. Those turns carry no value at
all, so any VALUE span there is a false positive that would send filler to the resolver.
The span node had exactly this bug until facet names were excluded from its dictionary.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/studies/evaluate_span_bert_heldout.py
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "experiments" / "studies"
CACHE = ROOT / ".v2_model_cache"
RUN = CACHE / os.environ.get("SPAN_RUN", "span_extractor_valrand")
TEST = V2 / "v1_turn_gated_bank" / "final_test.jsonl"
OUT = V2 / "results" / "span_bert_heldout_v2_64.json"

SLOT_RE = re.compile(r"\{(\w+)\}")
VALUE_SLOTS = {"a", "b"}
CATEGORY_SLOTS = {"category"}
SEED = 20260830

_b = None


def render_with_spans(template, values):
    out, spans, pos = [], [], 0
    for m in SLOT_RE.finditer(template):
        out.append(template[pos:m.start()])
        prefix = "".join(out)
        slot = m.group(1)
        text = str(values.get(slot, ""))
        spans.append((len(prefix), len(prefix) + len(text), slot))
        out.append(text)
        pos = m.end()
    out.append(template[pos:])
    return "".join(out), spans


def bio(text, spans):
    words, offsets = [], []
    for m in re.finditer(r"\S+", text):
        words.append(m.group())
        offsets.append((m.start(), m.end()))
    labels = ["O"] * len(words)
    for start, end, slot in spans:
        tag = ("VALUE" if slot in VALUE_SLOTS else
               "CATEGORY" if slot in CATEGORY_SLOTS else None)
        if tag is None:
            continue
        first = True
        for i, (ws, we) in enumerate(offsets):
            if ws < end and we > start:
                labels[i] = ("B-" if first else "I-") + tag
                first = False
    return words, labels


def spans_of(labels):
    out, start = set(), None
    for i, l in enumerate(list(labels) + ["O"]):
        if l == "B-VALUE":
            if start is not None:
                out.add((start, i))
            start = i
        elif l == "I-VALUE":
            if start is None:
                start = i
        else:
            if start is not None:
                out.add((start, i)); start = None
    return out


def score(rows, preds):
    tp = fp = fn = exact = total = 0
    for r, p in zip(rows, preds):
        g, h = spans_of(r["labels"]), spans_of(p)
        total += 1
        exact += int(g == h)
        gt = {i for s, e in g for i in range(s, e)}
        ht = {i for s, e in h for i in range(s, e)}
        tp += len(gt & ht); fp += len(ht - gt); fn += len(gt - ht)
    p_ = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    return {"span_exact": round(exact / max(total, 1), 4),
            "token_p": round(p_, 4), "token_r": round(r_, 4),
            "token_f1": round(2 * p_ * r_ / (p_ + r_), 4) if p_ + r_ else 0.0,
            "n": total}


def main() -> None:
    os.environ.setdefault("LLM_RESOLVE", "0")
    if not RUN.is_dir():
        print(f"no checkpoint at {RUN} -- train first."); return
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    rng = random.Random(SEED)
    LABELS = json.loads((RUN / "labels.json").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(RUN)
    model = AutoModelForTokenClassification.from_pretrained(RUN).to(device).eval()

    paraphrases = [" ".join(str(json.loads(l)["paraphrase"]).split())
                   for l in (V2 / "open_vocabulary_paraphrases.jsonl").open(encoding="utf-8")
                   if l.strip()]
    paraphrases = [p for p in paraphrases if p and p.lower() != "skip"]
    rows = [json.loads(l) for l in TEST.open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if SLOT_RE.search(str(r.get("template", "")))]
    print(f"held-out bank: {len(rows)} rows with slot templates")
    print(f"paraphrase pool: {len(paraphrases)} (never seen in training)\n")

    def build(condition):
        out = []
        for r in rows:
            slots = dict(r["slots"])
            if condition == "paraphrased":
                for s in list(slots):
                    if s in VALUE_SLOTS:
                        slots[s] = rng.choice(paraphrases)
            text, spans = render_with_spans(str(r["template"]), slots)
            words, labels = bio(text, spans)
            if words:
                out.append({"tokens": words, "labels": labels,
                            "action": r.get("action")})
        return out

    def predict(rows_, bs=64):
        out = []
        for i in range(0, len(rows_), bs):
            batch = rows_[i:i + bs]
            enc = tok([b["tokens"] for b in batch], is_split_into_words=True,
                      padding=True, truncation=True, max_length=96,
                      return_tensors="pt").to(device)
            with torch.no_grad():
                pred = model(**{k: v for k, v in enc.items()
                                if k in ("input_ids", "attention_mask")}).logits.argmax(-1)
            for j, b in enumerate(batch):
                lab, prev = ["O"] * len(b["tokens"]), None
                for pos, wid in enumerate(enc.word_ids(j)):
                    if wid is None or wid == prev:
                        continue
                    prev = wid
                    if wid < len(lab):
                        lab[wid] = LABELS[int(pred[j, pos])]
                out.append(lab)
        return out

    t0, report = time.time(), {}
    for condition in ("canonical", "paraphrased"):
        data = build(condition)
        preds = predict(data)
        m = score(data, preds)
        nothing = score(data, [["O"] * len(r["tokens"]) for r in data])
        every = score(data, [["B-VALUE"] + ["I-VALUE"] * (len(r["tokens"]) - 1)
                             for r in data])
        report[condition] = {"model": m, "predict_nothing": nothing,
                             "predict_everything": every}
        print(f"=== {condition} ===")
        print(f"  {'':<20}{'span-exact':>12}{'VALUE P':>10}{'R':>9}{'F1':>9}")
        print(f"  {'model':<20}{m['span_exact']:>12.4f}{m['token_p']:>10.4f}"
              f"{m['token_r']:>9.4f}{m['token_f1']:>9.4f}")
        print(f"  {'predict-nothing':<20}{nothing['span_exact']:>12.4f}"
              f"{nothing['token_p']:>10.4f}{nothing['token_r']:>9.4f}"
              f"{nothing['token_f1']:>9.4f}")
        print(f"  {'predict-everything':<20}{every['span_exact']:>12.4f}"
              f"{every['token_p']:>10.4f}{every['token_r']:>9.4f}"
              f"{every['token_f1']:>9.4f}")

        per = defaultdict(lambda: {"n": 0, "fp_rows": 0, "exact": 0})
        for r, p in zip(data, preds):
            g = per[r["action"]]
            g["n"] += 1
            gs, hs = spans_of(r["labels"]), spans_of(p)
            g["exact"] += int(gs == hs)
            if not gs and hs:
                g["fp_rows"] += 1          # predicted a value where there is none
        print(f"  {'action':<20}{'rows':>7}{'span-exact':>12}{'false spans':>13}")
        for a in sorted(per):
            g = per[a]
            print(f"  {a:<20}{g['n']:>7}{g['exact']/g['n']:>12.4f}"
                  f"{g['fp_rows']/g['n']:>13.4f}")
        report[condition]["per_action"] = {k: dict(v) for k, v in per.items()}
        print()

    c, p = report["canonical"]["model"], report["paraphrased"]["model"]
    print(f"  THE CONTRAST")
    print(f"  span-exact  canonical {c['span_exact']:.4f} -> paraphrased "
          f"{p['span_exact']:.4f}   ({p['span_exact']-c['span_exact']:+.4f})")
    print(f"  VALUE F1    canonical {c['token_f1']:.4f} -> paraphrased "
          f"{p['token_f1']:.4f}   ({p['token_f1']-c['token_f1']:+.4f})")
    print(f"\n  A small drop means the model learned SLOT POSITION and transfers to values")
    print(f"  it has never seen -- which is the entire point of it. A large drop means it")
    print(f"  memorised attribute-shaped vocabulary despite the randomisation, and would")
    print(f"  be useless on exactly the paraphrased case it was built for.")
    print(f"\n  {time.time()-t0:.0f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "V2.64 span extractor, held-out templates",
        "checkpoint": str(RUN), "bank": str(TEST.relative_to(ROOT)),
        "paraphrase_pool": len(paraphrases), "conditions": report,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
