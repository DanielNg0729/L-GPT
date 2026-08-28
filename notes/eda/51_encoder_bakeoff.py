"""EDA pass 51: does a BETTER PRETRAINED ENCODER improve the scaffolding tagger?

THE QUESTION, PRECISELY. Pass 49 measured our OWN MLM adaptation on the 50,000-product
catalogue and found it worthless (T1 0.88820 adapted vs 0.89300 plain). That is not the
same question as: is `distilbert-base-uncased` the right CHECKPOINT? The measured mechanism
says the tagger's power comes from GENERAL pretraining -- knowing "necklaces" is a product
word and "appreciate it" is politeness -- so an encoder pretrained on more or better text
should do better, and one pretrained on product text specifically should do better still.

ON DOMAIN-PRETRAINED ENCODERS: searched, and none suitable exists. The e-commerce
checkpoints on the Hub are CLIP-style image/text embedders (Marqo, Trendyol) or chat LLMs;
the one shopping-domain text model found is a MiniLM cross-encoder reranker trained on ESCI,
which is an architecture for scoring query-document pairs, not for tagging tokens. So the
"pretrained on a different corpus" arm is answered by absence rather than by measurement.

WHAT IS TESTED, all through the identical pipeline and the identical protocol:

    distilbert-base-uncased    66M, 6 layers    the shipped baseline
    bert-base-uncased         110M, 12 layers   pure capacity: same corpus, bigger model
    roberta-base              125M, 12 layers   ~10x BERT's pretraining data, better recipe
    microsoft/deberta-v3-small  ELECTRA-style pretraining, strong on token classification

Capacity is worth watching sceptically here. Pass 24 established on this task that CAPACITY
CONSUMES SIGNAL -- k=1 feature reproduced the best ranking result and every added feature
made it worse. If bert-base beats distilbert, capacity helps for tagging even though it hurt
for ranking, which is a genuinely useful thing to know. If it does not, that is the same
lesson again in a new place.

THE DECISIVE MEASUREMENT is the same one that killed the linear model in pass 47:
HELD-OUT-TRANSFORM lift, above all on T2, where the linear tagger scored -0.435 (below
majority class). Training accuracy is not evidence -- pass 47 had 0.837 of it and was
useless.

SIZE MATTERS TOO. The shipped distilbert tagger is already 266 MB. roberta-base and
deberta-v3-small are larger, and the submission rules allow "lightweight local assets".
A win has to be big enough to justify the weight, and is reported alongside it.

Run:  PYTHONIOENCODING=utf-8 python -u notes/eda/51_encoder_bakeoff.py
"""
from __future__ import annotations

# torch FIRST -- sklearn-before-torch breaks c10.dll on this Windows build (WinError 1114).
import torch  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notes" / "eda"))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent, raw_toks, recognised  # noqa: E402

_p30 = __import__("30_robustness_benchmark")
_p31 = __import__("31_paraphrase_stress")
_p49 = __import__("49_bert_scaffolding_tagger")
mint, SEEDS = _p30.mint, _p30.SEEDS
evaluate_transformed, TRANSFORMS = _p31.evaluate_transformed, _p31.TRANSFORMS
DEV = _p49.DEV

CANDIDATES = [
    ("distilbert (shipped)", "distilbert-base-uncased", {}),
    ("bert-base", "bert-base-uncased", {}),
    # RoBERTa's BPE needs an explicit leading space to align pre-split words.
    ("roberta-base", "roberta-base", {"add_prefix_space": True}),
    ("deberta-v3-small", "microsoft/deberta-v3-small", {}),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--sessions", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    t0 = time.time()

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    pub_t = {str(s["ground_truth"]["parent_asin"]) for s in samples}
    profiles = [s["user_profile"] for s in samples]

    pool = [a for a in prods if a not in pub_t]
    random.Random(3).shuffle(pool)
    tr_asins = pool[:args.sessions]
    te_asins = pool[args.sessions:args.sessions + 300]

    sets = {"clean": samples,
            "unseen800": mint(prods, pub_t, profiles, "reviews", 800, seed=SEEDS["reviews"])}
    PARA = ["T1 scaffold reworded", "T2 scaffold stripped", "T5 realistic (T1+T3)"]
    COLS = list(sets) + [p.split()[0] for p in PARA]

    def share(cls=Agent):
        o = object.__new__(cls)
        o.ix, o.sessions, o.llm, o.llm_extract, o.tagger = base.ix, {}, None, None, None
        return o

    def row(cls):
        r = {}
        for name, sub in sets.items():
            r[name] = evaluate(share(cls), sub, cid, cats, prods)[
                "recommended_technical_score"]
        for p in PARA:
            r[p.split()[0]] = evaluate_transformed(
                share(cls), samples, cid, cats, prods,
                TRANSFORMS[p])["recommended_technical_score"]
        return r

    OUT = {}
    ref = row(Agent)
    OUT["shipped (no tagger)"] = {"scores": ref}
    print(f"\n{'encoder':<24}{'params':>9}" + "".join(f"{c:>11}" for c in COLS)
          + f"{'T2 lift':>9}{'train s':>9}")
    print("-" * (24 + 9 + 11 * len(COLS) + 18))
    print(f"{'shipped (no tagger)':<24}{'-':>9}" + "".join(f"{ref[c]:>11.5f}" for c in COLS))

    for label, ckpt, tok_kw in CANDIDATES:
        try:
            tstart = time.time()
            tok = AutoTokenizer.from_pretrained(ckpt, **tok_kw)
            rows = _p49.gen(prods, tr_asins, _p49.TRAIN_TF, seed=1)
            model = AutoModelForTokenClassification.from_pretrained(
                ckpt, num_labels=2).to(DEV)
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            enc, labels = _p49.encode(tok, rows)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
            n = len(rows)
            order = list(range(n))
            for ep in range(args.epochs):
                model.train()
                random.Random(ep).shuffle(order)
                for k in range(0, n - args.batch, args.batch):
                    idx = order[k:k + args.batch]
                    out = model(input_ids=enc["input_ids"][idx].to(DEV),
                                attention_mask=enc["attention_mask"][idx].to(DEV),
                                labels=labels[idx].to(DEV))
                    out.loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            train_s = time.time() - tstart

            lifts = {}
            for tf in _p49.HELDOUT_TF:
                te = _p49.gen(prods, te_asins, [tf], seed=2)
                acc, majority = _p49.evaluate_tagger(model, tok, te)
                lifts[tf] = {"acc": acc, "majority": majority, "lift": acc - majority}

            model.eval()

            def make(mdl, tk, threshold=0.30):
                class Strip(Agent):
                    def _observe(self, st, msg):
                        if recognised(msg):
                            return super()._observe(st, msg)
                        words = raw_toks(msg)
                        if len(words) >= 3:
                            e = tk([words], is_split_into_words=True, truncation=True,
                                   max_length=96, return_tensors="pt")
                            with torch.no_grad():
                                p = torch.softmax(mdl(
                                    input_ids=e["input_ids"].to(DEV),
                                    attention_mask=e["attention_mask"].to(DEV)).logits,
                                    -1)[0, :, 1]
                            keep, prev = [], None
                            for pos, wid in enumerate(e.word_ids(0)):
                                if wid is None or wid == prev:
                                    continue
                                prev = wid
                                if wid < len(words) and float(p[pos]) >= threshold:
                                    keep.append(words[wid])
                            if len(keep) >= 2:
                                msg = " ".join(keep)
                        return super()._observe(st, msg)
                return Strip

            r = row(make(model, tok))
            OUT[label] = {"scores": r, "params_m": n_params, "lifts": lifts,
                          "train_s": train_s, "checkpoint": ckpt}
            t2 = lifts["T2 scaffold stripped"]["lift"]
            print(f"{label:<24}{n_params:>8.0f}M" + "".join(f"{r[c]:>11.5f}" for c in COLS)
                  + f"{t2:>+9.3f}{train_s:>9.0f}")
            del model
            torch.cuda.empty_cache()
        except Exception as exc:
            OUT[label] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"{label:<24}  FAILED: {type(exc).__name__}: {str(exc)[:60]}")

    print(f"\n{'encoder':<24}" + "".join(f"{c:>11}" for c in COLS) + "   held-out lifts")
    print("-" * (24 + 11 * len(COLS) + 18))
    for label, v in OUT.items():
        if "scores" not in v or label.startswith("shipped"):
            continue
        d = {c: v["scores"][c] - ref[c] for c in COLS}
        lift_s = " ".join(f"{k.split()[0]}{v['lifts'][k]['lift']:+.2f}" for k in v["lifts"])
        print(f"{label:<24}" + "".join(f"{d[c]:>+11.5f}" for c in COLS) + f"   {lift_s}")

    print("\n  clean and unseen800 MUST be +0.00000 for every row -- the recognition gate")
    print("  makes the tagger unreachable on unparaphrased traffic regardless of encoder.")
    (ROOT / "notes" / "eda" / "out_51.json").write_text(
        json.dumps(OUT, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n[saved] notes/eda/out_51.json   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
