"""EDA pass 49: B-a -- a BERT token-tagger for scaffolding, and MLM adaptation on the 50k corpus.

WHAT THIS IS TESTING, AND WHY IT IS NOT THE FOUR DEAD DENSE APPROACHES
---------------------------------------------------------------------
Every previous transformer attempt here tried to make the model MATCH PRODUCTS -- dense
bi-encoder (-0.047), ColBERT (predicted-fail on its own OOD mechanism), cross-encoder
(-0.030, and it got WORSE with training: 32.4% -> 8.8%, below random). They all failed for
one reason: the task is exact-substring provenance recovery, and embeddings blur the
lexical precision that solves it.

This does not ask BERT to match anything. It asks it to TAG TOKENS:

    "Appreciate it. I want to find Jewelry Necklaces. It absolutely has to be
     Material:alloy. Cheers."
      scaffolding -> appreciate it i want to find it absolutely has to be cheers
      content     -> jewelry necklaces material alloy

Exact matching then runs, unchanged, over the surviving text. Semantics is used only to
decide WHAT TO KEEP, never to decide what matches.

THE SPECIFIC HYPOTHESIS, FROM THE FAILURE OF PASS 47
----------------------------------------------------
Pass 47 (F2) did exactly this with a linear model over `df`-statistics features. It reached
0.837 train accuracy and then collapsed on the held-out transform T2 -- **-0.435 lift,
BELOW majority class** -- because T2 strips scaffolding entirely, so the correct behaviour
is "keep everything" and the model instead discarded content. It had learned OUR filler
vocabulary rather than the shape of scaffolding.

A pretrained transformer should not have that failure, because it already knows from
pretraining that "necklaces" and "alloy" are product words and "appreciate it" is politeness
-- which is precisely the knowledge the LLM extractor uses, and precisely what `df`
statistics cannot express.

    ARM 1  distilbert, token classification, trained on our paraphrase families
    ARM 2  ARM 1 + MLM domain-adaptation on all 50,000 catalogue listings first

Arm 2 is the "use the corpus" arm: continue masked-LM pretraining on the product text so
the encoder's vocabulary priors match this catalogue before it is asked to tag anything.

THE DECISIVE MEASUREMENT is not accuracy on the training transforms -- pass 47 already
showed that looks like success while being useless. It is HELD-OUT-TRANSFORM lift, above
all on T2, the transform that broke the linear model. Anything that fails T2 the same way
is the same failure with more parameters.

SHIPPING NOTE: a transformer at inference breaks the standard-library-only property. It
would run ONLY behind the recognition gate (0 clean messages reach it), so a failure there
costs nothing, but the dependency is a real cost and is only worth paying for a real gain.

Run:  PYTHONIOENCODING=utf-8 python -u notes/eda/49_bert_scaffolding_tagger.py --epochs 3
"""
from __future__ import annotations

# Import torch FIRST. On this Windows build importing sklearn first makes torch's c10.dll
# fail to initialise (WinError 1114) -- both ship an OpenMP runtime and the first one loaded
# wins. Reproducible in two lines; cost pass 27 its entire B2 arm.
import torch  # noqa: E402  (ordering matters, not the symbol)

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notes" / "eda"))

from evaluator.local_evaluator import (  # noqa: E402
    behavior_for, catalog_index, coarse_category, customer_reply, evaluate,
    initial_message, intent_card, load_jsonl,
)
from submission.agent import Agent, raw_toks, recognised  # noqa: E402

_p30 = __import__("30_robustness_benchmark")
_p31 = __import__("31_paraphrase_stress")
mint, SEEDS = _p30.mint, _p30.SEEDS
evaluate_transformed, TRANSFORMS = _p31.evaluate_transformed, _p31.TRANSFORMS

MODEL = "distilbert-base-uncased"
TRAIN_TF = ["T1 scaffold reworded", "T3 conversational noise"]
HELDOUT_TF = ["T2 scaffold stripped", "T5 realistic (T1+T3)", "T4 case/punctuation churn"]
N_TRAIN_SESSIONS = 1200
N_TEST_SESSIONS = 300
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def gen(prods, asins, transforms, seed=0):
    """(word list, content-word set) with EXACT labels, from the generator itself."""
    rng = random.Random(seed)
    rows = []
    for asin in asins:
        card = intent_card(prods[asin])
        content = set()
        for v in list(card["hard_constraints"]) + list(card["soft_preferences"]):
            content.update(raw_toks(str(v)))
        cat = coarse_category([str(x) for x in (prods[asin].get("categories") or [])])
        content.update(raw_toks(cat))
        scenario = rng.choice(["buying", "browsing"])
        eff = {"sample_id": asin, "scenario_type": scenario, "intent_card": card,
               "behavior": behavior_for(scenario, card, random.Random(asin))}
        disclosed, bu = set(), False
        msgs = [initial_message(eff, cat, disclosed)]
        for attr in ("feature", "material", "color", "other"):
            m, bu = customer_reply(eff, attr, disclosed, bu)
            msgs.append(m)
        for m in msgs:
            shown = TRANSFORMS[rng.choice(transforms)](m)
            words = raw_toks(shown)
            if 2 <= len(words) <= 64:
                rows.append((words, [1 if w in content else 0 for w in words]))
    return rows


def encode(tok, rows, max_len=96):
    enc = tok([r[0] for r in rows], is_split_into_words=True, truncation=True,
              max_length=max_len, padding="max_length", return_tensors="pt")
    labels = torch.full(enc["input_ids"].shape, -100, dtype=torch.long)
    for i, (_words, tags) in enumerate(rows):
        prev = None
        for pos, wid in enumerate(enc.word_ids(i)):
            if wid is None or wid == prev:
                continue                      # label the FIRST sub-token of each word only
            prev = wid
            if wid < len(tags):
                labels[i, pos] = tags[wid]
    return enc, labels


def evaluate_tagger(model, tok, rows, bs=64):
    model.eval()
    enc, labels = encode(tok, rows)
    correct = total = pos = 0
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            ids = enc["input_ids"][i:i + bs].to(DEV)
            am = enc["attention_mask"][i:i + bs].to(DEV)
            lb = labels[i:i + bs].to(DEV)
            pred = model(input_ids=ids, attention_mask=am).logits.argmax(-1)
            mask = lb != -100
            correct += int((pred[mask] == lb[mask]).sum())
            total += int(mask.sum())
            pos += int((lb[mask] == 1).sum())
    rate = pos / max(total, 1)
    return correct / max(total, 1), max(rate, 1 - rate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--mlm-steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    t0 = time.time()

    from transformers import (AutoModelForMaskedLM, AutoModelForTokenClassification,
                              AutoTokenizer, DataCollatorForLanguageModeling)

    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    pub_t = {str(s["ground_truth"]["parent_asin"]) for s in samples}
    profiles = [s["user_profile"] for s in samples]
    print(f"device {DEV}  [{time.time()-t0:.0f}s]")

    pool = [a for a in prods if a not in pub_t]
    random.Random(3).shuffle(pool)
    tr_asins = pool[:N_TRAIN_SESSIONS]
    te_asins = pool[N_TRAIN_SESSIONS:N_TRAIN_SESSIONS + N_TEST_SESSIONS]

    tok = AutoTokenizer.from_pretrained(MODEL)
    train_rows = gen(prods, tr_asins, TRAIN_TF, seed=1)
    print(f"train messages {len(train_rows):,}  [{time.time()-t0:.0f}s]")

    # ------------------------------------------------------- ARM 2 prep: MLM on the corpus
    def mlm_adapt():
        print(f"\nMLM domain-adaptation on the 50k catalogue ({args.mlm_steps} steps) ...")
        mdl = AutoModelForMaskedLM.from_pretrained(MODEL).to(DEV)
        texts = [base.ix.doc[a][:400] for a in list(prods)[:50_000] if base.ix.doc.get(a)]
        random.Random(0).shuffle(texts)
        coll = DataCollatorForLanguageModeling(tokenizer=tok, mlm_probability=0.15)
        opt = torch.optim.AdamW(mdl.parameters(), lr=5e-5)
        mdl.train()
        step = 0
        while step < args.mlm_steps:
            for i in range(0, len(texts) - args.batch, args.batch):
                if step >= args.mlm_steps:
                    break
                batch = tok(texts[i:i + args.batch], truncation=True, max_length=128,
                            padding="max_length", return_tensors="pt")
                feats = [{"input_ids": batch["input_ids"][j],
                          "attention_mask": batch["attention_mask"][j]}
                         for j in range(batch["input_ids"].size(0))]
                b = {k: v.to(DEV) for k, v in coll(feats).items()}
                loss = mdl(**b).loss
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 300 == 0:
                    print(f"    step {step}/{args.mlm_steps} loss {loss.item():.3f} "
                          f"[{time.time()-t0:.0f}s]", flush=True)
        out = ROOT / "notes" / "eda" / ".mlm_adapted"
        mdl.save_pretrained(out)
        tok.save_pretrained(out)
        return str(out)

    def train_tagger(init: str, tag: str):
        print(f"\ntraining tagger [{tag}] from {init} ...")
        mdl = AutoModelForTokenClassification.from_pretrained(init, num_labels=2).to(DEV)
        enc, labels = encode(tok, train_rows)
        opt = torch.optim.AdamW(mdl.parameters(), lr=3e-5)
        n = len(train_rows)
        order = list(range(n))
        for ep in range(args.epochs):
            mdl.train()
            random.Random(ep).shuffle(order)
            tot = 0.0
            for k in range(0, n - args.batch, args.batch):
                idx = order[k:k + args.batch]
                out = mdl(input_ids=enc["input_ids"][idx].to(DEV),
                          attention_mask=enc["attention_mask"][idx].to(DEV),
                          labels=labels[idx].to(DEV))
                out.loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                tot += out.loss.item()
            print(f"  epoch {ep+1}/{args.epochs} loss {tot/max(1,n//args.batch):.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        return mdl

    RESULTS: dict = {}

    def assess(mdl, tag):
        print(f"\n  HELD-OUT-TRANSFORM accuracy [{tag}] "
              f"(the measurement that killed the linear model):")
        row = {}
        for tf in HELDOUT_TF:
            rows = gen(prods, te_asins, [tf], seed=2)
            acc, majority = evaluate_tagger(mdl, tok, rows)
            row[tf] = {"acc": acc, "majority": majority, "lift": acc - majority}
            flag = "" if acc - majority > 0.02 else "   <-- no better than majority"
            print(f"    {tf:<28} acc {acc:.3f}  majority {majority:.3f}  "
                  f"lift {acc-majority:+.3f}{flag}")
        RESULTS[tag] = row
        return row

    arms = {"arm1 pretrained": MODEL}
    if args.mlm_steps > 0:
        arms["arm2 +MLM on 50k corpus"] = mlm_adapt()

    models = {}
    for tag, init in arms.items():
        mdl = train_tagger(init, tag)
        assess(mdl, tag)
        models[tag] = mdl

    # ----------------------------------------------------------------- end-to-end scoring
    def make(mdl, threshold):
        mdl.eval()

        class BertStrip(Agent):
            def _observe(self, st, msg):
                if recognised(msg):
                    return super()._observe(st, msg)     # clean path untouched by the gate
                words = raw_toks(msg)
                if len(words) >= 3:
                    e = tok([words], is_split_into_words=True, truncation=True,
                            max_length=96, return_tensors="pt")
                    with torch.no_grad():
                        p = torch.softmax(
                            mdl(input_ids=e["input_ids"].to(DEV),
                                attention_mask=e["attention_mask"].to(DEV)).logits, -1)[0, :, 1]
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
        return BertStrip

    sets = {"clean": samples,
            "unseen800": mint(prods, pub_t, profiles, "reviews", 800, seed=SEEDS["reviews"])}
    PARA = ["T1 scaffold reworded", "T2 scaffold stripped", "T5 realistic (T1+T3)"]
    COLS = list(sets) + [p.split()[0] for p in PARA]

    def share(cls=Agent):
        o = object.__new__(cls)
        o.ix, o.sessions, o.llm, o.llm_extract = base.ix, {}, None, None
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

    print(f"\n{'variant':<34}" + "".join(f"{c:>12}" for c in COLS))
    print("-" * (34 + 12 * len(COLS)))
    ref = row(Agent)
    RESULTS["shipped"] = ref
    print(f"{'shipped (no tagger)':<34}" + "".join(f"{ref[c]:>12.5f}" for c in COLS))
    for tag, mdl in models.items():
        for th in (0.30, 0.50):
            r = row(make(mdl, th))
            RESULTS[f"{tag} @ {th}"] = r
            print(f"{tag + f' keep>={th}':<34}"
                  + "".join(f"{r[c]:>12.5f}" for c in COLS))

    print(f"\n  reference points -- LLM extractor on T1: +0.0688 | linear tagger: -0.0976")
    (ROOT / "notes" / "eda" / "out_49.json").write_text(
        json.dumps(RESULTS, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n[saved] notes/eda/out_49.json   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
