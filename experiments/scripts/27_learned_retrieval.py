"""Experiment 27: B2 + B4 -- learned retrieval, re-motivated by the width-1 objective.

WHY THESE MATTER MORE NOW
-------------------------
Under width-1 disclosure the score algebra collapses to

    score = 0.8*HR + 0.2*Eff          (because MRR == HR when every hit is rank 1)
    MTTC  = the target's POSITION in our ranking

So recall carries 0.8 instead of 0.5, and "rank within a shown list" no longer exists.
Both remaining levers -- HR (2 misses) and MTTC (2.38 vs a 1.39 floor) -- are retrieval
quantities. Total remaining headroom to the 0.992 ceiling is ~0.028.

B2  fine-tuned bi-encoder: train an encoder on (evidence -> target) pairs minted from the
    known generator, then use it to re-order the retrieval POOL before coverage scoring.
    The off-the-shelf bi-encoder failed (-0.047) on a domain mismatch; this removes that
    excuse.

B4  learned sparse ("SPLADE-lite"): keep exact lexical matching but LEARN the term
    weights instead of hand-setting them, via a linear model over per-phrase properties.
    Sparse, so it cannot blur the provenance signal the way a dense encoder does.

Both are evaluated where it counts: HR@10 and MTTC under the width-1 policy.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/scripts/27_learned_retrieval.py --mint 8000
"""
from __future__ import annotations

# Import torch FIRST. On this Windows build, importing sklearn first makes torch's
# c10.dll fail to initialise (WinError 1114) -- both ship an OpenMP runtime and the
# first one loaded wins. Reproducible in two lines; cost pass 27 its B2 arm.
try:
    import torch  # noqa: F401  (ordering matters, not the symbol)
except Exception:
    torch = None  # type: ignore[assignment]

import argparse
import json
import math
import pickle
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    behavior_for, catalog_index, coarse_category, customer_reply, evaluate,
    initial_message, intent_card, load_jsonl,
)
from submission.agent import Agent, CAT, CONSTRAINT, raw_toks  # noqa: E402

W1 = (1,) * 9 + (10,)
PAIRS_CACHE = ROOT / "experiments" / "studies" / ".retr_pairs.pkl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mint", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    TUNE = [s for i, s in enumerate(samples) if i % 2 == 0]
    HOLD = [s for i, s in enumerate(samples) if i % 2 == 1]
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    public_targets = {str(s["ground_truth"]["parent_asin"]) for s in samples}

    def share(cls=Agent, **kw):
        o = object.__new__(cls)
        o.ix, o.sessions, o.llm = base.ix, {}, None
        o.DISCLOSURE = W1
        for k, v in kw.items():
            setattr(o, k, v)
        return o

    def run(ag, sub):
        r = evaluate(ag, sub, cid, cats, prods)
        return {"score": r["recommended_technical_score"], "hr": r["hit_rate_at_10"],
                "mrr": r["mrr"], "mttc": r["mttc"]}

    print(f"{'variant':<34}{'tune':>9}{'hold':>9}{'HR':>8}{'MTTC':>7}")
    print("-" * 68)
    OUT = {}
    bt, bh = run(share(), TUNE), run(share(), HOLD)
    OUT["baseline"] = {"tune": bt, "hold": bh}
    print(f"{'baseline (width-1)':<34}{bt['score']:>9.5f}{bh['score']:>9.5f}"
          f"{bh['hr']:>8.1%}{bh['mttc']:>7.2f}")

    # ---------------- mint (evidence -> target) pairs from the known generator
    if PAIRS_CACHE.exists():
        pairs = pickle.loads(PAIRS_CACHE.read_bytes())
        print(f"\nloaded {len(pairs):,} cached (evidence, target) pairs")
    else:
        def rn(a):
            try:
                return float(prods[a].get("rating_number") or 0)
            except (TypeError, ValueError):
                return 0.0

        cand = [a for a in prods if a not in public_targets]
        rng = random.Random(args.seed)
        minted = rng.choices(cand, weights=[rn(a) for a in cand], k=args.mint)
        pairs = []
        print(f"\nminting {len(minted):,} (evidence -> target) pairs ...")
        t0 = time.time()
        for a in minted:
            card = intent_card(prods[a])
            cons = [str(x) for x in card["hard_constraints"]] + \
                   [str(x) for x in card["soft_preferences"]]
            cons = [c for c in dict.fromkeys(cons) if c.strip()]
            cat = coarse_category([str(v) for v in (prods[a].get("categories") or [])])
            if len(cons) >= 2:
                k = rng.randint(1, len(cons))
                pairs.append({"q": "; ".join([cat] + cons[:k])[:320], "tgt": a})
        PAIRS_CACHE.write_bytes(pickle.dumps(pairs))
        print(f"  {len(pairs):,} pairs  [{time.time()-t0:.0f}s]")

    # ================= B4: learned sparse term weighting =====================
    # Learn how much a matched phrase should count, from its OWN properties
    # (length, df, tier, field) rather than from constants I chose. Stays exact-match.
    print("\nB4 learned sparse term weighting ...")
    rows, labels = [], []
    rng = random.Random(args.seed + 1)
    all_asins = list(prods)
    for p in pairs[:6000]:
        toks = [t for t in p["q"].split("; ") if t.strip()]
        tgt = p["tgt"]
        for phr in toks:
            ph = " ".join(raw_toks(phr)[:12])
            if not ph:
                continue
            df = base.ix.df(ph) or 1
            if not base.ix.covers(tgt, ph):
                continue
            # DISCRIMINATIVE label: does this phrase separate the target from a
            # competitor? A phrase the target and a rival BOTH contain is worthless
            # for ranking, however long or rare it looks. The earlier label ("does
            # the target contain it") was 93.7% positive -- trivially true, since the
            # phrases are lifted FROM the target -- so it taught nothing.
            rival = all_asins[rng.randrange(len(all_asins))]
            disc = 0 if base.ix.covers(rival, ph) else 1
            rows.append([len(ph.split()), math.log1p(df),
                         1.0 if base.ix.in_title(tgt, ph) else 0.0])
            labels.append(disc)
    Xs, ys = np.array(rows, dtype=np.float32), np.array(labels, dtype=np.int8)
    print(f"  phrase rows {len(Xs):,}  positive rate {ys.mean():.1%}")
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=400).fit(Xs, ys)
    coef = lr.coef_[0]
    print(f"  learned coefficients: len={coef[0]:+.3f}  log_df={coef[1]:+.3f}  "
          f"in_title={coef[2]:+.3f}")

    class LearnedSparse(Agent):
        def _weight(self, phrase, df, tier):
            base_w = {CONSTRAINT: self.W_CONSTRAINT, CAT: self.W_CATEGORY}.get(
                tier, self.W_MINED)
            n = len(phrase.split())
            z = coef[0] * n + coef[1] * math.log1p(max(df, 1))
            return base_w * float(1.0 / (1.0 + math.exp(-z)))

    t, h = run(share(LearnedSparse), TUNE), run(share(LearnedSparse), HOLD)
    OUT["B4 learned sparse"] = {"tune": t, "hold": h}
    print(f"{'B4 learned sparse weighting':<34}{t['score']:>9.5f}{h['score']:>9.5f}"
          f"{h['hr']:>8.1%}{h['mttc']:>7.2f}")

    # ================= B2: fine-tuned bi-encoder =============================
    print("\nB2 fine-tuned bi-encoder ...")
    try:
        if torch is None:
            raise ImportError("torch unavailable at module import")
        from torch import nn
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        print(f"  unavailable: {type(exc).__name__}: {exc}")
        OUT["B2"] = {"error": str(exc)}
        (ROOT / "experiments" / "results" / "out_27.json").write_text(
            json.dumps(OUT, indent=2) + "\n", encoding="utf-8")
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = AutoModel.from_pretrained(MODEL).to(dev)
    opt = torch.optim.AdamW(enc.parameters(), lr=2e-5)

    def embed(texts, grad=True):
        b = tok(texts, truncation=True, max_length=128, padding=True,
                return_tensors="pt").to(dev)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = enc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            v = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
            return nn.functional.normalize(v, dim=-1)

    BS = 32
    train = pairs[:6000]
    print(f"  in-batch contrastive on {len(train):,} pairs, batch {BS}, {args.epochs} ep")
    t0 = time.time()
    for ep in range(args.epochs):
        random.Random(ep).shuffle(train)
        tot = 0.0
        for i in range(0, len(train) - BS, BS):
            chunk = train[i:i + BS]
            qv = embed([c["q"] for c in chunk])
            dv = embed([base.ix.doc.get(c["tgt"], "")[:320] for c in chunk])
            logits = qv @ dv.T * 20.0
            loss = nn.functional.cross_entropy(
                logits, torch.arange(len(chunk), device=dev))
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            tot += loss.item()
            if (i // BS) % 40 == 0:
                print(f"    ep{ep+1} {i:,}/{len(train):,} loss {tot/max(1,i//BS+1):.4f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
    enc.eval()

    class BiEncoderRerank(Agent):
        """Re-order the retrieval POOL with the fine-tuned encoder before coverage."""
        DEPTH = 60

        def _candidates(self, st, message):
            pool = super()._candidates(st, message)
            if len(pool) < 3 or not st.evidence:
                return pool
            head = pool[:self.DEPTH]
            q = "; ".join(list(st.evidence)[:8])[:320]
            try:
                qv = embed([q], grad=False)
                dv = embed([self.ix.doc.get(a, "")[:320] for a in head], grad=False)
                sims = (qv @ dv.T).squeeze(0).tolist()
            except Exception:
                return pool
            order = sorted(range(len(head)), key=lambda i: -sims[i])
            return [head[i] for i in order] + pool[self.DEPTH:]

    t, h = run(share(BiEncoderRerank), TUNE), run(share(BiEncoderRerank), HOLD)
    OUT["B2 fine-tuned bi-encoder"] = {"tune": t, "hold": h}
    print(f"\n{'B2 fine-tuned bi-encoder':<34}{t['score']:>9.5f}{h['score']:>9.5f}"
          f"{h['hr']:>8.1%}{h['mttc']:>7.2f}")

    print(f"\n  baseline hold {bh['score']:.5f}")
    for k in ("B4 learned sparse", "B2 fine-tuned bi-encoder"):
        if k in OUT:
            d = OUT[k]["hold"]["score"] - bh["score"]
            print(f"  {k:<28} {d:+.5f} -> "
                  f"{'ADOPT' if d > 0.005 else 'inside noise' if d > -0.005 else 'REJECT'}")

    (ROOT / "experiments" / "results" / "out_27.json").write_text(
        json.dumps(OUT, indent=2) + "\n", encoding="utf-8")
    print("\n[saved] experiments/results/out_27.json")


if __name__ == "__main__":
    main()
