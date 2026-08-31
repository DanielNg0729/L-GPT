"""Experiment 23: a cross-encoder FINE-TUNED on this task, specialised on tie decisions.

Two distinct failures motivate this, and it is designed to fix both.

FAILURE 1 -- the cross-encoder we tested was zero-shot and out of domain.
`ms-marco-MiniLM-L-6-v2` is trained on natural-language web queries. We fed it a bag of
catalogue phrases against product listings and it lost 0.030 held-out. That is a domain
mismatch, not a verdict on learned text scoring. Here we fine-tune the same base model on
this task's actual distribution.

FAILURE 2 -- the feature model was trained on the wrong decision.
Pass 22 measured the learned ranker inside coverage ties: 32.4% target-first against
popularity's 57.4% -- worse than the LLM's 41.2%, barely above random. The cause is a
train/test mismatch one level up: rows were collected from EVERY turn, so ties are a
minority of training data and the model optimises average-case ranking, where coverage
already dominates. It has popularity as a feature and still loses to popularity alone --
which is only possible if it never specialised on the tie decision.

So this trains ONLY on coverage-tie groups, with a LISTWISE objective that directly
optimises the measured quantity: softmax over the group, cross-entropy against the true
target's position. "Put the target first" is the loss, not a proxy for it.

Bars, fixed in advance:
    within-tie target-first   popularity 57.4%  |  LLM 41.2%  |  feature-LTR 32.4%
    end-to-end                +0.005 held-out to adopt

Run:  PYTHONIOENCODING=utf-8 python -u experiments/log/23_tie_crossencoder.py --mint 12000
"""
from __future__ import annotations

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
    behavior_for, catalog_index, coarse_category, customer_reply, initial_message,
    intent_card, load_jsonl,
)
from submission.agent import Agent, CAT, CONSTRAINT  # noqa: E402

CATALOG = ROOT / "data" / "catalog.jsonl"
PUBLIC = ROOT / "data" / "public_set.jsonl"
GROUPS_CACHE = ROOT / "experiments" / "studies" / ".tie_groups.pkl"
BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def collect_groups(agent, sessions, prods, want_target=True, limit_turns=10):
    """Replay sessions; capture the top coverage-tie group whenever it holds the target."""
    out = []
    for s in sessions:
        tgt = str(s["ground_truth"]["parent_asin"])
        card = intent_card(prods[tgt])
        eff = {**s, "intent_card": card,
               "behavior": behavior_for(str(s["scenario_type"]), card,
                                        random.Random(f"{s['sample_id']}\0{s['scenario_type']}"))}
        disclosed, bu = set(), False
        sid = s["sample_id"]
        agent.reset(sid, s["user_profile"])
        st = agent.sessions[sid]
        msg = initial_message(eff, coarse_category(
            [str(v) for v in (prods[tgt].get("categories") or [])]), disclosed)
        applied = s["scenario_type"] != "intent_override"
        for turn in range(1, limit_turns + 1):
            st.turn += 1
            try:
                agent._observe(st, msg)
                pool = agent._candidates(st, msg)
                ranked = agent._rank(st, pool, 10) if pool else []
            except Exception:
                ranked = []
            if len(ranked) >= 2 and st.evidence:
                wm = {p: agent._weight(p, df, t) for p, (df, t) in st.evidence.items()}

                def cov(a):
                    return sum(w for p, w in wm.items() if agent.ix.covers(a, p))

                j = 1
                c0 = cov(ranked[0])
                while j < len(ranked) and abs(cov(ranked[j]) - c0) < 1e-12:
                    j += 1
                if j >= 2 and ((tgt in ranked[:j]) or not want_target):
                    grp = ranked[:j][:8]
                    if not want_target or tgt in grp:
                        reqs = [p for p, (_, t) in st.evidence.items()
                                if t in (CONSTRAINT, CAT)]
                        out.append({
                            "query": "; ".join(reqs)[:320],
                            "docs": [agent.ix.doc.get(a, "")[:320] for a in grp],
                            "asins": list(grp),
                            "label": grp.index(tgt) if tgt in grp else -1,
                        })
            if applied and tgt in ranked:
                break
            ov = eff.get("behavior", {}).get("override") or {}
            if not applied and turn + 1 == int(ov.get("turn", 3)):
                applied = True
                nv = str(ov.get("new_value", ""))
                if nv:
                    disclosed.add(nv)
                msg = str(ov.get("message", ""))
            else:
                msg, bu = customer_reply(eff, agent._next_probe(st), disclosed, bu)
        agent.sessions.pop(sid, None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mint", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=48, help="candidate pairs per step")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--maxlen", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from torch.nn import functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  torch={torch.__version__}")

    samples = load_jsonl(PUBLIC)
    TUNE = [s for i, s in enumerate(samples) if i % 2 == 0]
    HOLD = [s for i, s in enumerate(samples) if i % 2 == 1]
    cid, cats, prods = catalog_index(CATALOG)
    agent = Agent(CATALOG)
    public_targets = {str(s["ground_truth"]["parent_asin"]) for s in samples}

    # ---------------- synthetic tie groups, sampled to match the real target prior
    if GROUPS_CACHE.exists():
        train_groups = pickle.loads(GROUPS_CACHE.read_bytes())
        print(f"loaded {len(train_groups):,} cached tie groups")
    else:
        def rn(a):
            try:
                return float(prods[a].get("rating_number") or 0)
            except (TypeError, ValueError):
                return 0.0

        cand = [a for a in prods if a not in public_targets]
        w = [rn(a) for a in cand]
        rng = random.Random(args.seed)
        minted = rng.choices(cand, weights=w, k=args.mint)
        SCEN = (["buying"] * 40 + ["browsing"] * 40 +
                ["intent_override"] * 15 + ["boundary"] * 5)
        synth = [{"sample_id": f"syn_{i}", "scenario_type": SCEN[i % len(SCEN)],
                  "user_profile": {"preference_tags": []},
                  "ground_truth": {"parent_asin": a}} for i, a in enumerate(minted)]
        print(f"replaying {len(synth):,} synthetic sessions for TIE GROUPS ...")
        t0 = time.time()
        train_groups = collect_groups(agent, synth, prods)
        GROUPS_CACHE.write_bytes(pickle.dumps(train_groups))
        print(f"  {len(train_groups):,} tie groups  [{time.time()-t0:.0f}s]")

    sizes = [len(g["docs"]) for g in train_groups]
    print(f"  mean group size {statistics.fmean(sizes):.2f}   "
          f"pairs {sum(sizes):,}")

    # ---------------- model
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def score_group(g, train=True):
        enc = tok([g["query"]] * len(g["docs"]), g["docs"], truncation=True,
                  max_length=args.maxlen, padding=True, return_tensors="pt").to(dev)
        return model(**enc).logits.squeeze(-1)

    def evaluate_groups(groups, tag):
        """target-first rate under the model vs under the incoming (popularity) order."""
        model.eval()
        det, ml = [], []
        with torch.no_grad():
            for g in groups:
                if g["label"] < 0:
                    continue
                det.append(g["label"])
                s = score_group(g, train=False)
                order = torch.argsort(s, descending=True).tolist()
                ml.append(order.index(g["label"]))
        model.train()
        f_det = sum(1 for p in det if p == 0) / len(det)
        f_ml = sum(1 for p in ml if p == 0) / len(ml)
        m_det = statistics.fmean(1 / (p + 1) for p in det)
        m_ml = statistics.fmean(1 / (p + 1) for p in ml)
        print(f"  [{tag}] n={len(det)}  "
              f"popularity {f_det:.1%} (MRR {m_det:.4f})  |  "
              f"cross-encoder {f_ml:.1%} (MRR {m_ml:.4f})")
        return {"n": len(det), "det_first": f_det, "ce_first": f_ml,
                "det_mrr": m_det, "ce_mrr": m_ml}

    print("\ncollecting PUBLIC tie groups for evaluation ...")
    eval_tune = collect_groups(agent, TUNE, prods)
    eval_hold = collect_groups(agent, HOLD, prods)
    print(f"  tune {len(eval_tune)}  hold {len(eval_hold)}")

    print("\nbefore fine-tuning (zero-shot, the pass-11 condition):")
    OUT = {"zero_shot": {"tune": evaluate_groups(eval_tune, "tune"),
                         "hold": evaluate_groups(eval_hold, "hold")}}

    # ---------------- listwise training
    print(f"\ntraining: listwise softmax over each tie group, {args.epochs} epoch(s)")
    rng = random.Random(args.seed)
    order = [g for g in train_groups if g["label"] >= 0]
    print(f"  usable groups: {len(order):,}")
    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        rng.shuffle(order)
        run_loss, n = 0.0, 0
        for g in order:
            s = score_group(g)
            loss = F.cross_entropy(s.unsqueeze(0),
                                   torch.tensor([g["label"]], device=dev))
            loss.backward()
            run_loss += loss.item()
            n += 1
            step += 1
            if step % 8 == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
            if n % 2000 == 0:
                print(f"    ep{ep+1} {n:,}/{len(order):,}  loss {run_loss/n:.4f}  "
                      f"[{time.time()-t0:.0f}s]", flush=True)
        opt.step()
        opt.zero_grad(set_to_none=True)
        print(f"  epoch {ep+1} done, mean loss {run_loss/max(1,n):.4f}")
        OUT[f"after_epoch{ep+1}"] = {"tune": evaluate_groups(eval_tune, "tune"),
                                     "hold": evaluate_groups(eval_hold, "hold")}

    best = OUT[f"after_epoch{args.epochs}"]
    print(f"\n  BAR (popularity, tune): {best['tune']['det_first']:.1%}")
    print(f"  cross-encoder (tune):   {best['tune']['ce_first']:.1%}  -> "
          f"{'BEATS' if best['tune']['ce_first'] > best['tune']['det_first'] else 'FAILS'}")
    print(f"  cross-encoder (hold):   {best['hold']['ce_first']:.1%}  vs "
          f"popularity {best['hold']['det_first']:.1%}")
    print(f"  reference: LLM 41.2%, feature-LTR 32.4%, random 16.2%")

    out_dir = ROOT / "experiments" / "studies" / "tie_ce_model"
    if best["hold"]["ce_first"] > best["hold"]["det_first"]:
        out_dir.mkdir(exist_ok=True)
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        print(f"\n  beat the bar on held-out -> saved to {out_dir}")

    (ROOT / "experiments" / "results" / "out_23.json").write_text(
        json.dumps(OUT, indent=2) + "\n", encoding="utf-8")
    print("\n[saved] experiments/results/out_23.json")


if __name__ == "__main__":
    main()
