"""V2.40: semantic candidates as WEAK EVIDENCE, tuned on a concentration-resistant metric.

WHY RECALL@1 WAS THE WRONG TARGET
----------------------------------
Nodes 3/4/5 have been scored by Recall@1 over a 7,922-phrase dictionary. But Node 7 does
not consume a single canonical -- the roadmap says "add WEAK SEMANTIC EVIDENCE to V1 ranking
... a separate, pessimistically weighted tier". If the resolver hands V1 several candidates
at low weight and one is right, V1's coverage scorer rewards the product matching it, and
the wrong ones are broad, low-weight and largely inert -- exactly how `W_MINED` already
treats mined n-grams. Under that design the operative quantity is Recall@k (0.58 at k=5,
0.87 at k=100), not Recall@1 (0.37).

THE INSTRUMENT, AND ITS TWO KNOWN WEAKNESSES
---------------------------------------------
`public_value_only/` replays the REAL Official200 sessions -- same targets, profiles,
scenarios, wrappers, scoring -- changing only constraint VALUES (551 atoms, 192 sessions).
So it reports end-to-end TechnicalScore on the organizer's own harness, which is the metric
every other decision in this project rests on.

    canonical replay            0.970100   must not move (non-interference)
    literal V1 on paraphrases   0.777000   the number to beat
    headroom                    0.193100

Weakness 1 -- CONCENTRATION. 551 atoms come from 27 rules, and the top five (cotton 104,
imported 99, polyester 81, leather 52, rubber 34) are 67% of all rewrites. A configuration
that happens to fix cotton would dominate the session-weighted score while resolving almost
nothing.

Weakness 2 -- UNNATURAL PARAPHRASES. The substitutions are encyclopedic definitions
("100 cotton" -> "made from a soft plant fibre"), not shopper language.

NEITHER IS EXPLOITED, AND THE FIRST IS NEUTRALISED. Tuning is done on a MACRO metric:
hit-rate is computed per distinct substitution and then averaged across substitutions with
equal weight, so cotton's 104 sessions count exactly as much as breathable's 1. A
configuration that improves the session-weighted score without improving the macro metric
is fitting the instrument's concentration and is rejected on that basis. The 27-rule list
is never an input to the resolver, which sees only the message and the dictionary.

TRAIN/TEST HYGIENE: verified disjoint. Of 155 distinct (canonical -> paraphrase)
substitutions in this suite, 0 appear in the 317-pair training corpora.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/studies/evaluate_semantic_weak_evidence.py
"""
from __future__ import annotations

import argparse
import glob
import importlib.util as ilu
import json
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "experiments" / "studies"
OUT = V2 / "results" / "semantic_weak_evidence_v2_40.json"
PVO = V2 / "public_value_only"

_s = ilu.spec_from_file_location("_v2_32", V2 / "evaluate_node4_cluster_aware.py")
_m = ilu.module_from_spec(_s)
_s.loader.exec_module(_m)
normalise, load_encoder = _m.normalise, _m.load_encoder

SEM = "sem"


def substitutions(canon_rows, para_rows):
    """Per session, the set of paraphrase strings substituted in. Used ONLY for reporting."""
    per_session = []
    for c, p in zip(canon_rows, para_rows):
        cv = [str(x) for x in c["intent_card"]["hard_constraints"] + c["intent_card"]["soft_preferences"]]
        pv = [str(x) for x in p["intent_card"]["hard_constraints"] + p["intent_card"]["soft_preferences"]]
        subs = {normalise(b) for a, b in zip(cv, pv) if normalise(a) != normalise(b)}
        per_session.append(subs)
    return per_session


def macro_hit_rate(sessions, per_session_subs):
    """Hit-rate per distinct substitution, averaged with EQUAL weight per substitution.

    This is what stops a configuration that only fixes `cotton` (104 atoms) from looking
    better than one that fixes twenty rare concepts.
    """
    hit_by_sub: dict[str, list[int]] = {}
    for row, subs in zip(sessions, per_session_subs):
        for s in subs:
            hit_by_sub.setdefault(s, []).append(1 if row["hit"] else 0)
    if not hit_by_sub:
        return 0.0, 0
    rates = [sum(v) / len(v) for v in hit_by_sub.values()]
    return statistics.fmean(rates), len(rates)


def build_resolver(device, use_nli: bool):
    import numpy as np
    import torch
    canonicals = [json.loads(l)["canonical"]
                  for l in (V2 / "catalogue_attribute_dictionary.jsonl").open(encoding="utf-8")
                  if l.strip()]
    # V2.41: all-mpnet-base-v2 beat MiniLM-L6 on BOTH benchmarks (corpus MRR +0.0712,
    # dev200 MRR +0.0373) and was the only candidate to do so. BGE and E5 lost on dev200
    # because they are trained for asymmetric query->passage retrieval, while this task is
    # symmetric short-phrase matching.
    from transformers import AutoModel, AutoTokenizer
    repo = os.environ.get("V2_ENCODER", "sentence-transformers/all-mpnet-base-v2")
    _tok = AutoTokenizer.from_pretrained(repo, cache_dir=str(ROOT / ".v2_model_cache"))
    _mdl = AutoModel.from_pretrained(repo, cache_dir=str(ROOT / ".v2_model_cache")).to(device).eval()

    def encode(texts, bs=256):
        outs = []
        for i in range(0, len(texts), bs):
            b = _tok(texts[i:i + bs], padding=True, truncation=True, max_length=64,
                     return_tensors="pt").to(device)
            with torch.no_grad():
                h = _mdl(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                outs.append(torch.nn.functional.normalize(v, dim=-1).cpu())
        return torch.cat(outs).numpy()

    matrix = encode(canonicals)
    nli = ntok = None
    ent_idx = 0
    if use_nli:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = glob.glob(str(ROOT / ".v2_model_cache" /
                             "models--cross-encoder--nli-deberta-v3-small" /
                             "snapshots" / "*"))[0]
        ntok = AutoTokenizer.from_pretrained(path)
        nli = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
        ent_idx = next(i for i, l in nli.config.id2label.items()
                       if "entail" in str(l).lower())
    cache: dict[str, list[str]] = {}

    def resolve(phrase: str, k: int) -> list[str]:
        key = f"{phrase}\0{k}"
        if key in cache:
            return cache[key]
        sims = (encode([phrase]) @ matrix.T)[0]
        idx = np.argsort(-sims)[:max(k, 10)]
        cands = [canonicals[int(i)] for i in idx]
        if nli is not None and cands:
            b = ntok([f"The product is {c}." for c in cands],
                     [f"The product is {phrase}."] * len(cands),
                     padding=True, truncation=True, max_length=48,
                     return_tensors="pt").to(device)
            with torch.no_grad():
                ent = torch.softmax(nli(**b).logits, -1)[:, ent_idx].cpu().numpy()
            rs = 1.0 - np.argsort(np.argsort(-sims[idx])) / max(len(idx) - 1, 1)
            re_ = 1.0 - np.argsort(np.argsort(-ent)) / max(len(idx) - 1, 1)
            cands = [cands[i] for i in np.argsort(-(0.5 * rs + 0.5 * re_))]
        cache[key] = cands[:k]
        return cache[key]
    return resolve


def make_cls(resolve, topk: int, weight: float):
    from submission.agent import CONSTRAINT, Agent

    class SemanticWeakEvidence(Agent):
        SEM_TOPK, SEM_WEIGHT = topk, weight

        def _observe(self, st, msg):
            super()._observe(st, msg)
            for text, tier in self._extract_templated(msg):
                if tier != CONSTRAINT:
                    continue
                # G3 entry condition: no literal resolution for the whole phrase. Necessary,
                # not sufficient -- so the output is a separate weak tier, never a
                # replacement for literal evidence.
                if self.ix.df(normalise(text)) > 0:
                    continue
                for cand in resolve(" ".join(text.lower().split()), self.SEM_TOPK):
                    ph = normalise(cand)
                    if not ph or ph in st.evidence:
                        continue
                    df = self.ix.df(ph)
                    if df > 0:
                        st.evidence[ph] = (df, SEM)

        def _weight(self, phrase, df, tier):
            if tier == SEM:
                return self.SEM_WEIGHT / (1.0 + df) ** self.IDF_POW
            return super()._weight(phrase, df, tier)
    return SemanticWeakEvidence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["LLM_EXTRACT"] = "0"
    os.environ["BERT_EXTRACT"] = "0"     # wrappers are intact here; the tagger is not applicable
    import torch
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    from submission.agent import Agent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    canon = load_jsonl(PVO / "official200_canonical_replay.jsonl")
    para = load_jsonl(PVO / "official200_attribute_paraphrase_dev.jsonl")
    subs = substitutions(canon, para)
    print(f"suite: {len(para)} sessions, "
          f"{len({s for ss in subs for s in ss})} distinct substitutions, "
          f"{sum(len(s) for s in subs)} atoms")

    base = Agent(ROOT / "data" / "catalog.jsonl")

    def run(cls):
        out = {}
        for name, samples in (("canonical", canon), ("paraphrase", para)):
            a = object.__new__(cls)
            a.ix, a.sessions = base.ix, {}
            a.llm = a.llm_extract = a.tagger = None
            r = evaluate(a, samples, cid, cats, prods)
            out[name] = round(r["recommended_technical_score"], 6)
            if name == "paraphrase":
                mh, nsub = macro_hit_rate(r["sessions"], subs)
                out["macro_hit"] = round(mh, 6)
                out["n_subs"] = nsub
        return out

    ref = run(Agent)
    print(f"\nbaseline  canonical {ref['canonical']:.6f}  paraphrase {ref['paraphrase']:.6f}"
          f"  macro-hit {ref['macro_hit']:.4f} over {ref['n_subs']} substitutions")
    print(f"          headroom {ref['canonical'] - ref['paraphrase']:.6f}\n")

    resolve = build_resolver(device, args.nli)
    report = {"experiment": "V2.40 semantic weak evidence", "nli_rerank": args.nli,
              "baseline": ref, "grid": {}}

    print(f"{'k':>3}{'w':>7}{'canonical':>12}{'paraphrase':>12}{'d-para':>9}"
          f"{'macro':>9}{'d-macro':>9}  verdict")
    print("-" * 74)
    for topk in (1, 3, 5, 10):
        for weight in (0.05, 0.15, 0.30, 0.60):
            r = run(make_cls(resolve, topk, weight))
            dp = r["paraphrase"] - ref["paraphrase"]
            dm = r["macro_hit"] - ref["macro_hit"]
            dc = r["canonical"] - ref["canonical"]
            if abs(dc) > 1e-9:
                verdict = "REJECT canonical moved"
            elif dp > 0.002 and dm <= 0:
                verdict = "REJECT concentration-only"
            elif dm > 0.002:
                verdict = "gain"
            else:
                verdict = "no gain"
            report["grid"][f"k{topk}_w{weight}"] = {
                "topk": topk, "weight": weight, **r,
                "d_para": round(dp, 6), "d_macro": round(dm, 6),
                "d_canon": round(dc, 6), "verdict": verdict}
            print(f"{topk:>3}{weight:>7.2f}{r['canonical']:>12.6f}{r['paraphrase']:>12.6f}"
                  f"{dp:>+9.4f}{r['macro_hit']:>9.4f}{dm:>+9.4f}  {verdict}")

    ok = [v for v in report["grid"].values()
          if abs(v["d_canon"]) < 1e-9 and v["d_macro"] > 0 and v["d_para"] > 0]
    if ok:
        best = max(ok, key=lambda v: v["d_macro"])
        report["selected"] = best
        print(f"\n  selected on MACRO gain (not session-weighted): "
              f"k={best['topk']} w={best['weight']}")
        print(f"    paraphrase {ref['paraphrase']:.6f} -> {best['paraphrase']:.6f} "
              f"({best['d_para']:+.4f}, {best['d_para']/(ref['canonical']-ref['paraphrase']):.1%} "
              f"of headroom)")
        print(f"    macro-hit  {ref['macro_hit']:.4f} -> {best['macro_hit']:.4f} "
              f"({best['d_macro']:+.4f})")
        print(f"    canonical unchanged at {best['canonical']:.6f}")
    else:
        print("\n  Nothing gained on BOTH the session score and the macro metric while")
        print("  leaving canonical untouched. Configurations that moved only the")
        print("  session-weighted score were rejected as concentration artifacts.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
