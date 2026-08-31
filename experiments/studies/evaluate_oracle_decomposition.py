"""V2.43: where exactly does the attribute-paraphrase score go, and what could recover it?

THE QUESTION
------------
Nodes 3/4/5 exist to UNDO the attribute paraphrase. Before spending more on them it is
worth knowing what perfect undoing would even be worth, and which part of the loss is
recoverable at all. That is measurable rather than arguable, so this decomposes the gap.

    canonical replay        0.970100   nothing paraphrased
    paraphrase baseline     0.777000   what V1 scores today
    gap                     0.193100

FIVE ARMS, each isolating one mechanism:

  1 baseline          V1 as shipped on the paraphrased suite.
  2 ORACLE-RESOLVE    every paraphrased value is replaced by its TRUE canonical, injected
                      as evidence. This is a perfect Node 3/4/5 -- the ceiling of the whole
                      semantic programme. Uses ground truth and is a DIAGNOSTIC ONLY; it can
                      never ship.
  3 ORACLE-DROP       every paraphrased value is deleted instead of resolved. Separates two
                      very different diagnoses: if dropping recovers most of the gap, the
                      paraphrased text is actively POLLUTING the miner and the fix is to
                      suppress it, not to resolve it. If dropping recovers little, the loss
                      is genuinely missing evidence and only resolution helps.
  4 ORACLE-TOP1       the true canonical is placed in the candidate list at rank 1, but the
                      agent still integrates it through the normal weak-evidence path. This
                      separates "retrieval is wrong" from "integration is weak": if arm 4
                      is far below arm 2, even a perfect retriever cannot help because the
                      INTEGRATION discards the win.
  5 current resolver  mpnet + NLI at the best measured setting (k=1, w=0.15).

ALSO REPORTED: the resolver's top-1 accuracy against the true canonical, per atom. That
number is the direct measure of what Nodes 3/4/5 currently achieve, independent of how the
result is integrated.

Reading the result:
    arm2 - arm1   total value of a perfect semantic resolver
    arm5 - arm1   what we capture today
    arm3 - arm1   how much of the loss is pollution rather than missing evidence
    arm2 - arm4   how much is lost by the integration path rather than by retrieval

Run:  PYTHONIOENCODING=utf-8 python -u experiments/studies/evaluate_oracle_decomposition.py
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
PVO = V2 / "public_value_only"
OUT = V2 / "results" / "oracle_decomposition_v2_43.json"

_s = ilu.spec_from_file_location("_v2_32", V2 / "evaluate_node4_cluster_aware.py")
_m = ilu.module_from_spec(_s)
_s.loader.exec_module(_m)
normalise = _m.normalise

SEM = "sem"
BI = "sentence-transformers/all-mpnet-base-v2"


def paraphrase_map():
    """paraphrase text -> true canonical, derived by diffing the two released suites."""
    canon = [json.loads(l) for l in (PVO / "official200_canonical_replay.jsonl").open(encoding="utf-8") if l.strip()]
    para = [json.loads(l) for l in (PVO / "official200_attribute_paraphrase_dev.jsonl").open(encoding="utf-8") if l.strip()]
    mapping = {}
    for c, p in zip(canon, para):
        cv = [str(x) for x in c["intent_card"]["hard_constraints"] + c["intent_card"]["soft_preferences"]]
        pv = [str(x) for x in p["intent_card"]["hard_constraints"] + p["intent_card"]["soft_preferences"]]
        for a, b in zip(cv, pv):
            if normalise(a) != normalise(b):
                mapping[normalise(b)] = normalise(a)
    return mapping


def main() -> None:
    os.environ["LLM_EXTRACT"] = "0"
    os.environ["BERT_EXTRACT"] = "0"
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    import numpy as np
    import torch
    from transformers import (AutoModel, AutoModelForSequenceClassification,
                              AutoTokenizer)
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    from submission.agent import CONSTRAINT, Agent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    canon_s = load_jsonl(PVO / "official200_canonical_replay.jsonl")
    para_s = load_jsonl(PVO / "official200_attribute_paraphrase_dev.jsonl")
    pmap = paraphrase_map()
    print(f"suite: {len(para_s)} sessions, {len(pmap)} distinct paraphrase->canonical maps")

    base = Agent(ROOT / "data" / "catalog.jsonl")
    CACHE = ROOT / ".v2_model_cache"

    # ---- the real resolver (mpnet + NLI), for arm 5 and for the accuracy measurement
    canonicals = [json.loads(l)["canonical"]
                  for l in (V2 / "catalogue_attribute_dictionary.jsonl").open(encoding="utf-8")
                  if l.strip()]
    btok = AutoTokenizer.from_pretrained(BI, cache_dir=str(CACHE))
    bmdl = AutoModel.from_pretrained(BI, cache_dir=str(CACHE)).to(device).eval()

    def embed(texts, bs=256):
        outs = []
        for i in range(0, len(texts), bs):
            b = btok(texts[i:i + bs], padding=True, truncation=True, max_length=64,
                     return_tensors="pt").to(device)
            with torch.no_grad():
                h = bmdl(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                outs.append(torch.nn.functional.normalize(v, dim=-1).cpu())
        return torch.cat(outs).numpy()

    matrix = embed(canonicals)
    npath = glob.glob(str(CACHE / "models--cross-encoder--nli-deberta-v3-small" / "snapshots" / "*"))[0]
    ntok = AutoTokenizer.from_pretrained(npath)
    nmdl = AutoModelForSequenceClassification.from_pretrained(npath).to(device).eval()
    ent = next(i for i, l in nmdl.config.id2label.items() if "entail" in str(l).lower())
    rcache: dict[str, str] = {}

    def resolve_top1(phrase: str) -> str:
        if phrase in rcache:
            return rcache[phrase]
        sims = (embed([phrase]) @ matrix.T)[0]
        idx = np.argsort(-sims)[:10]
        cands = [canonicals[int(i)] for i in idx]
        b = ntok([f"The product is {c}." for c in cands],
                 [f"The product is {phrase}."] * len(cands),
                 padding=True, truncation=True, max_length=48, return_tensors="pt").to(device)
        with torch.no_grad():
            e = torch.softmax(nmdl(**b).logits, -1)[:, ent].cpu().numpy()
        rs = 1.0 - np.argsort(np.argsort(-sims[idx])) / max(len(idx) - 1, 1)
        re_ = 1.0 - np.argsort(np.argsort(-e)) / max(len(idx) - 1, 1)
        rcache[phrase] = cands[int(np.argmax(0.5 * rs + 0.5 * re_))]
        return rcache[phrase]

    # ---- resolver accuracy, independent of integration
    correct = sum(1 for p, c in pmap.items() if normalise(resolve_top1(p)) == c)
    print(f"resolver top-1 accuracy on this suite: {correct}/{len(pmap)} = "
          f"{correct/len(pmap):.1%}\n")

    def mk(mode: str, weight: float = 0.15):
        class Arm(Agent):
            def _observe(self, st, msg):
                if mode == "drop":
                    # Remove the paraphrased clause entirely before V1 ever sees it.
                    for text, tier in super()._extract_templated(msg):
                        if tier == CONSTRAINT and normalise(text) in pmap:
                            msg = msg.replace(text, "")
                    return super()._observe(st, msg)
                super()._observe(st, msg)
                if mode == "baseline":
                    return
                for text, tier in super()._extract_templated(msg):
                    if tier != CONSTRAINT:
                        continue
                    key = normalise(text)
                    if key not in pmap:
                        continue
                    if mode in ("oracle", "oracle_top1"):
                        ph = pmap[key]
                    else:
                        ph = normalise(resolve_top1(key))
                    df = self.ix.df(ph)
                    if df > 0 and ph not in st.evidence:
                        # oracle injects at CONSTRAINT strength; the others use the weak tier
                        st.evidence[ph] = (df, CONSTRAINT if mode == "oracle" else SEM)

            def _weight(self, phrase, df, tier):
                if tier == SEM:
                    return weight / (1.0 + df) ** self.IDF_POW
                return super()._weight(phrase, df, tier)
        return Arm

    def run(cls, samples):
        a = object.__new__(cls)
        a.ix, a.sessions = base.ix, {}
        a.llm = a.llm_extract = a.tagger = None
        return round(evaluate(a, samples, cid, cats, prods)["recommended_technical_score"], 6)

    canon_score = run(Agent, canon_s)
    arms = {
        "1 baseline (V1 on paraphrase)": run(mk("baseline"), para_s),
        "2 ORACLE-RESOLVE (perfect 3/4/5)": run(mk("oracle"), para_s),
        "3 ORACLE-DROP (delete paraphrase)": run(mk("drop"), para_s),
        "4 ORACLE-TOP1 (true canon, weak tier)": run(mk("oracle_top1"), para_s),
        "5 current resolver (mpnet+NLI)": run(mk("resolver"), para_s),
    }
    print(f"{'arm':<40}{'score':>10}{'vs baseline':>13}")
    print("-" * 63)
    print(f"{'0 canonical replay (no paraphrase)':<40}{canon_score:>10.6f}{'':>13}")
    b = arms["1 baseline (V1 on paraphrase)"]
    for k, v in arms.items():
        print(f"{k:<40}{v:>10.6f}{v-b:>+13.4f}")

    gap = canon_score - b
    o = arms["2 ORACLE-RESOLVE (perfect 3/4/5)"]
    d = arms["3 ORACLE-DROP (delete paraphrase)"]
    t = arms["4 ORACLE-TOP1 (true canon, weak tier)"]
    c = arms["5 current resolver (mpnet+NLI)"]
    print(f"\n  total gap to close                     {gap:.4f}")
    print(f"  a PERFECT resolver would recover       {o-b:+.4f}  ({(o-b)/gap:.1%} of the gap)")
    print(f"  merely DELETING the paraphrase gets    {d-b:+.4f}  ({(d-b)/gap:.1%})")
    print(f"  perfect retrieval, weak integration    {t-b:+.4f}  ({(t-b)/gap:.1%})")
    print(f"  we currently capture                   {c-b:+.4f}  ({(c-b)/gap:.1%})")
    print(f"\n  lost to INTEGRATION rather than retrieval: {o-t:.4f}")
    print(f"  lost to RETRIEVAL rather than integration: {t-c:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"experiment": "V2.43 oracle decomposition", "canonical": canon_score,
         "arms": arms, "gap": round(gap, 6),
         "resolver_top1_accuracy": round(correct / len(pmap), 4),
         "n_maps": len(pmap)}, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
