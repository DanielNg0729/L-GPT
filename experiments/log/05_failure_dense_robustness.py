"""
Experiment 5: failure analysis, dense fusion, and robustness stress tests.

Three questions that decide the final architecture:

  A. WHY does V4 still miss ~7.5% of sessions? (determines whether a dense
     retriever is the right fix, or whether the misses are a different pathology)
  B. Does a dense bi-encoder fused by RRF actually add anything on top?
  C. Does the whole approach survive the two documented threats --
     (i) organiser-added natural-language paraphrasing of customer messages,
     (ii) ask_attribute='other' behaving differently in the private simulator?

Run:  PYTHONIOENCODING=utf-8 python experiments/log/05_failure_dense_robustness.py
      add --dense to include the (slow) bi-encoder pass
"""
from __future__ import annotations

import json
import random
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index, coarse_category, evaluate, intent_card, load_jsonl,
)

sys.path.insert(0, str(ROOT / "experiments" / "log"))
from importlib import import_module  # noqa: E402

abl = import_module("04_ablation")
Index, Session, V4_Coverage, phrase_of, raw_toks = (
    abl.Index, abl.Session, abl.V4_Coverage, abl.phrase_of, abl.raw_toks)

CATALOG = ROOT / "data" / "catalog.jsonl"
PUBLIC = ROOT / "data" / "public_set.jsonl"
OUT = {}


def section(name: str) -> None:
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")


print("loading ...")
samples = load_jsonl(PUBLIC)
cid, cats, prods = catalog_index(CATALOG)
ix = Index(CATALOG)
print(f"indexed {ix.n:,}\n")

# =========================================================== A. failure analysis
section("A. FAILURE ANALYSIS OF V4")

agent = V4_Coverage(ix)
res = evaluate(agent, samples, cid, cats, prods)
print(f"V4 baseline: HR@10 {res['hit_rate_at_10']:.1%}  MRR {res['mrr']:.3f}  "
      f"SCORE {res['recommended_technical_score']:.4f}")

by_id = {s["sample_id"]: s for s in samples}
misses = [x for x in res["sessions"] if not x["hit"]]
late = [x for x in res["sessions"] if x["hit"] and x["first_hit_turn"] and x["first_hit_turn"] >= 4]
print(f"\nmisses: {len(misses)}/{len(samples)}   late hits (turn>=4): {len(late)}")

print("\nmiss composition:")
print(f"  by scenario   : {dict(Counter(by_id[m['sample_id']]['scenario_type'] for m in misses))}")
print(f"  by difficulty : {dict(Counter(by_id[m['sample_id']]['difficulty_bucket'] for m in misses))}")

# Is the target even reachable? Check whether its constraints are verbatim-present.
print("\nroot-cause probe on each miss (is the evidence even in the index?):")
causes = Counter()
for m in misses[:40]:
    s = by_id[m["sample_id"]]
    tgt = str(s["ground_truth"]["parent_asin"])
    card = intent_card(prods[tgt])
    cons = [str(x) for x in card["hard_constraints"]] + [str(x) for x in card["soft_preferences"]]
    cons = list(dict.fromkeys(cons))
    present = sum(1 for c in cons if ix.covers(tgt, phrase_of(c).strip('"')))
    cat = coarse_category([str(v) for v in (prods[tgt].get("categories") or [])])
    cat_ok = ix.covers(tgt, phrase_of(cat).strip('"'))
    # how many products share the FULL constraint conjunction?
    parts = [phrase_of(c) for c in cons if phrase_of(c).strip(' "')]
    n_and = len(ix.search(" AND ".join(parts), 500)) if parts else -1
    if present < len(cons):
        causes["constraint not verbatim in own text (truncation/normalisation drift)"] += 1
    elif n_and > 200:
        causes["constraints non-selective (huge conjunction pool)"] += 1
    elif not cat_ok:
        causes["category string absent from own text"] += 1
    else:
        causes["evidence present & selective -> RANKING failure, not recall"] += 1
print(f"{'cause':<64} {'n':>4}")
for c, n in causes.most_common():
    print(f"  {c:<62} {n:>4}")
OUT["miss_causes"] = dict(causes)

# recall ceiling: is the target anywhere in the pool at all?
print("\nrecall check -- is the target inside V4's candidate pool by the final turn?")
in_pool = 0
for m in misses:
    s = by_id[m["sample_id"]]
    tgt = str(s["ground_truth"]["parent_asin"])
    card = intent_card(prods[tgt])
    cons = list(dict.fromkeys([str(x) for x in card["hard_constraints"]] +
                              [str(x) for x in card["soft_preferences"]]))
    parts = [phrase_of(c) for c in cons if phrase_of(c).strip(' "')]
    pool = ix.search(" OR ".join(parts), 2000) if parts else []
    if tgt in pool:
        in_pool += 1
print(f"  target present in a 2000-deep OR pool for {in_pool}/{len(misses)} misses")
print("  => misses that ARE in the deep pool are RANKING failures (a reranker can fix them);")
print("     misses that are NOT are RECALL failures (need a different retrieval channel).")
OUT["misses_recoverable_by_rerank"] = f"{in_pool}/{len(misses)}"

# =========================================================== C. robustness
section("C. ROBUSTNESS STRESS TESTS")

PARA_PREFIX = ["Hmm, ", "Okay so ", "Right, ", "Well, ", "Let me think - ", ""]
PARA_REWRITE = [
    (re.compile(r"I'm looking for", re.I), ["I need", "I'm after", "I want to find", "Looking for"]),
    (re.compile(r"A key requirement is:", re.I), ["It has to have", "One thing I need:", "Must be"]),
    (re.compile(r"For that, what matters is:", re.I), ["What I care about:", "Mainly", "The important bit is"]),
    (re.compile(r"but I'm still exploring", re.I), ["though I'm just browsing", "not sure exactly yet"]),
]


class Paraphraser:
    """Wraps an agent and paraphrases the customer message BEFORE the agent sees it.

    This simulates the organiser's stated option to add natural-language
    paraphrasing to the simulator ("If natural-language paraphrasing is added by
    the organizer, it cannot decide correctness" -- competition_specification.md).
    """

    def __init__(self, inner, seed: int = 0, strength: str = "light"):
        self.inner = inner
        self.rng = random.Random(seed)
        self.strength = strength

    def reset(self, session_id, user_profile):
        self.inner.reset(session_id, user_profile)

    def _mangle(self, msg: str) -> str:
        out = msg
        for pat, opts in PARA_REWRITE:
            if pat.search(out):
                out = pat.sub(self.rng.choice(opts), out, count=1)
        out = self.rng.choice(PARA_PREFIX) + out
        if self.strength == "heavy":
            # also drop the colon structure entirely and shuffle clause order
            out = out.replace(":", "")
            parts = [p.strip() for p in out.split(".") if p.strip()]
            self.rng.shuffle(parts)
            out = ". ".join(parts) + "."
        return out

    def respond(self, session_id, user_message, turn, top_k):
        return self.inner.respond(session_id, self._mangle(user_message), turn, top_k)


class NoOtherProbe(V4_Coverage):
    """'other' is disabled: fall back to a rotating typed probe.

    Guards against the private simulator implementing `other` differently. Rotates
    across the buckets classify_constraint can actually emit, most-frequent first.
    """
    ROTATION = ["feature", "material", "color", "style", "size", "use_case"]

    def respond(self, session_id, user_message, turn, top_k):
        r = super().respond(session_id, user_message, turn, top_k)
        r["ask_attribute"] = self.ROTATION[(turn - 1) % len(self.ROTATION)]
        return r


class NoTemplates(V4_Coverage):
    """Template regexes disabled -- pure bag-of-tokens accumulation.

    This is the true worst case: the organiser paraphrases so heavily that NO
    structured extraction fires. Establishes the floor of the design.
    """

    def respond(self, session_id, user_message, turn, top_k):
        st = self.s.setdefault(session_id, Session())
        # bypass structured parse: treat the entire message as one evidence blob
        st.turn += 1
        if not st.category and st.turn == 1:
            st.category = user_message
        blob = user_message.strip(" .,")
        if blob and blob not in st.constraints:
            st.constraints.append(blob)
        for t in abl.toks(user_message):
            if t not in st.raw_tokens:
                st.raw_tokens.append(t)
        return {"message": "Here are the closest matches.", "ask_attribute": "other",
                "recommendations": [{"parent_asin": a} for a in self._query(st, top_k)],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}}


def run(name, ag):
    t0 = time.time()
    r = evaluate(ag, samples, cid, cats, prods)
    print(f"{name:<44} HR@10 {r['hit_rate_at_10']:>6.1%}  MRR {r['mrr']:>6.3f}  "
          f"MTTC {r['mttc']:>5.2f}  SCORE {r['recommended_technical_score']:>6.4f}  ({time.time()-t0:.0f}s)")
    return r["recommended_technical_score"]


print(f"{'condition':<44}")
print("-" * 100)
rob = {}
rob["clean (reference)"] = run("clean (reference)", V4_Coverage(ix))
rob["light paraphrase"] = run("light paraphrase", Paraphraser(V4_Coverage(ix), 1, "light"))
rob["heavy paraphrase"] = run("heavy paraphrase", Paraphraser(V4_Coverage(ix), 2, "heavy"))
rob["no 'other' (typed rotation)"] = run("no 'other' (typed probe rotation)", NoOtherProbe(ix))
rob["no templates (bag of tokens)"] = run("no templates (raw blob only)", NoTemplates(ix))
rob["no templates + heavy para"] = run("no templates + heavy paraphrase",
                                       Paraphraser(NoTemplates(ix), 3, "heavy"))
OUT["robustness"] = rob

print("\n=> The gap between 'clean' and 'no templates + heavy paraphrase' is the")
print("   exposure carried by template-matching. Design target: keep that gap small.")

# =========================================================== B. dense
if "--dense" in sys.argv:
    section("B. DENSE BI-ENCODER FUSION (fastembed / bge-small-en-v1.5)")
    import numpy as np
    from fastembed import TextEmbedding

    t0 = time.time()
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    asins, texts = [], []
    with CATALOG.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            asins.append(str(d["parent_asin"]))
            t = f"{Index._f(d.get('title'))} {Index._f(d.get('categories'))} {Index._f(d.get('features'))}"
            texts.append(t[:512])
    print(f"embedding {len(texts):,} products (this is the slow part) ...")
    mat = np.array(list(model.embed(texts, batch_size=256)), dtype=np.float32)
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    pos = {a: i for i, a in enumerate(asins)}
    print(f"embedded in {time.time()-t0:.0f}s -> matrix {mat.shape} "
          f"({mat.nbytes/1e6:.0f} MB in RAM)")

    class V5_Hybrid(V4_Coverage):
        """+ dense channel fused with the sparse ranking by Reciprocal Rank Fusion."""
        K_RRF = 60

        def _query(self, st, top_k):
            sparse = super()._query(st, max(top_k, 60))
            q = " ".join([st.category] + st.constraints)[:512]
            if not q.strip():
                return sparse[:top_k]
            qv = np.array(list(model.embed([q])), dtype=np.float32)[0]
            qv /= (np.linalg.norm(qv) + 1e-9)
            sims = mat @ qv
            dense = [asins[i] for i in np.argpartition(-sims, 60)[:60]]
            dense.sort(key=lambda a: -sims[pos[a]])
            score: dict[str, float] = {}
            for rank, a in enumerate(sparse):
                score[a] = score.get(a, 0.0) + 1.0 / (self.K_RRF + rank + 1)
            for rank, a in enumerate(dense):
                score[a] = score.get(a, 0.0) + 1.0 / (self.K_RRF + rank + 1)
            return sorted(score, key=lambda a: -score[a])[:top_k]

    class V5_DenseOnly(V4_Coverage):
        def _query(self, st, top_k):
            q = " ".join([st.category] + st.constraints)[:512]
            if not q.strip():
                return []
            qv = np.array(list(model.embed([q])), dtype=np.float32)[0]
            qv /= (np.linalg.norm(qv) + 1e-9)
            sims = mat @ qv
            idx = np.argpartition(-sims, top_k)[:top_k]
            return [asins[i] for i in sorted(idx, key=lambda i: -sims[i])]

    print()
    run("V5a dense only", V5_DenseOnly(ix))
    run("V5b sparse+dense RRF", V5_Hybrid(ix))
    run("V4 sparse only (reference)", V4_Coverage(ix))

Path(ROOT / "experiments" / "results" / "out_05.json").write_text(json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print("\n[saved] experiments/results/out_05.json")
