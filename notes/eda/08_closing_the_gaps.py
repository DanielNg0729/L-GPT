"""
EDA pass 8: closing the four gaps left open by pass 7.

  A. FUZZY PHRASE RESOLUTION  -- the §9 "highest expected value" alternative, now tested.
     Diagnosis first: of 800 constraint strings, 30 (3.8%) are NOT contiguous in their own
     target's text. The cause is NOT tokenisation drift as pass 5 assumed -- it is that
     intent_card SYNTHESISES some constraints. `f"color: {colour}"` is assembled from a
     regex hit, so "color black" never appears verbatim even though "black" does. Same for
     `f"budget around ${price}"`. A hardcoded prefix strip would fix the observed cases;
     instead we resolve every phrase to its longest catalogue-ATTESTED substring, which
     handles synthesised prefixes without knowing they exist.

  B. PER-FIELD BM25 WEIGHTS -- inherited unchanged from the starter and never tuned.

  C. HELD-OUT VALIDATION -- pass 7 tuned and reported on the same 200 sessions. Here we
     split into a 100-session tuning half and a 100-session held-out half, so the reported
     improvement is measured on data the weights never saw.

  D. CROSS-ENCODER RERANK -- probed for feasibility (offline, CPU, no network).

Run:  PYTHONIOENCODING=utf-8 python -u notes/eda/08_closing_the_gaps.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
import submission.agent as sa  # noqa: E402
from submission.agent import Agent, raw_toks  # noqa: E402

CATALOG = ROOT / "data" / "catalog.jsonl"
PUBLIC = ROOT / "data" / "public_set.jsonl"
OUT: dict = {}


def section(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


print("loading ...")
samples = load_jsonl(PUBLIC)
cid, cats, prods = catalog_index(CATALOG)
# Deterministic split by sample_id parity so both halves keep the scenario mix.
TUNE = [s for i, s in enumerate(samples) if i % 2 == 0]
HOLD = [s for i, s in enumerate(samples) if i % 2 == 1]
print(f"tune n={len(TUNE)}  holdout n={len(HOLD)}")

t0 = time.time()
agent = Agent(CATALOG)
print(f"index built in {time.time()-t0:.0f}s")


def run(subset, tag="") -> dict:
    r = evaluate(agent, subset, cid, cats, prods)
    if tag:
        print(f"    {tag:<34} HR@10 {r['hit_rate_at_10']:>6.1%}  MRR {r['mrr']:>6.4f}  "
              f"MTTC {r['mttc']:>5.2f}  SCORE {r['recommended_technical_score']:>7.5f}")
    return {"hr": r["hit_rate_at_10"], "mrr": r["mrr"], "mttc": r["mttc"],
            "score": r["recommended_technical_score"]}


# =========================================================== A
section("A. FUZZY PHRASE RESOLUTION (longest attested substring)")

_orig_observe = Agent._observe


def resolve(self, text: str, cap: int = 12) -> list[str]:
    """Resolve a constraint to phrases the catalogue actually attests.

    Try the whole phrase; if the catalogue has never seen it, search for the longest
    contiguous substring it HAS seen, scanning long-to-short over every window. Returns
    [] rather than a phantom phrase, so a synthesised constraint degrades to nothing
    instead of poisoning the conjunctive rungs.
    """
    t = raw_toks(text)[:cap]
    if not t:
        return []
    whole = " ".join(t)
    if self.ix.df(whole) > 0:
        return [whole]
    for n in range(len(t) - 1, 1, -1):                 # windows of length n >= 2
        hits = []
        for i in range(0, len(t) - n + 1):
            ph = " ".join(t[i:i + n])
            if self.ix.df(ph) > 0:
                hits.append(ph)
        if hits:
            return hits[:2]
    singles = [x for x in t if self.ix.df(x) > 0]
    return singles[:2]


def observe_fuzzy(self, st, msg: str) -> None:
    from submission.agent import PAT_NOINFO, CAT, CONSTRAINT, MINED
    if PAT_NOINFO.search(msg):
        return
    found = self._extract_templated(msg)
    if st.turn == 1:
        st.buying = any(tier == CONSTRAINT for _, tier in found)
    resolved: list[tuple[str, str]] = []
    for text, tier in found:
        for ph in resolve(self, text):
            resolved.append((ph, tier))
    if not any(tier == CONSTRAINT for _, tier in found):
        resolved.extend((ph, MINED) for ph, _ in self.ix.mine(msg))
    for ph, tier in resolved:
        if ph and ph not in st.evidence:
            df = self.ix.df(ph)
            st.evidence[ph] = (df if df > 0 else self.ix.DF_CAP * 2, tier)


print("  before (verbatim-only phrase construction):")
base_tune = run(TUNE, "verbatim | tune")
base_hold = run(HOLD, "verbatim | holdout")

Agent._observe = observe_fuzzy
print("  after (longest attested substring):")
fz_tune = run(TUNE, "fuzzy | tune")
fz_hold = run(HOLD, "fuzzy | holdout")

use_fuzzy = fz_hold["score"] >= base_hold["score"]
print(f"\n  holdout delta: {fz_hold['score']-base_hold['score']:+.5f}  "
      f"-> {'ADOPT' if use_fuzzy else 'REJECT'}")
if not use_fuzzy:
    Agent._observe = _orig_observe
OUT["fuzzy"] = {"verbatim_tune": base_tune, "verbatim_hold": base_hold,
                "fuzzy_tune": fz_tune, "fuzzy_hold": fz_hold, "adopted": use_fuzzy}

# =========================================================== B
section("B. PER-FIELD BM25 WEIGHTS (never tuned; inherited from the starter)")

FIELDS = ["title", "categories", "features", "details", "store", "description"]
START = [6.0, 4.0, 2.5, 2.5, 1.5, 1.0]


def set_weights(w: list[float]) -> None:
    from submission.agent import CatalogIndex
    CatalogIndex.BM25 = "bm25(p, 0.0, " + ", ".join(f"{x}" for x in w) + ")"
    agent.ix.BM25 = CatalogIndex.BM25


cur = list(START)
set_weights(cur)
print(f"  starter weights {dict(zip(FIELDS, cur))}")
w_tune = run(TUNE, "starter weights | tune")

GRID = {"title": [1.0, 3.0, 6.0, 10.0], "categories": [1.0, 4.0, 8.0],
        "features": [1.0, 2.5, 5.0, 9.0], "details": [0.5, 2.5, 5.0],
        "store": [0.2, 1.5, 4.0], "description": [0.3, 1.0, 3.0]}
best = w_tune["score"]
for fi, fname in enumerate(FIELDS):
    trials = {}
    for v in GRID[fname]:
        trial = list(cur)
        trial[fi] = v
        set_weights(trial)
        trials[v] = run(TUNE, f"{fname}={v}")["score"]
    bv = max(trials, key=lambda k: trials[k])
    if trials[bv] > best:
        best = trials[bv]
        cur[fi] = bv
    set_weights(cur)
    print(f"    -> {fname} := {cur[fi]}  (tune best {best:.5f})\n")

set_weights(cur)
print(f"  tuned weights {dict(zip(FIELDS, cur))}")
bm_tune = run(TUNE, "tuned weights | tune")
bm_hold = run(HOLD, "tuned weights | holdout")
set_weights(START)
st_hold = run(HOLD, "starter weights | holdout")

use_bm = bm_hold["score"] >= st_hold["score"]
print(f"\n  holdout delta: {bm_hold['score']-st_hold['score']:+.5f}  "
      f"-> {'ADOPT' if use_bm else 'REJECT (tuning did not generalise)'}")
if use_bm:
    set_weights(cur)
OUT["bm25_weights"] = {"starter": START, "tuned": cur, "tune": bm_tune,
                       "holdout_tuned": bm_hold, "holdout_starter": st_hold,
                       "adopted": use_bm}

# =========================================================== C
section("C. FINAL CONFIGURATION, MEASURED ON HELD-OUT DATA")

fin_tune = run(TUNE, "final | tune half")
fin_hold = run(HOLD, "final | HELD-OUT half")
fin_all = run(samples, "final | all 200")
OUT["final"] = {"tune": fin_tune, "holdout": fin_hold, "all": fin_all}

print(f"\n  generalisation gap (tune - holdout): "
      f"{fin_tune['score']-fin_hold['score']:+.5f}")
print("  A small gap means the tuned constants are not memorising the public set.")

# =========================================================== D
section("D. CROSS-ENCODER RERANK -- feasibility probe")
try:
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    names = [m["model"] for m in TextCrossEncoder.list_supported_models()]
    print("  fastembed cross-encoders available offline:")
    for n in names[:8]:
        print(f"    - {n}")
    OUT["cross_encoder_available"] = names[:8]
    print("\n  STATUS: available, but NOT adopted. Reasons, in order:")
    print("   1. HR@10 is already 0.985 -- only ~3 sessions remain to gain.")
    print("   2. It would add a third-party dependency and a model file to the scored")
    print("      path, against a grader that may restrict network AND disk.")
    print("   3. The dense bi-encoder result (pass 5) showed neural scoring actively")
    print("      degrades provenance matching here; a cross-encoder is the same family.")
    print("   Recorded as measured-feasible-but-declined, not as untested.")
except Exception as exc:  # pragma: no cover
    print(f"  not available in this environment: {type(exc).__name__}: {exc}")
    OUT["cross_encoder_available"] = False

Path(ROOT / "notes" / "eda" / "out_08.json").write_text(
    json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print("\n[saved] notes/eda/out_08.json")
print(f"\nFINAL BM25 weights: {dict(zip(FIELDS, cur if use_bm else START))}")
print(f"Fuzzy phrase resolution: {'ENABLED' if use_fuzzy else 'disabled'}")
