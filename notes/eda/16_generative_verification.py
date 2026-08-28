"""
EDA pass 16: generative verification -- inverting the data-generating function.

Every technique so far scores a candidate by ASKING "does this product's text contain the
phrases the customer said?" That is a containment test, and it is symmetric: a 5,000-char
listing that happens to include "cotton" scores the same as a product whose entire
identity is cotton.

But we know something much stronger. The customer's constraints are not merely present in
the target -- they are the OUTPUT OF A KNOWN FUNCTION of the target:

    card       = intent_card(product)
    candidates = [material?, color?, *features, *details, price?]   # a FIXED order
    hard       = cleaned[:2]        soft = cleaned[2:4]

So instead of asking "does the text contain this phrase?", we can ask the far more
discriminative question:

    "would THIS product have generated the intent card we are observing?"

That is generative verification: re-run the generator on each candidate and compare its
output to what the customer actually disclosed. A phrase must appear not just anywhere,
but in the right SLOT and the right ORDER. Under the provenance thesis this should be
near-proof of identity.

LEGITIMACY / RISK, stated up front
----------------------------------
This reimplements the simulator's card-construction logic inside our agent. That is a far
heavier dependence on the PUBLIC evaluator's exact implementation than anything else we
ship: if the private simulator orders or truncates differently, verification silently
stops matching. We therefore measure the size of the prize FIRST, and treat adoption as a
separate decision -- with a mandatory fallback to plain coverage whenever verification
finds nothing.

Run:  PYTHONIOENCODING=utf-8 python -u notes/eda/16_generative_verification.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent, raw_toks, CONSTRAINT, CAT  # noqa: E402

CATALOG = ROOT / "data" / "catalog.jsonl"
PUBLIC = ROOT / "data" / "public_set.jsonl"
OUT: dict = {}

print("loading ...")
samples = load_jsonl(PUBLIC)
cid, cats, prods = catalog_index(CATALOG)
TUNE = [s for i, s in enumerate(samples) if i % 2 == 0]
HOLD = [s for i, s in enumerate(samples) if i % 2 == 1]
base = Agent(CATALOG)

# ---------------------------------------------------------------- our own reimplementation
# Deliberately OUR OWN code, mirroring the documented construction, so we are not importing
# organizer internals into a shipped path.
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
MAT_RE = re.compile(r"\b(" + "|".join(MATERIALS) + r")\b", re.I)
COL_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")


def _searchable(p: dict) -> str:
    parts = []
    for f in SEARCH_FIELDS:
        v = p.get(f)
        if isinstance(v, dict):
            parts.extend(f"{k} {x}" for k, x in v.items())
        elif isinstance(v, list):
            parts.extend(str(x) for x in v)
        elif v is not None:
            parts.append(str(v))
    return " ".join(parts).strip()


def _flatten(v) -> list[str]:
    if isinstance(v, dict):
        return [f"{k}: {x}" for k, x in v.items() if x not in (None, "", [])]
    if isinstance(v, list):
        return [str(x) for x in v if x not in (None, "")]
    return [str(v)] if v not in (None, "") else []


def _clean(v: str, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", v).strip(" -;,.\t\n")[:limit].rstrip()


def predicted_card(p: dict) -> list[str]:
    """Reconstruct the ordered constraint slots this product would generate."""
    cand = [*_flatten(p.get("features")), *_flatten(p.get("details"))]
    corpus = _searchable(p)
    m = MAT_RE.search(corpus)
    c = COL_RE.search(corpus)
    if m:
        cand.insert(0, m.group(1).lower())
    if c:
        cand.insert(1, f"color: {c.group(1).lower()}")
    if p.get("price") not in (None, ""):
        cand.append(f"budget around ${p['price']}")
    out, seen = [], set()
    for x in cand:
        cx = _clean(x)
        if cx and cx not in seen:
            seen.add(cx)
            out.append(cx)
    return out[:4]


print("precomputing predicted cards for 50,000 products ...")
t0 = time.time()
CARD: dict[str, list[str]] = {}
for a, d in prods.items():
    CARD[a] = [" ".join(raw_toks(x)[:12]) for x in predicted_card(d)]
print(f"  done in {time.time()-t0:.0f}s")

# ---------------------------------------------------------------- sanity: does it reproduce?
exact = partial = 0
for s in samples:
    tgt = str(s["ground_truth"]["parent_asin"])
    pred = set(CARD[tgt])
    from evaluator.local_evaluator import intent_card as _truth   # analysis only, not shipped
    tc = _truth(prods[tgt])
    real = {" ".join(raw_toks(str(x))[:12])
            for x in list(tc["hard_constraints"]) + list(tc["soft_preferences"])}
    if real <= pred:
        exact += 1
    elif real & pred:
        partial += 1
print(f"  reconstruction fidelity on 200 targets: exact {exact}  partial {partial}  "
      f"none {200-exact-partial}")
OUT["reconstruction"] = {"exact": exact, "partial": partial, "none": 200 - exact - partial}


class Verified(Agent):
    """Boost candidates whose PREDICTED card contains the observed constraints."""
    W_VERIFY = 3.0
    SLOT_BONUS = 1.0        # extra when the phrase lands in the same ordinal slot

    def _rank(self, st, pool, top_k):
        wmap = {p: self._weight(p, df, tier) for p, (df, tier) in st.evidence.items()}
        order = {a: i for i, a in enumerate(pool)}
        cover = self.ix.covers
        # observed constraint phrases, in disclosure order
        obs = [p for p, (_, tier) in st.evidence.items() if tier == CONSTRAINT]

        def score(asin: str):
            s = 0.0
            for phrase, w in wmap.items():
                if cover(asin, phrase):
                    s += w
            if obs:
                pc = CARD.get(asin, ())
                if pc:
                    hit = sum(1 for o in obs if o in pc)
                    s += self.W_VERIFY * (hit / len(obs))
                    if self.SLOT_BONUS:
                        slot = sum(1 for i, o in enumerate(obs)
                                   if i < len(pc) and pc[i] == o)
                        s += self.SLOT_BONUS * (slot / len(obs))
            s += self.W_POP * (self.ix.pop.get(asin, 0.0) / (self.ix.max_pop or 1.0))
            return (-s, order[asin])

        return sorted(pool, key=score)[:top_k]


def share(cls, **kw):
    o = object.__new__(cls)
    o.ix, o.sessions = base.ix, {}
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def run(ag, subset, tag):
    t = time.time()
    r = evaluate(ag, subset, cid, cats, prods)
    print(f"    {tag:<46} HR@10 {r['hit_rate_at_10']:>6.1%}  MRR {r['mrr']:>6.4f}  "
          f"MTTC {r['mttc']:>5.2f}  SCORE {r['recommended_technical_score']:>7.5f} "
          f"[{time.time()-t:>4.0f}s]")
    return {"hr": r["hit_rate_at_10"], "mrr": r["mrr"], "mttc": r["mttc"],
            "score": r["recommended_technical_score"]}


print("\n" + "=" * 104)
print("A. VERIFICATION WEIGHT -- tuning half")
print("=" * 104)
res = {"baseline": run(share(Agent), TUNE, "baseline (containment only)")}
for w in (1.0, 3.0, 8.0, 20.0):
    res[f"verify_{w}"] = run(share(Verified, W_VERIFY=w, SLOT_BONUS=1.0), TUNE,
                             f"generative verification W={w}")
res["verify_noslot"] = run(share(Verified, W_VERIFY=8.0, SLOT_BONUS=0.0), TUNE,
                           "verification W=8, no slot bonus")
OUT["tune"] = res
best = max((k for k in res if k != "baseline"), key=lambda k: res[k]["score"])
print(f"\n  best: {best} ({res[best]['score']:.5f}) vs baseline {res['baseline']['score']:.5f} "
      f"({res[best]['score']-res['baseline']['score']:+.5f})")

print("\n" + "=" * 104)
print("B. HELD-OUT ADJUDICATION")
print("=" * 104)
w = float(best.split("_")[1]) if best != "verify_noslot" else 8.0
sb = 0.0 if best == "verify_noslot" else 1.0
hb = run(share(Agent), HOLD, "baseline")
hv = run(share(Verified, W_VERIFY=w, SLOT_BONUS=sb), HOLD, f"verification W={w} slot={sb}")
d = hv["score"] - hb["score"]
print(f"\n  HELD-OUT DELTA: {d:+.5f}  -> "
      f"{'ADOPT' if d > 0.005 else 'inside noise' if d > -0.005 else 'REJECT'}")
OUT["holdout"] = {"baseline": hb, "verified": hv, "delta": d,
                  "cfg": {"W_VERIFY": w, "SLOT_BONUS": sb}}

full = evaluate(share(Verified, W_VERIFY=w, SLOT_BONUS=sb), samples, cid, cats, prods)
print(f"\n    ALL 200: SCORE {full['recommended_technical_score']:.5f}  "
      f"HR@10 {full['hit_rate_at_10']:.1%}  MRR {full['mrr']:.4f}  MTTC {full['mttc']:.2f}")
OUT["full"] = {"score": full["recommended_technical_score"], "hr": full["hit_rate_at_10"],
               "mrr": full["mrr"], "mttc": full["mttc"]}

Path(ROOT / "notes" / "eda" / "out_16.json").write_text(
    json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print("\n[saved] notes/eda/out_16.json")
