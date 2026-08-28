"""
EDA pass 17: everything not yet measured, including things previously rejected on reasoning alone.

AUDIT NOTE, recorded because it was a real reporting failure:
  W_LEN (document-length penalty) and W_RATE (average_rating prior) were written into a
  first draft of pass 7, that file was then rewritten, and both were silently dropped.
  A later summary nevertheless asserted "length normalisation ... tested as W_LEN in
  pass 7 and it was harmful". That claim was FALSE -- neither was ever run. Both are
  tested here.

Batches:
  A  scoring features never measured   : W_LEN, W_RATE, positional slot weighting
  B  lexical matching variants         : porter stemming, NEAR proximity, trigram fuzzy,
                                         prefix matching for truncated constraints
  C  signal extraction never measured  : negative evidence, evidence-subset ensemble

Run:  PYTHONIOENCODING=utf-8 python -u notes/eda/17_everything_else.py
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import (  # noqa: E402
    Agent, CONSTRAINT, CAT, MINED, PAT_NOINFO, raw_toks, _flat, content_toks,
)

CATALOG = ROOT / "data" / "catalog.jsonl"
PUBLIC = ROOT / "data" / "public_set.jsonl"
OUT: dict = {}

print("loading ...")
samples = load_jsonl(PUBLIC)
cid, cats, prods = catalog_index(CATALOG)
TUNE = [s for i, s in enumerate(samples) if i % 2 == 0]
HOLD = [s for i, s in enumerate(samples) if i % 2 == 1]
base = Agent(CATALOG)

DOCLEN = {a: max(1, len(b)) for a, b in base.ix.blob.items()}
RATE = {}
for a, d in prods.items():
    try:
        RATE[a] = float(d.get("average_rating") or 0)
    except (TypeError, ValueError):
        RATE[a] = 0.0


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
          f"MTTC {r['mttc']:>5.2f}  SCORE {r['recommended_technical_score']:>7.5f} [{time.time()-t:>4.0f}s]")
    return {"hr": r["hit_rate_at_10"], "mrr": r["mrr"], "mttc": r["mttc"],
            "score": r["recommended_technical_score"]}


# ================================================================ BATCH A
class ScoreFeat(Agent):
    """W_LEN document-length penalty, W_RATE rating prior, W_SLOT positional weighting."""
    W_LEN = 0.0
    W_RATE = 0.0
    W_SLOT = 0.0

    def _rank(self, st, pool, top_k):
        wmap = {p: self._weight(p, df, tier) for p, (df, tier) in st.evidence.items()}
        # disclosure order: earlier-disclosed constraints come from earlier card slots
        slotw = {p: (1.0 / (1 + i)) for i, p in enumerate(st.evidence)}
        order = {a: i for i, a in enumerate(pool)}
        cover = self.ix.covers

        def score(a):
            s = 0.0
            for phrase, w in wmap.items():
                if cover(a, phrase):
                    s += w * (1.0 + self.W_SLOT * slotw[phrase])
            s += self.W_POP * (self.ix.pop.get(a, 0.0) / (self.ix.max_pop or 1.0))
            if self.W_RATE:
                s += self.W_RATE * (RATE.get(a, 0.0) / 5.0)
            if self.W_LEN:
                s -= self.W_LEN * math.log1p(DOCLEN[a]) / 10.0
            return (-s, order[a])

        return sorted(pool, key=score)[:top_k]


print("\n" + "=" * 106)
print("BATCH A -- scoring features that were never measured (incl. the two audit gaps)")
print("=" * 106)
A = {"baseline": run(share(Agent), TUNE, "baseline")}
for v in (0.1, 0.3, 0.8):
    A[f"W_LEN={v}"] = run(share(ScoreFeat, W_LEN=v), TUNE, f"A1 length penalty W_LEN={v}")
for v in (0.1, 0.3, 0.8):
    A[f"W_RATE={v}"] = run(share(ScoreFeat, W_RATE=v), TUNE, f"A2 rating prior W_RATE={v}")
for v in (0.3, 1.0):
    A[f"W_SLOT={v}"] = run(share(ScoreFeat, W_SLOT=v), TUNE, f"A3 slot weighting W_SLOT={v}")
OUT["batchA"] = A


# ================================================================ BATCH B
print("\n" + "=" * 106)
print("BATCH B -- lexical matching variants")
print("=" * 106)
print("  building porter-stemmed index ...")
t0 = time.time()
pcon = sqlite3.connect(":memory:", check_same_thread=False)
pcon.execute("CREATE VIRTUAL TABLE p USING fts5(asin UNINDEXED, title, categories, features, "
             "details, store, description, tokenize='porter unicode61 remove_diacritics 2')")
rows = []
with CATALOG.open(encoding="utf-8") as fh:
    for line in fh:
        d = json.loads(line)
        rows.append((str(d["parent_asin"]), _flat(d.get("title")), _flat(d.get("categories")),
                     _flat(d.get("features")), _flat(d.get("details")),
                     _flat(d.get("store")), _flat(d.get("description"))))
pcon.executemany("INSERT INTO p VALUES (?,?,?,?,?,?,?)", rows)
pcon.commit()
print(f"    done in {time.time()-t0:.0f}s")


class Stemmed(Agent):
    """Retrieval (pool generation) via a porter-stemmed index; coverage unchanged."""
    def _candidates(self, st, message):
        real = self.ix.con
        try:
            self.ix.con = pcon
            return super()._candidates(st, message)
        finally:
            self.ix.con = real


class Proximity(Agent):
    """NEAR(...) instead of strict phrase adjacency -- tolerates small drift."""
    NEAR_N = 5

    def _candidates(self, st, message):
        ev = sorted(st.evidence.items(), key=lambda kv: kv[1][0])
        strong = [p for p, (df, _) in ev if df <= self.STRONG_DF]
        pool, seen = [], set()

        def add(expr, lim):
            for a in self.ix.search(expr, lim):
                if a not in seen:
                    seen.add(a)
                    pool.append(a)

        def near(p):
            toks = p.split()
            if len(toks) < 2:
                return f'"{p}"'
            return "NEAR(" + " ".join(f'"{t}"' for t in toks[:8]) + f", {self.NEAR_N})"

        q = [near(p) for p in strong[:8]]
        if q:
            add(" AND ".join(q), self.POOL)
            for k in range(len(q) - 1, 0, -1):
                if len(pool) >= self.POOL:
                    break
                add(" AND ".join(q[:k]), self.POOL)
            add(" OR ".join(q), self.POOL)
        if len(pool) < self.POOL and ev:
            add(" OR ".join(f'"{p}"' for p, _ in ev[:14]), self.POOL)
        if not pool:
            return super()._candidates(st, message)
        return pool


class TrigramFuzzy(Agent):
    """Character-trigram Jaccard fallback when exact containment fails."""
    THRESH = 0.72

    @staticmethod
    def _tri(s: str) -> set:
        s = f"  {s} "
        return {s[i:i + 3] for i in range(len(s) - 2)}

    def _rank(self, st, pool, top_k):
        wmap = {p: self._weight(p, df, tier) for p, (df, tier) in st.evidence.items()}
        tri = {p: self._tri(p) for p in st.evidence}
        order = {a: i for i, a in enumerate(pool)}
        cover = self.ix.covers

        def score(a):
            s = 0.0
            blob = self.ix.blob.get(a, "")
            btri = None
            for phrase, w in wmap.items():
                if cover(a, phrase):
                    s += w
                elif len(phrase) > 12:
                    if btri is None:
                        btri = self._tri(blob[:4000])
                    t = tri[phrase]
                    j = len(t & btri) / max(1, len(t))
                    if j >= self.THRESH:
                        s += w * j
            s += self.W_POP * (self.ix.pop.get(a, 0.0) / (self.ix.max_pop or 1.0))
            return (-s, order[a])

        return sorted(pool, key=score)[:top_k]


class PrefixMatch(Agent):
    """Trailing-token prefix wildcard: tolerates the 180-char truncation cut."""
    def _candidates(self, st, message):
        ev = sorted(st.evidence.items(), key=lambda kv: kv[1][0])
        strong = [p for p, (df, _) in ev if df <= self.STRONG_DF]
        pool, seen = [], set()

        def add(expr, lim):
            for a in self.ix.search(expr, lim):
                if a not in seen:
                    seen.add(a)
                    pool.append(a)

        q = [f'"{p}" * ' if len(p.split()) > 2 else f'"{p}"' for p in strong[:8]]
        if q:
            add(" AND ".join(q), self.POOL)
            add(" OR ".join(q), self.POOL)
        if len(pool) < self.POOL and ev:
            add(" OR ".join(f'"{p}"' for p, _ in ev[:14]), self.POOL)
        if not pool:
            return super()._candidates(st, message)
        return pool


B = {"baseline": A["baseline"]}
B["stemmed"] = run(share(Stemmed), TUNE, "B1 porter-stemmed retrieval index")
for n in (2, 5, 10):
    B[f"near{n}"] = run(share(Proximity, NEAR_N=n), TUNE, f"B2 NEAR proximity N={n}")
for th in (0.65, 0.8):
    B[f"trigram{th}"] = run(share(TrigramFuzzy, THRESH=th), TUNE, f"B3 trigram fuzzy thresh={th}")
B["prefix"] = run(share(PrefixMatch), TUNE, "B4 prefix wildcard matching")
OUT["batchB"] = B


# ================================================================ BATCH C
PAT_NOPREF_ATTR = re.compile(r"preference for ([a-z_]+)", re.I)


class NegEvidence(Agent):
    """Use 'I don't have a preference for X' as a negative signal.

    It proves no UNDISCLOSED constraint classifies to X, so products whose distinguishing
    text is dominated by that attribute type are less likely to be the target.
    """
    W_NEG = 0.3
    ATTR_WORDS = {
        "material": ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon"),
        "color": ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey"),
        "size": ("size", "width", "wide", "narrow"),
        "style": ("style", "fit", "sleeve", "neck"),
    }

    def respond(self, session_id, user_message, turn, top_k):
        st = self.sessions.setdefault(session_id, __import__("submission.agent",
                                                             fromlist=["SessionState"]).SessionState())
        m = PAT_NOPREF_ATTR.search(user_message or "")
        if m:
            attr = m.group(1).lower()
            if not hasattr(st, "_neg"):
                pass
            self._neg = getattr(self, "_neg", {})
            self._neg.setdefault(session_id, set()).add(attr)
        return super().respond(session_id, user_message, turn, top_k)

    def _rank(self, st, pool, top_k):
        ranked = super()._rank(st, pool, max(top_k, 40))
        neg = set()
        for v in getattr(self, "_neg", {}).values():
            neg |= v
        if not neg:
            return ranked[:top_k]
        words = set()
        for a in neg:
            words |= set(self.ATTR_WORDS.get(a, ()))
        if not words:
            return ranked[:top_k]

        def pen(asin):
            blob = self.ix.blob.get(asin, "")
            return sum(1 for w in words if f" {w} " in blob)

        return sorted(ranked, key=lambda a: (pen(a) * self.W_NEG, ranked.index(a)))[:top_k]


class SubsetEnsemble(Agent):
    """Rank under several evidence subsets and fuse by reciprocal rank (bagging)."""
    K_RRF = 20

    def _rank(self, st, pool, top_k):
        ev = list(st.evidence)
        if len(ev) < 3:
            return super()._rank(st, pool, top_k)
        fused: dict[str, float] = {}
        subsets = [ev] + [[p for p in ev if p != drop] for drop in ev]
        for sub in subsets:
            saved = st.evidence
            st.evidence = {p: saved[p] for p in sub}
            try:
                r = super()._rank(st, pool, min(len(pool), 40))
            finally:
                st.evidence = saved
            for i, a in enumerate(r):
                fused[a] = fused.get(a, 0.0) + 1.0 / (self.K_RRF + i + 1)
        return sorted(fused, key=lambda a: -fused[a])[:top_k]


print("\n" + "=" * 106)
print("BATCH C -- signal extraction never measured")
print("=" * 106)
C = {"baseline": A["baseline"]}
C["negative_evidence"] = run(share(NegEvidence), TUNE, "C1 negative evidence from 'no preference'")
C["subset_ensemble"] = run(share(SubsetEnsemble), TUNE, "C2 evidence-subset ensemble (bagging)")
OUT["batchC"] = C

# ================================================================ HELD-OUT
print("\n" + "=" * 106)
print("HELD-OUT ADJUDICATION for anything that beat baseline on the tuning half")
print("=" * 106)
allres = {**{k: v for k, v in A.items() if k != "baseline"},
          **{k: v for k, v in B.items() if k != "baseline"},
          **{k: v for k, v in C.items() if k != "baseline"}}
bl = A["baseline"]["score"]
winners = [k for k, v in allres.items() if v["score"] > bl]
print(f"  beat baseline ({bl:.5f}) on tune: {winners or 'NONE'}\n")

CTOR = {
    **{f"W_LEN={v}": (ScoreFeat, dict(W_LEN=v)) for v in (0.1, 0.3, 0.8)},
    **{f"W_RATE={v}": (ScoreFeat, dict(W_RATE=v)) for v in (0.1, 0.3, 0.8)},
    **{f"W_SLOT={v}": (ScoreFeat, dict(W_SLOT=v)) for v in (0.3, 1.0)},
    "stemmed": (Stemmed, {}),
    **{f"near{n}": (Proximity, dict(NEAR_N=n)) for n in (2, 5, 10)},
    **{f"trigram{th}": (TrigramFuzzy, dict(THRESH=th)) for th in (0.65, 0.8)},
    "prefix": (PrefixMatch, {}),
    "negative_evidence": (NegEvidence, {}),
    "subset_ensemble": (SubsetEnsemble, {}),
}
hb = run(share(Agent), HOLD, "baseline")
OUT["holdout"] = {"baseline": hb}
for k in winners:
    cls, kw = CTOR[k]
    h = run(share(cls, **kw), HOLD, k)
    OUT["holdout"][k] = h
    d = h["score"] - hb["score"]
    print(f"        -> delta {d:+.5f}  "
          f"{'ADOPT' if d > 0.005 else 'inside noise' if d > -0.005 else 'REJECT'}")

Path(ROOT / "notes" / "eda" / "out_17.json").write_text(
    json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print("\n[saved] notes/eda/out_17.json")
