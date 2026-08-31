"""
Experiment 13: two RAG-family retrieval techniques that actually apply here.

We already run the retrieval half of a RAG stack (query construction -> candidate
generation -> rerank). What we have not tried are two techniques the RAG/IR literature
treats as standard, both fully deterministic:

  A. STRUCTURE-AWARE CHUNKING.
     Chunk granularity is repeatedly reported to dominate retrieval quality, and
     structure-aware chunking beats fixed-size and semantic chunking at lower cost
     (arXiv:2603.24556, arXiv:2606.00881). We currently index each product as ONE
     document: every field concatenated into a single blob. That creates a specific
     false positive -- a constraint phrase can match ACROSS a boundary between two
     unrelated feature bullets, because the blob glues them together with a space.

     This matters more here than in a generic RAG setting, because of where the
     constraints come from: `intent_card` lifts individual items out of the product's
     `features` / `details` lists. A genuine constraint therefore lives inside ONE
     bullet. Requiring single-chunk containment should remove cross-boundary matches
     without losing a single true one.

  B. RM3 PSEUDO-RELEVANCE FEEDBACK (Lavrenko & Croft 2001; Rocchio 1971).
     Retrieve, harvest discriminative terms from the top-k, expand the query, retrieve
     again. "BM25 + RM3 gives a very strong baseline, competitive even with modern
     approaches." The literature also names the failure mode we are most exposed to:
     PRF underperforms plain BM25 when the FIRST-PASS ranking is unreliable -- and our
     turn-1 browsing ranking is exactly that (category only, median 145 candidates).
     So this is tested per-turn, not just globally.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/log/13_rag_techniques.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent, raw_toks, _flat, STOP  # noqa: E402

CATALOG = ROOT / "data" / "catalog.jsonl"
PUBLIC = ROOT / "data" / "public_set.jsonl"
OUT: dict = {}

print("loading ...")
samples = load_jsonl(PUBLIC)
cid, cats, prods = catalog_index(CATALOG)
TUNE = [s for i, s in enumerate(samples) if i % 2 == 0]
HOLD = [s for i, s in enumerate(samples) if i % 2 == 1]
base = Agent(CATALOG)
print(f"indexed {len(base.ix.blob):,}")

# --------------------------------------------------------------- A: chunk store
print("building structure-aware chunk store (one chunk per feature bullet / detail / field) ...")
t0 = time.time()
CHUNKS: dict[str, list[str]] = {}
for asin, d in prods.items():
    parts: list[str] = []
    t = str(d.get("title") or "")
    if t:
        parts.append(t)
    for f in (d.get("features") or []):
        parts.append(str(f))
    for x in (d.get("description") or []):
        parts.append(str(x))
    det = d.get("details")
    if isinstance(det, dict):
        parts.extend(f"{k} {v}" for k, v in det.items())
    c = d.get("categories")
    if isinstance(c, list) and c:
        parts.append(" ".join(str(x) for x in c))
    s = str(d.get("store") or "")
    if s:
        parts.append(s)
    CHUNKS[asin] = [" " + " ".join(raw_toks(p)) + " " for p in parts if p.strip()]
n_chunks = sum(len(v) for v in CHUNKS.values())
print(f"  {n_chunks:,} chunks over {len(CHUNKS):,} products "
      f"({n_chunks/len(CHUNKS):.1f} per product) in {time.time()-t0:.0f}s")
OUT["chunk_stats"] = {"n_chunks": n_chunks, "per_product": n_chunks / len(CHUNKS)}


class ChunkAware(Agent):
    """`covers()` requires the phrase inside a SINGLE structural chunk.

    The shipped agent concatenates every field into one blob, so a phrase can satisfy
    containment by straddling the join between two unrelated bullets. Since every real
    constraint originates from a single bullet, that match is always spurious.
    """

    def _covers_chunk(self, asin: str, phrase: str) -> bool:
        needle = f" {phrase} "
        for ch in CHUNKS.get(asin, ()):
            if needle in ch:
                return True
        return False

    def _rank(self, st, pool, top_k):
        wmap = {p: self._weight(p, df, tier) for p, (df, tier) in st.evidence.items()}
        order = {a: i for i, a in enumerate(pool)}

        def score(asin: str):
            s = 0.0
            for phrase, w in wmap.items():
                if self._covers_chunk(asin, phrase):
                    s += w
            s += self.W_POP * (self.ix.pop.get(asin, 0.0) / (self.ix.max_pop or 1.0))
            return (-s, order[asin])

        return sorted(pool, key=score)[:top_k]


# how often does the blob match but no single chunk match? (the false-positive rate)
straddle = tot = 0
for s in samples[:60]:
    tgt = str(s["ground_truth"]["parent_asin"])
    for other in list(prods)[:400]:
        for ph in list(base.ix.blob.get(tgt, "").split()[:0]):
            pass
probe = ChunkAware.__new__(ChunkAware)
probe.ix = base.ix


# --------------------------------------------------------------- B: RM3
IDF: dict[str, float] = {}


class RM3(Agent):
    """Rocchio/RM3-style pseudo-relevance feedback over the sparse first pass."""
    FB_DOCS = 10
    FB_TERMS = 8
    MIN_TURN = 1          # turns below this skip expansion (unreliable first pass)

    def _candidates(self, st, message):
        pool = super()._candidates(st, message)
        if st.turn < self.MIN_TURN or len(pool) < 3:
            return pool
        # harvest discriminative terms from the pseudo-relevant head
        counts: Counter = Counter()
        for a in pool[:self.FB_DOCS]:
            for tok in set(self.ix.blob.get(a, "").split()):
                if len(tok) > 2 and tok not in STOP:
                    counts[tok] += 1
        have = set()
        for p in st.evidence:
            have.update(p.split())
        cand = [(t, c) for t, c in counts.items()
                if t not in have and c >= max(2, self.FB_DOCS // 3)]
        if not cand:
            return pool
        # prefer terms frequent in the feedback set but rare in the collection
        def w(item):
            t, c = item
            df = self.ix.df(t) or 1
            return (c / self.FB_DOCS) * math.log(50000.0 / df)
        expansion = [t for t, _ in sorted(cand, key=w, reverse=True)[:self.FB_TERMS]]
        if not expansion:
            return pool
        extra = self.ix.search(" OR ".join(f'"{t}"' for t in expansion), self.POOL)
        seen = set(pool)
        for a in extra:
            if a not in seen:
                seen.add(a)
                pool.append(a)
        return pool[:self.POOL * 2]


def share(cls, **kw):
    o = object.__new__(cls)
    o.ix, o.sessions = base.ix, {}
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def run(ag, subset, tag):
    t = time.time()
    r = evaluate(ag, subset, cid, cats, prods)
    print(f"    {tag:<44} HR@10 {r['hit_rate_at_10']:>6.1%}  MRR {r['mrr']:>6.4f}  "
          f"MTTC {r['mttc']:>5.2f}  SCORE {r['recommended_technical_score']:>7.5f} "
          f"[{time.time()-t:>4.0f}s]")
    return {"hr": r["hit_rate_at_10"], "mrr": r["mrr"], "mttc": r["mttc"],
            "score": r["recommended_technical_score"]}


print("\n" + "=" * 104)
print("TUNING HALF")
print("=" * 104)
res = {}
res["baseline"] = run(share(Agent), TUNE, "baseline (single-blob containment)")
res["chunk"] = run(share(ChunkAware), TUNE, "A structure-aware chunk containment")
res["rm3_all"] = run(share(RM3, MIN_TURN=1), TUNE, "B RM3 PRF, all turns")
res["rm3_late"] = run(share(RM3, MIN_TURN=3), TUNE, "B RM3 PRF, turn>=3 only")
res["chunk_rm3"] = run(share(type("CR", (RM3, ChunkAware), {}), MIN_TURN=3), TUNE,
                       "A+B chunk + RM3(turn>=3)")
OUT["tune"] = res

print("\n" + "=" * 104)
print("HELD-OUT ADJUDICATION")
print("=" * 104)
hold = {"baseline": run(share(Agent), HOLD, "baseline")}
for name, cls, kw in (("chunk", ChunkAware, {}),
                      ("rm3_all", RM3, dict(MIN_TURN=1)),
                      ("rm3_late", RM3, dict(MIN_TURN=3))):
    hold[name] = run(share(cls, **kw), HOLD, name)
OUT["holdout"] = hold

print("\n  verdicts vs baseline on HELD-OUT data:")
for k in ("chunk", "rm3_all", "rm3_late"):
    d = hold[k]["score"] - hold["baseline"]["score"]
    verdict = "ADOPT" if d > 0.005 else ("reject (inside noise)" if d > -0.005 else "REJECT")
    print(f"    {k:<12} {d:+.5f}  -> {verdict}")

Path(ROOT / "experiments" / "results" / "out_13.json").write_text(
    json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print("\n[saved] experiments/results/out_13.json")
