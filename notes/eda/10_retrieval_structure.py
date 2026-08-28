"""
EDA pass 10: three STRUCTURAL retrieval changes (not parameter tuning).

Each was listed as "untested" in the report's alternatives table. None is a sweep.

  A. FIELD-RESTRICTED MATCHING. A constraint lifted from `details` ("Material: alloy")
     currently matches anywhere in the document. FTS5 supports column filters, so we can
     require it to match the column it CAME from. Higher precision, at the cost of recall
     when our guess about provenance-field is wrong.

  B. SPARSE-SPARSE RRF ENSEMBLE. Pass 5 showed RRF fusion with a DENSE partner costs
     0.047, because embeddings blur the lexical precision this task needs. That argument
     does not apply to fusing several SPARSE formulations, which all preserve exact
     matching. Cormack's RRF (SIGIR 2009), k=60, over {full-doc, title-weighted,
     features-weighted} rankings.

  C. FULL CATEGORY PATH. `coarse_category()` keeps only the LAST TWO comma-separated
     parts of the category list. The catalogue carries the whole path. The deeper path is
     more selective; the question is whether the extra terms are attested on the target.

Run:  PYTHONIOENCODING=utf-8 python -u notes/eda/10_retrieval_structure.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent, CONSTRAINT, CAT, MINED  # noqa: E402

CATALOG = ROOT / "data" / "catalog.jsonl"
PUBLIC = ROOT / "data" / "public_set.jsonl"
OUT: dict = {}

print("loading ...")
samples = load_jsonl(PUBLIC)
cid, cats, prods = catalog_index(CATALOG)
TUNE = [s for i, s in enumerate(samples) if i % 2 == 0]
HOLD = [s for i, s in enumerate(samples) if i % 2 == 1]
t0 = time.time()
base = Agent(CATALOG)
print(f"index built in {time.time()-t0:.0f}s\n")


def share(cls, **kw):
    """Instantiate a variant reusing the already-built index."""
    obj = object.__new__(cls)
    obj.ix = base.ix
    obj.sessions = {}
    for k, v in kw.items():
        setattr(obj, k, v)
    return obj


def run(ag, subset, tag):
    r = evaluate(ag, subset, cid, cats, prods)
    print(f"    {tag:<44} HR@10 {r['hit_rate_at_10']:>6.1%}  MRR {r['mrr']:>6.4f}  "
          f"MTTC {r['mttc']:>5.2f}  SCORE {r['recommended_technical_score']:>7.5f}")
    return {"hr": r["hit_rate_at_10"], "mrr": r["mrr"], "mttc": r["mttc"],
            "score": r["recommended_technical_score"]}


# =============================================================== A
class FieldRestricted(Agent):
    """Route each evidence phrase to the column it most likely came from.

    intent_card draws hard/soft constraints from `features` and `details`, and the
    category clause from `categories`. Requiring the match to land in that column removes
    coincidental hits elsewhere in a long description.
    """
    COL = {CAT: "categories", CONSTRAINT: "{features details title}", MINED: None}

    def _candidates(self, st, message):
        ev = sorted(st.evidence.items(), key=lambda kv: kv[1][0])
        strong = [(p, tier) for p, (df, tier) in ev if df <= self.STRONG_DF]
        pool, seen = [], set()

        def add(expr, limit):
            for a in self.ix.search(expr, limit):
                if a not in seen:
                    seen.add(a)
                    pool.append(a)

        def q(p, tier):
            col = self.COL.get(tier)
            return f'{col} : "{p}"' if col else f'"{p}"'

        quoted = [q(p, t) for p, t in strong[:8]]
        if quoted:
            add(" AND ".join(quoted), self.POOL)
            for k in range(len(quoted) - 1, 0, -1):
                if len(pool) >= self.POOL:
                    break
                add(" AND ".join(quoted[:k]), self.POOL)
            add(" OR ".join(quoted), self.POOL)
        if len(pool) < self.POOL and ev:
            add(" OR ".join(f'"{p}"' for p, _ in ev[:14]), self.POOL)
        if not pool:                                   # unrestricted fallback
            return super()._candidates(st, message)
        return pool


# =============================================================== B
class SparseEnsemble(Agent):
    """RRF over several SPARSE formulations of the same evidence.

    The pass-5 objection to fusion was that a dense partner destroys lexical precision.
    Fusing sparse-with-sparse keeps every arm exact, so the objection does not transfer.
    """
    K_RRF = 60
    ARMS = (
        "bm25(p, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)",     # shipped weighting
        "bm25(p, 0.0, 12.0, 2.0, 1.0, 1.0, 0.5, 0.5)",    # title-dominant
        "bm25(p, 0.0, 1.0, 1.0, 8.0, 4.0, 0.5, 1.0)",     # features/details-dominant
    )

    def _candidates(self, st, message):
        ev = sorted(st.evidence.items(), key=lambda kv: kv[1][0])
        strong = [p for p, (df, _) in ev if df <= self.STRONG_DF]
        quoted = [f'"{p}"' for p in strong[:8]]
        exprs = []
        if quoted:
            exprs.append(" AND ".join(quoted))
            exprs.append(" OR ".join(quoted))
        if ev:
            exprs.append(" OR ".join(f'"{p}"' for p, _ in ev[:14]))
        if not exprs:
            return super()._candidates(st, message)

        fused: dict[str, float] = {}
        orig = self.ix.BM25
        try:
            for arm in self.ARMS:
                self.ix.BM25 = arm
                for expr in exprs:
                    for rank, asin in enumerate(self.ix.search(expr, self.POOL)):
                        fused[asin] = fused.get(asin, 0.0) + 1.0 / (self.K_RRF + rank + 1)
        finally:
            self.ix.BM25 = orig
        if not fused:
            return super()._candidates(st, message)
        return sorted(fused, key=lambda a: -fused[a])[:self.POOL]


# =============================================================== C
class FullCategoryPath(Agent):
    """Also mine the DEEPER category path, not just coarse_category's last two parts."""

    def _extract_templated(self, msg):
        out = super()._extract_templated(msg)
        for text, tier in list(out):
            if tier == CAT:
                parts = [x.strip() for x in text.replace("&", " ").split() if len(x.strip()) > 2]
                if len(parts) >= 3:
                    out.append((" ".join(parts[-3:]), CAT))
        return out


print("=" * 96)
print("A/B/C -- tuning half")
print("=" * 96)
res = {}
res["baseline"] = run(share(Agent), TUNE, "baseline (pass 8 agent)")
res["field_restricted"] = run(share(FieldRestricted), TUNE, "A field-restricted matching")
res["sparse_ensemble"] = run(share(SparseEnsemble), TUNE, "B sparse-sparse RRF ensemble")
res["full_category"] = run(share(FullCategoryPath), TUNE, "C full category path")
OUT["tune"] = res

winner = max(res, key=lambda k: res[k]["score"])
print(f"\n  best on tuning half: {winner} ({res[winner]['score']:.5f})")

print()
print("=" * 96)
print("HELD-OUT ADJUDICATION")
print("=" * 96)
cls = {"baseline": Agent, "field_restricted": FieldRestricted,
       "sparse_ensemble": SparseEnsemble, "full_category": FullCategoryPath}
hold = {}
hold["baseline"] = run(share(Agent), HOLD, "baseline")
for k in ("field_restricted", "sparse_ensemble", "full_category"):
    hold[k] = run(share(cls[k]), HOLD, k)
OUT["holdout"] = hold

print("\n  verdicts vs baseline on HELD-OUT data:")
for k in ("field_restricted", "sparse_ensemble", "full_category"):
    d = hold[k]["score"] - hold["baseline"]["score"]
    print(f"    {k:<22} {d:+.5f}  -> {'ADOPT' if d > 0.005 else 'reject (inside noise)' if d > -0.005 else 'REJECT'}")

Path(ROOT / "notes" / "eda" / "out_10.json").write_text(
    json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print("\n[saved] notes/eda/out_10.json")
