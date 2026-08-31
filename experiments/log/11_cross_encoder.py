"""
Experiment 11: cross-encoder reranking -- retracting a badly-argued decline.

Pass 8 declined this on three grounds. Two do not survive scrutiny:

  1. "HR@10 is already 0.990, only ~2 sessions remain to gain."
     WRONG TARGET. That is recall headroom. A cross-encoder is a RERANKER; it attacks
     MRR, where 0.067 of TechnicalScore is currently sitting (rank>1 among hits).

  2. "The grader may restrict network AND disk."
     UNSUPPORTED. submission_rules.md names CPU, memory, timeout and network
     restrictions -- disk is not among them -- and line 40 explicitly ALLOWS
     "lightweight local assets required by your agent". A pre-downloaded model bundled
     with the submission is permitted. Network is a non-issue once it is bundled.

  3. "The dense bi-encoder degraded provenance matching; a cross-encoder is the same
     family." CONFLATION. The bi-encoder failed at candidate GENERATION, where blurring
     lexical precision is fatal. A cross-encoder reranks an already lexically-retrieved
     pool -- different role, different failure mode.

What legitimately remains is timeout and memory, both explicitly named in the rules, plus
install risk. Those are reasons to MEASURE, which is what this pass does:

  * does cross-encoder reranking actually improve MRR here?
  * what does it cost in per-turn latency and RAM?

Run:  PYTHONIOENCODING=utf-8 python -u experiments/log/11_cross_encoder.py
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
from submission.agent import Agent, _flat  # noqa: E402

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

from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: E402

MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
t0 = time.time()
ce = TextCrossEncoder(model_name=MODEL)
print(f"cross-encoder {MODEL} loaded in {time.time()-t0:.1f}s")

# Document text is built once; rebuilding it per call would dominate the measurement.
DOC: dict[str, str] = {}
for a, d in prods.items():
    DOC[a] = (f"{_flat(d.get('title'))} {_flat(d.get('features'))}")[:320]

LAT: list[float] = []


class CEAgent(Agent):
    """Rerank the top-DEPTH of the lexical ranking with a cross-encoder.

    BLEND controls the role of the neural score:
      0.0 -> pure cross-encoder ordering
      w   -> coverage_score_rank + w * ce_rank fused by reciprocal rank
    """
    DEPTH = 20
    BLEND = 0.0
    K_RRF = 60

    def _rank(self, st, pool, top_k):
        ranked = super()._rank(st, pool, max(top_k, self.DEPTH))
        head = ranked[:self.DEPTH]
        if len(head) < 2 or not st.evidence:
            return ranked[:top_k]
        query = " ".join(sorted(st.evidence, key=lambda p: st.evidence[p][0]))[:320]
        t = time.perf_counter()
        try:
            scores = list(ce.rerank(query, [DOC.get(a, "") for a in head]))
        except Exception:
            return ranked[:top_k]
        LAT.append(time.perf_counter() - t)
        order = sorted(range(len(head)), key=lambda i: -scores[i])
        if self.BLEND <= 0.0:
            out = [head[i] for i in order]
        else:
            fused: dict[str, float] = {}
            for r, a in enumerate(head):                      # lexical arm
                fused[a] = fused.get(a, 0.0) + 1.0 / (self.K_RRF + r + 1)
            for r, i in enumerate(order):                     # neural arm
                a = head[i]
                fused[a] = fused.get(a, 0.0) + self.BLEND / (self.K_RRF + r + 1)
            out = sorted(fused, key=lambda a: -fused[a])
        return (out + ranked[self.DEPTH:])[:top_k]


def share(cls, **kw):
    o = object.__new__(cls)
    o.ix, o.sessions = base.ix, {}
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def run(ag, subset, tag):
    LAT.clear()
    t = time.time()
    r = evaluate(ag, subset, cid, cats, prods)
    wall = time.time() - t
    lat = f"{statistics.fmean(LAT)*1000:.0f}ms" if LAT else "-"
    print(f"    {tag:<38} HR@10 {r['hit_rate_at_10']:>6.1%}  MRR {r['mrr']:>6.4f}  "
          f"MTTC {r['mttc']:>5.2f}  SCORE {r['recommended_technical_score']:>7.5f}  "
          f"[{wall:>5.0f}s wall, {lat}/rerank]")
    return {"hr": r["hit_rate_at_10"], "mrr": r["mrr"], "mttc": r["mttc"],
            "score": r["recommended_technical_score"], "wall_s": wall,
            "mean_rerank_ms": (statistics.fmean(LAT) * 1000) if LAT else None}


print("\n" + "=" * 100)
print("TUNING HALF")
print("=" * 100)
res = {}
res["baseline"] = run(share(Agent), TUNE, "baseline (no cross-encoder)")
res["ce_pure_20"] = run(share(CEAgent, DEPTH=20, BLEND=0.0), TUNE, "CE pure, depth 20")
res["ce_blend_20"] = run(share(CEAgent, DEPTH=20, BLEND=1.0), TUNE, "CE+lexical RRF, depth 20")
res["ce_blend_20_w05"] = run(share(CEAgent, DEPTH=20, BLEND=0.5), TUNE, "CE+lexical RRF w=0.5, depth 20")
res["ce_blend_10"] = run(share(CEAgent, DEPTH=10, BLEND=1.0), TUNE, "CE+lexical RRF, depth 10")
OUT["tune"] = res

best = max((k for k in res if k != "baseline"), key=lambda k: res[k]["score"])
print(f"\n  best cross-encoder variant on tuning half: {best} ({res[best]['score']:.5f})")
print(f"  vs baseline {res['baseline']['score']:.5f}  ({res[best]['score']-res['baseline']['score']:+.5f})")

print("\n" + "=" * 100)
print("HELD-OUT ADJUDICATION")
print("=" * 100)
cfg = {"ce_pure_20": dict(DEPTH=20, BLEND=0.0), "ce_blend_20": dict(DEPTH=20, BLEND=1.0),
       "ce_blend_20_w05": dict(DEPTH=20, BLEND=0.5), "ce_blend_10": dict(DEPTH=10, BLEND=1.0)}
hb = run(share(Agent), HOLD, "baseline")
hc = run(share(CEAgent, **cfg[best]), HOLD, f"best CE: {best}")
OUT["holdout"] = {"baseline": hb, "best_ce": hc, "best_name": best}
delta = hc["score"] - hb["score"]
print(f"\n  HELD-OUT DELTA: {delta:+.5f}  -> "
      f"{'ADOPT' if delta > 0.005 else 'inside noise, reject' if delta > -0.005 else 'REJECT'}")

print("\n" + "=" * 100)
print("FEASIBILITY COST (the objections that DO survive: memory + timeout)")
print("=" * 100)
try:
    import os
    import psutil  # noqa
    rss = psutil.Process(os.getpid()).memory_info().rss / 1e6
    print(f"    process RSS with index + cross-encoder: {rss:.0f} MB")
    OUT["rss_mb"] = rss
except Exception:
    print("    (psutil unavailable; RSS not measured)")
ml = hc.get("mean_rerank_ms")
if ml:
    print(f"    mean rerank latency: {ml:.0f} ms/turn at depth {cfg[best]['DEPTH']}")
    print(f"    projected private-set cost: 800 sessions x ~2 turns x {ml:.0f}ms "
          f"= ~{800*2*ml/1000/60:.1f} min of pure reranking")
print(f"    model asset must be bundled (allowed: submission_rules.md:40 "
      f"'lightweight local assets')")

Path(ROOT / "experiments" / "results" / "out_11.json").write_text(
    json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print("\n[saved] experiments/results/out_11.json")
