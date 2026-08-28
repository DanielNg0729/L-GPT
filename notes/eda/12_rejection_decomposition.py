"""
EDA pass 12: is rejection feedback's +0.0103 real, or is it the §8.5 metric artifact again?

Filtering out previously-shown items shortens the returned list (measured: 8 items instead
of 10 on turn 2). MRR is computed over the list WE return, so a shorter list inflates MRR
mechanically -- the exact effect we declined to exploit in pass 9. If most of rejection
feedback's gain comes from that shortening rather than from better ordering, then we are
taking through the back door what we refused at the front, and should say so.

Decomposition:

  A. no rejection                 -- baseline, always 10 items
  B. rejection, no backfill       -- SHIPPED. drops known-wrong items, list can be < 10
  C. rejection, backfilled to 10  -- known-wrong items are demoted to the TAIL rather than
                                     removed, so the list is always exactly 10.
                                     C isolates the ORDERING benefit with the denominator
                                     held constant.

If C ≈ B, the gain is genuine reranking and the shipped behaviour is fine.
If C ≈ A, the gain was the denominator, and we should ship C instead.

Run:  PYTHONIOENCODING=utf-8 python -u notes/eda/12_rejection_decomposition.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent, PAT_OVERRIDE, PAT_OVERRIDE_CUE, SessionState  # noqa: E402

CATALOG = ROOT / "data" / "catalog.jsonl"
PUBLIC = ROOT / "data" / "public_set.jsonl"
OUT: dict = {}

print("loading ...")
samples = load_jsonl(PUBLIC)
cid, cats, prods = catalog_index(CATALOG)
base = Agent(CATALOG)
TUNE = [s for i, s in enumerate(samples) if i % 2 == 0]
HOLD = [s for i, s in enumerate(samples) if i % 2 == 1]

SIZES: list[int] = []


class Variant(Agent):
    MODE = "none"           # 'none' | 'filter' | 'backfill'

    def respond(self, session_id, user_message, turn, top_k):
        st = self.sessions.setdefault(session_id, SessionState())
        st.turn += 1
        msg = user_message or ""
        probe = "other"
        if PAT_OVERRIDE.search(msg) or PAT_OVERRIDE_CUE.search(msg):
            st.rejected.clear()
        try:
            self._observe(st, msg)
            probe = self._next_probe(st)
            pool = self._candidates(st, msg)
            ranked = self._rank(st, pool, top_k) if pool else list(st.last_rank[:top_k])
            if st.rejected and self.MODE != "none":
                fresh = [a for a in ranked if a not in st.rejected]
                stale = [a for a in ranked if a in st.rejected]
                if self.MODE == "filter":
                    ranked = fresh or ranked
                else:                                  # backfill: demote, never drop
                    ranked = (fresh + stale) or ranked
            if ranked:
                st.last_rank = ranked
        except Exception:
            ranked = list(st.last_rank[:top_k])
        ranked = ranked[:top_k]
        SIZES.append(len(ranked))
        if self.MODE != "none":
            st.rejected.update(ranked)
        st.asked.append(probe)
        return {"message": self._question_text(probe), "ask_attribute": probe,
                "recommendations": [{"parent_asin": a} for a in ranked],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}}


def share(mode):
    o = object.__new__(Variant)
    o.ix, o.sessions, o.MODE = base.ix, {}, mode
    return o


def run(mode, subset, tag):
    SIZES.clear()
    r = evaluate(share(mode), subset, cid, cats, prods)
    mean_len = statistics.fmean(SIZES) if SIZES else 0
    print(f"    {tag:<40} HR@10 {r['hit_rate_at_10']:>6.1%}  MRR {r['mrr']:>6.4f}  "
          f"SCORE {r['recommended_technical_score']:>7.5f}   mean list len {mean_len:>4.1f}")
    return {"hr": r["hit_rate_at_10"], "mrr": r["mrr"],
            "score": r["recommended_technical_score"], "mean_list_len": mean_len}


for half, name in ((TUNE, "TUNING HALF"), (HOLD, "HELD-OUT HALF"), (samples, "ALL 200")):
    print(f"\n{'='*100}\n{name}\n{'='*100}")
    a = run("none", half, "A  no rejection (always 10)")
    b = run("filter", half, "B  rejection, filtered  [SHIPPED]")
    c = run("backfill", half, "C  rejection, backfilled to 10")
    OUT[name] = {"A_none": a, "B_filter": b, "C_backfill": c}
    gain_b = b["score"] - a["score"]
    gain_c = c["score"] - a["score"]
    share_ordering = (gain_c / gain_b * 100) if gain_b else float("nan")
    print(f"\n    gain from filtering  (B−A): {gain_b:+.5f}")
    print(f"    gain from ordering   (C−A): {gain_c:+.5f}")
    print(f"    -> {share_ordering:.0f}% of the gain survives with the denominator held at 10")

print("\nINTERPRETATION")
print("  C close to B  => the gain is genuine reranking; shipped behaviour is fine.")
print("  C close to A  => the gain was the shortened denominator, i.e. the §8.5 artifact")
print("                   arriving through the back door. Ship C instead.")

Path(ROOT / "notes" / "eda" / "out_12.json").write_text(
    json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print("\n[saved] notes/eda/out_12.json")
