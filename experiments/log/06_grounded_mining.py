"""
Experiment 6: catalog-grounded evidence mining + information-ordered probing.

Pass 5 produced two results that overturn the pass-4 design:

  (1) Template parsing is BRITTLE and, under partial paraphrase, actively harmful:
      clean 0.8257 -> light paraphrase 0.5092, which is WORSE than running with no
      parser at all (0.7736). Cause: when a template misses, the fallback promotes
      the whole noisy message to a "constraint" and phrase-matches it, which can
      never hit; the filler words then also pollute the bag-of-words channel.

  (2) A TYPED probe rotation beat ask_attribute='other' (0.8285 vs 0.8257) despite
      being slower to full disclosure. Cause: 'other' returns constraints in card
      order, and hard_constraints[0] is a near-useless bare material word 76.5% of
      the time; asking 'feature' first pulls the LONG, highly selective free-text
      bullets forward. Ordering probes by expected selectivity beats greedy maximal
      extraction -- precisely the claim of the information-gain CRS line.

This pass builds CATALOG-GROUNDED N-GRAM MINING: mine n-grams from the message and
keep only those the catalogue actually attests at usable document frequency. Filler
has no catalogue support and self-eliminates; real product text survives.

Two refinements were needed to make mining competitive (see FINDINGS in the header
of each class):
  * evidence weight must scale with PHRASE LENGTH, not just rarity -- short junk
    collocations ("what i", df=15) are rare but not product-identifying, whereas real
    provenance yields long df=1 phrases;
  * templates, WHEN THEY FIRE, are higher precision than mining -- so the right
    structure is template-first with mining as the fallback, never raw-message dump.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/log/06_grounded_mining.py
"""
from __future__ import annotations

import json
import sys
import time
from functools import lru_cache
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "log"))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402

abl = import_module("04_ablation")
rob = import_module("05_failure_dense_robustness")
Index, V4_Coverage, raw_toks, toks = (abl.Index, abl.V4_Coverage, abl.raw_toks, abl.toks)
PAT_REQUIREMENT, PAT_MATTERS = abl.PAT_REQUIREMENT, abl.PAT_MATTERS
PAT_OVERRIDE, PAT_LOOKING, PAT_NOINFO = abl.PAT_OVERRIDE, abl.PAT_LOOKING, abl.PAT_NOINFO
Paraphraser = rob.Paraphraser

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


class Miner:
    """Greedy longest-match segmentation against the catalogue as dictionary."""

    MAXN, MINN = 9, 3
    MAX_DF = 4000
    IDEAL_DF = 500

    def __init__(self, index: Index):
        self.ix = index
        self._df = lru_cache(maxsize=400_000)(self._df_uncached)

    def _df_uncached(self, phrase_tokens: str) -> int:
        """Document frequency, CAPPED.

        An uncapped count(*) over an FTS5 phrase intersects full position lists, which
        is ruinous for common phrases -- and anything above MAX_DF is rejected anyway.
        Bounding the scan with a LIMIT subquery makes cost proportional to the cap
        rather than the collection: ~19s per full evaluation instead of intractable.
        """
        try:
            return self.ix.con.execute(
                "SELECT count(*) FROM (SELECT 1 FROM p WHERE p MATCH ? LIMIT ?)",
                (f'"{phrase_tokens}"', self.MAX_DF + 1)).fetchone()[0]
        except Exception:
            return 0

    def mine(self, text: str) -> list[tuple[str, int]]:
        t = raw_toks(text)
        out, i = [], 0
        while i < len(t):
            best = None
            for n in range(min(self.MAXN, len(t) - i), self.MINN - 1, -1):
                ph = " ".join(t[i:i + n])
                df = self._df(ph)
                if 0 < df <= self.MAX_DF:
                    best = (ph, df, n)
                    break
            if best:
                out.append((best[0], best[1]))
                i += best[2]
            else:
                i += 1
        return out


miner = Miner(ix)


def weight(phrase: str, df: int) -> float:
    """Evidence weight = phrase length x inverse document frequency.

    FINDING: rarity alone is not enough. Junk collocations such as "what i" (df=15) or
    "care about quality" (df=3) are RARE but not product-identifying, and under a pure
    1/sqrt(1+df) weight they outrank genuine provenance. Real provenance is long: the
    mined phrase "long torso camisole for extra coverage with spagetti" has df=1 at 8
    tokens. Scaling by token count separates the two without any hand-built stoplist.
    """
    return len(phrase.split()) / (1.0 + df) ** 0.5


class Sess:
    __slots__ = ("evidence", "turn", "asked", "last_rank")

    def __init__(self):
        self.evidence: dict[str, int] = {}
        self.turn = 0
        self.asked: list[str] = []
        self.last_rank: list[str] = []


class Grounded(V4_Coverage):
    """Template-first extraction, catalogue-grounded mining as fallback.

    FINDING: templates, when they fire, are higher precision than mining (they return
    the constraint exactly). The pass-5 collapse under paraphrase was NOT caused by
    using templates -- it was caused by the FALLBACK, which promoted the entire noisy
    message to a phrase constraint. Replacing that fallback with mining keeps clean
    precision and removes the paraphrase cliff.
    """

    PROBE_ORDER = ["feature", "color", "style", "size", "use_case", "material", "other"]
    POOL = 400
    MODE = "hybrid"          # 'hybrid' | 'mine' | 'template'

    def __init__(self, index, probe_order=None, mode="hybrid"):
        super().__init__(index)
        self.ss: dict[str, Sess] = {}
        self.order = probe_order if probe_order is not None else self.PROBE_ORDER
        self.mode = mode

    def reset(self, session_id, user_profile):
        self.ss[session_id] = Sess()

    def _pick(self, st: Sess):
        for a in self.order:
            if a not in st.asked:
                return a
        return "other"

    def _extract(self, msg: str) -> list[str]:
        """Template hits, PLUS the category clause.

        BUG FIXED HERE: the category was previously extracted only when no other
        template fired. But a `buying` turn-1 message matches PAT_REQUIREMENT, so its
        category clause -- the single most reliable channel, median 145 matching
        products -- was silently discarded on exactly the 40% of sessions that state a
        constraint up front. That alone cost ~0.10 of TechnicalScore against pass 4.
        Category and constraints are independent channels; always take both.
        """
        found: list[str] = []
        m = PAT_LOOKING.search(msg)
        if m and len(raw_toks(m.group(1))) >= 2:
            found.append(m.group(1).strip())
        for pat in (PAT_REQUIREMENT, PAT_MATTERS, PAT_OVERRIDE):
            mm = pat.search(msg)
            if mm:
                found.extend(p.strip() for p in mm.group(1).split(";") if p.strip())
        return found

    def _observe(self, st: Sess, msg: str) -> None:
        if PAT_NOINFO.search(msg):
            return
        got: list[tuple[str, int]] = []
        if self.mode in ("hybrid", "template"):
            for c in self._extract(msg):
                ph = " ".join(raw_toks(c)[:12])
                if ph:
                    got.append((ph, miner._df(ph)))
        # hybrid: mine only when templates produced no CONSTRAINT (category alone is
        # not enough evidence to skip mining on a paraphrased message)
        if self.mode == "mine" or (self.mode == "hybrid" and len(got) < 2):
            for ph, df in miner.mine(msg):
                got.append((ph, df))
        for ph, df in got:
            # Broad phrases (df > MAX_DF) are kept, not dropped: a bare material word is
            # useless for RANKING but still contributes recall and coverage tie-breaks.
            # weight() already discounts them by 1/sqrt(df).
            if ph and ph not in st.evidence:
                st.evidence[ph] = df if df > 0 else miner.MAX_DF * 2

    def respond(self, session_id, user_message, turn, top_k):
        st = self.ss.setdefault(session_id, Sess())
        st.turn += 1
        try:
            self._observe(st, user_message)
        except Exception:
            pass

        ev = sorted(st.evidence.items(), key=lambda kv: -weight(*kv))
        wmap = {p: weight(p, d) for p, d in ev}
        strong = [p for p, d in ev if d <= miner.IDEAL_DF]

        pool, seen = [], set()

        def add(expr, lim):
            for a in self.ix.search(expr, lim):
                if a not in seen:
                    seen.add(a)
                    pool.append(a)

        qs = [f'"{p}"' for p in strong[:8]]
        qa = [f'"{p}"' for p, _ in ev[:14]]
        if qs:
            add(" AND ".join(qs), self.POOL)
            for k in range(len(qs) - 1, 0, -1):
                if len(pool) >= self.POOL:
                    break
                add(" AND ".join(qs[:k]), self.POOL)
            add(" OR ".join(qs), self.POOL)
        if qa and len(pool) < self.POOL:
            add(" OR ".join(qa), self.POOL)
        if not pool:
            tk = list(dict.fromkeys(toks(user_message)))[:40]
            if tk:
                add(" OR ".join(f'"{t}"' for t in tk), self.POOL)

        probe = self._pick(st)
        st.asked.append(probe)
        if not pool:
            return {"message": "Could you tell me a bit more?", "ask_attribute": probe,
                    "recommendations": [{"parent_asin": a} for a in st.last_rank[:top_k]],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

        base = {a: i for i, a in enumerate(pool)}

        def score(a):
            return (-sum(wmap[p] for p in st.evidence if self.ix.covers(a, p)), base[a])

        ranked = sorted(pool, key=score)[:top_k]
        st.last_rank = ranked
        return {"message": "Here are the closest matches I found.", "ask_attribute": probe,
                "recommendations": [{"parent_asin": a} for a in ranked],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}}


def run(name, ag):
    t0 = time.time()
    r = evaluate(ag, samples, cid, cats, prods)
    print(f"{name:<48} HR@10 {r['hit_rate_at_10']:>6.1%}  MRR {r['mrr']:>6.3f}  "
          f"MTTC {r['mttc']:>5.2f}  SCORE {r['recommended_technical_score']:>6.4f}  ({time.time()-t0:.0f}s)")
    return r["recommended_technical_score"]


section("1. MINING SANITY CHECK")
for d in [
    "I'm looking for Tops & Tees Tanks & Camis. Long torso camisole for extra coverage with spagetti adjustable strap",
    "Hmm, What I care about: Quality soft cottonblend camisole, 95% cotton, 5% spandex, Neon Colors.",
]:
    print(f"\n  msg: {d[:86]}")
    for ph, df in miner.mine(d)[:6]:
        print(f"      df={df:<6} w={weight(ph, df):>5.2f}  {ph[:60]}")

section("2. EXTRACTION MODE x PARAPHRASE (the robustness matrix)")
print(f"{'system x condition':<48}")
print("-" * 108)
M = {}
for mode in ("template", "mine", "hybrid"):
    M[f"{mode}|clean"] = run(f"{mode:<9}| clean", Grounded(ix, mode=mode))
    M[f"{mode}|light"] = run(f"{mode:<9}| light paraphrase",
                             Paraphraser(Grounded(ix, mode=mode), 1, "light"))
    M[f"{mode}|heavy"] = run(f"{mode:<9}| heavy paraphrase",
                             Paraphraser(Grounded(ix, mode=mode), 2, "heavy"))
    print()
OUT["extraction_matrix"] = M

section("3. PROBE-ORDER ABLATION (best extraction mode)")
best_mode = max(("template", "mine", "hybrid"),
                key=lambda m: min(M[f"{m}|clean"], M[f"{m}|light"], M[f"{m}|heavy"]))
print(f"best worst-case mode: {best_mode}\n")
orders = {
    "'other' only (greedy maximal)": ["other"],
    "feature -> other": ["feature", "other"],
    "feature,material,color,style,...": ["feature", "material", "color", "style", "size", "use_case", "other"],
    "feature,color,style,size,use_case,material": ["feature", "color", "style", "size", "use_case", "material", "other"],
    "material first (worst prior)": ["material", "other"],
}
print(f"{'probe order':<48}")
print("-" * 108)
PO = {}
for nm, o in orders.items():
    PO[nm] = run(nm, Grounded(ix, probe_order=o, mode=best_mode))
OUT["probe_order"] = PO
OUT["best_mode"] = best_mode

Path(ROOT / "experiments" / "results" / "out_06.json").write_text(json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print("\n[saved] experiments/results/out_06.json")
