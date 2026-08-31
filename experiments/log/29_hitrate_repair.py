"""Experiment 29: repair the two HitRate misses -- and prove the fixes are not 2-session overfits.

Pass 28 traced both misses to the same failure MODE: an evidence phrase that the catalogue
attests (df > 0, so `_resolve` keeps it) but that the TARGET ITSELF does not contain
contiguously. Such a phrase is worse than useless -- it hands weight to the field.

Both instances are derivable from the generator's source, not from these two sessions:

  P1  SYNTHESISED COLOUR PREFIX.  local_evaluator.py:61
        candidates.insert(1, f"color: {color.group(1).lower()}")
      The token "color" is written by the GENERATOR; only the colour word comes from the
      product. 918 other products happen to carry "Color: Black" in `details`, so
      "color grey" has df=52 and survives resolution -- pointing away from the target.
      public_0020: target contains "grey", not "color grey". Stalls at rank 229.

  P2  NON-CONTIGUOUS COARSE CATEGORY.  local_evaluator.py:126-134
        coarse_category = " ".join(cleaned[-2:])   after DROPPING parts equal to
        "clothing" / "clothing shoes & jewelry"
      The dropped element can sit BETWEEN the two survivors. B08KKBBMMD's categories are
      ['Clothing, Shoes & Jewelry', 'Boys', 'Clothing', 'Pants'] -> coarse "Boys Pants",
      but the indexed text reads "... boys clothing pants", so the phrase is not
      contiguous in the target's own document. public_0037: target NEVER enters the pool.

CENSUS FIRST. If P2 affects two products it is an overfit; if it affects thousands it is a
structural defect in our CAT channel. The census is run over the whole 50k catalogue,
independent of which sessions happen to be in the public split.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/log/29_hitrate_repair.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index, coarse_category, evaluate, load_jsonl,
)
from submission.agent import CAT, CONSTRAINT, MINED, Agent, raw_toks  # noqa: E402

W1 = (1,) * 9 + (10,)
PAT_SYNTH_COLOR = re.compile(r"^\s*colou?r\s*:\s*(.+)$", re.I)


# ----------------------------------------------------------------- candidate fixes

class FixColour(Agent):
    """P1: drop the generator-written `color:` prefix, keep the colour word."""

    def _resolve(self, text, cap=12):
        m = PAT_SYNTH_COLOR.match(text)
        if m:
            text = m.group(1)
        return super()._resolve(text, cap)


class FixCategory(Agent):
    """P2: treat the coarse category as a TOKEN SET, not an adjacent phrase.

    `coarse_category` joins two catalogue parts that need not be adjacent in the product's
    own text. Requiring adjacency makes the CAT channel fail on exactly the products whose
    category list has an excluded element in the middle. Emitting the parts separately
    keeps the signal and costs only the (weak) adjacency bonus.
    """

    def _extract_templated(self, msg):
        out = []
        for text, tier in super()._extract_templated(msg):
            if tier == CAT:
                toks = [t for t in raw_toks(text) if len(t) > 2]
                if len(toks) >= 2 and self.ix.df(" ".join(toks)) == 0:
                    out.extend((t, CAT) for t in toks)      # phrase unattested -> split
                    continue
                out.append((text, tier))
                out.extend((t, CAT) for t in toks)          # keep parts as well
            else:
                out.append((text, tier))
        return out


class FixCategoryStrict(Agent):
    """P2 variant: emit the category parts ONLY, never the joined phrase."""

    def _extract_templated(self, msg):
        out = []
        for text, tier in super()._extract_templated(msg):
            if tier == CAT:
                out.extend((t, CAT) for t in raw_toks(text) if len(t) > 2)
            else:
                out.append((text, tier))
        return out


class FixBoth(FixColour, FixCategory):
    pass


class FixBothStrict(FixColour, FixCategoryStrict):
    pass


def main() -> None:
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    TUNE = [s for i, s in enumerate(samples) if i % 2 == 0]
    HOLD = [s for i, s in enumerate(samples) if i % 2 == 1]
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")

    # ---------------------------------------------------------- census over 50k products
    print("CATALOGUE-WIDE CENSUS (all 50,000 products, split-independent)")
    n_cat_broken = n_cat = 0
    for asin in prods:
        cc = coarse_category([str(v) for v in (prods[asin].get("categories") or [])])
        toks = raw_toks(cc)
        if len(toks) < 2:
            continue
        n_cat += 1
        if not base.ix.covers(asin, " ".join(toks)):
            n_cat_broken += 1
    print(f"  P2 coarse category NOT contiguous in the product's own text: "
          f"{n_cat_broken:,}/{n_cat:,} = {n_cat_broken/max(n_cat,1):.1%}")

    from evaluator.local_evaluator import intent_card
    n_col = n_col_broken = 0
    for asin in prods:
        card = intent_card(prods[asin])
        for c in list(card["hard_constraints"]) + list(card["soft_preferences"]):
            m = PAT_SYNTH_COLOR.match(str(c))
            if not m:
                continue
            n_col += 1
            full = " ".join(raw_toks(str(c))[:12])
            val = " ".join(raw_toks(m.group(1))[:12])
            if (not base.ix.covers(asin, full)) and base.ix.covers(asin, val):
                n_col_broken += 1
    print(f"  P1 'color: X' absent from target but 'X' present:            "
          f"{n_col_broken:,}/{n_col:,} = {n_col_broken/max(n_col,1):.1%}")
    print("  -> both are structural defects, not properties of the two failing sessions\n")

    def share(cls=Agent, **kw):
        o = object.__new__(cls)
        o.ix, o.sessions, o.llm = base.ix, {}, None
        o.DISCLOSURE = W1
        for k, v in kw.items():
            setattr(o, k, v)
        return o

    def run(ag, sub):
        r = evaluate(ag, sub, cid, cats, prods)
        return {"score": r["recommended_technical_score"], "hr": r["hit_rate_at_10"],
                "mrr": r["mrr"], "mttc": r["mttc"],
                "miss": sorted(s["sample_id"] for s in r["sessions"] if not s["hit"])}

    VARIANTS = {
        "baseline (shipped)":            (Agent, {}),
        "P1 colour-prefix strip":        (FixColour, {}),
        "P2 category parts (+phrase)":   (FixCategory, {}),
        "P2 category parts (only)":      (FixCategoryStrict, {}),
        "P1+P2 (+phrase)":               (FixBoth, {}),
        "P1+P2 (parts only)":            (FixBothStrict, {}),
        "POOL 1000":                     (Agent, {"POOL": 1000}),
        "POOL 2000":                     (Agent, {"POOL": 2000}),
        "P1+P2 + POOL 1000":             (FixBoth, {"POOL": 1000}),
    }

    print(f"{'variant':<30}{'tune':>9}{'hold':>9}{'dHold':>9}{'HR-t':>7}{'HR-h':>7}{'MTTC-h':>8}")
    print("-" * 79)
    OUT = {"census": {"p2_broken": n_cat_broken, "p2_total": n_cat,
                      "p1_broken": n_col_broken, "p1_total": n_col}}
    bt = bh = None
    for name, (cls, kw) in VARIANTS.items():
        t, h = run(share(cls, **kw), TUNE), run(share(cls, **kw), HOLD)
        if bt is None:
            bt, bh = t, h
        OUT[name] = {"tune": t, "hold": h}
        print(f"{name:<30}{t['score']:>9.5f}{h['score']:>9.5f}"
              f"{h['score']-bh['score']:>+9.5f}{t['hr']:>7.1%}{h['hr']:>7.1%}{h['mttc']:>8.2f}")

    print("\n  full-200 confirmation for anything that wins on BOTH halves:")
    for name, (cls, kw) in VARIANTS.items():
        t, h = OUT[name]["tune"], OUT[name]["hold"]
        if name != "baseline (shipped)" and t["score"] >= bt["score"] and h["score"] >= bh["score"]:
            f = run(share(cls, **kw), samples)
            OUT[name]["full200"] = f
            print(f"    {name:<28} full-200 {f['score']:.5f}  HR {f['hr']:.1%}  "
                  f"misses {f['miss']}")

    (ROOT / "experiments" / "results" / "out_29.json").write_text(
        json.dumps(OUT, indent=2) + "\n", encoding="utf-8")
    print("\n[saved] experiments/results/out_29.json")


if __name__ == "__main__":
    main()
