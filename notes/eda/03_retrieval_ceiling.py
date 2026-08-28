"""
EDA pass 3: retrieval ceiling.

Pass 2 showed the dialogue game is solved in <=3 probes (ask 'other'). So the
score is decided almost entirely by one question:

    given a coarse category string + <=4 verbatim constraint strings,
    how well can we rank ONE target out of 50,000?

Crucially, every constraint the simulator utters is lifted VERBATIM out of the
target's own catalog text (intent_card reads product['features'/'details'],
regex-matches material/colour over searchable_text, and formats price). So this
is not open-ended semantic search -- it is provenance recovery, and exact phrase
matching is the natural weapon. This script quantifies exactly how much
selectivity each channel buys.

Run:  PYTHONIOENCODING=utf-8 python notes/eda/03_retrieval_ceiling.py
"""
from __future__ import annotations

import json
import re
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    coarse_category,
    intent_card,
    load_jsonl,
    searchable_text,
)

CATALOG = ROOT / "data" / "catalog.jsonl"
PUBLIC = ROOT / "data" / "public_set.jsonl"
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
OUT = {}


def section(name: str) -> None:
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")


def toks(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def phrase(text: str, cap: int = 12) -> str:
    t = toks(text)[:cap]
    return '"' + " ".join(t) + '"' if t else ""


# ---------------------------------------------------------------- index
print("building FTS5 index over 50k products ...")
con = sqlite3.connect(":memory:")
con.execute(
    "CREATE VIRTUAL TABLE p USING fts5(asin UNINDEXED, title, categories, features, "
    "details, store, description, tokenize='unicode61 remove_diacritics 2')"
)


def flat(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, dict):
        return " ".join(f"{k} {x}" for k, x in v.items())
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return str(v)


products: dict[str, dict] = {}
rows = []
with CATALOG.open(encoding="utf-8") as fh:
    for line in fh:
        d = json.loads(line)
        a = str(d["parent_asin"])
        products[a] = d
        rows.append((a, flat(d.get("title")), flat(d.get("categories")), flat(d.get("features")),
                     flat(d.get("details")), flat(d.get("store")), flat(d.get("description"))))
con.executemany("INSERT INTO p VALUES (?,?,?,?,?,?,?)", rows)
con.commit()
print(f"indexed {len(rows):,}")

samples = load_jsonl(PUBLIC)
cards = {s["sample_id"]: intent_card(products[str(s["ground_truth"]["parent_asin"])]) for s in samples}
cats = {s["sample_id"]: coarse_category([str(v) for v in (products[str(s["ground_truth"]["parent_asin"])].get("categories") or [])])
        for s in samples}


def count_match(expr: str) -> int:
    if not expr.strip(' "'):
        return -1
    try:
        return con.execute("SELECT count(*) FROM p WHERE p MATCH ?", (expr,)).fetchone()[0]
    except sqlite3.OperationalError:
        return -1


# ---------------------------------------------------------------- 1
section("1. SELECTIVITY OF EACH DISCLOSED CHANNEL (how many of 50k products match?)")

chan = {"coarse_category": [], "hard[0]": [], "hard[1]": [], "soft[0]": [], "soft[1]": []}
for s in samples:
    sid = s["sample_id"]
    c = cards[sid]
    hc = [str(x) for x in c["hard_constraints"]]
    sp = [str(x) for x in c["soft_preferences"]]
    chan["coarse_category"].append(count_match(phrase(cats[sid])))
    for i, key in ((0, "hard[0]"), (1, "hard[1]")):
        if i < len(hc):
            chan[key].append(count_match(phrase(hc[i])))
    for i, key in ((0, "soft[0]"), (1, "soft[1]")):
        if i < len(sp):
            chan[key].append(count_match(phrase(sp[i])))

print(f"{'channel':<18} {'n':>5} {'median':>9} {'mean':>10} {'p10':>8} {'p90':>9}  interpretation")
print("-" * 92)
for k, v in chan.items():
    v = [x for x in v if x >= 0]
    if not v:
        continue
    vs = sorted(v)
    med = statistics.median(vs)
    note = ("near-useless" if med > 2000 else
            "weak" if med > 300 else
            "useful" if med > 25 else
            "highly selective")
    print(f"{k:<18} {len(v):>5} {med:>9,.0f} {statistics.fmean(vs):>10,.0f} "
          f"{vs[len(vs)//10]:>8,} {vs[9*len(vs)//10]:>9,}  {note}")
OUT["channel_selectivity"] = {k: statistics.median([x for x in v if x >= 0]) for k, v in chan.items() if any(x >= 0 for x in v)}

print("\n=> hard[0] is a bare material word in 76.5% of sessions ('polyester', 'leather'),")
print("   which matches thousands of products. The DISCRIMINATIVE signal lives in the")
print("   soft_preferences -- the long free-text feature bullets -- which are ONLY")
print("   released once you probe. This is why a stateless turn-1-only agent cannot win.")

# ---------------------------------------------------------------- 2
section("2. CONJUNCTIVE NARROWING (AND-ing channels as the dialogue progresses)")

stages = {
    "turn1 browsing (category only)":            lambda sid, hc, sp: [cats[sid]],
    "turn1 buying (category + hard[0])":         lambda sid, hc, sp: [cats[sid]] + hc[:1],
    "after 1 probe (cat + hard[0..1])":          lambda sid, hc, sp: [cats[sid]] + hc[:2],
    "after 2 probes (cat + hard + soft[0])":     lambda sid, hc, sp: [cats[sid]] + hc[:2] + sp[:1],
    "after 3 probes (cat + ALL 4 constraints)":  lambda sid, hc, sp: [cats[sid]] + hc[:2] + sp[:2],
    "ALL constraints, no category":              lambda sid, hc, sp: hc[:2] + sp[:2],
}

stage_stats = {}
for name, fn in stages.items():
    sizes, zero, uniq = [], 0, 0
    for s in samples:
        sid = s["sample_id"]
        c = cards[sid]
        hc = [str(x) for x in c["hard_constraints"]]
        sp = [str(x) for x in c["soft_preferences"]]
        parts = [phrase(x) for x in fn(sid, hc, sp)]
        parts = [x for x in parts if x.strip(' "')]
        if not parts:
            continue
        n = count_match(" AND ".join(parts))
        if n < 0:
            continue
        sizes.append(n)
        if n == 0:
            zero += 1
        if n == 1:
            uniq += 1
    ss = sorted(sizes)
    stage_stats[name] = {
        "median": statistics.median(ss), "mean": statistics.fmean(ss),
        "pct_empty": zero / len(ss), "pct_unique": uniq / len(ss),
        "pct_le_10": sum(1 for x in ss if 0 < x <= 10) / len(ss),
    }

print(f"{'dialogue stage':<42} {'median':>8} {'≤10 hits':>9} {'unique':>8} {'EMPTY':>7}")
print("-" * 82)
for k, v in stage_stats.items():
    print(f"{k:<42} {v['median']:>8,.0f} {v['pct_le_10']:>8.1%} {v['pct_unique']:>7.1%} {v['pct_empty']:>6.1%}")
OUT["conjunctive_stages"] = stage_stats

print("\n=> Strict AND collapses to ZERO matches often (tokenisation/truncation drift),")
print("   so a pure boolean filter is brittle. Ranking must degrade gracefully:")
print("   score by how many constraints match, never hard-filter to nothing.")

# ---------------------------------------------------------------- 3
section("3. BM25 RANK OF THE TARGET UNDER PROGRESSIVE DISCLOSURE")

W = "bm25(p, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)"


def ranked(expr: str, limit: int = 200) -> list[str]:
    try:
        return [r[0] for r in con.execute(
            f"SELECT asin FROM p WHERE p MATCH ? ORDER BY {W} LIMIT ?", (expr, limit)).fetchall()]
    except sqlite3.OperationalError:
        return []


def eval_stage(builder, mode: str) -> dict:
    ranks = []
    for s in samples:
        sid = s["sample_id"]
        tgt = str(s["ground_truth"]["parent_asin"])
        c = cards[sid]
        hc = [str(x) for x in c["hard_constraints"]]
        sp = [str(x) for x in c["soft_preferences"]]
        parts = [x for x in builder(sid, hc, sp) if x and str(x).strip()]
        if mode == "phrase_or":
            expr = " OR ".join(p for p in (phrase(x) for x in parts) if p.strip(' "'))
        elif mode == "token_or":
            tk = list(dict.fromkeys(t for x in parts for t in toks(str(x))))[:40]
            expr = " OR ".join(f'"{t}"' for t in tk)
        else:  # phrase_and_backoff
            ps = [p for p in (phrase(x) for x in parts) if p.strip(' "')]
            expr = " AND ".join(ps)
            if not ranked(expr, 1):
                expr = " OR ".join(ps)
        if not expr:
            ranks.append(None)
            continue
        res = ranked(expr, 200)
        ranks.append(res.index(tgt) + 1 if tgt in res else None)
    hit10 = sum(1 for r in ranks if r and r <= 10) / len(ranks)
    mrr = statistics.fmean(0.0 if not r else 1.0 / r for r in ranks)
    mrr10 = statistics.fmean(0.0 if (not r or r > 10) else 1.0 / r for r in ranks)
    return {"hit@10": hit10, "hit@200": sum(1 for r in ranks if r) / len(ranks),
            "mrr": mrr, "mrr@10": mrr10,
            "median_rank_when_found": statistics.median([r for r in ranks if r]) if any(ranks) else None}


builders = {
    "cat only (browsing turn1)":     lambda sid, hc, sp: [cats[sid]],
    "cat + hard[0] (buying turn1)":  lambda sid, hc, sp: [cats[sid]] + hc[:1],
    "cat + hard[0:2]":               lambda sid, hc, sp: [cats[sid]] + hc[:2],
    "cat + ALL 4 constraints":       lambda sid, hc, sp: [cats[sid]] + hc[:2] + sp[:2],
    "ALL 4 constraints (no cat)":    lambda sid, hc, sp: hc[:2] + sp[:2],
}

for mode in ("token_or", "phrase_or", "phrase_and_backoff"):
    print(f"\n--- retrieval mode: {mode} ---")
    print(f"{'evidence available':<32} {'HR@10':>7} {'MRR':>7} {'HR@200':>8} {'medRank':>8}")
    print("-" * 68)
    for name, b in builders.items():
        r = eval_stage(b, mode)
        OUT.setdefault("bm25_progressive", {})[f"{mode} | {name}"] = r
        mr = r["median_rank_when_found"]
        print(f"{name:<32} {r['hit@10']:>6.1%} {r['mrr']:>7.3f} {r['hit@200']:>7.1%} "
              f"{mr if mr is not None else '-':>8}")

print("\n=> Read the last row of each block as the ceiling for a PERFECT dialogue agent")
print("   using that retrieval mode alone. Everything above it is what the baseline")
print("   throws away by being stateless.")

Path(ROOT / "notes" / "eda" / "out_03.json").write_text(json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print("\n\n[saved] notes/eda/out_03.json")
