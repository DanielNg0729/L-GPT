"""
Experiment 1: catalog structure + reverse-engineering the simulator's information channel.

The central question this script answers: the evaluator synthesises every customer
utterance deterministically from the *target product's own metadata*. So what,
exactly, does the agent get told, and how identifying is it?

Run:  python experiments/log/01_catalog_and_leak.py
"""
from __future__ import annotations

import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Reuse the ACTUAL simulator functions so our analysis cannot drift from the harness.
from evaluator.local_evaluator import (  # noqa: E402
    ALLOWED_ATTRIBUTES,
    behavior_for,
    classify_constraint,
    coarse_category,
    initial_message,
    intent_card,
    load_jsonl,
    searchable_text,
)

CATALOG = ROOT / "data" / "catalog.jsonl"
PUBLIC = ROOT / "data" / "public_set.jsonl"

OUT = {}


def section(name: str) -> None:
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")


# ---------------------------------------------------------------- load
products: dict[str, dict] = {}
with CATALOG.open(encoding="utf-8") as fh:
    for line in fh:
        p = json.loads(line)
        products[str(p["parent_asin"])] = p

samples = load_jsonl(PUBLIC)

section("1. CATALOG SHAPE")
print(f"products: {len(products):,}")
print(f"public sessions: {len(samples):,}")

field_present = Counter()
field_empty = Counter()
for p in products.values():
    for k, v in p.items():
        if v in (None, "", [], {}):
            field_empty[k] += 1
        else:
            field_present[k] += 1

print(f"\n{'field':<20} {'non-empty':>10} {'coverage':>10}")
for k in sorted(field_present, key=lambda x: -field_present[x]):
    cov = field_present[k] / len(products)
    print(f"{k:<20} {field_present[k]:>10,} {cov:>9.1%}")
OUT["field_coverage"] = {k: field_present[k] / len(products) for k in field_present}

# text length stats
lens = defaultdict(list)
for p in products.values():
    lens["title"].append(len(str(p.get("title") or "")))
    lens["features"].append(len(" ".join(str(x) for x in (p.get("features") or []))))
    lens["description"].append(len(" ".join(str(x) for x in (p.get("description") or []))))
    lens["searchable_total"].append(len(searchable_text(p)))

print(f"\n{'field':<20} {'mean':>8} {'median':>8} {'p90':>8} {'max':>8}")
for k, vals in lens.items():
    vals_sorted = sorted(vals)
    p90 = vals_sorted[int(0.9 * len(vals_sorted))]
    print(f"{k:<20} {statistics.fmean(vals):>8.0f} {statistics.median(vals):>8.0f} "
          f"{p90:>8} {max(vals):>8}")

# categories
cat_counter = Counter()
for p in products.values():
    cat_counter[coarse_category([str(v) for v in (p.get("categories") or [])])] += 1
print(f"\ndistinct coarse_category values: {len(cat_counter):,}")
print("top 15 coarse categories (what turn-1 messages actually say):")
for c, n in cat_counter.most_common(15):
    print(f"  {n:>6,}  {c}")
OUT["n_coarse_categories"] = len(cat_counter)

# price
prices = [float(p["price"]) for p in products.values()
          if p.get("price") not in (None, "") and str(p["price"]).replace(".", "").isdigit()]
print(f"\nprice present: {len(prices):,} ({len(prices)/len(products):.1%})")
if prices:
    ps = sorted(prices)
    print(f"  median ${statistics.median(ps):.2f}  p10 ${ps[len(ps)//10]:.2f}  p90 ${ps[9*len(ps)//10]:.2f}")

# ---------------------------------------------------------------- intent cards
section("2. WHAT THE SIMULATOR ACTUALLY REVEALS (reconstructed intent cards)")

cards = {}
for s in samples:
    tgt = str(s["ground_truth"]["parent_asin"])
    cards[s["sample_id"]] = intent_card(products[tgt])

hard0_kind = Counter()
MATERIAL_WORDS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")


def kind_of(c: str) -> str:
    cl = c.lower().strip()
    if cl in MATERIAL_WORDS:
        return "bare material word"
    if cl.startswith("color:"):
        return "color: X"
    if cl.startswith("budget around"):
        return "budget"
    if ":" in c[:24]:
        return "details key: value"
    return "feature bullet (free text)"


hard_lens, soft_lens = [], []
for sid, card in cards.items():
    hc = card["hard_constraints"]
    sp = card["soft_preferences"]
    if hc:
        hard0_kind[kind_of(str(hc[0]))] += 1
    hard_lens.extend(len(str(x)) for x in hc)
    soft_lens.extend(len(str(x)) for x in sp)

print("hard_constraints[0] composition  -  this is the ONLY constraint in a Buying turn-1 message:")
for k, n in hard0_kind.most_common():
    print(f"  {n:>4} ({n/len(cards):>5.1%})  {k}")
OUT["hard0_kind"] = dict(hard0_kind)

print(f"\nhard constraint string length: mean {statistics.fmean(hard_lens):.0f} chars, median {statistics.median(hard_lens):.0f}")
print(f"soft preference string length: mean {statistics.fmean(soft_lens):.0f} chars, median {statistics.median(soft_lens):.0f}")

print("\n--- 8 sample intent cards (verbatim from the simulator) ---")
for sid in list(cards)[:8]:
    s = next(x for x in samples if x["sample_id"] == sid)
    print(f"\n[{sid}] scenario={s['scenario_type']} difficulty={s['difficulty_bucket']}")
    print(f"  target_category : {cards[sid]['target_category'][:90]}")
    print(f"  hard_constraints: {[str(x)[:70] for x in cards[sid]['hard_constraints']]}")
    print(f"  soft_preferences: {[str(x)[:70] for x in cards[sid]['soft_preferences']]}")

# ---------------------------------------------------------------- turn 1
section("3. THE TURN-1 MESSAGE (all 200, by scenario)")

turn1 = {}
for s in samples:
    tgt = str(s["ground_truth"]["parent_asin"])
    eff = {**s, "intent_card": cards[s["sample_id"]],
           "behavior": behavior_for(str(s["scenario_type"]), cards[s["sample_id"]],
                                    random.Random(f"{s['sample_id']}\0{s['scenario_type']}"))}
    disclosed: set[str] = set()
    cat = coarse_category([str(v) for v in (products[tgt].get("categories") or [])])
    turn1[s["sample_id"]] = initial_message(eff, cat, disclosed)

by_scen = defaultdict(list)
for s in samples:
    by_scen[s["scenario_type"]].append(turn1[s["sample_id"]])

t1_lens = []
for scen, msgs in sorted(by_scen.items()):
    ls = [len(m.split()) for m in msgs]
    t1_lens.extend(ls)
    print(f"\n{scen:<16} n={len(msgs):<4} mean {statistics.fmean(ls):.1f} words")
    for m in msgs[:3]:
        print(f"     | {m[:150]}")

print(f"\noverall turn-1 length: mean {statistics.fmean(t1_lens):.1f} words, median {statistics.median(t1_lens):.0f}")
OUT["turn1_mean_words"] = statistics.fmean(t1_lens)

# ---------------------------------------------------------------- attribute reachability
section("4. WHICH ask_attribute VALUES CAN EVER PAY OUT?")

reachable = Counter()
per_session_reachable = []
for sid, card in cards.items():
    all_c = [str(v) for v in card["hard_constraints"]] + [str(v) for v in card["soft_preferences"]]
    kinds = {classify_constraint(c) for c in all_c}
    per_session_reachable.append(len(kinds))
    for k in kinds:
        reachable[k] += 1

print("For each attribute: in how many of the 200 sessions does ANY constraint classify to it?")
print("(i.e. an upper bound on how often asking it can possibly return information)\n")
print(f"{'attribute':<12} {'sessions':>9} {'share':>8}   note")
for a in sorted(ALLOWED_ATTRIBUTES):
    n = reachable.get(a, 0)
    note = ""
    if n == 0:
        note = "<-- STRUCTURALLY UNREACHABLE (classify_constraint never emits it)"
    print(f"{a:<12} {n:>9} {n/len(cards):>7.1%}   {note}")
print(f"\n'other' is special-cased: it matches ANY undisclosed constraint (bypasses classifier).")
print(f"mean distinct payable attributes per session: {statistics.fmean(per_session_reachable):.2f}")
OUT["attr_reachability"] = {a: reachable.get(a, 0) / len(cards) for a in sorted(ALLOWED_ATTRIBUTES)}

# ---------------------------------------------------------------- override turns
section("5. INTENT-OVERRIDE TIMING (hard floor on MTTC)")

ov_turns = Counter()
for s in samples:
    if s["scenario_type"] != "intent_override":
        continue
    rng = random.Random(f"{s['sample_id']}\0{s['scenario_type']}")
    b = behavior_for("intent_override", cards[s["sample_id"]], rng)
    ov_turns[b["override"]["turn"]] += 1

print(f"override turn distribution across the {sum(ov_turns.values())} intent_override sessions: {dict(sorted(ov_turns.items()))}")
mean_ov = sum(t * n for t, n in ov_turns.items()) / max(1, sum(ov_turns.values()))
print(f"mean earliest-possible hit turn for those sessions: {mean_ov:.3f}")

mix = Counter(s["scenario_type"] for s in samples)
best_mttc = sum(
    (mean_ov if scen == "intent_override" else 1.0) * n for scen, n in mix.items()
) / len(samples)
best_eff = max(0.0, min(1.0, (11 - best_mttc) / 10))
print(f"\nPERFECT-PLAY CEILING on the public set:")
print(f"  MTTC        = {best_mttc:.4f}")
print(f"  Efficiency  = {best_eff:.4f}")
print(f"  TechnicalScore_max = 0.5*1 + 0.3*1 + 0.2*{best_eff:.4f} = {0.5 + 0.3 + 0.2*best_eff:.4f}")
OUT["ceiling"] = {"mttc": best_mttc, "efficiency": best_eff,
                  "technical_score": 0.5 + 0.3 + 0.2 * best_eff}

# what does each 1-turn delay cost?
print(f"\nMarginal cost of being one turn slower on EVERY session: "
      f"0.2 * (1/10) = {0.2*0.1:.4f} TechnicalScore")
print(f"Marginal cost of dropping one session from rank 1 to rank 5: "
      f"0.3 * (1 - 0.2)/200 = {0.3*0.8/200:.5f} per session "
      f"({0.3*0.8:.4f} if it happened to all)")
print(f"Marginal cost of one session going from hit to miss: "
      f"(0.5*1 + 0.3*1)/200 + efficiency drag = ~{0.8/200:.5f}+ per session")

Path(ROOT / "experiments" / "results" / "out_01.json").write_text(json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print(f"\n\n[saved] experiments/results/out_01.json")
