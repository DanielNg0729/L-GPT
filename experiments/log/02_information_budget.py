"""
Experiment 2: the simulator's total information budget, and the dominance of ask_attribute='other'.

customer_reply() resolves an ask against:
    constraints = hard_constraints + soft_preferences        (<= 4 short strings)
    matches = [c for c in constraints
               if c not in disclosed
               and (attribute == "other" or classify_constraint(c) == attribute)][:2]

Two consequences worth proving empirically rather than asserting:
  (a) the hidden state is TINY  -- a handful of short strings, fully enumerable;
  (b) attribute == "other" bypasses the classifier, so it is a strictly
      dominant probe: it returns the first 2 undisclosed constraints whatever
      they are, whereas a typed ask returns nothing unless the classifier agrees.

Run:  PYTHONIOENCODING=utf-8 python experiments/log/02_information_budget.py
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    ALLOWED_ATTRIBUTES,
    behavior_for,
    classify_constraint,
    customer_reply,
    intent_card,
    load_jsonl,
)

CATALOG = ROOT / "data" / "catalog.jsonl"
PUBLIC = ROOT / "data" / "public_set.jsonl"

products: dict[str, dict] = {}
with CATALOG.open(encoding="utf-8") as fh:
    for line in fh:
        p = json.loads(line)
        products[str(p["parent_asin"])] = p
samples = load_jsonl(PUBLIC)

OUT = {}


def section(name: str) -> None:
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")


def card_for(s: dict) -> dict:
    return intent_card(products[str(s["ground_truth"]["parent_asin"])])


# ------------------------------------------------------------------ 1
section("1. TOTAL HIDDEN INFORMATION PER SESSION")

n_constraints, total_chars, uniq_counts = [], [], []
for s in samples:
    c = card_for(s)
    cons = [str(x) for x in c["hard_constraints"]] + [str(x) for x in c["soft_preferences"]]
    uniq = list(dict.fromkeys(cons))
    n_constraints.append(len(cons))
    uniq_counts.append(len(uniq))
    total_chars.append(sum(len(x) for x in uniq))

print(f"constraint slots per session : mean {statistics.fmean(n_constraints):.2f} (hard<=2 + soft<=2)")
print(f"DISTINCT constraints         : mean {statistics.fmean(uniq_counts):.2f}, "
      f"distribution {dict(sorted(Counter(uniq_counts).items()))}")
print(f"total hidden text per session: mean {statistics.fmean(total_chars):.0f} chars, "
      f"median {statistics.median(total_chars):.0f}, max {max(total_chars)}")
print("\n=> The ENTIRE hidden intent is a few hundred characters. There is no deep")
print("   latent preference to discover; the dialogue is a bounded extraction game.")
OUT["distinct_constraints_mean"] = statistics.fmean(uniq_counts)
OUT["hidden_chars_mean"] = statistics.fmean(total_chars)

# ------------------------------------------------------------------ 2
section("2. PROBE-STRATEGY COMPARISON (exhaustive, against the real customer_reply)")

STRATEGIES: dict[str, list[str] | None] = {
    "always 'other'":            ["other"] * 10,
    "always 'feature'":          ["feature"] * 10,
    "always 'material'":         ["material"] * 10,
    "cycle feature/material/color": ["feature", "material", "color"] * 4,
    "cycle ALL 10 enum values":  sorted(ALLOWED_ATTRIBUTES) + ["other"],
    "always 'category' (dead)":  ["category"] * 10,
    "always 'brand' (dead)":     ["brand"] * 10,
    "always null (no ask)":      [None] * 10,
}

results = {}
for name, plan in STRATEGIES.items():
    turns_to_full, frac_at_turn = [], defaultdict(list)
    for s in samples:
        card = card_for(s)
        eff = {**s, "intent_card": card}
        cons = list(dict.fromkeys(
            [str(x) for x in card["hard_constraints"]] + [str(x) for x in card["soft_preferences"]]))
        disclosed: set[str] = set()
        boundary_used = False
        full_at = None
        for t in range(10):
            attr = plan[t] if t < len(plan) else plan[-1]
            _, boundary_used = customer_reply(eff, attr, disclosed, boundary_used)
            got = len([c for c in cons if c in disclosed])
            frac_at_turn[t + 1].append(got / max(1, len(cons)))
            if full_at is None and got >= len(cons):
                full_at = t + 1
        turns_to_full.append(full_at if full_at else 11)
    results[name] = {
        "mean_turns_to_full_disclosure": statistics.fmean(turns_to_full),
        "pct_fully_disclosed_by_turn_3": sum(1 for t in turns_to_full if t <= 3) / len(turns_to_full),
        "frac_known_after_1_probe": statistics.fmean(frac_at_turn[1]),
        "frac_known_after_2_probes": statistics.fmean(frac_at_turn[2]),
        "frac_known_after_3_probes": statistics.fmean(frac_at_turn[3]),
    }

print(f"{'probe strategy':<30} {'after1':>7} {'after2':>7} {'after3':>7} {'turns→all':>10} {'%all≤3':>8}")
print("-" * 78)
for name, r in sorted(results.items(), key=lambda kv: -kv[1]["frac_known_after_2_probes"]):
    print(f"{name:<30} {r['frac_known_after_1_probe']:>6.1%} {r['frac_known_after_2_probes']:>6.1%} "
          f"{r['frac_known_after_3_probes']:>6.1%} {r['mean_turns_to_full_disclosure']:>10.2f} "
          f"{r['pct_fully_disclosed_by_turn_3']:>7.1%}")
OUT["probe_strategies"] = results

print("\n=> 'other' dominates: it bypasses classify_constraint entirely and returns")
print("   the first 2 undisclosed constraints unconditionally. No information-gain")
print("   machinery can beat a probe that is already extracting the maximum per turn.")

# ------------------------------------------------------------------ 3
section("3. WHY TYPED ASKS UNDERPERFORM: classifier collision")

kind_counts = Counter()
for s in samples:
    card = card_for(s)
    cons = list(dict.fromkeys(
        [str(x) for x in card["hard_constraints"]] + [str(x) for x in card["soft_preferences"]]))
    for c in cons:
        kind_counts[classify_constraint(c)] += 1
tot = sum(kind_counts.values())
print("classify_constraint() over every constraint string in the public set:\n")
print(f"{'bucket':<12} {'count':>7} {'share':>8}")
for k, n in kind_counts.most_common():
    print(f"{k:<12} {n:>7} {n/tot:>7.1%}")
print(f"\nnever emitted: {sorted(ALLOWED_ATTRIBUTES - set(kind_counts))}")
print("\n=> 'feature' is the classifier's fallback bucket and absorbs the majority of")
print("   constraints, so a typed ask is mostly a coin-flip between feature/material.")
OUT["constraint_kind_distribution"] = dict(kind_counts)

# ------------------------------------------------------------------ 4
section("4. BOUNDARY + OVERRIDE INTERACTION")

b = [s for s in samples if s["scenario_type"] == "boundary"]
print(f"boundary sessions: {len(b)} ({len(b)/len(samples):.1%})")
print("  first typed ask is ALWAYS deflected ('use your judgment'), whatever it is.")
print("  => costs exactly one probe; unavoidable, affects every strategy equally.")

ov = [s for s in samples if s["scenario_type"] == "intent_override"]
ovt = Counter()
for s in ov:
    rng = random.Random(f"{s['sample_id']}\0{s['scenario_type']}")
    ovt[behavior_for("intent_override", card_for(s), rng)["override"]["turn"]] += 1
print(f"\nintent_override sessions: {len(ov)} ({len(ov)/len(samples):.1%}), override turn {dict(sorted(ovt.items()))}")
print("  hits are GATED until override_applied; and on the override turn the harness")
print("  REPLACES the user message, discarding that turn's ask_attribute entirely.")
print("  => in those sessions one probe is silently voided, and the new_value string")
print("     is added to `disclosed` without ever being spoken to the agent.")

lost = 0
for s in ov:
    rng = random.Random(f"{s['sample_id']}\0{s['scenario_type']}")
    beh = behavior_for("intent_override", card_for(s), rng)
    nv = str(beh["override"]["new_value"])
    card = card_for(s)
    cons = list(dict.fromkeys(
        [str(x) for x in card["hard_constraints"]] + [str(x) for x in card["soft_preferences"]]))
    if nv in cons:
        lost += 1
print(f"\n  override new_value is drawn from hard_constraints[0] and is marked disclosed")
print(f"  without disclosure in {lost}/{len(ov)} sessions -- but it IS quoted verbatim")
print(f"  inside the override message, so a stateful agent still receives the text.")

Path(ROOT / "experiments" / "results" / "out_02.json").write_text(json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print("\n\n[saved] experiments/results/out_02.json")
