"""V2.53: does the shipped span node need the route classifier, or is the gate enough?

THE DECISION THIS SETTLES
-------------------------
The V2 prototype gated its exact category and short-span lookups on a TRAINED six-route
classifier's action label. The shipped integration uses the agent's own literal recognition
gate instead: the lookups run on any message matching none of the simulator's known shapes.
That removed a 257 MB model dependency from the submission -- which already carries 254 MB
for the scaffolding tagger -- but it gave up whatever precision the action label was buying.

The action label's job is to say WHICH extraction a message can support:

    buying_opening / override_opening / plain_opening   may carry a CATEGORY
    buying_opening / override_opening /
      constraint_update / override_update               may carry ATTRIBUTES
    no_evidence                                         carries NOTHING

Without it, the shipped node runs both lookups on every unrecognised message, including
the `no_evidence` ones -- "I don't have a preference for colour" reworded. V1's `PAT_NOINFO`
catches those only in their LITERAL form, and a reworded one sails past it. So the concrete
risk is spurious attribute evidence mined out of a message that states the customer has no
requirement at all, which is worse than useless: it is confident evidence for the wrong
thing.

WHAT IS MEASURED. Every row of the held-out wrapper bank, grouped by its TRUE action, run
through the shipped span node. Two quantities per group:

    recall-side    on rows that DO carry a value, is the value recovered
    precision-side on `no_evidence` rows, how much evidence is invented

The second is the number that decides the 257 MB. If reworded no-evidence messages yield
almost nothing, the recognition gate is sufficient and the classifier is 257 MB spent on a
problem that does not occur. If they yield a lot, the classifier is buying real precision
and the size is justified.

WHAT THIS IS NOT. It is not a score. Spurious evidence on a no-evidence turn does not
straightforwardly cost score -- the agent may recover on a later turn, and the ranker
attenuates broad phrases by idf. The end-to-end grid measures cost; this measures the
mechanism, so that a cost measured there can be attributed rather than guessed at.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/studies/audit_span_node_precision.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "experiments" / "studies"
BANK = V2 / "v1_turn_gated_bank" / "final_test.jsonl"
OUT = V2 / "results" / "span_node_precision_v2_53.json"

CARRIES_VALUE = {"buying_opening", "override_opening", "constraint_update",
                 "override_update"}
CARRIES_CATEGORY = {"buying_opening", "override_opening", "plain_opening"}


def main() -> None:
    os.environ["LLM_RERANK"] = "0"
    os.environ["LLM_EXTRACT"] = "0"
    os.environ["LLM_RESOLVE"] = "0"
    os.environ["BERT_EXTRACT"] = "1"
    from submission.agent import Agent, raw_toks, recognised

    agent = Agent(ROOT / "data" / "catalog.jsonl")
    node = agent.span_node
    if node is None or not node.ok:
        print("span node unavailable."); return
    tagger = agent.tagger
    print(f"span node: {node.stats()}")
    print(f"tagger enabled: {tagger is not None and tagger.enabled}\n")

    rows = [json.loads(l) for l in BANK.open(encoding="utf-8") if l.strip()]
    per = defaultdict(lambda: {"n": 0, "recognised": 0, "category": 0, "attrs": 0,
                               "value_hit": 0, "value_rows": 0, "rows_with_attrs": 0})
    for r in rows:
        action, msg = r["action"], r["message"]
        g = per[action]
        g["n"] += 1
        if recognised(msg):
            # The shipped node never runs here: V1's template path already handles it.
            g["recognised"] += 1
            continue
        cleaned = None
        if tagger is not None and tagger.enabled:
            try:
                cleaned = tagger.strip(msg)
            except Exception:
                cleaned = None
        category, attrs = node.extract(raw_toks(cleaned or msg))
        g["category"] += 1 if category else 0
        g["attrs"] += len(attrs)
        g["rows_with_attrs"] += 1 if attrs else 0
        # SLOT NAME DEPENDS ON THE ACTION. An override OPENING carries its value in slot
        # `b` (the released opening's second field is `soft_preferences[-1]`), not `a`.
        # The first version of this audit read `a` for every action and therefore reported
        # override_opening recall as 0.0000 -- a measurement bug, not a node failure.
        slot = "b" if r["action"] == "override_opening" else "a"
        truth = str(r.get("slots", {}).get(slot) or "").strip()
        if action in CARRIES_VALUE and truth:
            g["value_rows"] += 1
            norm = " ".join(raw_toks(truth))
            g["value_hit"] += 1 if norm in attrs else 0

    print(f"{'action':<20}{'rows':>6}{'recog':>7}{'val recall':>12}"
          f"{'cat rate':>10}{'attrs/row':>11}{'rows w/ attrs':>15}")
    print("-" * 81)
    for action in ("buying_opening", "override_opening", "constraint_update",
                   "override_update", "plain_opening", "no_evidence"):
        g = per[action]
        live = g["n"] - g["recognised"]
        rec = g["value_hit"] / g["value_rows"] if g["value_rows"] else float("nan")
        print(f"{action:<20}{g['n']:>6}{g['recognised']:>7}"
              f"{rec:>12.4f}" if g["value_rows"] else
              f"{action:<20}{g['n']:>6}{g['recognised']:>7}{'--':>12}", end="")
        print(f"{(g['category'] / live if live else 0):>10.3f}"
              f"{(g['attrs'] / live if live else 0):>11.3f}"
              f"{(g['rows_with_attrs'] / live if live else 0):>15.3f}")

    ne = per["no_evidence"]
    ne_live = ne["n"] - ne["recognised"]
    carry = [a for a in CARRIES_VALUE if per[a]["value_rows"]]
    mean_carry = (sum(per[a]["attrs"] for a in carry)
                  / max(sum(per[a]["n"] - per[a]["recognised"] for a in carry), 1))
    print(f"\n  VERDICT INPUT")
    print(f"  value-bearing actions: {mean_carry:.3f} attributes per message")
    print(f"  no_evidence:           "
          f"{(ne['attrs'] / ne_live if ne_live else 0):.3f} attributes per message, "
          f"{(ne['rows_with_attrs'] / ne_live if ne_live else 0):.1%} of rows affected")
    ratio = (ne["attrs"] / ne_live) / mean_carry if ne_live and mean_carry else 0.0
    print(f"  ratio:                 {ratio:.2f}")
    print(f"\n  A ratio near 0 means the recognition gate is sufficient and the 257 MB")
    print(f"  route classifier would be buying precision against a problem that does not")
    print(f"  occur. A ratio near or above 1 means no_evidence messages are producing as")
    print(f"  much spurious evidence as real messages produce real evidence, and the")
    print(f"  classifier is load-bearing rather than optional.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"experiment": "V2.53 span node precision without the route classifier",
         "bank": str(BANK.relative_to(ROOT)), "per_action": {k: dict(v) for k, v in per.items()},
         "no_evidence_attrs_per_message": round(ne["attrs"] / ne_live, 4) if ne_live else None,
         "value_action_attrs_per_message": round(mean_carry, 4),
         "ratio": round(ratio, 4)}, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
