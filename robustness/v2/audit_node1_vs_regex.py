"""V2.55: Node 1's trained classifier against the lexical cues, on the same held-out bank.

WHY THIS RUN EXISTS
-------------------
Node 1 was declined on a measurement that turned out to cover only half its job. The
classifier does two things:

  ROUTING     which extraction a message can support. Measured unnecessary: the span node
              self-routes from message content (V2.53, spurious-evidence ratio 0.00).
  STATE       whether this turn is an intent OVERRIDE (clear the rejection set) or a
              NO-EVIDENCE turn (contribute nothing). Nothing in V1 recovers this on
              reworded traffic -- both literal patterns fire 0/1600.

An attempt to recover STATE with widened regexes was made and it half-failed. Its first
version scored 100% on the test bank and that number was FITTED: seven override cues and
eight no-preference cues were strings present only in the test set. Rebuilt from
train-attested vocabulary alone the honest held-out result is 37.5% override and 0.0%
no-evidence -- and even that 37.5% is measured within our own synthetic style, so it is an
upper bound on transfer to any real rewording.

This measures the classifier on exactly the same rows, so the comparison is like for like.

PROVENANCE, because it decides whether this is a fair test. The checkpoint was trained on
the train bank with full-wrapper-disjoint augmentation and previously scored 0.990938
masked accuracy on this same 9,600-row test split. Train and test share ZERO templates
(verified). So the classifier never saw these wrappers, exactly as the regexes never did.

WHAT WOULD JUSTIFY 257 MB. Not overall accuracy -- the interesting quantity is recall on
the two STATE actions the regexes cannot recover, and the false-positive rate on the one
that is dangerous. A false OVERRIDE merely forgets negatives, which is safe. A false
NO-EVIDENCE discards a real turn's evidence, which is not.

THE TURN MASK IS PART OF THE SYSTEM, not a favour to the model. `classify` restricts
openings to turn 1 and updates to later turns, which is a released runtime invariant
rather than a learned one. Each row is therefore evaluated at the turn its action implies,
which is what the runtime would supply.

Run:  PYTHONIOENCODING=utf-8 python -u robustness/v2/audit_node1_vs_regex.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "robustness" / "v2"
TEST = V2 / "v1_turn_gated_bank" / "final_test.jsonl"
OUT = V2 / "results" / "node1_vs_regex_v2_55.json"

OPENING_ACTIONS = {"buying_opening", "plain_opening", "override_opening"}


def main() -> None:
    os.environ["LLM_RERANK"] = "0"
    os.environ["LLM_EXTRACT"] = "0"
    os.environ["LLM_RESOLVE"] = "0"
    os.environ["BERT_EXTRACT"] = "0"
    from submission.agent import PAT_NOINFO, PAT_OVERRIDE_CUE, recognised
    from robustness.v2.route_node import StrictGatedRouteNode

    node = StrictGatedRouteNode()
    rows = [json.loads(l) for l in TEST.open(encoding="utf-8") if l.strip()]
    print(f"held-out bank: {len(rows)} rows")

    t0 = time.time()
    per = defaultdict(lambda: defaultdict(int))
    confusion = defaultdict(lambda: defaultdict(int))
    for r in rows:
        msg, truth = r["message"], r["action"]
        turn = 1 if truth in OPENING_ACTIONS else 2
        g = per[truth]
        g["n"] += 1
        g["recognised"] += bool(recognised(msg))
        g["rx_ovr"] += bool(PAT_OVERRIDE_CUE.search(msg))
        g["rx_ni"] += bool(PAT_NOINFO.search(msg))
        pred = node.classify(msg, turn)
        if pred is None:
            g["pred_none"] += 1
        else:
            confusion[truth][pred] += 1
            g["correct"] += int(pred == truth)
            g["nn_ovr"] += int(pred == "override_update")
            g["nn_ni"] += int(pred == "no_evidence")
    elapsed = time.time() - t0

    if node.disabled_reason:
        print(f"classifier unavailable: {node.disabled_reason}")
        print("cannot compare -- the 257 MB checkpoint is not on this machine.")
        return

    print(f"\n{'action':<20}{'rows':>6}{'accuracy':>10}"
          f"{'regex OVR':>11}{'NN OVR':>9}{'regex NI':>10}{'NN NI':>8}")
    print("-" * 74)
    for a in ("buying_opening", "plain_opening", "override_opening",
              "constraint_update", "override_update", "no_evidence"):
        g = per[a]
        if not g["n"]:
            continue
        print(f"{a:<20}{g['n']:>6}{g['correct']/g['n']:>10.4f}"
              f"{g['rx_ovr']:>11}{g['nn_ovr']:>9}{g['rx_ni']:>10}{g['nn_ni']:>8}")

    ov, ne = per["override_update"], per["no_evidence"]
    # False positives, on the actions where each signal must NOT fire.
    fp_rx_ovr = sum(per[a]["rx_ovr"] for a in per if a not in
                    ("override_update", "override_opening"))
    fp_nn_ovr = sum(per[a]["nn_ovr"] for a in per if a not in
                    ("override_update", "override_opening"))
    fp_rx_ni = sum(per[a]["rx_ni"] for a in per if a != "no_evidence")
    fp_nn_ni = sum(per[a]["nn_ni"] for a in per if a != "no_evidence")
    n_not_ovr = sum(per[a]["n"] for a in per if a not in
                    ("override_update", "override_opening"))
    n_not_ni = sum(per[a]["n"] for a in per if a != "no_evidence")

    print(f"\n  THE TWO STATE SIGNALS, recall on the held-out bank")
    print(f"  {'signal':<28}{'regex':>12}{'classifier':>13}")
    print("  " + "-" * 53)
    print(f"  {'override -> clear rejection':<28}"
          f"{ov['rx_ovr']/ov['n']:>11.1%}{ov['nn_ovr']/ov['n']:>13.1%}")
    print(f"  {'no-evidence -> skip turn':<28}"
          f"{ne['rx_ni']/ne['n']:>11.1%}{ne['nn_ni']/ne['n']:>13.1%}")
    print(f"\n  false positives (lower is better; the NO-EVIDENCE one is the dangerous one)")
    print(f"  {'override FP':<28}{fp_rx_ovr}/{n_not_ovr:<10}{fp_nn_ovr}/{n_not_ovr}")
    print(f"  {'no-evidence FP':<28}{fp_rx_ni}/{n_not_ni:<10}{fp_nn_ni}/{n_not_ni}")

    overall = sum(per[a]["correct"] for a in per) / max(sum(per[a]["n"] for a in per), 1)
    print(f"\n  overall six-way accuracy under the turn mask: {overall:.4f}")
    print(f"  {node.inferences} inferences in {elapsed:.0f}s "
          f"({1000*elapsed/max(node.inferences,1):.1f} ms each, device {node.stats()['device']})")

    print(f"\n  confusion (rows = truth):")
    labels = ("buying_opening", "plain_opening", "override_opening",
              "constraint_update", "override_update", "no_evidence")
    print("  " + " " * 20 + "".join(f"{l[:9]:>11}" for l in labels))
    for a in labels:
        print(f"  {a:<20}" + "".join(f"{confusion[a][b]:>11}" for b in labels))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "V2.55 Node 1 classifier vs lexical cues, held-out bank",
        "rows": len(rows), "overall_accuracy": round(overall, 6),
        "state_recall": {
            "override": {"regex": round(ov["rx_ovr"] / ov["n"], 4),
                         "classifier": round(ov["nn_ovr"] / ov["n"], 4)},
            "no_evidence": {"regex": round(ne["rx_ni"] / ne["n"], 4),
                            "classifier": round(ne["nn_ni"] / ne["n"], 4)}},
        "false_positives": {"override": {"regex": fp_rx_ovr, "classifier": fp_nn_ovr,
                                         "of": n_not_ovr},
                            "no_evidence": {"regex": fp_rx_ni, "classifier": fp_nn_ni,
                                            "of": n_not_ni}},
        "per_action": {k: dict(v) for k, v in per.items()},
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "ms_per_inference": round(1000 * elapsed / max(node.inferences, 1), 2),
        "node_stats": node.stats(),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
