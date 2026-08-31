"""Experiment 41: can we PROVE an LLM extraction channel never fires on clean messages?

The requirement for a third extraction channel is that it must not degrade the agent when
the organizer adds no paraphrasing at all. A confidence threshold cannot promise that --
any scorer fires sometimes, and "sometimes" on a path that already scores 0.96960 with
HR 99.5% is pure downside.

A RECOGNITION GATE can promise it structurally. The simulator emits a closed set of message
shapes, all of them literal format strings in `local_evaluator.py`:

    initial_message()   "I'm looking for {c}. A key requirement is: {v}."
                        "I'm looking for {c}, but I'm still exploring."
                        "I'm looking for {c}. {old_value}"
    customer_reply()    "For that, what matters is: {a}; {b}."
                        "I don't have an additional preference for {attr}."
                        "I don't have a preference for {attr}; please use your judgment."
                        "Those options are not quite right yet. Ask me about one specific attribute."
    behavior_for()      "Actually, ignore my earlier preference. What I need is: {v}."

Anchor a regex to each FULL message. Then:

    message matches a known shape  -> clean; run today's path only, never call the LLM
    message matches nothing        -> paraphrased or novel; the LLM channel is allowed

If every message in a clean run matches, the LLM is unreachable at zero paraphrase and
"never degrades" stops being a hope and becomes a property of the control flow.

This pass measures exactly that, on both clean and paraphrased traffic:
  * clean coverage   -- what fraction of real clean messages match a known shape?
                        Anything below 100% is a hole the LLM would fire into.
  * paraphrase recall -- what fraction of PARAPHRASED messages correctly fail to match?
                        This is the gate's sensitivity: messages it lets through to the LLM.

The unmatched rate is also a free, label-free paraphrase DETECTOR: if the organizer ships
clean templates we will observe ~0% unmatched, and if they paraphrase we will see it
immediately -- the same trick as the population detector in `_w_pop_effective()`.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/log/41_recognition_gate.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "log"))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS, TOP_K, catalog_index, coarse_category, customer_reply, initial_message,
    load_jsonl, materialize_hidden_fields, normalize_recommendations,
)
from submission.agent import Agent  # noqa: E402

_p31 = __import__("31_paraphrase_stress")
TRANSFORMS = _p31.TRANSFORMS

# Anchored to the WHOLE message. Order matters only for reporting which shape hit.
KNOWN_SHAPES = [
    ("buying_open",   re.compile(r"^I'm looking for .+\. A key requirement is: .+\.$")),
    ("browsing_open", re.compile(r"^I'm looking for .+, but I'm still exploring\.$")),
    ("reply_matters", re.compile(r"^For that, what matters is: .+\.$")),
    ("reply_none",    re.compile(r"^I don't have an additional preference for [a-z_]+\.$")),
    ("boundary",      re.compile(r"^I don't have a preference for [a-z_]+; please use your judgment\.$")),
    ("nudge",         re.compile(r"^Those options are not quite right yet\. Ask me about one specific attribute\.$")),
    ("override",      re.compile(r"^Actually, ignore my earlier preference\. What I need is: .+\.$")),
    ("override_dflt", re.compile(r"^Actually, please ignore my earlier preference\.$")),
    # The intent_override opening is "I'm looking for {cat}. {old_value}" where old_value is
    # arbitrary catalogue text, so it must be matched last and loosely.
    ("override_open", re.compile(r"^I'm looking for .+\. .+$")),
]


def recognised(msg: str) -> str | None:
    for name, pat in KNOWN_SHAPES:
        if pat.match(msg.strip()):
            return name
    return None


def collect_messages(agent, samples, cid, cats, prods, transform):
    """Replay sessions and return every message the AGENT actually sees."""
    import uuid
    seen = []
    for sample in samples:
        session_id = f"gate_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, prods)
        eff = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        bu = False
        applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(eff, coarse_category(cats.get(target, [])), disclosed)
        for turn in range(1, MAX_TURNS + 1):
            shown = transform(user_message)
            seen.append(shown)
            try:
                resp = agent.respond(session_id, shown, turn, TOP_K)
            except Exception:
                resp = {"message": "", "ask_attribute": None, "recommendations": []}
            ranked = normalize_recommendations(resp.get("recommendations"), cid)
            if applied and target in ranked:
                break
            if turn == MAX_TURNS:
                break
            ov = eff.get("behavior", {}).get("override") or {}
            if not applied and turn + 1 == int(ov.get("turn", 3)):
                applied = True
                nv = str(ov.get("new_value", ""))
                if nv:
                    disclosed.add(nv)
                user_message = str(ov.get("message", "Actually, please ignore my earlier preference."))
            else:
                user_message, bu = customer_reply(eff, resp.get("ask_attribute"), disclosed, bu)
    return seen


def main() -> None:
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")

    def share():
        o = object.__new__(Agent)
        o.ix, o.sessions, o.llm = base.ix, {}, None
        return o

    OUT = {}
    print(f"{'condition':<34}{'messages':>10}{'recognised':>12}{'unmatched':>11}"
          f"{'-> LLM':>9}")
    print("-" * 76)
    for tname in ("T0 identity (control)", "T1 scaffold reworded", "T2 scaffold stripped",
                  "T3 conversational noise", "T4 case/punctuation churn",
                  "T5 realistic (T1+T3)"):
        msgs = collect_messages(share(), samples, cid, cats, prods, TRANSFORMS[tname])
        hits = [recognised(m) for m in msgs]
        n = len(msgs)
        rec = sum(1 for h in hits if h)
        OUT[tname] = {"n": n, "recognised": rec, "unmatched": n - rec,
                      "shapes": dict(Counter(h for h in hits if h))}
        print(f"{tname:<34}{n:>10}{rec/n:>11.1%}{(n-rec)/n:>11.1%}{n-rec:>9}")

    clean = OUT["T0 identity (control)"]
    print(f"\n  CLEAN COVERAGE: {clean['recognised']}/{clean['n']} "
          f"= {clean['recognised']/clean['n']:.2%}")
    if clean["unmatched"] == 0:
        print("  -> the gate is AIRTIGHT: at zero paraphrase no message reaches the LLM,")
        print("     so a third channel behind this gate cannot change the clean score.")
    else:
        print(f"  -> {clean['unmatched']} clean messages would leak to the LLM. Not airtight;")
        print("     tighten the shapes before trusting the guarantee.")
    print(f"  shape histogram (clean): {clean['shapes']}")

    print("\n  gate sensitivity -- share of PARAPHRASED messages routed to the LLM:")
    for t in OUT:
        if t.startswith("T0"):
            continue
        o = OUT[t]
        print(f"    {t:<32}{o['unmatched']/o['n']:>7.1%}")

    print("\n  the unmatched rate doubles as a free paraphrase detector: ~0% means the")
    print("  organizer shipped clean templates, anything higher says they did not.")

    (ROOT / "experiments" / "results" / "out_41.json").write_text(
        json.dumps(OUT, indent=2) + "\n", encoding="utf-8")
    print("\n[saved] experiments/results/out_41.json")


if __name__ == "__main__":
    main()
