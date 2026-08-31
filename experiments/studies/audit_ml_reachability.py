"""Does any learned component actually FIRE on the two decision suites?

THE QUESTION. The agent's defence is a recognition gate: a message matching the simulator's
released shapes is handled by exact machinery, and the learned layers are never consulted.
"Never consulted" has been asserted from the gate's 463/463 match on clean traffic, but that
is a statement about MESSAGES. Two components are not governed by it:

  * the deparaphraser is consulted per VALUE, not per message. `intent_card()` truncates
    long feature bullets mid-word, so genuine catalogue prose can arrive as a clause the
    catalogue cannot attest -- inside a perfectly recognised message.
  * the span node and category matcher run wherever templates yielded no constraint.

So this audits reachability directly, per suite, per component.

NO NETWORK. The deparaphraser is DISABLED and its hook instrumented to count the calls it
WOULD have made. That is the reachability number, and it costs nothing. Whether those calls
change the score is a separate, already-measured question, reported alongside.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/studies/audit_ml_reachability.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Local models ON, every network layer OFF. The point is what the LOCAL stack does.
os.environ["V2_ROUTE"] = "1"
os.environ["BERT_EXTRACT"] = "1"
os.environ["LLM_RESOLVE"] = "0"
os.environ["LLM_EXTRACT"] = "0"
os.environ["LLM_RERANK"] = "0"

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import CAT, Agent, recognised  # noqa: E402

OUT = ROOT / "experiments" / "results" / "out_73_ml_reachability.json"
SUITES = (
    ("official200", ROOT / "data" / "public_set.jsonl"),
    ("unseen800 (catalog_review_distinct_800)",
     ROOT / "experiments" / "datasets" / "sets" / "catalog_review_distinct_800.jsonl"),
)


class AuditAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_messages = 0
        self.n_unrecognised = 0
        self.n_deparaphrase_reached = 0
        self.n_span_fired = 0
        self.n_route_called = 0

    def _observe(self, st, msg):
        self.n_messages += 1
        if not recognised(msg):
            self.n_unrecognised += 1
        return super()._observe(st, msg)

    def _deparaphrase(self, text):
        # Counted, never called: the layer is disabled, so this records reachability only.
        self.n_deparaphrase_reached += 1
        return super()._deparaphrase(text)

    def _route(self, msg, turn):
        self.n_route_called += 1
        return super()._route(msg, turn)


def main() -> None:
    ids, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    report = {}

    for name, path in SUITES:
        if not path.exists():
            print(f"{name}: MISSING {path}")
            continue
        samples = load_jsonl(path)
        agent = AuditAgent(ROOT / "data" / "catalog.jsonl")
        result = evaluate(agent, samples, ids, cats, prods)

        route = agent.route_node.stats() if getattr(agent, "route_node", None) else {}
        tag = agent.tagger.stats() if getattr(agent, "tagger", None) else {}
        span = getattr(agent, "span_node", None)
        res = agent.resolver.stats() if getattr(agent, "resolver", None) else {}

        row = {
            "sessions": len(samples),
            "score": result["recommended_technical_score"],
            "messages_seen": agent.n_messages,
            "messages_unrecognised": agent.n_unrecognised,
            "route_hook_entered": agent.n_route_called,
            "route_model_loads": int(route.get("model_loads", 0)),
            "route_inferences": int(route.get("inferences", 0)),
            "tagger_enabled": bool(tag.get("enabled", False)),
            "tagger_calls": int(tag.get("calls", 0)),
            "span_node_ok": bool(getattr(span, "ok", False)),
            "deparaphrase_reached": agent.n_deparaphrase_reached,
            "deparaphrase_enabled": bool(res.get("enabled", False)),
            "deparaphrase_calls": int(res.get("calls", 0)),
        }
        report[name] = row

        print(f"\n{name}   ({len(samples)} sessions)   score {row['score']:.6f}")
        print(f"  messages seen                 {row['messages_seen']}")
        print(f"  of those, UNRECOGNISED        {row['messages_unrecognised']}")
        print(f"  -- learned components --")
        print(f"  route classifier  loads       {row['route_model_loads']}")
        print(f"  route classifier  inferences  {row['route_inferences']}")
        print(f"  scaffolding tagger calls      {row['tagger_calls']}")
        print(f"  -- value-level layer (not governed by the message gate) --")
        print(f"  deparaphrase hook REACHED     {row['deparaphrase_reached']}")
        print(f"  deparaphrase actual calls     {row['deparaphrase_calls']} "
              f"(layer disabled for this audit)")

    print("\n" + "=" * 70)
    for name, row in report.items():
        fired = [k for k, v in (("route", row["route_inferences"]),
                                ("tagger", row["tagger_calls"])) if v]
        verdict = ", ".join(fired) if fired else "NONE"
        print(f"{name}")
        print(f"  learned models that fired: {verdict}")
        print(f"  deparaphrase reachable   : {row['deparaphrase_reached']} times")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "learned-component reachability on the two decision suites",
        "network": "disabled; deparaphrase hook counted, never called",
        "suites": report,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
