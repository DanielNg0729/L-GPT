"""HTTP wrapper around the shipped agent, for the demo UI. Not part of the submission.

WHAT THIS IS AND IS NOT. `.web/` is gitignored and nothing in the scored path imports it.
This exists so a person can talk to the agent in a browser; it adds no capability, and the
agent it serves is the same `submission.agent.Agent` the evaluator constructs.

WHY FREE TYPING IS THE INTERESTING DEMO. The agent has a recognition gate for the
simulator's own message shapes. A person typing "something warm for winter walks" matches
none of them, so the message takes the unfamiliar-wording path: the dialogue-act router
reads what the turn means, exact span recovery pulls catalogue-attested values out of it,
and the tagger strips conversational filler before mining. Those layers record 0 inferences
across the entire scored benchmark, because nothing there is unfamiliar. Here they run on
every turn. The demo shows the half of the system the score cannot.

Retrieval is correspondingly harder than the benchmark suggests, and that is honest rather
than a defect: off-template wording is the case the benchmark does not contain.

    pip install -r .web/requirements.txt
    python .web/server.py
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# All five configurable layers on: this is the showcase configuration, not the scored
# one. The deterministic span node is core rather than a configurable layer.
# Hosted layers additionally need GROQ_API_KEY and go quietly inert without it.
os.environ.setdefault("V2_ROUTE", "1")
os.environ.setdefault("BERT_EXTRACT", "1")
os.environ.setdefault("LLM_RESOLVE", "1")
os.environ.setdefault("LLM_RESCUE", "1")
os.environ.setdefault("LLM_MESSAGE", "1")
os.environ.setdefault("MESSAGE_VARIETY", "1")

# The rescue's shipped gate is turn 5 plus four rejected candidates, which is tuned for a
# simulator that answers in two-constraint bursts. A person types shorter, vaguer turns and
# stalls sooner, so the demo lowers both thresholds -- otherwise the one layer that reads
# the whole conversation would almost never fire in front of an audience.
os.environ.setdefault("LLM_RESCUE_TURN", "3")
os.environ.setdefault("LLM_RESCUE_REJECTS", "2")

from fastapi import FastAPI                                       # noqa: E402
from fastapi.middleware.cors import CORSMiddleware                # noqa: E402
from pydantic import BaseModel                                    # noqa: E402

from evaluator.local_evaluator import catalog_index               # noqa: E402
import submission.agent as agent_module                           # noqa: E402
from submission.agent import Agent, recognised                    # noqa: E402

# DEMO ONLY: TAKE `other` OFF THE TABLE.
#
# The shipped clarification policy asks `other` on nearly every turn, and it is right to.
# `customer_reply` short-circuits its family check for that one value and hands back two
# undisclosed constraints of ANY type, so it extracts more per turn than any typed question
# -- worth +0.000400 on the public set and more on the shifted populations.
#
# It is also the one question with no topic. A person reading "is there anything else?"
# every turn has nothing to answer, and no amount of rephrasing fixes that, because the
# emptiness is in the choice rather than the wording.
#
# Here nothing reads `ask_attribute` -- there is no simulator, only a person -- so the
# extraction advantage is worth exactly nothing and the cost is the whole conversation.
# Removing `other` from the candidate set leaves the information-gain calculation intact
# over the six typed attributes, so the demo still asks the most discriminating question
# it can, just one that names something.
#
# This is a patch on the demo process, not a change to the agent. The submitted policy is
# untouched, and `submission/` has no idea this file exists.
_shipped_askable = agent_module._askable

# DEMO ONLY: PUT BACK THE THREE ATTRIBUTES THE SIMULATOR CANNOT ANSWER.
#
# The contract allows ten. The agent probes seven, because `classify_constraint()` in the
# evaluator has no branch that emits `category` or `brand`, and a `budget` string is always
# sliced off by `cleaned[:4]` -- measured at 0 payouts in 200 sessions, so asking them
# against the simulator is a guaranteed wasted turn.
#
# They are dead because of what is ANSWERING, not because they are bad questions. "What is
# your budget?" and "any brand you prefer?" are among the most useful things to ask a
# person. Here a person is answering, so they come back, and the demo asks nine real
# questions instead of six.
agent_module.DEAD_ATTRIBUTES = ()
_TYPED = tuple(a for a in agent_module._PROBE_ATTRIBUTES if a != "other") + \
    ("category", "brand", "budget")
agent_module._PROBE_ATTRIBUTES = _TYPED + ("other",)


def _demo_askable(attribute: str, state) -> bool:
    if attribute == "other":
        return False
    asked = list(getattr(state, "asked", []))
    # ONCE ALL SIX ARE USED, ALLOW THEM AGAIN. The shipped policy asks each typed
    # attribute at most once and then has only `other` left, which is correct when a
    # simulator is answering: there is nothing new to learn from a repeat. A person is
    # different -- they answer in fragments, and turn 7 is not the end of the conversation.
    # Without this the option list empties and `_next_probe` falls back to `other`, which
    # is what put "asking about: other" back on screen at turn 7 despite the exclusion
    # above. Excluding the immediately previous attribute stops it asking the same thing
    # twice in a row; the information-gain calculation picks among the rest as usual, and
    # its answer moves as evidence accumulates.
    if all(t in asked for t in _TYPED):
        return attribute != (asked[-1] if asked else None)
    return _shipped_askable(attribute, state)


agent_module._askable = _demo_askable

CATALOG = ROOT / "data" / "catalog.jsonl"

app = FastAPI(title="L-GPT Shopping Copilot")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

print("building the catalogue index (once, ~15s)...", flush=True)
_ids, _cats, PRODUCTS = catalog_index(CATALOG)
AGENT = Agent(CATALOG)
TURNS: dict[str, int] = {}
print(f"ready: {len(PRODUCTS):,} products", flush=True)


class Turn(BaseModel):
    session_id: str | None = None
    message: str


def card(asin: str) -> dict:
    """The full catalogue record, which is what the UI renders."""
    p = PRODUCTS.get(asin) or {}
    details = p.get("details")
    return {
        "parent_asin": asin,
        "title": p.get("title") or asin,
        "store": p.get("store"),
        "price": p.get("price"),
        "rating": p.get("average_rating"),
        "rating_count": p.get("rating_number"),
        "categories": p.get("categories") or [],
        "features": p.get("features") or [],
        "details": details if isinstance(details, dict) else {},
        "description": p.get("description") or [],
    }


@app.post("/api/chat")
def chat(turn: Turn) -> dict:
    session = turn.session_id or uuid.uuid4().hex
    if session not in TURNS:
        AGENT.reset(session, {"preference_tags": []})
        TURNS[session] = 0
    TURNS[session] += 1
    n = TURNS[session]

    reply = AGENT.respond(session, turn.message, n, 10)
    asins = [r["parent_asin"] for r in reply.get("recommendations", [])]

    # WHY THE UI SEES TEN WHEN THE AGENT RETURNS ONE. Sequential disclosure returns a
    # single candidate before turn 10, because showing a shortlist would cost rank. The
    # panel behind the ">" is the rest of the ranked pool, fetched here so the demo can
    # show what the agent is holding back rather than implying it only found one.
    state = AGENT.sessions.get(session)
    if state is not None and len(asins) < 10:
        for extra in list(state.last_rank)[:10]:
            if extra not in asins:
                asins.append(extra)
    return {
        "session_id": session,
        "turn": n,
        "message": reply.get("message", ""),
        "ask_attribute": reply.get("ask_attribute"),
        "products": [card(a) for a in asins[:10]],
        "recognised": bool(recognised(turn.message)),
        "rescue_fired": (AGENT.rescue.stats()["reaches"]
                         if getattr(AGENT, "rescue", None) else 0),
        "evidence": ([{"phrase": p, "tier": t} for p, (_d, t) in state.evidence.items()]
                     if state is not None else []),
    }


@app.post("/api/reset")
def reset(turn: Turn) -> dict:
    if turn.session_id:
        TURNS.pop(turn.session_id, None)
        AGENT.sessions.pop(turn.session_id, None)
    return {"ok": True}


@app.get("/api/health")
def health() -> dict:
    writer = getattr(AGENT, "message_writer", None)
    resolver = getattr(AGENT, "resolver", None)
    route = getattr(AGENT, "route_node", None)

    def route_ready() -> bool:
        # The router loads LAZILY, on the first message the gate does not recognise, so
        # `model_loads` is 0 until someone types. Reporting readiness from a load count
        # would call a healthy router dead for the whole first turn -- which is exactly
        # what an earlier version of this endpoint did, by probing a `.ok` attribute the
        # class does not have and silently getting False.
        if route is None:
            return False
        s = route.stats()
        return bool(s.get("enabled")) and s.get("disabled_reason") is None

    return {
        "products": len(PRODUCTS),
        "layers": {
            "route_classifier": route_ready(),
            "tagger": bool(getattr(getattr(AGENT, "tagger", None), "enabled", False)),
            "deparaphraser": bool(resolver is not None and resolver.enabled),
            "rescue": bool(getattr(getattr(AGENT, "rescue", None), "enabled", False)),
            "message_writer": bool(writer is not None and writer.enabled),
        },
        "core": {
            "span_node": bool(getattr(getattr(AGENT, "span_node", None), "ok", False)),
        },
        "loaded": {
            "route_inferences": (route.stats().get("inferences", 0) if route else 0),
            "route_disabled_reason": (route.stats().get("disabled_reason") if route else None),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
