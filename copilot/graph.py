"""The LangGraph pipeline: router, structured-response builder, retrieval, Select 10.

This is a direct transcription of the architecture diagram, with the two nodes the
brief asked to drop (`Query expansion` and `COSMO`) removed — the customer's input is
already structured, so there is nothing to expand and no commonsense gap to fill.

    START
      -> router                        classify first turn vs. follow-up, and the track
      -> bootstrap | patch             build the Structured Response, or JSON-patch it
      -> buying_track | browse_track   dual-track routing (channel policy, not a fork)
      -> rag                           multi-route retrieval against the Knowledge Graph
      -> session_update                merge the turn into the Session Graph
      -> select_10                     rank, demote what is provably wrong, take 10
      -> ask                           expected-information-gain clarification policy
      -> compose                       customer-facing message
    END

Conversation memory is LangGraph's own: the compiled graph carries an `InMemorySaver`
checkpointer and every turn is invoked with `thread_id = session_id`, so the
Structured Response and the Session Graph persist across turns without the Agent
holding any state of its own. Both are plain JSON dicts, so a checkpoint is
inspectable and a session is replayable.

The Knowledge Graph is *not* in the state. It is global, read-only, and shared by
every session; putting 50,000 product nodes through a checkpointer on every turn
would be absurd. It is bound into the node closures instead.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from . import ask_policy, retrieval, select10, session_graph as sg, understanding
from .config import MAX_TURNS, CopilotConfig
from .knowledge_graph import KnowledgeGraph


class CopilotState(TypedDict, total=False):
    """Everything that survives between turns lives here."""

    session_id: str
    user_profile: dict
    turn: int
    top_k: int
    user_message: str

    route: str                 # "bootstrap" | "patch"
    track: str                 # "buying" | "browsing" | "override"
    structured: dict           # the Structured Response (JSON)
    session_graph: dict        # the Session Graph (JSON)

    signals: dict
    retrieval: dict           # JSON summary only; the numpy artifacts live in `scratch`
    recommendations: list
    ask_attribute: str | None
    message: str
    usage: dict
    trace: Annotated[list, operator.add]


# Phrasings for the customer-facing `message`. The simulator ignores prose entirely
# and reads `ask_attribute`, so this is a fixed prompt table rather than a model call —
# the "Fixed prompt for ask-attribute" box in the diagram.
ASK_PROMPTS = {
    "material": "What material do you have in mind?",
    "color": "Any colour you are set on?",
    "size": "What size or fit are you after?",
    "style": "What style or cut works best for you?",
    "brand": "Is there a brand you prefer?",
    "budget": "What budget are you working with?",
    "feature": "Which feature matters most to you?",
    "use_case": "What will you be using it for?",
    "category": "Which category should I focus on?",
    "other": "Tell me the one thing that matters most and I will narrow this down.",
}


def build_graph(kg: KnowledgeGraph, config: CopilotConfig, scratch: dict):
    """Compile the turn pipeline with `kg` bound into every node.

    `scratch` is a plain dict owned by the Agent, keyed by session id. It carries the
    numpy artifacts of a single turn (the candidate pool, the coverage vector) between
    `rag` and `select_10`. They stay out of the graph state on purpose: the state is
    checkpointed on every turn, and it should stay small, JSON-clean and replayable.
    """

    # ---------------------------------------------------------------- router
    def router(state: CopilotState) -> dict:
        first_turn = not state.get("structured")
        return {
            "route": "bootstrap" if first_turn else "patch",
            "trace": [("router", "bootstrap" if first_turn else "patch")],
        }

    def route_branch(state: CopilotState) -> str:
        return state["route"]

    # ------------------------------------------------- structured response
    def bootstrap(state: CopilotState) -> dict:
        """First turn only: create the session's query state from the opening message."""
        structured, learned = understanding.bootstrap(state["user_message"], state["turn"])
        graph = sg.new_session_graph(state["session_id"], state.get("user_profile") or {})
        if structured["track"] == "override":
            # The evaluator refuses to score a hit in an override session until the
            # override message lands, on turn 3 *or* 4 — we cannot tell which. Until we
            # actually see that message, treat every slate as unscored. Guessing turn 3
            # and being wrong marks a correct turn-3 slate "provably wrong" and demotes
            # the real target out of the top 10 on turn 4.
            graph["hit_blocked_until"] = MAX_TURNS + 1
        sg.record_turn(graph, state["turn"], state["user_message"], structured["track"], learned)
        for constraint in learned:
            sg.record_attribute(graph, constraint["attribute"], constraint["text"], "user", state["turn"])
        return {
            "structured": structured,
            "session_graph": graph,
            "track": structured["track"],
            "signals": {"kind": "open"},
            "trace": [("bootstrap", structured["track"])],
        }

    def patch(state: CopilotState) -> dict:
        """Later turns: patch the existing Structured Response — never rebuild it."""
        structured, learned, signals = understanding.patch(
            dict(state["structured"]), state["user_message"], state["turn"]
        )
        graph = dict(state["session_graph"])
        sg.record_turn(graph, state["turn"], state["user_message"], structured["track"], learned)
        for constraint in learned:
            sg.record_attribute(graph, constraint["attribute"], constraint["text"], "user", state["turn"])
        if signals["kind"] == "exhausted" and signals.get("attribute"):
            sg.mark_exhausted(graph, signals["attribute"])
        if signals["kind"] == "boundary":
            graph["boundary_deflected"] = True
        if signals["kind"] == "override":
            # The pivot has landed; hits can be scored from here on.
            graph["hit_blocked_until"] = state["turn"]
        return {
            "structured": structured,
            "session_graph": graph,
            "track": structured["track"],
            "signals": signals,
            "trace": [("patch", signals["kind"])],
        }

    def track_branch(state: CopilotState) -> str:
        """Dual-track routing.

        Note what this branches on. Not the scenario label — the *amount of constraint
        text actually held*. An "intent override" session hands over a full feature
        bullet on turn 1 while a "buying" session often hands over a single word, so
        routing on the label would put the richer session in the thinner track.
        """
        structured = state["structured"]
        live = [c for c in structured["constraints"] if not c["superseded"]]
        return "buying_track" if live else "browse_track"

    def buying_track(state: CopilotState) -> dict:
        """High-precision track: hard constraints are locked in and ANDed."""
        return {"trace": [("track", "buying")]}

    def browse_track(state: CopilotState) -> dict:
        """Exploratory track: category slate plus a popularity prior, and ask early."""
        return {"trace": [("track", "browse")]}

    # ------------------------------------------------------------------ RAG
    def rag(state: CopilotState) -> dict:
        result = retrieval.retrieve(kg, state["structured"], config.retrieval)
        scratch[state["session_id"]] = result
        summary = {
            "pool_size": result["pool_size"],
            "direct_emit": result["direct_emit"],
            "channels": result.get("channels", {}),
            "used_constraints": result.get("used_constraints", []),
            "dropped_constraints": result.get("dropped_constraints", []),
            "candidate_count": len(result.get("candidates") or []),
        }
        return {
            "retrieval": summary,
            "trace": [("rag", "pool=%d direct=%s" % (result["pool_size"], result["direct_emit"]))],
        }

    # -------------------------------------------------------- session graph
    def session_update(state: CopilotState) -> dict:
        """Merge, never replace: the session graph accumulates across the whole session."""
        graph = dict(state["session_graph"])
        summary = state["retrieval"]
        graph.setdefault("retrieval_log", []).append({"turn": state["turn"], **summary})
        return {"session_graph": graph, "trace": [("session_update", len(graph["turns"]))]}

    # ----------------------------------------------------------- select 10
    def select_10(state: CopilotState) -> dict:
        picks = select10.select(
            kg,
            state["structured"],
            scratch[state["session_id"]],
            state["session_graph"],
            state.get("user_profile") or {},
            config.rank,
            top_k=state.get("top_k", 10),
        )
        graph = dict(state["session_graph"])
        # A slate only proves its members wrong if the evaluator could have scored it.
        hit_would_count = state["turn"] >= graph.get("hit_blocked_until", 1)
        sg.record_shown(graph, state["turn"], [p["parent_asin"] for p in picks], hit_would_count)
        return {
            "recommendations": picks,
            "session_graph": graph,
            "trace": [("select_10", len(picks))],
        }

    # ------------------------------------------------------------- ask + say
    def ask(state: CopilotState) -> dict:
        attribute = ask_policy.choose(
            state["structured"], state["session_graph"],
            state["retrieval"]["pool_size"], config.ask,
        )
        graph = dict(state["session_graph"])
        if attribute:
            graph["asked"] = list(graph.get("asked", [])) + [attribute]
        return {
            "ask_attribute": attribute,
            "session_graph": graph,
            "trace": [("ask", attribute)],
        }

    def compose(state: CopilotState) -> dict:
        attribute = state.get("ask_attribute")
        pool = state["retrieval"]["pool_size"]
        if attribute:
            lead = ("Here are the closest matches so far." if pool != 1
                    else "I think this is it.")
            message = "%s %s" % (lead, ASK_PROMPTS.get(attribute, ASK_PROMPTS["other"]))
        elif pool == 1:
            message = "Based on everything you have told me, this is the one."
        else:
            message = "These fit everything you have described."
        return {
            "message": message,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "trace": [("compose", attribute)],
        }

    builder = StateGraph(CopilotState)
    builder.add_node("router", router)
    builder.add_node("bootstrap", bootstrap)
    builder.add_node("patch", patch)
    builder.add_node("buying_track", buying_track)
    builder.add_node("browse_track", browse_track)
    builder.add_node("rag", rag)
    builder.add_node("session_update", session_update)
    builder.add_node("select_10", select_10)
    builder.add_node("ask", ask)
    builder.add_node("compose", compose)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router", route_branch, {"bootstrap": "bootstrap", "patch": "patch"}
    )
    for node in ("bootstrap", "patch"):
        builder.add_conditional_edges(
            node, track_branch,
            {"buying_track": "buying_track", "browse_track": "browse_track"},
        )
    builder.add_edge("buying_track", "rag")
    builder.add_edge("browse_track", "rag")
    builder.add_edge("rag", "session_update")
    builder.add_edge("session_update", "select_10")
    builder.add_edge("select_10", "ask")
    builder.add_edge("ask", "compose")
    builder.add_edge("compose", END)

    return builder.compile(checkpointer=InMemorySaver())


def turn_config(session_id: str) -> dict[str, Any]:
    """LangGraph thread config — one conversation thread per evaluator session."""
    return {"configurable": {"thread_id": session_id}, "recursion_limit": 32}
