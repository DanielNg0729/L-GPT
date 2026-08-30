"""The per-session graph: written every turn, dies with the session, plain JSON.

Unlike the knowledge graph this object has no gate — whatever the user says is
ground truth here. It is the only thing `Select 10` consults to avoid re-serving a
list it has already served.

One rule in here is subtle enough to be worth stating loudly:

    A product that was shown on an earlier turn is *not* automatically known to be
    wrong. The evaluator suppresses hits in an intent-override session until the
    override turn arrives, so a product shown at turn 1 of such a session may still
    be the target. Only turns where a hit *would have counted* license the demotion.

That is why `record_shown()` takes `hit_would_count` and why the exclusion is a
demotion (`demoted_asins`) rather than a deletion.
"""
from __future__ import annotations

import json
from pathlib import Path


def new_session_graph(session_id: str, user_profile: dict) -> dict:
    """A fresh, JSON-serialisable session graph."""
    return {
        "kind": "session_graph",
        "session_id": session_id,
        "user_profile": dict(user_profile or {}),
        "turns": [],            # one entry per respond() call
        "product_nodes": {},    # parent_asin -> {"shown_at": [...], "best_rank": int, ...}
        "attribute_nodes": {},  # "material:cotton" -> {"source": ..., "turn": ...}
        "edges": [],            # {"type": ..., "from": ..., "to": ..., "turn": ...}
        "exhausted_attributes": [],
        "asked": [],            # ask_attribute values already spent
        "boundary_deflected": False,
        "hit_blocked_until": 1,  # first turn on which a hit can be scored
    }


def record_turn(graph: dict, turn: int, user_message: str, opening_type: str,
                constraints_added: list[dict]) -> None:
    graph["turns"].append({
        "turn": turn,
        "user_message": user_message,
        "opening_type": opening_type,
        "constraints_added": [c["text"] for c in constraints_added],
    })


def record_attribute(graph: dict, key: str, value: str, source: str, turn: int) -> None:
    """Attribute node tagged with provenance: `user`, `catalog`, or `profile`."""
    node_id = "%s:%s" % (key, value)
    node = graph["attribute_nodes"].setdefault(
        node_id, {"key": key, "value": value, "source": source, "turn": turn, "count": 0}
    )
    node["count"] += 1
    node["last_turn"] = turn


def record_shown(graph: dict, turn: int, asins: list[str], hit_would_count: bool) -> None:
    """Log the slate served on this turn.

    `hit_would_count` is False exactly when the evaluator cannot score a hit yet
    (an intent-override session before its override turn). Slates served under that
    condition teach us nothing and must not demote their members.
    """
    for rank, asin in enumerate(asins, start=1):
        node = graph["product_nodes"].setdefault(
            asin, {"shown_at": [], "best_rank": rank, "provably_wrong": False}
        )
        node["shown_at"].append({"turn": turn, "rank": rank, "scored": hit_would_count})
        node["best_rank"] = min(node["best_rank"], rank)
        if hit_would_count:
            # It was scored and the session continued, so it is not the target.
            node["provably_wrong"] = True
        graph["edges"].append(
            {"type": "shown_at_turn", "from": "session", "to": asin, "turn": turn, "rank": rank}
        )


def record_rejection(graph: dict, asin: str, turn: int, reason: str) -> None:
    """Marked, never deleted — deleting it means the next retrieval returns it again."""
    node = graph["product_nodes"].setdefault(
        asin, {"shown_at": [], "best_rank": 999, "provably_wrong": False}
    )
    node["rejected"] = True
    graph["edges"].append(
        {"type": "rejected_by_user", "from": "user", "to": asin, "turn": turn, "reason": reason}
    )


def record_preference(graph: dict, winner: str, loser: str, turn: int) -> None:
    """`preferred_over` edge — what makes 'the cheaper one' resolvable next turn."""
    graph["edges"].append(
        {"type": "preferred_over", "from": winner, "to": loser, "turn": turn}
    )


def mark_exhausted(graph: dict, attribute: str) -> None:
    if attribute and attribute not in graph["exhausted_attributes"]:
        graph["exhausted_attributes"].append(attribute)


def demoted_asins(graph: dict) -> set[str]:
    """Products we can prove are not the target, plus anything the user rejected."""
    return {
        asin for asin, node in graph["product_nodes"].items()
        if node.get("provably_wrong") or node.get("rejected")
    }


def already_asked(graph: dict) -> list[str]:
    return list(graph["asked"])


def dump(graph: dict, directory: str | Path) -> Path:
    path = Path(directory) / ("%s.json" % graph["session_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
