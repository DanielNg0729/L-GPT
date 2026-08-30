"""`Agent` — the object the official evaluator constructs and calls.

Implements `docs/agent_api_contract.json` exactly: `reset(session_id, user_profile)`
followed by `respond(session_id, user_message, turn, top_k)` returning `message`,
`ask_attribute`, `recommendations` and `usage`.

The agent itself is deliberately thin. It owns the Knowledge Graph (built once, shared
by every session) and delegates each turn to the LangGraph pipeline, which owns the
conversation memory. `respond()` never raises: the evaluator counts an exception as a
miss, so every failure degrades to a popularity-ranked slate instead.
"""
from __future__ import annotations

import time
from pathlib import Path

from .config import DEFAULT_CATALOG, MAX_TURNS, CopilotConfig
from .graph import build_graph, turn_config
from .knowledge_graph import KnowledgeGraph
from .session_graph import dump as dump_session_graph


class Agent:
    """Conversational shopping copilot over the frozen Amazon catalog."""

    def __init__(
        self,
        catalog_path: str | Path = DEFAULT_CATALOG,
        config: CopilotConfig | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
    ) -> None:
        self.config = config or CopilotConfig()
        if catalog_path is not None:
            self.config = _with_catalog(self.config, Path(catalog_path))
        started = time.perf_counter()
        # `knowledge_graph` lets an ablation sweep build the 50k-row index once and
        # share it across configurations. The evaluator never passes it.
        self.kg = knowledge_graph or KnowledgeGraph(self.config.catalog_path)
        if self.config.enable_lsa:
            self.kg.enable_lsa(self.config.lsa_components)
        self.init_seconds = time.perf_counter() - started

        self._scratch: dict = {}
        self.graph = build_graph(self.kg, self.config, self._scratch)
        self._sessions: dict[str, dict] = {}

    # ------------------------------------------------------------------ API

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Start a new conversation. The profile is anonymised aggregate data."""
        self._sessions[session_id] = dict(user_profile or {})
        self._scratch.pop(session_id, None)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        # Hard guard: the harness zeroes a session that runs past turn 10.
        turn = max(1, min(int(turn), MAX_TURNS))
        top_k = max(1, int(top_k))

        try:
            state = self.graph.invoke(
                {
                    "session_id": session_id,
                    "user_profile": self._sessions[session_id],
                    "turn": turn,
                    "top_k": top_k,
                    "user_message": str(user_message or ""),
                },
                config=turn_config(session_id),
            )
            recommendations = [
                {"parent_asin": rec["parent_asin"]} for rec in state.get("recommendations", [])
            ][:top_k]
            if self.config.session_graph_dir and state.get("session_graph"):
                dump_session_graph(state["session_graph"], self.config.session_graph_dir)
            return {
                "message": state.get("message") or "Here are the closest matches I found.",
                "ask_attribute": state.get("ask_attribute"),
                "recommendations": recommendations,
                "usage": state.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0},
            }
        except Exception:  # noqa: BLE001 - a raise is scored as a miss, so never raise
            return self._fallback(top_k)

    # ------------------------------------------------------------- internals

    def _fallback(self, top_k: int) -> dict:
        """Popularity-ranked slate. Better than an empty list, which scores nothing."""
        import numpy as np

        order = np.argsort(-self.kg.popularity)[:top_k]
        return {
            "message": "Here are some popular options while I narrow this down.",
            "ask_attribute": "other",
            "recommendations": [{"parent_asin": self.kg.asins[int(i)]} for i in order],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    # --------------------------------------------------------------- helpers

    def session_state(self, session_id: str) -> dict:
        """Inspect a live conversation — the LangGraph checkpoint for this thread."""
        snapshot = self.graph.get_state(turn_config(session_id))
        return dict(snapshot.values) if snapshot and snapshot.values else {}


def _with_catalog(config: CopilotConfig, catalog_path: Path) -> CopilotConfig:
    import dataclasses

    return dataclasses.replace(config, catalog_path=catalog_path)
