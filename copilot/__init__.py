"""Shopping Copilot — TikTok TechJam 2026, Track 4.

A conversational retrieval agent over a frozen 50,000-row Amazon clothing catalog.

    from copilot import Agent
    agent = Agent("provided/techjam-conversational-search/data/catalog.jsonl")
    agent.reset("s1", user_profile)
    agent.respond("s1", "I'm looking for Jewelry Necklaces, but I'm still exploring.", 1, 10)
"""
from .agent import Agent
from .config import CopilotConfig, RankConfig, RetrievalConfig, AskConfig
from .knowledge_graph import KnowledgeGraph

__all__ = [
    "Agent",
    "CopilotConfig",
    "RetrievalConfig",
    "RankConfig",
    "AskConfig",
    "KnowledgeGraph",
]
