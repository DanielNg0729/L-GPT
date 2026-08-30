"""Last-resort LLM rescue: re-read the conversation when the deterministic path stalls.

Everything else in this agent is deterministic and offline. This module is the single
exception, and it is off by default.

**When it runs.** Only from turn `llm_rescue_turn` (default 5) onward. If we are still
being asked at turn 5, the regular pipeline has not converged — on the clean public set
that happens in 3 of 200 conversations, so this is genuinely a rescue, not a workload.

**What it does.** Exactly one job: *re-read the shopper's own words*. It is handed the
session graph — every message the shopper sent, verbatim — and asked to return the
requirements as structured JSON. Those replace the intent our regex parser produced, and
retrieval runs again.

That narrow scope is deliberate. Query understanding is the one stage that actually
breaks when the shopper stops speaking in templates (measured: a reworded public set
costs ~0.05 even after the robustness fixes, and all of it is parsing). The stages an
LLM would be *worse* at are left alone:

* it never ranks — picking 10 rows out of 50,000 is set intersection, not language;
* it never chooses the question — `ask_attribute` is one value from a 10-item enum,
  picked by arithmetic;
* it never writes the scored output — `parent_asin` values come from the catalog.

**Failure is free.** Any exception, timeout, or unusable answer returns `None` and the
turn proceeds exactly as it would have. The scored path never depends on a model being
reachable, which matters because `submission_rules.md` warns that *"organizer policy may
disable network access"* during final scoring.
"""
from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = (
    "You extract shopping requirements for a product search engine. "
    "You are given a conversation with a shopper who is looking for one specific "
    "product in a clothing, shoes and jewellery catalogue.\n\n"
    "Return the concrete requirements they stated. Rules:\n"
    "1. Copy the shopper's own wording for each requirement, word for word. The search "
    "engine matches these against product listing text, so paraphrasing them breaks it.\n"
    "2. One requirement per distinct thing they asked for. Split combined sentences.\n"
    "3. Do not invent attributes they never mentioned, and do not add generic filler "
    "like 'good quality' or 'comfortable' unless they said it.\n"
    "4. If they said they do not care about something, put it in `exclude`, not "
    "`requirements`."
)


def _schema():
    """Pydantic model for the structured response. Imported lazily."""
    from pydantic import BaseModel, Field

    class RescuedIntent(BaseModel):
        """Requirements recovered from the conversation."""

        category: str = Field(default="", description="Product category the shopper named")
        requirements: list[str] = Field(
            default_factory=list,
            description="Exact phrases the shopper used to describe what they need",
        )
        color: str | None = Field(default=None, description="Colour, if stated")
        material: str | None = Field(default=None, description="Material or fabric, if stated")
        price: float | None = Field(default=None, description="Target price in dollars, if stated")
        exclude: list[str] = Field(
            default_factory=list, description="Things the shopper said they do not care about"
        )

    return RescuedIntent


def build_model(config: Any):
    """Construct the chat model with structured output bound. Never called if disabled."""
    provider = getattr(config, "llm_provider", "ollama")
    model_name = getattr(config, "llm_model", "gpt-oss:20b")
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model = ChatOllama(
            model=model_name,
            base_url=getattr(config, "llm_base_url", "http://localhost:11434"),
            temperature=0.0,
            num_predict=getattr(config, "llm_max_tokens", 3072),
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq

        # `max_tokens` has to be generous, and this is not a detail. gpt-oss is a
        # *reasoning* model: it thinks before it answers, and the thinking is billed
        # against the same budget as the structured output. At 512 tokens two thirds of
        # calls failed - either "model did not call a tool" (the entire budget went to
        # reasoning, nothing came back) or "failed to parse tool call arguments as JSON"
        # (the object was cut off mid-write). Low reasoning effort keeps it short too.
        model = ChatGroq(
            model=model_name,
            temperature=0.0,
            max_tokens=getattr(config, "llm_max_tokens", 3072),
            reasoning_effort="low",
        )
    else:
        raise ValueError("unknown llm_provider: %r" % provider)
    return model.with_structured_output(_schema(), include_raw=True)


def transcript(session_graph: dict) -> str:
    """The shopper's own words, in order — the whole point of the session graph."""
    lines = []
    for entry in session_graph.get("turns", []):
        lines.append("Turn %d, shopper: %s" % (entry["turn"], entry["user_message"]))
    shown = len(session_graph.get("product_nodes", {}))
    asked = session_graph.get("asked", [])
    if asked:
        lines.append("")
        lines.append("We already asked about: %s" % ", ".join(asked))
    exhausted = session_graph.get("exhausted_attributes", [])
    if exhausted:
        lines.append("They said they have nothing more to add on: %s" % ", ".join(exhausted))
    lines.append("We have shown %d products and none was right." % shown)
    return "\n".join(lines)


def rescue(session_graph: dict, intent: dict, config: Any, model: Any = None) -> dict | None:
    """Re-read the conversation. Returns a plain dict, or None if anything goes wrong.

    The returned dict is deliberately provider-agnostic so the harness can swap in an
    oracle or a stub to measure the ceiling of this approach without a model running.
    """
    custom = getattr(config, "rescue_fn", None)
    if custom is not None:
        try:
            return custom(session_graph, intent)
        except Exception:  # noqa: BLE001
            return None

    if model is None:
        return None
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", "Conversation so far:\n\n%s\n\nExtract the requirements."
                  % transcript(session_graph)),
    ]
    try:
        result = None
        for _ in range(2):      # one retry: a refused or truncated tool call is transient
            try:
                result = model.invoke(messages)
                break
            except Exception:  # noqa: BLE001
                result = None
        if result is None:
            return None
        parsed = result.get("parsed") if isinstance(result, dict) else result
        if parsed is None:
            return None
        raw = result.get("raw") if isinstance(result, dict) else None
        usage = {}
        if raw is not None and getattr(raw, "usage_metadata", None):
            usage = {
                "prompt_tokens": int(raw.usage_metadata.get("input_tokens", 0)),
                "completion_tokens": int(raw.usage_metadata.get("output_tokens", 0)),
            }
        payload = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
        payload["usage"] = usage
        return payload
    except Exception:  # noqa: BLE001 - a stalled model must never fail the turn
        return None


def apply(intent: dict, rescued: dict, turn: int) -> dict:
    """Fold a rescued extraction into the intent, without discarding what we had.

    Rescued requirements are added at full weight; anything the parser already found
    stays, because the regex path is exact when it fires and the model is not.
    """
    from .understanding import _add_constraint, _set_category

    if rescued.get("category") and not intent.get("category_terms"):
        _set_category(intent, str(rescued["category"]))
    for requirement in rescued.get("requirements") or []:
        text = str(requirement).strip()
        if text:
            _add_constraint(intent, text, turn, weight=1.0)
    colour = rescued.get("color")
    if colour and str(colour).lower() not in intent["facets"]["color"]:
        intent["facets"]["color"].append(str(colour).lower())
    material = rescued.get("material")
    if material and str(material).lower() not in intent["facets"]["material"]:
        intent["facets"]["material"].append(str(material).lower())
    if rescued.get("price") is not None:
        try:
            intent["facets"]["price"] = float(rescued["price"])
        except (TypeError, ValueError):
            pass
    intent["rescued_at_turn"] = turn
    return intent
