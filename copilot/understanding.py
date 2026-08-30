"""Query understanding: build and patch the session's Structured Response.

This is the box the architecture diagram calls *Structured response*, and it is the
only thing downstream stages read — never the raw user text.

The original design put an LLM *Query expansion* step and a *COSMO* commonsense
enrichment step in front of it. Both are removed here, deliberately:

  * the customer's utterances are already intent — they are generated from a
    small set of fixed templates and their payload is verbatim catalog text
    (measured: 760/760 revealed constraints are punctuation-insensitive substrings
    of the target row), so there is nothing for an expander to recover; and
  * widening a query that already quotes the answer strictly *hurts* — the starter
    agent's `" OR ".join(terms)` is exactly that mistake, and it scores 0.125.

What replaces them is a deterministic parser plus a span-extraction fallback, so a
paraphrased private set degrades to BM25-over-extracted-spans instead of failing.
"""
from __future__ import annotations

import re

from .text import COLOR_RE, MATERIAL_RE, normalize_phrase, tokens, unique

# --- the simulator's fixed templates -------------------------------------------------
RE_OPENING = re.compile(r"^\s*i'm looking for (?P<category>.+?)(?P<tail>[.,]\s*(?:but i'm still exploring\.?|.*))?\s*$", re.I | re.S)
RE_BUYING = re.compile(r"^\s*i'm looking for (?P<category>.+?)\.\s*a key requirement is:\s*(?P<constraint>.+?)\.?\s*$", re.I | re.S)
RE_BROWSING = re.compile(r"^\s*i'm looking for (?P<category>.+?),\s*but i'm still exploring\.?\s*$", re.I | re.S)
RE_OVERRIDE_OPEN = re.compile(r"^\s*i'm looking for (?P<category>.+?)\.\s*(?P<constraint>.+?)\s*$", re.I | re.S)

RE_REVEAL = re.compile(r"^\s*for that,?\s*what matters is:\s*(?P<body>.+?)\.?\s*$", re.I | re.S)
RE_NO_MORE = re.compile(r"^\s*i don't have an additional preference for\s*(?P<attr>[a-z_]+)\.?\s*$", re.I)
RE_BOUNDARY = re.compile(r"^\s*i don't have a preference for\s*(?P<attr>[a-z_]+);\s*please use your judgment\.?\s*$", re.I)
RE_NUDGE = re.compile(r"not quite right yet\.\s*ask me about one specific attribute", re.I)
RE_OVERRIDE = re.compile(r"^\s*actually,?\s*ignore my earlier preference\.\s*what i need is:\s*(?P<value>.+?)\.?\s*$", re.I | re.S)

# A reworded change of mind still has to signal itself somehow. These are the cues, and
# the marker that introduces the replacement. Matching on cues rather than on one exact
# sentence is what keeps override handling alive if the private set is paraphrased -
# and mis-detecting an override is expensive, see `hit_blocked_until` in graph.py.
RE_OVERRIDE_CUE = re.compile(
    r"(actually|forget what|forget everything|scratch that|never ?mind|"
    r"changed my mind|change of plan|ignore (?:my|what|the))", re.I)
RE_NEW_VALUE = re.compile(
    r"(?:what i (?:really )?need is|i (?:really )?need|the thing i need is|"
    r"it has to be|it needs to be|what i want is|i want)\s*:?\s*(?P<value>.+?)\.?\s*$",
    re.I | re.S)

# --- synthesised constraints the simulator writes rather than quotes -----------------
RE_COLOR_C = re.compile(r"^\s*color:\s*(?P<value>.+?)\s*$", re.I)
RE_BUDGET_C = re.compile(r"^\s*budget around \$\s*(?P<value>[0-9][0-9.,]*)\s*$", re.I)

_SIZE_WORDS = ("size", "sizing", "width", "wide", "narrow", "length", "inseam")
_STYLE_WORDS = ("department", "style", "fit", "sleeve", "neck", "collar", "closure")
_USE_WORDS = ("hiking", "running", "gym", "winter", "outdoor", "work", "travel", "wedding")


def classify(value: str) -> str:
    """Constraint -> `ask_attribute` bucket.

    Mirrors the simulator's own `classify_constraint`, because the simulator uses it
    to decide whether a typed ask matches an undisclosed constraint. Getting this
    wrong makes typed asks whiff.
    """
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if MATERIAL_RE.search(lowered):
        return "material"
    if "color" in lowered or COLOR_RE.search(lowered):
        return "color"
    if any(word in lowered for word in _SIZE_WORDS):
        return "size"
    if any(word in lowered for word in _STYLE_WORDS):
        return "style"
    if any(word in lowered for word in _USE_WORDS):
        return "use_case"
    return "feature"


def new_intent(opening_type: str = "browsing") -> dict:
    """The empty intent object, created once per conversation."""
    return {
        "kind": "shopper_intent",
        # What the opening line looked like: buying / browsing / override. Recorded for
        # the session graph only - the search mode is decided separately, every turn.
        "opening_type": opening_type,
        "category_raw": "",
        "category_terms": [],
        "constraints": [],       # [{text, norm, attribute, weight, turn, superseded}]
        "facets": {"material": [], "color": [], "department": [], "price": None},
        # Facets the customer stated outright, as opposed to ones inferred from the
        # wording of a feature bullet. Only stated ones are trusted as hard filters.
        "stated_facets": [],
        "boundary": False,
        "nothing_left_to_learn": False,
        "exhausted": [],
        "changed_mind": False,
        "last_reply_kind": None,
    }


def _add_constraint(intent: dict, text: str, turn: int, weight: float = 1.0) -> dict | None:
    """Add one constraint, splitting synthesised ones off into facets instead."""
    text = re.sub(r"\s+", " ", text).strip(" -;,.\t\n")
    if not text:
        return None

    color = RE_COLOR_C.match(text)
    if color:
        value = color.group("value").strip().lower()
        if value not in intent["facets"]["color"]:
            intent["facets"]["color"].append(value)
        if "color:%s" % value not in intent["stated_facets"]:
            intent["stated_facets"].append("color:%s" % value)
        return None

    budget = RE_BUDGET_C.match(text)
    if budget:
        try:
            intent["facets"]["price"] = float(budget.group("value").replace(",", ""))
            intent["stated_facets"].append("price")
        except ValueError:
            pass
        return None

    norm = normalize_phrase(text)
    for existing in intent["constraints"]:
        if existing["norm"] == norm:
            existing["weight"] = max(existing["weight"], weight)
            existing["superseded"] = False
            return None

    entry = {
        "text": text,
        "norm": norm,
        "attribute": classify(text),
        "weight": weight,
        "turn": turn,
        "superseded": False,
    }
    intent["constraints"].append(entry)

    # opportunistic facet harvest — a feature bullet often names its own material
    for material in {m.lower() for m in MATERIAL_RE.findall(text)}:
        if material not in intent["facets"]["material"]:
            intent["facets"]["material"].append(material)
    for colour in {c.lower() for c in COLOR_RE.findall(text)}:
        if colour not in intent["facets"]["color"]:
            intent["facets"]["color"].append(colour)
    return entry


def _set_category(intent: dict, raw: str) -> None:
    raw = raw.strip().strip(".,")
    intent["category_raw"] = raw
    intent["category_terms"] = unique(tokens(raw))


def read_first_message(user_message: str, turn: int) -> tuple[dict, list[dict]]:
    """First turn: create the Structured Response from the opening utterance.

    Returns the intent response and the constraints it learned.
    """
    message = user_message.strip()

    match = RE_BUYING.match(message)
    if match:
        intent = new_intent("buying")
        _set_category(intent, match.group("category"))
        added = _add_constraint(intent, match.group("constraint"), turn)
        intent["last_reply_kind"] = "buying_open"
        return intent, [c for c in (added,) if c]

    match = RE_BROWSING.match(message)
    if match:
        intent = new_intent("browsing")
        _set_category(intent, match.group("category"))
        intent["last_reply_kind"] = "browsing_open"
        return intent, []

    match = RE_OVERRIDE_OPEN.match(message)
    if match and match.group("constraint").strip():
        # "I'm looking for {category}. {old_value}" — the override opener. The payload
        # is a full feature bullet, which is the richest turn-1 signal in the whole set.
        intent = new_intent("override")
        _set_category(intent, match.group("category"))
        added = _add_constraint(intent, match.group("constraint"), turn)
        intent["last_reply_kind"] = "override_open"
        return intent, [c for c in (added,) if c]

    # Paraphrase fallback: keep the whole utterance as a low-weight span.
    intent = new_intent("browsing")
    opening = RE_OPENING.match(message)
    if opening:
        _set_category(intent, opening.group("category"))
    else:
        _set_category(intent, message)
    added = _add_constraint(intent, message, turn, weight=0.4)
    intent["last_reply_kind"] = "fallback_open"
    return intent, [c for c in (added,) if c]


def update_with_new_info(intent: dict, user_message: str, turn: int) -> tuple[dict, list[dict], dict]:
    """Later turns: apply a patch to the existing Structured Response.

    Never rebuilds. Returns (intent, newly-learned constraints, signals).
    `signals` reports what the utterance was, so the ask policy can react:
    `reveal`, `exhausted`, `boundary`, `nudge`, `override`, `fallback`.
    """
    message = user_message.strip()
    signals: dict = {"kind": "fallback", "attribute": None, "duplicate": False}
    learned: list[dict] = []

    match = RE_REVEAL.match(message)
    if match:
        parts = [p.strip() for p in match.group("body").split(";") if p.strip()]
        # "X; X" means the card had fewer than three usable items and the simulator
        # padded soft_preferences with a copy of hard_constraints[0]: nothing is left.
        if len(parts) >= 2 and len({normalize_phrase(p) for p in parts}) == 1:
            intent["nothing_left_to_learn"] = True
            signals["duplicate"] = True
        for part in parts:
            added = _add_constraint(intent, part, turn)
            if added:
                learned.append(added)
        if not learned and not signals["duplicate"]:
            intent["nothing_left_to_learn"] = True
        signals["kind"] = "reveal"
        intent["last_reply_kind"] = "reveal"
        return intent, learned, signals

    match = RE_NO_MORE.match(message)
    if match:
        attribute = match.group("attr").lower()
        if attribute not in intent["exhausted"]:
            intent["exhausted"].append(attribute)
        if attribute == "other":
            # The unrestricted ask returned nothing, so the card is empty.
            intent["nothing_left_to_learn"] = True
        signals.update(kind="exhausted", attribute=attribute)
        intent["last_reply_kind"] = "exhausted"
        return intent, [], signals

    match = RE_BOUNDARY.match(message)
    if match:
        intent["boundary"] = True
        signals.update(kind="boundary", attribute=match.group("attr").lower())
        intent["last_reply_kind"] = "boundary"
        return intent, [], signals

    if RE_NUDGE.search(message):
        # We sent ask_attribute=null and burned a turn. Never do that again.
        signals["kind"] = "nudge"
        intent["last_reply_kind"] = "nudge"
        return intent, [], signals

    match = RE_OVERRIDE.match(message)
    if match:
        value = match.group("value").strip()
        norm = normalize_phrase(value)
        known = {c["norm"] for c in intent["constraints"]}
        # Is this a real pivot, or the same intent restated?
        #
        # The public simulator builds the "new" intent from `hard_constraints[0]` of the
        # *same* target product, so it is usually something the customer already told us.
        # Treating that as a pivot and down-weighting everything else destroys a ranking
        # that was often already correct — measured: it dropped a target from rank 1 to
        # outside the top 10. So a restatement is a re-affirmation: boost it, keep the
        # rest at full weight.
        #
        # A genuinely new value is treated as a real pivot, and even then the earlier
        # constraints are down-weighted rather than deleted. Deleting them is only right
        # if the customer contradicted them, and "ignore my earlier preference" does not
        # say which one.
        genuine_pivot = norm not in known
        if genuine_pivot:
            for existing in intent["constraints"]:
                if existing["turn"] < turn:
                    existing["superseded"] = True
                    existing["weight"] = min(existing["weight"], 0.35)
        added = _add_constraint(intent, value, turn, weight=1.3)
        if added:
            learned.append(added)
        else:
            for existing in intent["constraints"]:
                if existing["norm"] == norm:
                    existing["weight"] = 1.3
                    existing["superseded"] = False
        intent["changed_mind"] = genuine_pivot
        signals["kind"] = "override"
        signals["genuine_pivot"] = genuine_pivot
        intent["last_reply_kind"] = "override"
        return intent, learned, signals

    if RE_OVERRIDE_CUE.search(message):
        # Reworded change of mind. Pull out the replacement if we can see a marker,
        # otherwise fall back to the whole sentence, and reuse the same
        # re-affirmation-vs-real-pivot logic as the exact template above.
        marker = RE_NEW_VALUE.search(message)
        value = (marker.group("value") if marker else message).strip()
        norm = normalize_phrase(value)
        genuine_pivot = norm not in {c["norm"] for c in intent["constraints"]}
        if genuine_pivot and marker:
            for existing in intent["constraints"]:
                if existing["turn"] < turn:
                    existing["superseded"] = True
                    existing["weight"] = min(existing["weight"], 0.35)
        added = _add_constraint(intent, value, turn, weight=1.0 if marker else 0.4)
        if added:
            learned.append(added)
        intent["changed_mind"] = genuine_pivot
        signals["kind"] = "override"
        signals["genuine_pivot"] = genuine_pivot
        signals["inferred"] = True
        intent["last_reply_kind"] = "override_inferred"
        return intent, learned, signals

    # Unrecognised utterance (a paraphrased private set lands here): treat the whole
    # thing as a low-weight span so BM25 and LSA still get something to work with.
    added = _add_constraint(intent, message, turn, weight=0.4)
    if added:
        learned.append(added)
    intent["last_reply_kind"] = "fallback"
    return intent, learned, signals


def active_constraints(intent: dict) -> list[dict]:
    """Constraints ordered strongest-first: live before superseded, recent before old."""
    return sorted(
        intent["constraints"],
        key=lambda c: (c["superseded"], -c["weight"], -c["turn"]),
    )
