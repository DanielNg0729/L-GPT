"""Text normalisation shared by the knowledge graph and the query understanding layer.

One rule matters above all: `normalize()` must render a catalog row the same way the
evaluator's own `searchable_text()` does, then strip punctuation. That single change
lifts the measured constraint-is-a-verbatim-substring rate from 0.9934 to 1.0000,
because the evaluator flattens the `details` dict as ``"Department Womens"`` while the
intent card quotes it as ``"Department: Womens"``.
"""
from __future__ import annotations

import re

# Field order is copied from evaluator.local_evaluator.searchable_text so that a
# phrase spanning two adjacent fields still matches.
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")

_PUNCT = re.compile(r"[^a-z0-9]+")
_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "s", "t",
})

MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "fabric", "denim", "linen", "cashmere", "suede", "velvet",
    "acrylic", "elastane", "viscose", "alloy", "sterling", "silver", "gold",
    "brass", "stainless", "titanium", "rubber", "canvas", "mesh", "satin",
)
COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "beige", "navy", "gold", "silver", "ivory",
    "burgundy", "teal", "khaki", "maroon", "cream", "tan",
)

MATERIAL_RE = re.compile(r"\b(%s)\b" % "|".join(MATERIALS), re.I)
COLOR_RE = re.compile(r"\b(%s)\b" % "|".join(COLORS), re.I)


def flatten(value: object) -> str:
    """Render one catalog field exactly as the evaluator does."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join("%s %s" % (key, item) for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def searchable_text(product: dict) -> str:
    """Byte-compatible with evaluator.local_evaluator.searchable_text."""
    parts = [flatten(product.get(field)) for field in SEARCH_FIELDS]
    return " ".join(part for part in parts if part).strip()


def normalize(text: str) -> str:
    """Lowercase, punctuation to space, whitespace collapsed.

    Padded with single spaces at both ends so that ``" cotton "`` is a safe
    word-boundary containment test.
    """
    return " %s " % _PUNCT.sub(" ", text.lower()).strip()


def normalize_phrase(text: str) -> str:
    """Same normalisation, for the needle side of a containment test."""
    return " %s " % _PUNCT.sub(" ", text.lower()).strip()


def tokens(text: str, drop_stopwords: bool = True) -> list[str]:
    out = _TOKEN.findall(text.lower())
    if drop_stopwords:
        return [t for t in out if len(t) > 1 and t not in STOPWORDS]
    return [t for t in out if len(t) > 1]


def unique(seq: list[str]) -> list[str]:
    return list(dict.fromkeys(seq))
