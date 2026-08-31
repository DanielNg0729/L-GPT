"""
Exact catalogue span recovery for unfamiliar wrappers. The other half of the BERT tagger.

WHY THE TAGGER ALONE IS NOT ENOUGH
----------------------------------
The scaffolding tagger strips filler so that mining sees only product text, and it does
that job well: measured on the held-out wrapper bank it retained 99.25% of the canonical
constraint slots. It then hands the cleaned text to n-gram mining, and mining recovered
**0.00% of the short constraints**.

That is the whole problem in one number. Mining keeps an n-gram only when
`0 < df <= DF_CAP`, and it is built to find distinctive multi-word phrases. A constraint
value like `cotton`, `zipper closure` or `pull on closure` is one to three tokens long and
far too common to survive that filter, so a perfectly cleaned message yields nothing. The
tagger cleans the text and the miner then throws away exactly the part that mattered.

Measured consequence, on the same held-out bank:

    raw V1                                            0.720525
    route classification only                         0.738198
    route + exact category + short spans              0.939900

Shipping the tagger without this module captures almost none of that: measured at +0.00315
against a -0.287 template-paraphrase gap, which is the tagger cleaning text for a miner
that cannot use it.

WHAT THIS ADDS, AND WHY IT CANNOT INVENT EVIDENCE
--------------------------------------------------
Two exact lookups over frozen, catalogue-derived vocabularies. Neither generates text.

  CATEGORY   the longest `coarse_category()` phrase, from the visible catalogue, that
             appears verbatim in the message. Matching consumes those tokens so they
             cannot also be read as an attribute -- without that, "women s shoes" would
             contribute `shoes` twice through two different channels at two weights.
  ATTRIBUTE  every contiguous 1-, 2- or 3-token span that is an EXACT member of the
             catalogue attribute dictionary. Width 3 is the cap because the dictionary is
             built from attribute values and longer spans are prose, and because mining
             already covers the long-phrase case.

Both are G1 exclusions in the gate taxonomy: an exact string attested in the frozen
catalogue, or nothing. There is no threshold, no similarity, and no model decision -- the
tagger's output is used only as a candidate SOURCE, and every phrase that survives is one
the catalogue itself contains. That is what makes this safe to run on text the agent does
not understand.

DIVERGENCE FROM THE V2 RESEARCH IMPLEMENTATION, STATED BECAUSE IT MATTERS. The V2 prototype
gated these lookups on a trained route classifier's action label, using it to decide when a
message can carry a category or an attribute. This module uses the agent's own recognition
gate instead: the lookups run only on a message that matches none of the simulator's known
shapes. That removes a trained-model dependency from the shipped path at the cost of some
precision the action label was buying, so the 0.939900 above should be read as the
prototype's number and not as a prediction for this one.
"""
from __future__ import annotations

import json
from pathlib import Path


# FACET NAMES ARE NOT VALUES. The dictionary is mined from `features`/`details`, and some
# products carry a detail whose VALUE is literally the facet word -- so `material`, `color`
# and `size` are all dictionary members. That makes the span node extract the facet a
# customer is DECLINING as though it were a requirement:
#
#     "material is not important to me."   ->   attrs = {material}
#
# Measured on the held-out wrapper bank before this exclusion: every one of 1,600 reworded
# no-preference messages produced exactly one spurious attribute, always the facet name.
# V1's `PAT_NOINFO` catches those messages only in their literal form, so a reworded one
# reached the node unguarded.
#
# The names are the evaluator's own constraint families (`PROBE_ORDER` plus the attributes
# with no emitting branch). Excluding them costs nothing that could matter: a bare facet
# word is the least selective phrase in the catalogue, so it carries almost no ranking
# signal even when it does appear as a genuine value. This stays a G1 exclusion -- it
# removes non-values from a value dictionary rather than adding any threshold.
FACET_NAMES = frozenset({
    "feature", "features", "material", "materials", "color", "colour", "colors",
    "colours", "style", "styles", "size", "sizes", "use case", "use_case", "usecase",
    "category", "categories", "brand", "brands", "budget", "price", "other", "attribute",
})


def _coarse_category(values: list[str]) -> str:
    """Match the public category normalization without importing the evaluator."""
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned = [part.strip() for value in values for part in value.split(",")
               if part.strip() and part.strip().lower() not in excluded]
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


class ExactCatalogueSpanNode:
    """Exact category and short-attribute recovery. Frozen vocabularies, no inference."""

    MAX_ATTRIBUTE_TOKENS = 3
    MIN_CATEGORY_TOKENS = 2          # a one-word "category" is an attribute, not a category

    def __init__(self, catalog_path: str | Path, dictionary_path: str | Path | None = None,
                 toks=None, coarse=None) -> None:
        self.ok = False
        self.categories: list[tuple[str, ...]] = []
        self.attributes: frozenset[str] = frozenset()
        self.hits = {"category": 0, "attribute": 0}
        try:
            self._build(catalog_path, dictionary_path, toks, coarse)
            self.ok = bool(self.attributes)
        except Exception:
            # A missing dictionary must degrade to "this layer does nothing", never to a
            # broken agent. The caller checks `ok`.
            self.ok = False

    def _build(self, catalog_path, dictionary_path, toks, coarse) -> None:
        if toks is None or coarse is None:               # pragma: no cover - direct use
            from submission.agent import raw_toks as toks  # type: ignore
            coarse = _coarse_category
        seen: set[tuple[str, ...]] = set()
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                phrase = tuple(toks(coarse(
                    [str(x) for x in product.get("categories") or []])))
                if len(phrase) >= self.MIN_CATEGORY_TOKENS:
                    seen.add(phrase)
        # Longest first: "women s running shoes" must win over "running shoes".
        self.categories = sorted(seen, key=len, reverse=True)

        if dictionary_path is None:
            here = Path(__file__).resolve().parent
            for candidate in (here / "catalogue_attribute_dictionary.jsonl",
                              here.parent / "experiments" / "datasets"
                              / "catalogue_attribute_dictionary.jsonl"):
                if candidate.exists():
                    dictionary_path = candidate
                    break
        self.attributes = frozenset(
            phrase
            for line in Path(dictionary_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
            for phrase in (json.loads(line)["canonical"],)
            if phrase not in FACET_NAMES)

    def extract(self, tokens: list[str]) -> tuple[str | None, set[str]]:
        """Return (category phrase or None, set of attribute phrases) found verbatim."""
        if not self.ok or not tokens:
            return None, set()
        category = None
        for pattern in self.categories:
            width = len(pattern)
            if width > len(tokens):
                continue
            hit = next((i for i in range(len(tokens) - width + 1)
                        if tuple(tokens[i:i + width]) == pattern), None)
            if hit is not None:
                category = " ".join(pattern)
                # Consume the category tokens so they cannot ALSO be read as an attribute.
                tokens = tokens[:hit] + tokens[hit + width:]
                self.hits["category"] += 1
                break
        found = {
            " ".join(tokens[i:i + w])
            for w in range(1, self.MAX_ATTRIBUTE_TOKENS + 1)
            for i in range(len(tokens) - w + 1)
            if " ".join(tokens[i:i + w]) in self.attributes
        }
        self.hits["attribute"] += len(found)
        return category, found

    def stats(self) -> dict:
        return {"ok": self.ok, "categories": len(self.categories),
                "attributes": len(self.attributes), **self.hits}
