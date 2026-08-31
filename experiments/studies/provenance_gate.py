"""Literal-provenance assessment used before a V2 semantic branch may influence ranking."""
from __future__ import annotations

from dataclasses import dataclass

from submission.agent import raw_toks


@dataclass(frozen=True)
class Provenance:
    phrase: str
    full_attested: bool
    coverage: float
    specificity: float
    known_construction: bool
    source_continuation: bool
    eligible: bool

    @property
    def confidence(self) -> float:
        if self.full_attested or self.known_construction or self.source_continuation:
            return 1.0
        return min(1.0, 0.70 * self.coverage + 0.30 * self.specificity)


def assess(agent, text: str) -> Provenance:
    """Assess whether lexical output is provenance or an incidental corpus collision.

    This is deliberately not an empty-result check. A long, rare literal span is strong
    provenance; a short or broad n-gram merely found somewhere in the 50k catalogue is not.
    """
    phrase = " ".join(raw_toks(text))
    if not phrase:
        return Provenance("", False, 0.0, 0.0, False, False, False)
    full = agent.ix.df(phrase) > 0
    known = phrase.startswith("color ") or phrase.startswith("budget around ")
    resolved = agent._resolve(text)
    length = max(1, len(phrase.split()))
    coverage = max((len(item.split()) / length for item in resolved), default=0.0)
    # A phrase attested in most catalogue products is weak even if it has several words.
    best_df = min((agent.ix.df(item) for item in resolved), default=agent.ix.DF_CAP)
    specificity = max(0.0, 1.0 - min(1.0, best_df / agent.ix.DF_CAP))
    # `intent_card` caps raw feature prose at 180 characters. Such a prefix can be absent
    # from FTS after normalisation while its long, rare continuation is still literal target
    # evidence. V2 paraphrase atoms are deliberately short, so retain these source passages.
    source_continuation = (
        (length >= 12 and coverage >= 0.50 and specificity >= 0.95)
        or (any(ord(char) > 127 for char in text) and coverage >= 0.45)
    )
    eligible = not full and not known and not source_continuation and (coverage < 0.80 or specificity < 0.85)
    return Provenance(phrase, full, coverage, specificity, known, source_continuation, eligible)
