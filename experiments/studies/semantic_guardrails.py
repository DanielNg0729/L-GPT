"""Independent, fail-closed guardrails for V2 semantic attribute evidence.

This module does not rank products.  It decides whether a semantic model may contribute a
single catalogue-attested attribute to the unchanged V1 evidence store.  Each node records
its own verdict so a partial semantic shift cannot silently bypass a failed predecessor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import json

from experiments.studies.provenance_gate import Provenance, assess

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DICTIONARY = ROOT / "experiments" / "datasets" / "catalogue_attribute_dictionary.jsonl"
Reason = Literal[
    "provenance", "dictionary", "catalogue_attestation", "family", "similarity",
    "margin", "verifier", "rank", "abstain", "accepted"
]


@dataclass(frozen=True)
class AttributePrediction:
    """One semantic model candidate, always expressed in canonical catalogue vocabulary."""

    canonical: str
    family: str
    similarity: float
    margin: float
    # These two fields are intentionally optional while Nodes 4 and 5 are still being
    # rebuilt.  A production semantic resolver must supply both before Node 6 can admit
    # a prediction.  Their defaults preserve the old audit scaffold only.
    verifier_score: float | None = None
    retrieval_rank: int | None = None


@dataclass(frozen=True)
class FamilyPrediction:
    """An independently produced coarse family route for the customer phrase."""

    family: str
    confidence: float


@dataclass(frozen=True)
class GuardrailConfig:
    min_similarity: float = 0.45
    min_margin: float = 0.03
    full_margin: float = 0.12
    min_family_confidence: float = 0.60
    # Production defaults fail closed: a proposed mapping must include outputs from
    # both Nodes 4 and 5.  The legacy provenance audit opts into its own labelled
    # scaffold configuration below; it cannot become a runtime default by accident.
    min_verifier_score: float = 0.80
    # Node 4 returns a small candidate set, Node 5 verifies each pair, and Node 6 admits
    # the best verified mapping.  Rank one would discard recoverable candidates before
    # verification; top five is the frozen retrieval evaluation budget.
    max_retrieval_rank: int = 5
    provenance_mode: Literal["strict", "soft"] = "soft"


@dataclass(frozen=True)
class GuardrailDecision:
    routed: bool
    allowed: bool
    reason: Reason
    provenance: Provenance
    # Node 6 confidence is a correctness estimate for one candidate mapping.  It is
    # deliberately not a ranking weight.  Node 7 owns ranking influence.
    semantic_confidence: float
    canonical: str | None
    trace: tuple[str, ...]


def load_dictionary(path: Path = DEFAULT_DICTIONARY) -> frozenset[str]:
    """Load only canonical phrases harvested from the fixed visible catalogue."""
    return frozenset(
        json.loads(line)["canonical"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _reject(reason: Reason, provenance: Provenance, trace: list[str]) -> GuardrailDecision:
    return GuardrailDecision(False, False, reason, provenance, 0.0, None, tuple(trace))


def evaluate(
    agent,
    customer_text: str,
    attribute: AttributePrediction,
    family: FamilyPrediction,
    dictionary: frozenset[str],
    config: GuardrailConfig = GuardrailConfig(),
) -> GuardrailDecision:
    """Fail closed unless every independently auditable semantic node succeeds.

    This is Node 6 only.  ``semantic_confidence`` estimates confidence in the proposed
    canonical mapping; it has no authority to change ranking.  Node 7 separately turns
    an accepted mapping into ranking evidence, and defaults to a strict zero weight.
    """
    provenance = assess(agent, customer_text)
    trace = [f"provenance:{'eligible' if provenance.eligible else 'lexical'}:{provenance.confidence:.3f}"]
    # The current canonical index and local encoder operate on the English normalised
    # vocabulary.  An empty normalisation is neither catalogue-grounded nor safely
    # comparable, even under soft provenance.  This prevents non-ASCII literals such as
    # a verbatim imported-feature string from being mistaken for unseen semantic content.
    if not provenance.phrase:
        trace.append("provenance:empty_normalisation")
        return _reject("provenance", provenance, trace)
    if config.provenance_mode == "strict" and not provenance.eligible:
        return _reject("provenance", provenance, trace)

    canonical = " ".join(attribute.canonical.lower().split())
    trace.append(f"dictionary:{canonical in dictionary}")
    if canonical not in dictionary:
        return _reject("dictionary", provenance, trace)

    attested = agent.ix.df(canonical) > 0
    trace.append(f"catalogue_attestation:{attested}")
    if not attested:
        return _reject("catalogue_attestation", provenance, trace)

    family_ok = bool(family.family) and family.family == attribute.family and family.confidence >= config.min_family_confidence
    trace.append(f"family:{family_ok}")
    if not family_ok:
        return _reject("family", provenance, trace)

    similarity_ok = attribute.similarity >= config.min_similarity
    trace.append(f"similarity:{similarity_ok}")
    if not similarity_ok:
        return _reject("similarity", provenance, trace)

    margin_ok = attribute.margin >= config.min_margin
    trace.append(f"margin:{margin_ok}")
    if not margin_ok:
        return _reject("margin", provenance, trace)

    # The old frozen baselines do not provide a verifier or a rank.  They therefore
    # remain usable as *scaffold* examples only by leaving these checks disabled in the
    # supplied config.  Any actual Node 4/5 experiment must enable both checks.
    if config.min_verifier_score > 0.0:
        verifier_ok = attribute.verifier_score is not None and attribute.verifier_score >= config.min_verifier_score
        trace.append(f"verifier:{verifier_ok}")
        if not verifier_ok:
            return _reject("verifier", provenance, trace)
    if config.max_retrieval_rank > 0:
        rank_ok = attribute.retrieval_rank is not None and attribute.retrieval_rank <= config.max_retrieval_rank
        trace.append(f"rank:{rank_ok}")
        if not rank_ok:
            return _reject("rank", provenance, trace)

    margin_confidence = min(1.0, attribute.margin / config.full_margin)
    verifier_confidence = attribute.verifier_score if attribute.verifier_score is not None else 1.0
    # Provenance is a safety condition, not a reward.  It is excluded from correctness
    # confidence so unfamiliar but precise wording is not automatically up-weighted.
    semantic_confidence = attribute.similarity * margin_confidence * family.confidence * verifier_confidence
    trace.append("accepted")
    return GuardrailDecision(True, True, "accepted", provenance, semantic_confidence, canonical, tuple(trace))
