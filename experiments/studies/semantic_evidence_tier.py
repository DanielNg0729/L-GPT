"""A disabled-by-default semantic evidence tier for V2 integration experiments.

The submitted V1 agent does not import this module.  Later experiments may pass only
catalogue-attested candidates and an explicit weight.  The default weight is exactly zero,
which proves that adding the harness cannot alter canonical V1 rankings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SemanticEvidence:
    canonical: str
    confidence: float
    product_ids: frozenset[str]


@dataclass(frozen=True)
class SemanticIntegrationPolicy:
    """Node 7 policy, separated from semantic correctness.

    ``maximum_weight=0`` is the required default and is a strict identity transform.
    A later experiment may set a small positive maximum only after Node 6 is calibrated
    on a sealed value-paraphrase holdout and passes Official200 and Unseen800
    non-interference checks.  The monotonic schedule prevents a low-confidence mapping
    from receiving more influence than a high-confidence mapping.
    """

    maximum_weight: float = 0.0
    minimum_confidence: float = 1.0

    def weight_for(self, confidence: float) -> float:
        if self.maximum_weight < 0:
            raise ValueError("maximum semantic weight must be non-negative")
        if self.maximum_weight == 0.0 or confidence < self.minimum_confidence:
            return 0.0
        if self.minimum_confidence >= 1.0:
            return self.maximum_weight
        scaled = (min(1.0, confidence) - self.minimum_confidence) / (1.0 - self.minimum_confidence)
        return self.maximum_weight * max(0.0, scaled)


def apply_semantic_tier(
    base_scores: Mapping[str, float], evidence: SemanticEvidence | None, weight: float = 0.0
) -> dict[str, float]:
    """Return a new score map; zero weight is a strict identity transformation."""
    if weight < 0:
        raise ValueError("semantic weight must be non-negative")
    result = dict(base_scores)
    if not evidence or weight == 0.0 or evidence.confidence <= 0.0:
        return result
    bonus = weight * min(1.0, evidence.confidence)
    for product_id in evidence.product_ids:
        if product_id in result:
            result[product_id] += bonus
    return result


def apply_policy(
    base_scores: Mapping[str, float], evidence: SemanticEvidence | None,
    policy: SemanticIntegrationPolicy = SemanticIntegrationPolicy(),
) -> dict[str, float]:
    """Apply Node 7 with an explicit policy; default is byte-for-byte score identity."""
    return apply_semantic_tier(base_scores, evidence, policy.weight_for(evidence.confidence) if evidence else 0.0)
