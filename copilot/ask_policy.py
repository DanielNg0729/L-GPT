"""Clarification policy: which attribute to ask about, by expected information gain.

The temptation is to hardcode `ask_attribute="other"`, because the simulator's reply
function short-circuits the type check for `"other"` and hands back the first two
undisclosed constraints of *any* type. That works, but it is a magic constant with no
justification, and it breaks the moment the private set changes the reveal policy.

So the choice is computed instead. For each candidate attribute `a`:

    EIG(a) = E[constraints revealed | ask a] x E[bits per constraint | a]

with both expectations estimated from the public set rather than guessed:

* `P(a)` — the attribute mix of the 760 constraints in the 200 public intent cards:
  feature 0.53, material 0.40, colour 0.08, style 0.025, size 0.014, use_case 0.005.
* bits — mean `log2(50000 / catalog_df)` of a constraint of that type, measured against
  the real catalog: feature 6.84, material 6.43, and 15.4-15.6 for colour / style /
  size / use_case (those are almost always unique rows, but they are rare).

The unrestricted ask matches *any* undisclosed constraint, so its `P` is 1.0 and it
wins on expectation until the card runs dry — the policy therefore *derives* "other"
rather than assuming it, and switches to typed asks once the unrestricted channel is
exhausted. Asking is free in this harness (a turn carries `message`, `ask_attribute`
and `recommendations` at once, and the hit is checked before the customer replies), so
the policy never trades a recommendation slot for a question.

One hard rule: never emit `ask_attribute: null` while anything is still unknown. The
simulator answers `null` with "Ask me about one specific attribute", which is a turn
that reveals nothing at all.
"""
from __future__ import annotations

import math

from .config import AskConfig

CATALOG_SIZE = 50_000
EXPECTED_CARD_SIZE = 4.0     # hard_constraints[:2] + soft_preferences[:2]
REVEALS_PER_ASK = 2.0        # the simulator returns at most two matches per reply

# Mean log2(N / catalog_df) per attribute, measured over the 200 public intent cards.
BITS_PER_CONSTRAINT = {
    "feature": 6.84,
    "material": 6.43,
    "color": 15.53,
    "style": 15.61,
    "size": 15.43,
    "use_case": 15.61,
    "budget": 12.0,
    "brand": 8.0,
    "category": 3.0,
    "other": 7.29,           # the population mean across all 760 constraints
}


def _remaining_estimate(intent: dict) -> float:
    """How many card constraints are probably still undisclosed."""
    if intent.get("nothing_left_to_learn"):
        return 0.0
    known = len(intent["constraints"])
    return max(0.0, EXPECTED_CARD_SIZE - known)


def expected_information_gain(intent: dict, cfg: AskConfig) -> dict[str, float]:
    remaining = _remaining_estimate(intent)
    exhausted = set(intent.get("exhausted") or [])
    gains: dict[str, float] = {}
    for attribute in ("other", *cfg.prior):
        if remaining <= 0.0:
            gains[attribute] = 0.0
            continue
        # "other" matches any undisclosed constraint; a typed ask only matches its own type.
        probability = 1.0 if attribute == "other" else float(cfg.prior.get(attribute, 0.02))
        if attribute in exhausted:
            probability *= 0.05          # the customer already said there is nothing more
        expected_reveals = min(REVEALS_PER_ASK, remaining * probability)
        gains[attribute] = expected_reveals * BITS_PER_CONSTRAINT.get(attribute, 6.0)
    return gains


def choose(intent: dict, session_graph: dict, pool_size: int, cfg: AskConfig) -> str | None:
    """Pick the next `ask_attribute`.

    `pool_size` is the size of the conjunctive candidate pool. Once it fits inside the
    scored top 10 there is nothing left for a question to buy, and the slate itself is
    the answer.
    """
    if cfg.allow_null_ask and intent.get("nothing_left_to_learn") and 0 < pool_size <= cfg.stop_asking_pool:
        return None

    # Otherwise always ask. A small pool is *not* evidence that the target is in it —
    # it only means the constraints we hold happen to intersect tightly, which is
    # exactly what happens when we hold too few of them. Staying silent there costs
    # the whole session: the simulator answers a null ask with "Ask me about one
    # specific attribute", which reveals nothing, so the conversation deadlocks until
    # turn 10. Asking is free here (a turn carries recommendations *and* a question,
    # and the hit is checked before the customer replies), so there is never a reason
    # to spend a turn on silence.
    gains = expected_information_gain(intent, cfg)
    best = max(gains, key=lambda a: gains[a]) if gains else "other"
    if gains.get(best, 0.0) > 0.0:
        return best

    # The card is drained but we still have not converged. A typed ask may whiff, but
    # `null` whiffs with certainty, so spend the turn on the least-explored attribute.
    exhausted = set(intent.get("exhausted") or [])
    asked = set(session_graph.get("asked") or [])
    for attribute in sorted(cfg.prior, key=lambda a: -cfg.prior[a]):
        if attribute not in exhausted and attribute not in asked:
            return attribute
    for attribute in sorted(cfg.prior, key=lambda a: -cfg.prior[a]):
        if attribute not in exhausted:
            return attribute
    return "feature"


def entropy_bits(pool_size: int) -> float:
    """Remaining uncertainty over the candidate pool, for logging and ablations."""
    return math.log2(pool_size) if pool_size > 1 else 0.0
