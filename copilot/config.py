"""Tunables for the shopping copilot.

Every number here was chosen from a measurement in `analysis/` (outside this repo),
not from intuition. Where a value encodes a measured fact, the comment names it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KIT_ROOT = REPO_ROOT / "provided" / "techjam-conversational-search"
DEFAULT_CATALOG = KIT_ROOT / "data" / "catalog.jsonl"

# The evaluator's allowed `ask_attribute` enum (docs/agent_api_contract.json).
ALLOWED_ATTRIBUTES = (
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
)

MAX_TURNS = 10
TOP_K = 10


@dataclass(frozen=True)
class RetrievalConfig:
    """Channel weights and thresholds for the multi-route retriever."""

    # Reciprocal Rank Fusion constant. 60 is the standard value from Cormack et al.
    rrf_k: int = 60

    # Per-channel RRF weights. The exact/conjunctive channel dominates because the
    # simulated customer quotes catalog text verbatim (measured: 760/760 constraints
    # are punctuation-insensitive substrings of the target row).
    w_conjunctive: float = 6.0
    w_phrase: float = 3.0
    w_bm25: float = 1.5
    w_facet: float = 1.0
    w_category: float = 0.8
    w_lsa: float = 0.6

    # A phrase appearing in more than this many rows carries almost no signal on its
    # own ("Imported" = 13,994 rows). It still participates in the AND, but it is the
    # first thing dropped during backoff.
    boilerplate_df: int = 1500

    # Candidate pool size handed to the reranker.
    candidate_pool: int = 400

    # If the conjunctive pool is at most this size we emit it directly: measured
    # median pool size is 1 and 60% of sessions collapse to a single row.
    direct_emit_max: int = 10

    # Whether the customer's stated colour/price facets also act as hard AND filters.
    # Off: it tightens the pool but converts slightly *earlier* at a slightly worse
    # rank, and rank is worth ~13x a turn here. Measured on the public set,
    # on -> 0.8830, off -> 0.8862. They still contribute through the facet channel and
    # through `RankConfig.w_facet`.
    fold_facets_into_and: bool = False


@dataclass(frozen=True)
class RankConfig:
    """Weights for the Select-10 reranker (see copilot/select10.py)."""

    w_coverage: float = 10.0        # constraints satisfied — by far the strongest signal
    w_facet: float = 2.0            # colour / price / material / department agreement
    w_category: float = 3.0         # category-path overlap with the stated category
    w_popularity: float = 0.9       # targets are real purchases, so they skew popular
    demote_shown: float = 4.0       # previously shown *and* provably not the target
    superseded_weight: float = 0.35  # an overridden constraint is down-weighted, never deleted

    # --- switches for the alternative scoring formula (see docs/ARCHITECTURE.md) ---
    # Weight each satisfied requirement by ln(1 + N/df) instead of treating them all
    # equally, so a rare phrase counts for more than "Imported".
    idf_coverage: bool = False
    # Sort strictly by "how many requirements does this product satisfy" first, and use
    # the blended score only to break ties, instead of one weighted sum.
    lexicographic: bool = False
    # Scale the category bonus by how narrow the category is, rather than a flat weight.
    idf_category: bool = False

    # The anonymised `preference_tags` scored as noise and is therefore off. Measured
    # on the held-out split: 0.0 -> 0.8856, 0.3 -> 0.8811, 1.2 -> 0.8300. The tags are
    # generic ("fit", "comfort", "durability") and match most of the catalog, so they
    # dilute the signals that do discriminate. The code path is kept for the private
    # set, where richer tags would make it worth turning back on.
    w_profile: float = 0.0


@dataclass(frozen=True)
class AskConfig:
    """Clarification policy (copilot/ask_policy.py)."""

    # Measured attribute mix over the 200 public cards: feature 404, material 302,
    # color 60, style 19, size 11, use_case 4. A typed ask therefore usually whiffs;
    # the unrestricted ask has the highest expected yield until the card runs dry.
    prior: dict = field(default_factory=lambda: {
        "feature": 0.53, "material": 0.40, "color": 0.08,
        "style": 0.025, "size": 0.014, "use_case": 0.005,
    })
    # Stop asking once the candidate pool is this small AND the card is known to be
    # drained. Guarded by `allow_null_ask`, which is off: a null `ask_attribute` draws
    # the reply "Ask me about one specific attribute", i.e. a turn that reveals
    # nothing. Measured cost of leaving it on: 6 of 200 sessions deadlocked to turn 10.
    stop_asking_pool: int = 10
    allow_null_ask: bool = False
    max_unrestricted_asks: int = 4


@dataclass(frozen=True)
class CopilotConfig:
    catalog_path: Path = DEFAULT_CATALOG
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rank: RankConfig = field(default_factory=RankConfig)
    ask: AskConfig = field(default_factory=AskConfig)

    # Latent-semantic channel (TF-IDF + TruncatedSVD). Off by default: it ships no
    # model weights but costs init time, and the public set is 100% lexical so it
    # earns nothing there. Turn it on as a paraphrase hedge if the private set
    # rewrites the simulator's utterances.
    enable_lsa: bool = False
    lsa_components: int = 192

    # Optional LLM polish of the customer-facing `message` string only. Never touches
    # ranking, so the scored path stays deterministic and offline.
    enable_llm_message: bool = False

    # Dump each session graph to this directory for transcript debugging.
    session_graph_dir: Path | None = None
