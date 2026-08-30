"""Multi-route retrieval over the knowledge graph, fused with Reciprocal Rank Fusion.

Channel design follows the measurement, not the textbook. On this task the customer
quotes the target row verbatim, so the exact/conjunctive channel is not one channel
among equals — it is the answer, and everything else is a hedge for the turns where
too little has been disclosed yet.

Measured on the 200 public sessions, with the full intent card revealed:

    AND of every disclosed constraint      -> pool contains the target  100.0%
                                              median pool size          1
                                              pool <= 10                74.0%
                                              never needed a backoff drop

so `conjunctive` runs first and, when it lands a small pool, short-circuits the rest.

The remaining channels carry the early turns and the paraphrase risk:

    phrase      union of per-constraint exact matches, scored by weighted coverage
    bm25        field-weighted BM25 over constraint + category tokens
    facet       intent filters: material / colour / price / department
    category    category-path match ranked by a popularity prior (browsing cold start)
    lsa         optional TF-IDF + SVD cosine; ships no model weights
"""
from __future__ import annotations

import math

import numpy as np

from .config import RetrievalConfig
from .knowledge_graph import KnowledgeGraph
from .text import tokens, unique
from .understanding import active_constraints

_EMPTY = np.zeros(0, dtype=np.int32)


class RetrievalResult(dict):
    """Channels, the conjunctive pool, and the per-document constraint coverage."""


def _intersect(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.size == 0 or right.size == 0:
        return _EMPTY
    return np.intersect1d(left, right, assume_unique=True)


def _rrf(channels: dict[str, list[int]], weights: dict[str, float], k: int) -> list[int]:
    scores: dict[int, float] = {}
    for name, ranked in channels.items():
        weight = weights.get(name, 1.0)
        if weight <= 0.0:
            continue
        for rank, doc in enumerate(ranked, start=1):
            scores[doc] = scores.get(doc, 0.0) + weight / (k + rank)
    return [doc for doc, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def conjunctive_pool(
    kg: KnowledgeGraph, intent: dict, cfg: RetrievalConfig
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """AND every disclosed constraint, dropping the least selective one when empty.

    Returns (pool, phrases actually used, phrases dropped). Backoff order is
    superseded-first, then highest document frequency first — a superseded slot is
    down-weighted rather than deleted, and boilerplate like "Imported" (13,994 rows)
    is the cheapest thing to give up.
    """
    entries: list[dict] = []
    for constraint in active_constraints(intent):
        docs = kg.phrase_docs(constraint["norm"])
        if docs.size == 0:
            continue
        entries.append({
            "constraint": constraint,
            "docs": docs,
            "df": int(docs.size),
            "superseded": bool(constraint["superseded"]),
        })

    # Structured filters join the AND as first-class members. Two of the four intent-card
    # slots are *synthesised* by the simulator rather than quoted — "color: black" and
    # "budget around $12.99" — so they never match as phrases, but they are exactly the
    # kind of constraint a filter handles. Folding them in is worth a lot: the exact
    # price alone collapses most pools to a single row.
    for name, docs in (facet_members(kg, intent, stated_only=True)
                       if cfg.fold_facets_into_and else []):
        entries.append({
            "constraint": {"text": name, "norm": name, "attribute": "facet",
                           "weight": 1.0, "turn": 0, "superseded": False},
            "docs": docs,
            "df": int(docs.size),
            "superseded": False,
        })

    # The stated category deliberately does *not* join the AND. It looks like a verbatim
    # phrase but is not one: the simulator builds it from `categories[-2:]` after
    # dropping generic segments, so "Women Bodysuits" is stitched from a path that reads
    # "Women Clothing Bodysuits" in the row itself. ANDing it therefore matches a
    # handful of unrelated rows and silently excludes the target. Category enters as a
    # ranking signal and as its own channel instead.

    if not entries:
        return _EMPTY, [], []

    # Most selective first, so the intersection collapses in as few steps as possible.
    entries.sort(key=lambda e: (e["superseded"], e["df"]))
    dropped: list[dict] = []
    work = list(entries)
    while work:
        pool = work[0]["docs"]
        for entry in work[1:]:
            pool = _intersect(pool, entry["docs"])
            if pool.size == 0:
                break
        if pool.size:
            return pool, [e["constraint"] for e in work], [e["constraint"] for e in dropped]
        # give up the least useful member and retry
        victim = max(range(len(work)), key=lambda i: (work[i]["superseded"], work[i]["df"]))
        dropped.append(work.pop(victim))
    return _EMPTY, [], [e["constraint"] for e in dropped]


def coverage_vector(kg: KnowledgeGraph, intent: dict) -> dict:
    """How well each product satisfies the requirements the shopper has given.

    Returns three views of the same thing, because the reranker can be configured to
    use either:

    * ``coverage``   sum of requirement weights satisfied (every requirement equal)
    * ``idf``        the same sum, but each requirement scaled by ln(1 + N/df) so a
                     rare phrase counts for more than boilerplate
    * ``count``      plain integer count, for a strict "more requirements always wins"
                     sort

    The IDF view only changes anything when products differ in *which* requirements
    they satisfy. Inside a conjunctive pool every member satisfies all of them, so the
    three agree; it matters on the early turns, where the pool is still wide.
    """
    n = float(len(kg))
    coverage = np.zeros(len(kg), dtype=np.float32)
    idf_coverage = np.zeros(len(kg), dtype=np.float32)
    count = np.zeros(len(kg), dtype=np.int16)
    total = 0.0
    idf_total = 0.0
    for constraint in intent["constraints"]:
        weight = float(constraint["weight"])
        total += weight
        docs = kg.phrase_docs(constraint["norm"])
        idf = math.log(1.0 + n / max(1, docs.size))
        idf_total += weight * idf
        if docs.size:
            coverage[docs] += weight
            idf_coverage[docs] += weight * idf
            count[docs] += 1
    return {
        "coverage": coverage,
        "idf": idf_coverage,
        "count": count,
        "max": total,
        "idf_max": idf_total,
    }


def facet_members(kg: KnowledgeGraph, intent: dict,
                  stated_only: bool = False) -> list[tuple[str, np.ndarray]]:
    """Each intent filter as its own droppable posting list.

    These are kept separate rather than pre-intersected so the conjunctive backoff can
    give up one filter at a time instead of all of them at once.

    `stated_only` restricts the result to facets the customer actually stated — the
    simulator's synthesised "color: X" and "budget around $X" card slots. Facets merely
    *inferred* from the wording of a feature bullet are redundant with the phrase that
    produced them and measurably noisier, so they stay out of the AND and are used only
    for ranking and for the facet channel.
    """
    facets = intent["facets"]
    stated = set(intent.get("stated_facets") or [])
    members: list[tuple[str, np.ndarray]] = []

    price = facets.get("price")
    if price is not None and (not stated_only or "price" in stated):
        # The simulator leaks the target's *exact* price, which is close to a primary
        # key: only ~21% of the catalog carries a price at all. Widen only if no row
        # sits on that value.
        docs = kg.price_docs(float(price), tolerance=0.005)
        if docs.size == 0:
            docs = kg.price_docs(float(price), tolerance=max(0.5, float(price) * 0.02))
        if docs.size:
            members.append(("price=%s" % price, docs))

    for colour in facets.get("color") or []:
        if stated_only and "color:%s" % colour not in stated:
            continue
        docs = kg.facet_docs("color", colour)
        if docs.size:
            members.append(("color=%s" % colour, docs))
    if stated_only:
        return members
    for material in facets.get("material") or []:
        docs = kg.facet_docs("material", material)
        if docs.size:
            members.append(("material=%s" % material, docs))
    for department in facets.get("department") or []:
        docs = kg.facet_docs("department", department)
        if docs.size:
            members.append(("department=%s" % department, docs))
    return members


def facet_pool(kg: KnowledgeGraph, intent: dict) -> np.ndarray:
    """Intersection of the intent filters, relaxed rather than emptied."""
    members = facet_members(kg, intent)
    if not members:
        return _EMPTY
    members.sort(key=lambda kv: len(kv[1]))
    pool = members[0][1]
    for _, other in members[1:]:
        nxt = _intersect(pool, other)
        if nxt.size == 0:
            break          # relax: keep the tighter pool we already have
        pool = nxt
    return pool


def query_terms(intent: dict) -> list[str]:
    terms: list[str] = []
    for constraint in active_constraints(intent):
        if constraint["superseded"]:
            continue
        terms.extend(tokens(constraint["text"]))
    terms.extend(intent.get("category_terms") or [])
    return unique(terms)[:64]


def retrieve(kg: KnowledgeGraph, intent: dict, cfg: RetrievalConfig) -> RetrievalResult:
    """Run every channel and fuse. `pool` is the authoritative narrow candidate set."""
    pool, used, dropped = conjunctive_pool(kg, intent, cfg)
    cov = coverage_vector(kg, intent)
    coverage, coverage_max = cov["coverage"], cov["max"]

    result = RetrievalResult(
        pool=pool,
        pool_size=int(pool.size),
        used_constraints=[c["text"] for c in used],
        dropped_constraints=[c["text"] for c in dropped],
        coverage=coverage,
        coverage_max=coverage_max,
        idf_coverage=cov["idf"],
        idf_coverage_max=cov["idf_max"],
        match_count=cov["count"],
        channels={},
        direct_emit=bool(0 < pool.size <= cfg.direct_emit_max),
    )
    if result["direct_emit"]:
        # A pool this small is already the complete answer set; fusing other channels
        # into it could only push the target out of the scored top 10.
        result["candidates"] = pool.tolist()
        return result

    restrict = pool if 0 < pool.size <= 20 * cfg.candidate_pool else None
    channels: dict[str, list[int]] = {}

    if pool.size:
        channels["conjunctive"] = pool[np.argsort(-coverage[pool])][: cfg.candidate_pool].tolist()

    hit_docs = np.flatnonzero(coverage)
    if hit_docs.size:
        order = hit_docs[np.argsort(-coverage[hit_docs])][: cfg.candidate_pool]
        channels["phrase"] = order.tolist()

    terms = query_terms(intent)
    if terms:
        channels["bm25"] = [d for d, _ in kg.bm25(terms, restrict=restrict, top_n=cfg.candidate_pool)]

    facets = facet_pool(kg, intent)
    if facets.size:
        ordered = facets[np.argsort(-(coverage[facets] * 5.0 + kg.popularity[facets]))]
        channels["facet"] = ordered[: cfg.candidate_pool].tolist()

    category_terms = intent.get("category_terms") or []
    if category_terms:
        cats = kg.category_docs(category_terms)
        if cats.size:
            ordered = cats[np.argsort(-(coverage[cats] * 5.0 + kg.popularity[cats]))]
            channels["category"] = ordered[: cfg.candidate_pool].tolist()

    if kg._lsa is not None:
        query = " ".join([intent.get("category_raw", "")]
                         + [c["text"] for c in active_constraints(intent) if not c["superseded"]])
        if query.strip():
            channels["lsa"] = [d for d, _ in kg.lsa_docs(query, top_n=cfg.candidate_pool)]

    weights = {
        "conjunctive": cfg.w_conjunctive,
        "phrase": cfg.w_phrase,
        "bm25": cfg.w_bm25,
        "facet": cfg.w_facet,
        "category": cfg.w_category,
        "lsa": cfg.w_lsa,
    }
    result["channels"] = {name: len(ids) for name, ids in channels.items()}
    result["candidates"] = _rrf(channels, weights, cfg.rrf_k)[: cfg.candidate_pool]
    return result
