"""The global Knowledge Graph: the catalog, indexed once, read-only at runtime.

Scope and write rules (per the architecture note): this object is built before the
first session and is *never* written to during a conversation. Everything a turn
learns goes into the per-session graph instead.

The graph is a plain JSON-shaped structure — product nodes carrying facet values,
plus reverse indexes from facet/category/store/token back to product nodes. There is
no graph database; `to_json()` round-trips the whole thing.

Three retrieval structures are derived from it at build time:

* ``postings``    token -> sorted product ids, used to seed exact-phrase lookup
* ``matrix``      field-weighted term-frequency CSR, used for BM25F
* ``doc_norm``    punctuation-stripped rendering of each row, used to *verify*
                  that a candidate really contains a phrase verbatim
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .text import (
    STOPWORDS,
    COLOR_RE,
    MATERIAL_RE,
    normalize,
    normalize_phrase,
    searchable_text,
    tokens,
)

# Field weights folded into the term-frequency matrix, so BM25F is a plain BM25 over
# a weighted tf. Mirrors the ordering the organizer's own starter uses.
FIELD_WEIGHTS = {
    "title": 6.0,
    "features": 4.0,
    "categories": 2.5,
    "details": 2.5,
    "description": 1.5,
    "store": 1.0,
}

BM25_K1 = 1.4
BM25_B = 0.6

# Category segments that carry no discriminative signal — every row has them.
_GENERIC_CATEGORIES = {
    "clothing", "shoes", "jewelry", "clothing shoes & jewelry",
    "clothing, shoes & jewelry", "novelty & more",
}


def _flatten_field(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join("%s %s" % (k, v) for k, v in value.items())
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


class KnowledgeGraph:
    """Read-only catalog graph plus its derived retrieval indexes."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.asins: list[str] = []
        self.index_of: dict[str, int] = {}
        self.nodes: list[dict] = []          # JSON-shaped product nodes
        self.doc_norm: list[str] = []        # punctuation-free searchable text
        self.postings: dict[str, np.ndarray] = {}
        self.idf: dict[str, float] = {}
        self.facet_index: dict[str, dict[str, np.ndarray]] = {}
        self.category_postings: dict[str, np.ndarray] = {}
        self.popularity: np.ndarray = np.zeros(0, dtype=np.float32)
        self.price: np.ndarray = np.zeros(0, dtype=np.float32)
        self._matrix = None                  # scipy CSR, built lazily
        self._vocab: dict[str, int] = {}
        self._doc_len: np.ndarray = np.zeros(0, dtype=np.float32)
        self._lsa = None                     # (vectorizer, svd, embeddings)
        self._build()

    # ------------------------------------------------------------------ build

    def _build(self) -> None:
        raw_tf: list[Counter] = []
        postings: dict[str, set[int]] = defaultdict(set)
        facets: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        cat_post: dict[str, list[int]] = defaultdict(list)
        pops: list[float] = []
        prices: list[float] = []

        with self.catalog_path.open(encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                product = json.loads(line)
                asin = str(product["parent_asin"])
                self.asins.append(asin)
                self.index_of[asin] = i

                text = searchable_text(product)
                norm = normalize(text)
                self.doc_norm.append(norm)

                # weighted term frequency, one field at a time
                counter: Counter = Counter()
                for field, weight in FIELD_WEIGHTS.items():
                    for token in tokens(_flatten_field(product.get(field))):
                        counter[token] += weight
                raw_tf.append(counter)
                for token in counter:
                    postings[token].add(i)

                node = self._product_node(product, norm)
                self.nodes.append(node)

                for kind in ("material", "color", "department", "store"):
                    for value in node["facets"].get(kind, ()):
                        facets[kind][value].append(i)
                for segment in node["category_path"]:
                    cat_post[segment].append(i)

                pops.append(node["popularity"])
                prices.append(node["price"] if node["price"] is not None else float("nan"))

        n = len(self.asins)
        self.popularity = np.asarray(pops, dtype=np.float32)
        self.price = np.asarray(prices, dtype=np.float32)
        self.postings = {t: np.fromiter(sorted(d), dtype=np.int32) for t, d in postings.items()}
        self.idf = {
            t: math.log(1.0 + (n - len(d) + 0.5) / (len(d) + 0.5))
            for t, d in self.postings.items()
        }
        self.facet_index = {
            kind: {v: np.fromiter(sorted(set(ids)), dtype=np.int32) for v, ids in values.items()}
            for kind, values in facets.items()
        }
        self.category_postings = {
            seg: np.fromiter(sorted(set(ids)), dtype=np.int32) for seg, ids in cat_post.items()
        }
        self._build_matrix(raw_tf)

    def _product_node(self, product: dict, norm_text: str) -> dict:
        """One JSON-shaped node of the knowledge graph."""
        categories = [str(c) for c in (product.get("categories") or [])]
        path: list[str] = []
        for value in categories:
            for part in value.split(","):
                part = part.strip().lower()
                if part and part not in _GENERIC_CATEGORIES:
                    path.append(part)

        details = product.get("details") or {}
        department = []
        if isinstance(details, dict):
            for key in ("Department", "department"):
                if details.get(key):
                    department.append(str(details[key]).strip().lower())

        price = product.get("price")
        try:
            price_value = float(price) if price not in (None, "") else None
        except (TypeError, ValueError):
            price_value = None

        rating_number = int(product.get("rating_number") or 0)
        average_rating = float(product.get("average_rating") or 0.0)

        return {
            "parent_asin": str(product["parent_asin"]),
            "title": str(product.get("title") or ""),
            "category_path": path,
            "price": price_value,
            "rating_number": rating_number,
            "average_rating": average_rating,
            # log-damped popularity scaled by rating quality: the target is a real
            # purchase record, so it skews toward well-reviewed, frequently-bought rows.
            "popularity": math.log1p(rating_number) * (average_rating / 5.0 if average_rating else 0.5),
            "facets": {
                "material": sorted({m.lower() for m in MATERIAL_RE.findall(norm_text)}),
                "color": sorted({c.lower() for c in COLOR_RE.findall(norm_text)}),
                "department": sorted(set(department)),
                "store": [str(product.get("store") or "").strip().lower()] if product.get("store") else [],
            },
        }

    def _build_matrix(self, raw_tf: list[Counter]) -> None:
        from scipy.sparse import csr_matrix

        self._vocab = {token: j for j, token in enumerate(sorted(self.postings))}
        indptr = [0]
        indices: list[int] = []
        data: list[float] = []
        lengths: list[float] = []
        for counter in raw_tf:
            total = 0.0
            for token, weight in counter.items():
                indices.append(self._vocab[token])
                data.append(weight)
                total += weight
            indptr.append(len(indices))
            lengths.append(total)
        self._doc_len = np.asarray(lengths, dtype=np.float32)
        self._matrix = csr_matrix(
            (np.asarray(data, dtype=np.float32), np.asarray(indices, dtype=np.int32),
             np.asarray(indptr, dtype=np.int64)),
            shape=(len(raw_tf), len(self._vocab)),
        ).tocsc()

    # -------------------------------------------------------------- retrieval

    def __len__(self) -> int:
        return len(self.asins)

    def phrase_docs(self, phrase: str, limit: int | None = None) -> np.ndarray:
        """Product ids whose text contains `phrase` verbatim (punctuation-insensitive).

        Seeds candidates from the three rarest tokens of the phrase, then verifies
        containment on the normalised document text. Falls back to an empty result
        when any token is out of vocabulary — an out-of-vocabulary token cannot be
        present in any row, so the phrase cannot match either.
        """
        needle = normalize_phrase(phrase)
        terms = tokens(needle, drop_stopwords=False)
        if not terms:
            return np.zeros(0, dtype=np.int32)
        # Seed only from tokens the index actually holds. The postings are built from
        # `tokens()` with stopwords dropped, so a phrase like "Zipper fly WITH button
        # closure" has no posting list for "with" — and requiring one made the whole
        # phrase return nothing. Correctness does not depend on seeding with every
        # token: the candidates are verified by substring against `doc_norm` below, so
        # any indexed token is enough to narrow the search.
        if any(t not in self.postings and t not in STOPWORDS for t in terms):
            # a genuinely unknown word cannot appear in any row, so neither can the phrase
            return np.zeros(0, dtype=np.int32)
        seeds = [self.postings[t] for t in terms if t in self.postings]
        if not seeds:
            # every token is a stopword — such a phrase carries no signal anyway
            return np.zeros(0, dtype=np.int32)
        seeds.sort(key=len)
        candidates = seeds[0]
        for other in seeds[1:3]:
            candidates = np.intersect1d(candidates, other, assume_unique=True)
            if candidates.size == 0:
                return candidates
        docs = self.doc_norm
        hits = [int(d) for d in candidates if needle in docs[d]]
        if limit is not None:
            hits = hits[:limit]
        return np.asarray(hits, dtype=np.int32)

    def phrase_df(self, phrase: str) -> int:
        return int(self.phrase_docs(phrase).size)

    def bm25(self, query_terms: list[str], restrict: np.ndarray | None = None,
             top_n: int = 400) -> list[tuple[int, float]]:
        """Field-weighted BM25 over the term-frequency matrix."""
        cols = [(t, self._vocab[t]) for t in query_terms if t in self._vocab]
        if not cols:
            return []
        avg_len = float(self._doc_len.mean()) or 1.0
        scores = np.zeros(len(self.asins), dtype=np.float32)
        denom_norm = BM25_K1 * (1.0 - BM25_B + BM25_B * self._doc_len / avg_len)
        for token, col in cols:
            column = self._matrix.getcol(col).tocoo()
            rows, tf = column.row, column.data
            contrib = self.idf[token] * (tf * (BM25_K1 + 1.0)) / (tf + denom_norm[rows])
            scores[rows] += contrib.astype(np.float32)
        if restrict is not None and restrict.size:
            mask = np.zeros(len(self.asins), dtype=bool)
            mask[restrict] = True
            scores = np.where(mask, scores, 0.0)
        nonzero = np.flatnonzero(scores)
        if nonzero.size == 0:
            return []
        order = nonzero[np.argsort(-scores[nonzero])][:top_n]
        return [(int(i), float(scores[i])) for i in order]

    def category_docs(self, category_terms: list[str]) -> np.ndarray:
        """Rows whose category path overlaps the stated category, best overlap first."""
        pools = [self.category_postings[t] for t in category_terms if t in self.category_postings]
        if not pools:
            return np.zeros(0, dtype=np.int32)
        counts: Counter = Counter()
        for pool in pools:
            counts.update(pool.tolist())
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], -float(self.popularity[kv[0]])))
        return np.asarray([i for i, _ in ranked], dtype=np.int32)

    def facet_docs(self, kind: str, value: str) -> np.ndarray:
        return self.facet_index.get(kind, {}).get(value, np.zeros(0, dtype=np.int32))

    def price_docs(self, target: float, tolerance: float = 0.01) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            mask = np.abs(self.price - target) <= tolerance
        return np.flatnonzero(np.nan_to_num(mask, nan=False)).astype(np.int32)

    # ------------------------------------------------------------------- LSA

    def enable_lsa(self, n_components: int = 192) -> None:
        """Build the optional latent-semantic channel (no model weights shipped)."""
        if self._lsa is not None:
            return
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize as l2

        vectorizer = TfidfVectorizer(
            min_df=3, max_features=250_000, ngram_range=(1, 2), sublinear_tf=True
        )
        tfidf = vectorizer.fit_transform(self.doc_norm)
        svd = TruncatedSVD(n_components=n_components, random_state=0, n_iter=4)
        embeddings = l2(svd.fit_transform(tfidf)).astype(np.float32)
        self._lsa = (vectorizer, svd, embeddings)

    def lsa_docs(self, query: str, top_n: int = 200) -> list[tuple[int, float]]:
        if self._lsa is None:
            return []
        from sklearn.preprocessing import normalize as l2

        vectorizer, svd, embeddings = self._lsa
        vector = l2(svd.transform(vectorizer.transform([normalize(query)]))).astype(np.float32)
        scores = embeddings @ vector[0]
        order = np.argpartition(-scores, min(top_n, scores.size - 1))[:top_n]
        order = order[np.argsort(-scores[order])]
        return [(int(i), float(scores[i])) for i in order]

    # ------------------------------------------------------------------ JSON

    def to_json(self, include_indexes: bool = False) -> dict:
        """JSON-shaped view of the graph. Postings are omitted unless asked for."""
        payload: dict = {
            "kind": "knowledge_graph",
            "source": str(self.catalog_path),
            "product_nodes": len(self.nodes),
            "facet_nodes": {k: len(v) for k, v in self.facet_index.items()},
            "category_nodes": len(self.category_postings),
            "token_nodes": len(self.postings),
            "nodes": self.nodes,
        }
        if include_indexes:
            payload["facet_edges"] = {
                kind: {value: ids.tolist() for value, ids in values.items()}
                for kind, values in self.facet_index.items()
            }
            payload["category_edges"] = {
                seg: ids.tolist() for seg, ids in self.category_postings.items()
            }
        return payload

    def dump(self, path: str | Path, include_indexes: bool = False) -> None:
        Path(path).write_text(
            json.dumps(self.to_json(include_indexes), ensure_ascii=False), encoding="utf-8"
        )
