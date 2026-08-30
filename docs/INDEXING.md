# Indexing: turning 50,000 products into something we can search in 19 ms

This document covers what happens at startup, before any conversation begins. Its
companion, [TEXT_MATCHING.md](TEXT_MATCHING.md), covers what happens when a query arrives.

Code: [`copilot/knowledge_graph.py`](../copilot/knowledge_graph.py).

---

## 1. The constraint that shapes everything

The rules say the index must live **in memory** — no external vector service, no database.
The evaluator builds our `Agent` once and then calls it about 330 times in a row, so the
trade is obvious: pay once at startup, pay nothing per turn.

What we actually pay:

| | |
|---|---|
| Build time | **16.3 s**, single core |
| Products | 50,000 |
| Distinct words | 100,999 |
| Total word→product entries | 4,380,853 |
| Memory: cleaned text | 47 MB |
| Memory: word lists | 18 MB |
| Memory: keyword-score table | 35 MB |
| Per-turn cost afterwards | **19 ms** average |

About 100 MB of index, built once, then nothing touches the disk again.

---

## 2. What we build, and why each piece exists

The catalog is read line by line. Each row produces one **product node** plus entries in
four lookup tables.

### 2.1 The product node

A plain dictionary — the whole thing is JSON, there is no graph database:

```python
{
  "parent_asin": "B07K34RX5J",
  "title": "Kandinsky Statement Earrings for Women by Spirit Hoops, Fabric, ...",
  "category_path": ["women", "jewelry", "earrings", "hoop"],
  "price": None,
  "rating_number": 871,
  "average_rating": 4.1,
  "popularity": 5.57,
  "facets": {"material": ["spandex"], "color": [], "department": ["womens"],
             "store": ["spirit hoops"]}
}
```

Two fields are computed rather than copied:

**`category_path`** splits each category string on commas, lowercases, and drops segments
that every row shares (`clothing`, `shoes`, `jewelry`, `clothing, shoes & jewelry`,
`novelty & more`). Keeping them would mean every product matches every category query.
875 distinct segments survive.

**`popularity`** = `ln(1 + rating_number) × (average_rating / 5)`.

That formula is doing more work than it looks like. The hidden target is a **real purchase
record**, so it is biased towards products that many people actually bought. Ranking a
fully-narrowed candidate set by popularity *alone* reaches Hit@10 0.945 and MRR 0.861 —
without a single word of text matching. The `ln` keeps one viral product with 80,000
reviews from swamping everything; multiplying by the star rating stops a heavily-reviewed
but disliked product from floating up.

We compared eight variants (raw count, log, square root, Wilson-style, stars alone, title
length). They land within 0.005 MRR of one another, so the exact shape does not matter —
only that popularity is used at all.

### 2.2 Word lists (postings)

`word → sorted array of product ids`, as `numpy.int32`.

Sorted matters: intersecting two sorted arrays is a linear merge, which is what makes the
"must match all requirements" search fast.

Stopwords (`the`, `and`, `with`, `on`, …) are dropped here. That saves memory and search
time, but it caused a real bug in the lookup — see
[TEXT_MATCHING.md §4](TEXT_MATCHING.md#4-the-stopword-trap).

We also store each word's **IDF** — `ln(1 + (N − df + 0.5) / (df + 0.5))` — which the
keyword scorer uses and which tells us how rare a word is.

### 2.3 Cleaned text (`doc_norm`)

One string per product: every searchable field flattened, lowercased, punctuation replaced
by spaces, and padded with a space at each end.

This is the file's most important 47 MB. Word lists can only tell you a product *contains*
some words; they cannot tell you the words appeared **next to each other, in order**.
`doc_norm` is what we check a phrase against to confirm a genuine match. The details of
that check are in [TEXT_MATCHING.md](TEXT_MATCHING.md).

### 2.4 Keyword-score table (BM25F)

A sparse matrix of 4.4 million non-zero entries — 0.1% dense, 35 MB.

Rather than storing six separate per-field tables, we fold the field weight into the word
count as we read:

```python
for field, weight in FIELD_WEIGHTS.items():        # title 6.0, features 4.0,
    for token in tokens(product[field]):           # categories 2.5, details 2.5,
        counter[token] += weight                   # description 1.5, store 1.0
```

So a word in the title contributes 6 to its count and the same word in the description
contributes 1.5. Field-weighted BM25 then becomes ordinary BM25 over the weighted counts —
one matrix instead of six, and the weighting costs nothing at query time.

The weights mirror the ones the organizer's own starter agent uses, so field importance is
not something we invented.

Stored column-major (CSC), because a query touches a handful of *words* — that is, a
handful of columns — and never a whole product row.

### 2.5 Filters

Three more lookups, each `value → product ids`:

| Filter | Distinct values | Source |
|---|---|---|
| material | 28 | regex over the cleaned text (cotton, leather, alloy, …) |
| colour | 23 | regex over the cleaned text (black, navy, ivory, …) |
| department | 136 | the `details` dict, when it has a `Department` key |
| store | 19,749 | the `store` field |

Plus two plain arrays: `price` and `popularity`, one entry per product, so filtering by
price is a single vectorised comparison.

**Price is only present on 20.8% of products.** That sounds like a weakness and is actually
an advantage: when the shopper *does* state a price, matching it exactly narrows 50,000
products to a handful, because most rows cannot match any price at all.

---

## 3. What we deliberately did **not** build

**No dense vector index.** There is an optional latent-semantic channel (TF-IDF +
TruncatedSVD, no downloaded model weights) and it is switched off. It adds 45 s to startup
and measures +0.001 — which is not a broken feature, it is a measurement of how little
meaning-based signal this data contains. It stays available as a spare tyre in case the
shopper's wording changes.

**No pre-built phrase index.** We cannot know in advance which phrases a shopper will
quote, and indexing every n-gram of 50,000 product listings would be enormous. Instead we
find candidates with word lists and confirm with a substring check — cheap at build time,
1 ms at query time.

**No offline LLM enrichment.** Tagging all 50,000 rows with an LLM was considered. It would
cost real money, add a build artifact to ship, and the measurements say the catalog's own
text is already enough.

**Nothing is written back.** The index is built before the first conversation and is
read-only for the rest of the run. Everything a conversation learns goes into its own
session graph, which is thrown away when that conversation ends. That separation is what
lets one index be shared safely by every session.

---

## 4. Checking the index is right

```bash
python analysis/leakage_audit.py     # field fill rates, boilerplate, selectivity
python analysis/ceiling_probe.py     # what perfect matching could achieve
python analysis/rank_study.py        # which ranking signal separates the target
```

These live outside the repository, in `analysis/`. They import the **real** evaluator
functions and run against the **real** 50,000-row catalog — nothing is mocked.

The most important check is the one in [TEXT_MATCHING.md §4](TEXT_MATCHING.md#4-the-stopword-trap):
comparing the fast lookup against a brute-force scan of all 50,000 rows. That comparison
caught a bug that had been silently discarding whole requirements.
