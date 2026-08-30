# Retrieval: from what the shopper said to a shortlist

[TEXT_MATCHING.md](TEXT_MATCHING.md) covers how one phrase finds products. This covers how
we combine everything the shopper has told us into a single candidate list, ready for
[RANKING.md](RANKING.md) to order.

Code: [`copilot/retrieval.py`](../copilot/retrieval.py).

---

## 1. The main search: keep only products that have all of it

The shopper quotes the product's listing, so the most powerful move is not scoring — it is
**intersection**. Take everything they have said, and keep only the products containing all
of it.

```
"73% Cotton, 25% Polyester, 2% Spandex"  ->      8 products
"Zipper fly with button closure"         ->     16 products
"Imported"                               -> 15,300 products
                          intersection   ->      2 products
```

How well it works, measured over all 200 public conversations once the shopper has said
everything they are going to say:

| | |
|---|---|
| The right product is in the set | **100.0%** |
| Typical set size | **1 product** |
| Exactly one product left | 60% of conversations |
| 10 or fewer left (a guaranteed hit) | 74% |
| Times we had to give up a requirement | **0** |

### The order matters

We sort requirements **most specific first** and intersect in that order. Starting from the
8-product list collapses the set in one step; starting from `Imported` means carrying
15,300 candidates through every stage.

Specificity is just the size of each requirement's product list — we already know it from
the lookup.

### Giving up gracefully

If the intersection empties out, we drop one requirement and try again, choosing the
**vaguest** one first: anything the shopper has since overridden, then whatever matches the
most products.

That order is deliberate. `Imported` appears in 15,300 listings and tells us almost
nothing, so it is the cheapest thing to lose. A phrase in 8 listings is nearly a serial
number, and we keep it to the last.

On the public set this never fires — the intersection is never empty. It exists for a
private set where a shopper might say something the catalog does not contain.

### Stopping early

If the intersection leaves **10 products or fewer, we stop there and show them.** No other
search runs.

This is not laziness, it is arithmetic: only 10 products get scored. If the correct answer
is already in a set of 10, blending in results from other searches can only push it out.

---

## 2. Two things deliberately kept out of the intersection

Both of these looked obviously right and measured wrong.

### The stated category

The shopper says *"I'm looking for Women Bodysuits"*, and that looks like a phrase we could
require. It is not. The simulator builds that string from the last two category levels after
removing generic ones — but the product's own listing reads `Women Clothing Bodysuits`,
with `Clothing` still in the middle.

So `"women bodysuits"` is **not** a contiguous string in the product we are looking for. It
matches a handful of unrelated rows and **silently excludes the right answer**. In one
conversation this reduced the candidate set to a single wrong product and the agent sat
there for nine turns.

Category still matters enormously — it is worth 0.070 of the final score, more than any
other single signal. It just enters as a *ranking* signal and its own search, never as a
hard requirement.

### The shopper's stated colour and price

Requiring these tightens the set and converts slightly *earlier* — at a slightly worse
rank. Measured: 0.8901 with them required, 0.8927 without. Rank is worth roughly 13× a turn
here (see [RANKING.md §5](RANKING.md#5-why-rank-matters-more-than-speed)), so they stay
out of the requirement list and contribute through filtering and ranking instead.

The switch is `RetrievalConfig.fold_facets_into_and`, default off.

---

## 3. The four backup searches

The main search needs requirements. On turn 1 of a browsing conversation we have none — the
shopper has only named a category. These searches carry those turns.

| Search | Weight | What it does |
|---|---|---|
| **partial match** | 3.0 | products matching *some* requirements, ranked by how many |
| **keyword (BM25F)** | 1.5 | word-level scoring, for when the exact phrase fails |
| **filters** | 1.0 | colour, material, department, and exact price |
| **category** | 0.8 | products on the named shelf, most popular first |
| **meaning-based** | 0.6 | optional, off — worth +0.0001, kept as a spare tyre |

A note on the price filter: only **20.8%** of products have a price at all. That makes an
exact price match extremely powerful when the shopper states one — most of the catalog
cannot match any price, so the survivors are few. We match to the cent, and widen only if
nothing in the catalog sits on that exact value.

---

## 4. Merging them: Reciprocal Rank Fusion

Five searches, five completely different number scales. A BM25 score of 47.3 and a cosine
similarity of 0.81 cannot be added together.

RRF sidesteps this by throwing the scores away and keeping only the **positions**:

```
score(product) = Σ   weight(search) / (60 + rank in that search)
              searches
```

A product ranked 1st by the main search and 5th by keyword scoring gets
`6.0/(60+1) + 1.5/(60+5)`.

Three reasons this is the right choice:

- **No calibration.** Adding a sixth search later needs no rescaling of the other five.
- **Robust to a blown-up score.** One search returning a wild number cannot dominate; being
  ranked 1st is worth the same regardless of how confident that search was.
- **Agreement wins.** A product several searches all like beats one that a single search
  loves — which is exactly what we want when no single search is trustworthy alone.

The constant 60 is the standard value from the original RRF paper. It flattens the
difference between rank 1 and rank 2 so a narrow win in one search does not decide the
whole ordering.

The weights say the main search is worth about four times keyword scoring. That follows
directly from the 100% / 760-out-of-760 measurement: when the shopper is quoting the
catalog, exact matching is not one opinion among five, it is the answer.

---

## 5. What comes out

Either:

- **10 or fewer products** from the intersection, passed straight through, or
- **up to 400 candidates** from the merge, ordered by fused rank.

Either way the result is a shortlist plus, for each product, a record of exactly how many
of the shopper's requirements it satisfies. [RANKING.md](RANKING.md) takes it from there.

---

## 6. The whole thing on one conversation

```text
turn 1   "Hi, I need Women Jeans. It has to have cotton."
         requirements: [cotton]
         intersection: 9,775 products  -> too many, run all five searches
         merged shortlist -> 400 candidates
         we ask a question

turn 2   "What matters to me: 73% Cotton, 25% Polyester, 2% Spandex and Imported."
         requirements: [cotton, 73% Cotton..., Imported]
         intersection: sort by size -> [8, 9775, 15300], intersect
                       -> 2 products
         2 <= 10, stop and show them.  Found.
```

Two turns, no model, about 19 ms of work.
