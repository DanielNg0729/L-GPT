# Text matching: how a shopper's words find a product

[INDEXING.md](INDEXING.md) covers what we build at startup. This covers what happens when
the shopper says something and we go looking.

Code: [`copilot/text.py`](../copilot/text.py),
[`copilot/knowledge_graph.py`](../copilot/knowledge_graph.py).

---

## 1. The fact that decides the whole design

The shopper does not describe the product in their own words. **They quote its listing,
word for word.** We measured this across all 200 public conversations:

> **760 out of 760** things the shopper said are exact substrings of the target product's
> own catalog text.

So this is not a "find something similar" problem. It is a "find the row containing this
exact string" problem. Everything below follows from that.

Full evidence: [EVALUATOR_ANALYSIS.md](EVALUATOR_ANALYSIS.md).

---

## 2. Cleaning the text so both sides agree

A match only works if the shopper's sentence and the product's listing are written the same
way. Two rules get us there.

### Rule 1 — flatten the product exactly as the evaluator does

Product fields are joined in this order: `title`, `features`, `details`, `description`,
`categories`, `store`. Lists are joined with spaces; a `details` dictionary becomes
`"key value key value"`.

This ordering is copied from the evaluator's own code, not chosen by us. It matters because
a quoted phrase can straddle two neighbouring fields, and it will only match if we join
them in the same order.

### Rule 2 — punctuation becomes a space

```python
def normalize(text):
    return " %s " % re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
```

This one line is worth 5 of the 760 matches, for a reason worth spelling out.

The evaluator flattens a `details` entry as `"Department Womens"` — key and value with a
space between them. But when it quotes that same entry back to the shopper, it writes
`"Department: Womens"`, **with a colon**. A literal comparison fails. Replace punctuation
with spaces on both sides and both become `department womens`.

| Matching method | Requirements matched |
|---|---|
| exact string comparison | 755 / 760 (99.34%) |
| **punctuation-insensitive** | **760 / 760 (100%)** |

The leading and trailing spaces matter too. Searching for `" cotton "` inside
`" ...soft cotton fleece... "` cannot accidentally match the middle of *cottonseed*. It is
word-boundary matching without needing a regex.

---

## 3. Finding a phrase, fast

A word list tells you a product *contains* some words. It cannot tell you those words
appeared **next to each other, in order**. So the lookup runs in two stages: narrow with
the word lists, then confirm against the cleaned text.

```
phrase_docs("Zipper fly with button closure")

  1. clean it          -> " zipper fly with button closure "
  2. split into words  -> [zipper, fly, with, button, closure]
  3. unknown word?     -> if any word is in no product at all, stop: the phrase
                          cannot match either
  4. seed              -> take the word lists we have, sort by length:
                              fly     586 products
                              button  4,149
                              zipper  4,166
                              closure 19,303
                          intersect the 3 shortest -> a few dozen candidates
  5. confirm           -> keep only those whose cleaned text actually contains
                          " zipper fly with button closure "
                       -> 16 products
```

**Why the three rarest words.** Intersecting sorted arrays costs time proportional to their
length, so starting with the shortest list collapses the candidate set fastest. Three is
enough: after that the set is small enough that the substring check is cheaper than another
intersection.

**Why we can stop at three.** Correctness does not depend on using every word — step 5
verifies the complete phrase against the real text. The word lists are only there to avoid
scanning all 50,000 rows. This is exactly why the fix in §4 is safe.

Measured cost: **1.05 ms per lookup**.

---

## 4. The stopword trap

This is the bug worth knowing about, because the failure was silent.

The word lists are built with stopwords dropped — `the`, `and`, `with`, `on`, `for`. That
is a normal, sensible optimisation. But the lookup originally demanded a word list for
**every** word in the phrase, and bailed out if one was missing:

```python
seeds = [self.postings.get(t) for t in terms]
if any(s is None for s in seeds):
    return EMPTY            # <- any phrase containing "with" or "on" died here
```

So any requirement containing a stopword returned **zero products** and was quietly dropped
from the search:

| Phrase | Returned | Actually matches |
|---|---|---|
| `Zipper fly **with** button closure` | 0 | 16 |
| `Lace Slip **On** Sneaker` | 0 | 2 |
| `Pull **On** closure` | 0 | **7,405** |

Nothing crashed. Nothing logged. The requirement simply stopped counting.

**The fix:** seed from the words the index *does* hold, and bail out only on a genuinely
unknown word — a word in no product at all really does mean the phrase cannot match.

```python
if any(t not in postings and t not in STOPWORDS for t in terms):
    return EMPTY
seeds = [postings[t] for t in terms if t in postings]
```

**How we know it is right now:** every lookup is compared against a brute-force scan of all
50,000 rows.

| | before | after |
|---|---|---|
| phrases agreeing with brute force | 4 / 7 | **10 / 10** |
| TechnicalScore | 0.890686 | **0.892686** |
| MRR | 0.6896 | **0.6943** |

The lesson we took from it: any "fast path" over an index needs a slow path to check
against. The brute-force comparison is three lines and it caught something that had been
wrong for the whole project.

---

## 5. Keyword scoring, for when the exact phrase fails

Exact matching is our best tool, not our only one. If a shopper's wording differs — a real
risk if the private set rewrites their sentences — we fall back to word-level scoring with
**BM25F**.

```
                    tf × (k₁ + 1)
score(word) = IDF × ──────────────────────────────────,   k₁ = 1.4,  b = 0.6
                    tf + k₁ × (1 − b + b × len/avg_len)
```

Three things this does that plain word-counting does not:

- **IDF** — `ln(1 + (N − df + 0.5) / (df + 0.5))`. A word in 50 products is worth far more
  than one in 15,000. This is what stops `Imported` from drowning out
  `73% Cotton, 25% Polyester, 2% Spandex`.
- **Saturation** (`k₁`) — the 10th occurrence of a word adds much less than the 2nd.
  Keyword-stuffed listings do not win.
- **Length normalisation** (`b`) — a long listing has more chances to contain any given
  word, so it is discounted. Without this, verbose descriptions dominate everything.

The `tf` here already has field weighting baked in — title counts 6×, features 4×,
description 1.5× — because the weight was folded into the counts at build time
([INDEXING.md §2.4](INDEXING.md#24-keyword-score-table-bm25f)).

Keyword scoring is the **spare tyre that actually works**. When we stress-tested with the
shopper's sentences reworded, the score fell from 0.8927 to 0.8440 — but Hit@10 only fell
from 0.995 to 0.985. That is BM25F carrying the conversations where exact matching failed.
The meaning-based search, by contrast, contributed +0.001.

---

## 6. Why not embeddings

Not ideology — measurement.

We built the meaning-based channel (TF-IDF plus TruncatedSVD, no downloaded model weights)
and switched it on:

| | score |
|---|---|
| off | 0.8927 |
| on | 0.8928 |

**+0.0001**, for 45 seconds of extra startup. That is not a broken channel. It is a
measurement of how much meaning-based signal this data contains: when the shopper is
quoting the catalog verbatim, there is nothing for semantic similarity to add that exact
matching has not already found.

It stays in the code, switched off, as a hedge in case the private set paraphrases.

---

## 7. Summary

| Job | Tool | Cost |
|---|---|---|
| find an exact quoted phrase | word lists + substring confirm | 1 ms |
| combine several requirements | intersect sorted arrays | microseconds |
| cope with different wording | BM25F over weighted counts | ~5 ms |
| structured values (colour, price) | direct lookups and array comparisons | microseconds |
| meaning-based similarity | available, off, worth +0.0001 | 45 s startup |

Next: [RETRIEVAL.md](RETRIEVAL.md) — how these are combined into one candidate set.
