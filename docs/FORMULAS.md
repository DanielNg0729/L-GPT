# Every formula, and why each number is what it is

One place listing every calculation in the agent, with the reasoning and — where we have it
— the measurement behind each coefficient.

All coefficients live in [`copilot/config.py`](../copilot/config.py).

> **A note on the ablation figures.** Deltas quoted below come from a single sweep over the
> 200 public conversations using a fixed 140/60 split. That sweep ran *before* the
> phrase-lookup fix in [TEXT_MATCHING.md §4](TEXT_MATCHING.md#4-the-stopword-trap), so the
> absolute values sit about 0.002 below current. Every variant shifted by the same amount,
> so the comparisons between them are unaffected. The shipped configuration measures
> **0.892686** today. Re-run `python harness/ablate.py` for exact current values.

---

## 1. Indexing

### 1.1 Popularity prior

```
popularity(p) = ln(1 + rating_number) × (average_rating / 5)
```

**Why this exists.** The hidden target is a real purchase record, so it is biased towards
products many people actually bought. Ranking a fully-narrowed candidate set by popularity
*alone* reaches **Hit@10 0.945, MRR 0.861** with no text matching whatsoever.

**Why `ln`.** Review counts span roughly 0 to 80,000. Used raw, a single viral product
dominates every list it appears in. The logarithm compresses that: 100 reviews to 1,000 is a
meaningful jump, 40,000 to 80,000 is barely anything — which matches how the signal actually
behaves.

**Why multiply by the rating.** A product with 8,000 reviews at 2.1 stars is popular but not
bought *on purpose*. Scaling by `average_rating / 5` keeps the range at 0–1 so the shape of
the log term is preserved.

**Why the exact shape barely matters.** We compared eight variants on real candidate sets:

| Variant | MRR |
|---|---|
| `sqrt(n) × stars` | 0.6539 |
| raw count | 0.6529 |
| **`ln(1+n) × stars`** (shipped) | **0.6491** |
| Wilson-style | 0.6467 |
| stars alone | 0.4349 |
| shorter title | 0.4272 |

The top four are within 0.005 of each other. What matters is *using* popularity, not which
curve. Stars alone collapses because it ignores volume entirely.

### 1.2 IDF (word rarity)

```
IDF(word) = ln(1 + (N − df + 0.5) / (df + 0.5)),    N = 50,000
```

Standard smoothed inverse document frequency. `df` is how many products contain the word.

**Why the `+0.5` terms.** They stop the expression blowing up for a word in every product
(`df = N`) or in none. **Why the outer `+1`.** It keeps IDF non-negative, so a very common
word contributes ~0 rather than a negative score that would actively penalise products for
containing it.

Scale in practice: a word in 50 products scores ~6.9; a word in 15,000 scores ~1.2.

### 1.3 Field weights

```
tf(word, product) = Σ  weight(field) × count(word in field)
                   fields
```

| Field | Weight | Reasoning |
|---|---|---|
| title | 6.0 | short and deliberately written; a word here is what the product *is* |
| features | 4.0 | the bullet list, where most quoted requirements come from |
| categories | 2.5 | short and structured, but shared by thousands of products |
| details | 2.5 | structured key/value pairs, mostly reliable |
| description | 1.5 | long marketing prose, easy to match by accident |
| store | 1.0 | a brand name; useful only when the shopper names it |

These mirror the weights in the organizer's own starter agent, so field importance is not
something we invented. Folding them into the counts at build time turns field-weighted BM25
into ordinary BM25 over one matrix.

---

## 2. Text matching

### 2.1 Normalisation

```python
normalize(text) = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "
```

**Why punctuation becomes a space.** The evaluator renders a details entry as
`"Department Womens"` but quotes it back as `"Department: Womens"`. Under literal comparison
those differ. Under this rule both become `department womens`.

| Method | Requirements matched |
|---|---|
| exact comparison | 755 / 760 |
| **punctuation-insensitive** | **760 / 760** |

**Why the padding spaces.** Searching `" cotton "` inside `" ...soft cotton fleece... "`
cannot match the middle of *cottonseed*. Word-boundary matching without a regex.

### 2.2 BM25F

```
                     tf × (k₁ + 1)
score = IDF × ────────────────────────────────────,    k₁ = 1.4,  b = 0.6
              tf + k₁ × (1 − b + b × len / avg_len)
```

**`k₁ = 1.4` — how fast repetition stops helping.** With `k₁ = 0` a word counts once no
matter how often it appears; as `k₁ → ∞` the score grows linearly. The standard range is
1.2–2.0. We sit slightly low because our `tf` is already field-weighted and therefore
inflated — a title word arrives with a count of 6, not 1 — so saturation should kick in
sooner than it would on raw counts.

**`b = 0.6` — how hard long listings are penalised.** `b = 0` ignores length entirely;
`b = 1` divides fully by it. The usual default is 0.75. We use less because catalog length
here is mostly a function of *product type* rather than verbosity — jewellery listings are
naturally short, shoe listings naturally long — so penalising length in full would bias
against entire categories.

---

## 3. Retrieval

### 3.1 Reciprocal Rank Fusion

```
score(product) = Σ  weight(search) / (k + rank in that search),    k = 60
```

**Why fuse ranks and not scores.** BM25 returns numbers around 10–50, cosine similarity
returns 0–1, an exact match is boolean. They cannot be added. Positions are always
comparable.

**`k = 60`** is the value from the original RRF paper. It flattens the gap between rank 1
and rank 2 (`1/61` vs `1/62`) so a narrow win in one search does not decide everything, and
it bounds any single search's contribution — one search returning a wild score cannot
dominate.

**The weights:**

| Search | Weight | Reasoning |
|---|---|---|
| exact / intersection | 6.0 | the shopper quotes the listing; measured 760/760, right answer present 100% of the time |
| partial match | 3.0 | same evidence, weaker form — some requirements rather than all |
| keyword BM25F | 1.5 | the fallback that actually works when wording differs |
| filters | 1.0 | precise but narrow; only fires when a colour or price is stated |
| category | 0.8 | broad; thousands of products share a shelf |
| meaning-based | 0.6 | off by default; measured worth +0.0001 |

The 4:1 ratio between exact matching and keyword scoring is the direct consequence of the
760/760 measurement. When the shopper quotes the catalog, exact matching is not one opinion
among five — it is the answer.

### 3.2 Requirement ordering and backoff

```
sort requirements by |matching products| ascending, then intersect
if empty: drop max(superseded, |matching products|), retry
```

**Why most-specific-first.** Intersection cost is proportional to the lists involved.
Starting from an 8-product list collapses the set in one step; starting from `Imported`
(15,300) carries them through every stage.

**Why the vaguest goes first on backoff.** `Imported` appears in 15,300 listings and carries
almost no information. A phrase in 8 listings is nearly a serial number. Losing the vague one
costs least. Overridden requirements go before both, since the shopper already moved on.

### 3.3 Thresholds

| Constant | Value | Reasoning |
|---|---|---|
| `direct_emit_max` | 10 | only 10 products are scored. If the right answer is already in a set of 10, mixing in other searches can only push it out. |
| `candidate_pool` | 400 | ranking is linear in this; 400 is well past where the fused tail stops containing anything real. |
| price tolerance | $0.005 | prices are floats — "same price" means within half a cent. |
| widened tolerance | `max(0.5, 2%)` | used only if no product sits on that exact value. |
| `boilerplate_df` | 1,500 | above this a phrase is treated as vague for backoff ordering. `Imported` is 15,300. |

---

## 4. Ranking

### 4.1 The scoring formula

```
score(p) =  10.0 × coverage(p)          coverage ∈ [0,1]
         +   2.0 × facet_agreement(p)
         +   3.0 × category_overlap(p)
         +   0.9 × popularity(p)        normalised to [0,1]
         +   0.0 × profile_match(p)
         −   4.0 × [already shown and provably wrong]

coverage(p) = Σ weight(c) for satisfied c  /  Σ weight(c) for all c
weight(c)   = 1.0 normally, 0.35 if the shopper has since overridden it
```

Ties break on popularity.

### 4.2 Why each coefficient

**`coverage = 10.0`.** An order of magnitude above everything else because it is an order of
magnitude more informative: the shopper *told us* this, and it is a verbatim quote from the
listing of the product we are hunting. Its absolute size does not matter — 6, 10 and 16 all
score identically, because it is the *ratio* to the other terms that decides ordering.

**`category = 3.0`.** Category cannot be a hard requirement (it is not a contiguous string in
the listing — see [RETRIEVAL.md §2](RETRIEVAL.md#2-two-things-deliberately-kept-out-of-the-intersection)),
so it must earn its keep in ranking.

| Weight | Score |
|---|---|
| 0.0 | 0.8210 |
| 1.5 | 0.8895 |
| **3.0** | **0.8927** |
| 6.0 | 0.8927 |

Removing it costs **0.070** — the largest single contribution we measured. Above 3.0 it
plateaus, so we take the lower end of the flat region.

**`popularity = 0.9`.**

| Weight | Score |
|---|---|
| 0.4 | 0.8743 |
| **0.9** | **0.8927** |
| 1.6 | 0.8824 |
| 2.5 | 0.8916 |

Too low and we lose the purchase-record prior. Too high and it starts overruling real
requirements. 0.9 is roughly one third of a matched requirement — enough to decide ties,
never enough to overturn evidence.

**`facet = 2.0`.** Colour, price, material and department. Deliberately below category
because these are derived by regex over the listing text and are occasionally wrong, whereas
the category path is structured data.

**`profile = 0.0`.**

| Weight | Held-out score |
|---|---|
| **0.0** | **0.8872** |
| 0.3 | 0.8828 |
| 1.2 | 0.8310 |

The anonymised tags are generic — "fit", "comfort", "durability" — and match most of the
catalog, diluting signals that do discriminate. Switched off; code kept in case the private
set ships sharper tags.

**`demote_shown = −4.0`.**

| Weight | Score |
|---|---|
| 0.0 | 0.8731 |
| **4.0** | **0.8927** |
| 12.0 | 0.8927 |

Worth **0.018**. The exact size does not matter (4 and 12 are identical) because it only has
to exceed the popularity term to push a known-wrong product below an unseen one. 4.0 is the
smallest value that reliably does that. It is a demotion, never a deletion — a deleted
product can never come back.

**`superseded_weight = 0.35`.** When the shopper changes their mind, earlier requirements
drop to 0.35 rather than 0. Roughly: three superseded requirements together still outweigh
one current one, so old evidence keeps contributing without being able to overrule the new
intent. Deleting instead of down-weighting once dropped a product from rank 1 to outside the
top 10.

---

## 5. The ask policy

```
value(a)     = expected_reveals(a) × bits(a)
expected_reveals(a) = min(2, remaining × P(a))
remaining    = max(0, 4 − requirements already known)
bits(a)      = mean log₂(50000 / matching products) for requirements of type a
```

**`4`** — the simulator's card always has at most four slots (two hard, two soft).

**`2`** — one reply returns at most two requirements.

**`P(a)`**, the measured attribute mix of all 760 public requirements: feature 0.53,
material 0.40, colour 0.08, style 0.025, size 0.014, use case 0.005. The unrestricted ask
matches *any* undisclosed requirement, so `P = 1.0`.

**`bits(a)`**, measured against the real catalog: colour 15.53, style 15.61, size 15.43, use
case 15.61, feature 6.84, material 6.43, overall mean 7.29.

Colour, style and size requirements are almost always **unique in 50,000 products** — but
they are rare. Feature and material requirements are common but weak, because so many are
boilerplate.

Putting it together:

```
unrestricted:  min(2, r × 1.00) × 7.29  = 14.6   ← wins
material:      min(2, r × 0.40) × 6.43  =  5.1
colour:        min(2, r × 0.08) × 15.5  =  2.5
```

The policy therefore **derives** the unrestricted ask rather than hardcoding it, and switches
to typed questions on its own once the shopper runs out. If the private set changes how much
a question reveals, the arithmetic re-derives a different answer with no code change.

---

## 6. The competition's own scoring, and what it implies

```
Hit@10     = conversations where we found it / N
MRR        = mean(1 / rank), 0 for a miss
MTTC       = mean first winning turn, misses counted as 11
Efficiency = clip((11 − MTTC) / 10, 0, 1)

TechnicalScore = 0.50 × Hit@10 + 0.30 × MRR + 0.20 × Efficiency
```

Substituting `MTTC = h·t̄ + (1−h)·11` into Efficiency gives an exact identity:

```
Efficiency     = Hit@10 × (11 − average winning turn) / 10
TechnicalScore = Hit@10 × (0.50 + 0.02 × (11 − t̄)) + 0.30 × MRR
```

Two consequences that shaped every decision in this project:

**Finding it bounds everything.** Since `MRR ≤ Hit@10` and `Efficiency ≤ Hit@10`:

```
TechnicalScore ≤ Hit@10
```

MRR and Efficiency are only discount factors on recall. Never trade a find for a faster one.

**Once you are finding it, rank beats speed by about 13×.** At 0.995 Hit@10, converting at
rank 1 on turn 2 instead of rank 8 on turn 1 gains `0.3 × 0.875 = 0.26` of MRR weight and
costs `0.02` of efficiency weight.

That ratio is why several tempting optimisations were rejected — they converted a fraction of
a turn sooner at a slightly worse rank, which is a bad trade.

It also explains the starter agent. Decomposing its published `h = 0.125, MTTC = 9.81` gives
an average winning turn of 1.48, which reconstructs its Efficiency to 0.119000 exactly. Its
ranking and its speed were fine. It simply missed 87.5% of conversations.

---

## 7. Optional LLM rescue

| Constant | Value | Reasoning |
|---|---|---|
| `llm_rescue_turn` | 5 | by turn 5 the shopper has revealed everything the card holds; if we have not converged, re-reading is the only thing left. Fires 3 times in 200 conversations. |
| `llm_max_tokens` | 3072 | **not a detail.** gpt-oss reasons before answering and the thinking shares the output budget. At 512, two thirds of calls failed — empty responses or JSON cut off mid-object. At 3072 with low reasoning effort: 24/24. |
| `temperature` | 0.0 | as close to reproducible as a hosted model allows. |
| retries | 1 | a refused or truncated structured response is usually transient. |

Measured cost across 200 conversations: 15 calls, 6,947 prompt + 2,829 completion tokens,
roughly **$0.002**.
