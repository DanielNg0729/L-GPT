# What the simulated customer actually does

**Question:** is the conversation in the public set semantic, or is it exact word
matching?

**Answer:** it is exact word matching. Not mostly — entirely. Every piece of information
the simulated customer discloses is a **literal substring of the target product's own
catalog row**. Measured: **760 out of 760** revealed constraints, across all 200 public
sessions.

This is not a criticism of the kit. The competition specification says so plainly:
*"Customer messages are simulated from a hidden intent card derived from product
metadata; the source dataset does not contain real shopping conversations."* This
document just measures what that means in practice, because it determines the entire
retrieval design.

Reproduce every number here with the scripts in `analysis/` (outside this repo):

```bash
python analysis/leakage_audit.py     # leakage, selectivity, degenerate rows, fill rates
python analysis/ceiling_probe.py     # the achievable ceiling
python analysis/rank_study.py        # which ranking signal separates the target
```

They import the **real** evaluator functions and run against the **real** 50,000-row
catalog. Nothing is mocked.

---

## 1. Where the customer's words come from

`intent_card()` in the evaluator builds the hidden card mechanically from the target
product's own fields:

```python
candidates = [material_regex_hit, "color: <hit>", *features, *details, f"budget around ${price}"]
hard_constraints = candidates[:2]
soft_preferences = candidates[2:4] or candidates[:1]
```

`_clean_constraint()` only collapses whitespace and truncates at 180 characters. So a
disclosed constraint is a **literal feature bullet or detail line copied out of the
target row**.

There are five utterance templates, and that is all:

| Template | Payload |
|---|---|
| `I'm looking for {category}. A key requirement is: {hard[0]}.` | buying, turn 1 |
| `I'm looking for {category}. {soft[-1]}` | intent_override, turn 1 |
| `I'm looking for {category}, but I'm still exploring.` | browsing / boundary, turn 1 — no payload |
| `For that, what matters is: {A}; {B}.` | the reply to any ask |
| `Actually, ignore my earlier preference. What I need is: {hard[0]}.` | the override |

The customer never paraphrases, never negates, never compares two items, and never uses
a pronoun. There is no anaphora to resolve and no implicit intent to infer.

### Worked example — `public_0027`

```text
turn 1  "I'm looking for Women Jeans. A key requirement is: cotton."

the full hidden card, as the simulator will disclose it:
  cotton                                    appears in  9,804 of 50,000 rows
  73% Cotton, 25% Polyester, 2% Spandex     appears in      7 rows
  Imported                                  appears in 15,314 rows
  Zipper fly with button closure            appears in     16 rows

AND of all four  ->  2 rows, one of which is the target
```

Nothing here needs to be *understood*. It needs to be **intersected**.

---

## 2. The measurements

### Verbatim leakage — 100%

| | |
|---|---|
| Non-synthetic constraints across 200 sessions | 760 |
| Verbatim substrings of the target row | **760 (100.0%)** |

Two of the four card slots are *synthesised* rather than quoted — `"color: grey"` and
`"budget around $12.99"`. Those are handled as structured facets, not phrases. Every
other slot is a copy.

One detail is load-bearing: matching must be **punctuation-insensitive**. The evaluator
renders the `details` dict as `"Department Womens"` while the card quotes it as
`"Department: Womens"`. Ignoring that costs 5 of the 760; handling it reaches 100%.

### Selectivity — bimodal, which is why naive BM25 fails

| | |
|---|---|
| Constraints unique in the catalog (df = 1) | 23.6% |
| df ≤ 10 | 28.7% |
| **Median df** | **2,403** |
| df > 1,000 | 53.8% |

The distribution has two humps. Roughly a quarter of constraints are near-unique keys;
more than half are pure boilerplate:

```text
"Imported"        13,994 rows        "hand wash only"   4,999 rows
"machine wash"     9,030 rows        "100% cotton"      2,965 rows
"pull on closure"  7,144 rows        "100% polyester"   2,483 rows
"rubber sole"      5,685 rows        "zipper closure"   2,479 rows
```

This explains the starter agent's 0.125 exactly. It builds `" OR ".join(terms)`, so a
target identified by a 7-row phrase is drowned by a 14,000-row one. **AND, don't OR** —
and when the AND must give something up, give up the highest-df member first.

### The ceiling — 0.945 hit rate from exact matching alone

Taking the AND of everything the customer will eventually disclose:

| | |
|---|---|
| Pool contains the target | **100.0%** of sessions |
| Median pool size | **1** |
| Pool is exactly 1 row | 60.0% |
| Pool ≤ 10 (a guaranteed hit) | 74.0% |
| Sessions ever needing a backoff drop | 0.0% |
| Pool ranked by popularity → HitRate@10 | **0.945** |
| Pool ranked by popularity → MRR | **0.861** |

No embeddings. No LLM. No semantics.

### Information arrives on turn 2, not turn 1

Turn-1 payload alone is nearly worthless — a guaranteed hit in **8%** of sessions:

| Scenario | Turn-1 payload | Guaranteed hit@10 from turn 1 |
|---|---|---|
| intent_override | a full feature bullet | 0.367 |
| buying | usually one bare word (`cotton`) | 0.062 |
| browsing | category only | 0.000 |
| boundary | category only, plus a deflected ask | 0.000 |

Note the inversion: an *"intent override"* session — nominally the hard scenario —
hands over roughly six times more retrievable text on turn 1 than a *"buying"* session
does. **Routing on the scenario label puts the richer session in the thinner track.**
The agent routes on how much constraint text it actually holds instead.

The corollary is that **the ask policy is a recall intervention, not a speed one**.
Without a question, 90 of the 200 sessions (browsing + boundary) start with nothing but
a category.

### Floors imposed by the data

| | |
|---|---|
| Catalog rows yielding < 2 unique constraints | 0.53% |
| Rows with a non-null price | 21.1% |
| Rows with a non-empty description | 52.2% |
| Rows with ≥ 4 feature bullets | 69.9% |
| Rows with no feature bullets at all | 10.4% |
| Distinct feature bullets / all feature bullets | 0.543 |

When a row is too sparse, `soft_preferences` falls back to a duplicate of
`hard_constraints[0]` and the customer's reply repeats itself verbatim:

```text
"For that, what matters is: Adjustable strap; Adjustable strap."
```

That duplication is a reliable end-of-card signal, and the agent detects it. It also
bounds what any retriever can do: 0.53% of rows carry no distinguishing text at all.

---

## 3. What this means for the design

Each of these turned into a component, documented separately:

1. **Exact matching carries the competition.** In production a dense model trained on
   purchase logs would carry this weight. Here no behavioural data exists and the shopper
   quotes the catalog, so exact matching takes its place.
   → [TEXT_MATCHING.md](TEXT_MATCHING.md)
2. **Intersect, do not score.** Keeping only products that satisfy every stated
   requirement finds the right answer 100% of the time, with a typical set size of one.
   → [RETRIEVAL.md](RETRIEVAL.md)
3. **Query expansion and commonsense enrichment are the wrong tools.** Both widen a query
   that already contains the answer. Removed.
   → [ARCHITECTURE.md](ARCHITECTURE.md#why-query-expansion-and-cosmo-were-removed)
4. **Ask early, and ask unrestricted.** Asking is free here — one reply carries
   recommendations *and* a question, and the win is checked before the shopper replies.
   → [ASK_POLICY.md](ASK_POLICY.md)
5. **Popularity is a real signal, because the target is a real purchase.** Ranking a
   narrowed set by popularity alone reaches Hit@10 0.945.
   → [RANKING.md](RANKING.md)
6. **Meaning-based retrieval earns nothing here — and we can prove it.** With the
   latent-semantic channel on: 0.8928. Off: 0.8927. That is not a broken channel, it is a
   measurement of how much semantic content this data contains.

---

## 4. Honest disclosure, and the risk to the private set

This analysis reads the evaluator's mechanics and designs against them, so it should be
stated plainly rather than buried. The competition ships the simulator source in the
participant kit; understanding it is expected work, and every conclusion here is
reproducible from the scripts named above.

The stated risk is that the private set differs. The specification reserves the right to
paraphrase: *"If natural-language paraphrasing is added by the organizer, it cannot
decide correctness."* Three points on that:

- **The reveal policy lives in the evaluator, not in a paraphraser.** The same
  information would still arrive; only exact *matching* would degrade.
- **It is measured, not hoped for.** `harness/paraphrase_stress.py` rewords the
  simulator's output and re-scores: 0.8927 clean, **0.8440** with the carrier sentence
  reworded, **0.8136** with synonym swaps on top. Recall holds up far better than the
  score — 0.995 -> 0.985 -> 0.950 — so we still find the product, just lower down.
  Turning the optional LLM rescue on recovers the reworded case to **0.8623** at
  hit@10 0.995.
- **The hedge that works is BM25F, not the semantic channel.** With LSA on, the reworded
  score is 0.8403 against 0.8440 with it off. Earlier drafts of this document credited
  LSA as the paraphrase hedge; the measurement says otherwise.
- **That stress test found a real bug.** Reworded, `intent_override` collapsed to
  hit@10 0.067 — when no template matched we assumed hits were scored from turn 1, so we
  marked the *correct* product "proven wrong" during the turns the evaluator was not
  scoring, and buried it. Fixed by detecting a change of mind from cue words, and by
  holding demotion until turn 5 when the opening line is unrecognised.
- **Nothing is keyed to a specific sample.** No per-sample rules, no use of
  `difficulty_bucket`, and all tuning was gated on a held-out split (0.8872 held-out vs
  0.8951 dev).

The deeper point is a known one in the conversational-recommendation literature: a
policy tuned against a user simulator overfits that simulator. What protects the design
here is that the *architecture* — hybrid retrieval, conjunctive filtering, cascade
ranking, information-gain clarification — is the shape production systems use. Only the
channel weights are tuned to this harness's reality.
