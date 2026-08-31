# Catalogue & Session Data Profile

Early-phase research (2026-08-26, before the agent existed): a structural profile of the
frozen 50,000-product catalogue and the 200 public sessions, produced by the scripts in
[`experiments/profile/`](../../experiments/profile/). Two of its findings became
load-bearing components of the shipped agent — the three-line probe policy (§6) and the
popularity prior (§7).

## 1. Top-level fields — only 11

| field | present % | empty/null % | types |
|---|---|---|---|
| `parent_asin` | 100.0 | 0.0 | str |
| `title` | 100.0 | 0.0 | str |
| `features` | 100.0 | 10.44 | list |
| `description` | 100.0 | 47.77 | list |
| `price` | 100.0 | **78.95** | NoneType, float, str |
| `categories` | 100.0 | 0.0 | list |
| `details` | 100.0 | 3.34 | dict |
| `average_rating` | 100.0 | 0.0 | float |
| `rating_number` | 100.0 | 0.0 | int |
| `store` | 100.0 | 0.63 | str, NoneType |

## 2. `details` — the real attribute space: 287 distinct keys

- Keys per product: median **4.0**, p90 **6**, max **30**
- Keys covering >10% of the catalogue: **7** · >1%: **22** · long tail under 100 items: **234**

Highest-coverage keys: `Date First Available` 93.8%, `Department` 87.2%,
`Item model number` 55.5%, `Package Dimensions` 54.1%, `Manufacturer` 47.0%. The
shopper-meaningful attributes are sparse: `Color` 4.9% (1,165 values), `Brand` 4.7%,
`Material` 4.1% (463 values), `Style` 3.5%, `Size` 1.9%, `Fabric Type` 0.4%,
`Fit Type` 0.4%, `Occasion` 0.4%. Attribute-key filtering therefore cannot carry
retrieval — free-text `features`/`title` matching has to.

## 3. Categories

- Depth: median **5.0**, max **8**; distinct leaf pairs: **1,105**
- Level-2 split: Women **26,406** · Men **9,901** · Novelty & More 3,376 · Girls 1,716 ·
  Boot Shop 1,131 · Boys 1,101 · Baby 1,031 — the catalogue is 53% women's products.

## 4. Numeric shape

- **`price` is null for 79% of the catalogue** → a budget constraint can only exist for
  the remaining fifth (see §7: but 89% of *targets* have one).
- `price` (n=10,410): median $22.88, p90 $80.10, max $4,119
- `average_rating`: median 4.2, mean 4.09
- `rating_number`: median **12**, p75 59, p90 260, p99 3,332, max 408,371 — an extreme
  long tail
- title words: median 11 · features per product: median 5 · description words: median 8
  (47.8% empty)
- distinct `store` values: 19,855 (69.3% have exactly one product)
- regex extractability: material 57.1% · color 38.8% · size-hint 47.8%

## 5. What the simulated customer actually reveals

Constraint-bucket distribution through the evaluator's own `classify_constraint`:

| bucket | random sample % | 200 public targets % |
|---|---:|---:|
| `feature` | 52.3 | 50.5 |
| `material` | 28.4 | 37.8 |
| `color` | 12.3 | 7.5 |
| `style` | 4.6 | 2.4 |
| `size` | 1.9 | 1.4 |
| `use_case` | 0.5 | 0.5 |
| `budget` | 0.1 | 0.0 |

Distinct buckets per intent card: mostly 2–3 (public set: 1 bucket ×18, 2 ×137, 3 ×45).

## 6. Expected value of each probe → the three-line ask policy

[`probe_expected_value.py`](../../experiments/profile/probe_expected_value.py)
simulates `customer_reply()` exactly: if the agent asks attribute X, what is the
probability the customer reveals at least one constraint, and how many on average?

| ask_attribute | P(reveal ≥1) | constraints / turn |
|---|---:|---:|
| `other` | **100.0%** | **2.00** |
| `feature` | 96.0% | 1.73 |
| `material` | 76.5% | 1.43 |
| `color` | 25.5% | 0.29 |
| `style` | 9.0% | 0.10 |
| `size` | 4.5% | 0.05 |
| `use_case` | 2.0% | 0.02 |
| `budget` | **0.0%** | 0.00 |
| `category` | **0.0%** | 0.00 |
| `brand` | **0.0%** | 0.00 |

The probe policy collapses to three lines: **`other` → `other` → `feature`/`material`**.
The remaining seven attributes are essentially dead turns. `budget` pays 0% even though
89% of targets *have* a price, because `intent_card()` appends the price after slicing
`candidates[:4]` — it almost never enters the constraint list. (The shipped agent's
`DEAD_ATTRIBUTES` and probe order encode exactly this table.)

## 7. Target-vs-catalogue prior — the single biggest lever

[`target_priors.py`](../../experiments/profile/target_priors.py) compares the 200
ground-truth products with the other 49,800:

| statistic | TARGETS (200) | CATALOGUE (49,800) |
|---|---:|---:|
| `rating_number` median | **6,846** | **12** |
| `rating_number` p25 | 986 | 3 |
| has `price` | **89.0%** | **20.5%** |
| `average_rating` median | 4.4 | 4.2 |
| `features` count median | 8 | 5 |

Targets are **actual best-sellers**; three quarters of the catalogue is long-tail stock
that essentially never sold. Popularity prefiltering cuts the pool by an order of
magnitude at almost no recall cost:

| filter | pool | % catalogue | target recall |
|---|---:|---:|---:|
| none | 50,000 | 100% | 100% |
| `rating_number ≥ 50` | 13,715 | 27.4% | **96.5%** |
| `rating_number ≥ 100` | 9,381 | 18.8% | 95.0% |
| `rating_number ≥ 300` | 4,535 | 9.1% | 85.5% |
| `price` not null | 10,410 | 20.8% | 89.0% |
| `rating_number ≥ 1000` | 1,533 | 3.1% | 74.5% |

**Recommendation as written at the time:** use `log1p(rating_number)` as a soft additive
prior with a tunable coefficient rather than any hard threshold. (This is the prior the
shipped ranker uses, later made self-calibrating.)

> Overfit warning, also as written at the time: these are statistics over the 200 public
> sessions — but they are a *structural* property of how the organizer samples (targets
> come from real purchase records; a product that sold has reviews and a price), so they
> were expected to hold on the 800 private sessions. The independent population folds
> later confirmed the direction, and the inverse-popularity fold (0.868 vs 0.954)
> measures exactly what this bias costs when it is deliberately broken.
