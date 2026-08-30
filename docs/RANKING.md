# Ranking: scoring, sorting, and picking the final 10

This is the stage that decides the score. [RETRIEVAL.md](RETRIEVAL.md) produces a
shortlist; this puts it in order and cuts it to ten.

Code: [`copilot/select10.py`](../copilot/select10.py).

---

## 1. The scoring formula

Every candidate gets one number. Higher wins.

```
score(product) =  10.0 × requirement coverage      (0 to 1)
               +   2.0 × colour / price / material / department agreement
               +   3.0 × category overlap
               +   0.9 × popularity                (0 to 1)
               −   4.0   if we already showed it and know it was wrong
```

Ties are broken by popularity.

### Why coverage is weighted 10

**Requirement coverage** is the share of what the shopper has told us that this product
actually satisfies, weighted so an overridden requirement counts 0.35 instead of 1.0.

It is weighted an order of magnitude above everything else because it is an order of
magnitude more informative. Satisfying one more of the shopper's stated requirements is not
a mild preference — the shopper *told us* that thing, and it is a verbatim quote from the
listing of the product we are hunting for.

### Why popularity is only 0.9, but matters far more than it looks

The hidden target is a **real purchase record**. Real purchases skew heavily towards
products many people bought. Ranking a fully-narrowed candidate set by popularity *alone*
reaches **Hit@10 0.945, MRR 0.861** — with no text matching at all.

So why weight it at only 0.9? Because by the time we are ranking, the requirements have
usually already isolated one product. Popularity is not doing the finding; it is breaking
ties among products that all satisfy the same requirements — and there, it is decisive.

Turning it down to 0.4 costs 0.019. Turning it up to 2.5 costs 0.012, because it starts
overruling real requirements.

### Why category is weighted 3.0

Category cannot be a hard requirement ([RETRIEVAL.md §2](RETRIEVAL.md#2-two-things-deliberately-kept-out-of-the-intersection)),
so it has to carry its weight in the ranking instead. Removing it entirely drops the score
from 0.8927 to **0.8210** — the single largest contribution of any one signal we measured.

### Why the shopper profile is weighted 0.0

We tried. `preference_tags` are generic — "fit", "comfort", "durability" — and match most
of the catalog, so they dilute the signals that actually discriminate.

| weight | held-out score |
|---|---|
| **0.0** | **0.8872** |
| 0.3 | 0.8828 |
| 1.2 | 0.8310 |

It is switched off and the code is kept, in case the private set ships sharper tags.

---

## 2. The −4.0 penalty, and the trap inside it

If we showed a product on an earlier turn and the conversation carried on, that product was
not the answer. Push it down.

This is worth 0.018 — real, and worth having.

**But the inference is only valid if the evaluator was actually scoring that turn.** In a
change-of-mind conversation it suppresses scoring until the new intent arrives, on turn 3
or 4. A product we showed on turn 1 of such a conversation may well be the right answer.

We got this wrong twice, and both times it was expensive:

- **Assuming turn 3** when the change of mind actually landed on turn 4 marked a correct
  turn-3 list "already proven wrong" and pushed the right answer out of the top 10 on turn
  4. Fixed by never assuming: we wait until we *see* the change-of-mind message.
- **Assuming the conversation was ordinary** when we failed to parse the opening line. On a
  reworded set this collapsed change-of-mind conversations to Hit@10 **0.067**. Fixed by
  holding the penalty until turn 5 whenever we did not understand the opening.

The rule we settled on: **only penalise a product when we can prove the list it appeared in
was scored.** When unsure, do not penalise. Losing 0.018 is much cheaper than burying the
correct answer.

The penalty is also a *demotion*, never a deletion. A deleted product can never come back;
a demoted one resurfaces if the candidate set turns out to be thin.

---

## 3. Sorting and cutting to ten

```python
scored.sort()                    # by score, then popularity
return scored[:10]
```

The evaluator keeps the first 10 valid, unique product ids and ignores the rest, so there
is no benefit to returning more.

**The list is never empty.** If nothing matched at all, we fall back to the most popular
products in the stated category, and failing that, the most popular products overall. Every
turn is scored, so an empty list throws away a free chance at no benefit.

---

## 4. We tried a different formula, and measured it

A parallel design ranked by two keys: **M**, an integer count of how many of the shopper's
sentences a product matches, then **S**, a weighted score. We ported its ideas in one at a
time:

| Idea | Score |
|---|---|
| **ours as shipped** | **0.8927** |
| weight requirements by rarity | 0.8927 — no change |
| sort by match count first, score second | 0.8927 — no change |
| both together | 0.8927 — no change |
| scale the category bonus by rarity | 0.8849 — worse |
| all of them at once | 0.8501 — much worse |

The first three are **exact** no-ops, and the reason is structural. We checked every turn of
a full run:

> Across all **330 turns**, the ten products we show always satisfy **the same number** of
> requirements as each other. 330 out of 330, no exceptions.

Counting matches cannot reorder a list where every entry has the same count. That is not
luck — our intersection has already *filtered* to products satisfying every requirement,
before ranking begins. Filtering is strictly stronger than counting whenever the right
answer really does satisfy everything, which here is 100% of the time.

The other design spends its primary sort key recovering a property we enforce as a
precondition.

---

## 5. Why rank matters more than speed

The score is `0.5 × Hit@10 + 0.3 × MRR + 0.2 × Efficiency`, and Efficiency is driven by how
many turns we take. Substituting the definitions gives an exact identity:

```
Efficiency = Hit@10 × (11 − average winning turn) / 10
```

Two consequences that shaped every decision here:

**Finding it beats everything.** Since MRR ≤ Hit@10 and Efficiency ≤ Hit@10, the whole
score is bounded by Hit@10. Never trade a find for a faster find.

**Once you are finding it, rank beats speed by about 13×.** At 0.995 Hit@10, converting at
rank 1 on turn 2 instead of rank 8 on turn 1 gains `0.3 × 0.875 = 0.26` of MRR weight and
costs `0.02` of efficiency weight.

That ratio is why several tempting optimisations were rejected: they converted a fraction of
a turn sooner at a slightly worse rank, which is a bad trade.

---

## 6. Where we currently stand

| | |
|---|---|
| Hit@10 | 0.995 |
| MRR | 0.694 |
| Rank 1 | 55% of conversations |
| Top 3 | 78% |
| Median rank when found | **1** |

**MRR is the remaining headroom.** Ranking a fully-narrowed set by popularity reaches MRR
0.861; we are at 0.694 because we often convert on turn 1 or 2, before the shopper has told
us everything, from a wider set. Closing that would mean answering *later* on purpose —
which the arithmetic in §5 says could genuinely pay, and which we have not tested.
