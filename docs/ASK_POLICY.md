# Asking questions: which one is worth the most

Every turn the agent may ask the shopper about one attribute. This decides which.

Code: [`copilot/ask_policy.py`](../copilot/ask_policy.py).

---

## 1. Why this stage exists at all

On turn 1 of a browsing conversation we know only the category. Nothing else. Of the 200
public conversations, **90 start with no requirements at all** (browsing plus boundary).
Without a question, those are unwinnable — there is nothing to search with.

So asking is not a politeness feature. It is **the recall intervention**: it is how a
conversation with no information becomes one with four verbatim quotes from the target
product's listing.

---

## 2. Asking is free here

Worth stating plainly, because it inverts the usual advice.

In a real product, a clarifying question costs the customer patience, so it has to earn its
place. In this harness it costs nothing:

- one reply carries `message`, `ask_attribute` **and** `recommendations` together, so
  asking does not cost us a recommendation slot;
- the evaluator checks whether we found the product **before** the shopper replies, so a
  question never delays a win.

Every turn is a free shot at the answer plus a free question. So the policy never trades a
recommendation for a question — it always does both.

---

## 3. Choosing by expected information gain

The easy move is to hardcode the unrestricted ask (`"other"`), because the simulator's reply
function short-circuits its type check for it and hands back the first two undisclosed
requirements of any kind. That works — and it is a magic constant with no justification that
breaks the moment the private set changes its reveal policy.

So we compute it instead:

```
value(attribute) = how many requirements we expect to learn
                 × how much each one narrows the catalog
```

Both halves are measured, not guessed.

**How likely an attribute is to match something undisclosed** — the mix of the 760
requirements across the 200 public conversations:

| Attribute | Share |
|---|---|
| feature | 0.53 |
| material | 0.40 |
| colour | 0.08 |
| style | 0.025 |
| size | 0.014 |
| use case | 0.005 |

**How much a requirement of that type narrows the catalog** — mean `log₂(50000 / matches)`,
measured against the real catalog:

| Attribute | Bits |
|---|---|
| colour | 15.53 |
| style | 15.61 |
| size | 15.43 |
| use case | 15.61 |
| feature | 6.84 |
| material | 6.43 |

That table is interesting on its own. Colour, style and size requirements are almost always
**unique in the catalog** — 15.6 bits is a single product out of 50,000. They are gold. They
are also *rare*: only about 12% of requirements combined.

Feature and material requirements are common but much weaker, because so many are
boilerplate (`Imported` appears in 15,300 listings).

### What the arithmetic concludes

The unrestricted ask matches *any* undisclosed requirement, so its probability is 1.0:

```
unrestricted:   min(2, remaining × 1.00) × 7.29 bits  =  14.6
material:       min(2, remaining × 0.40) × 6.43 bits  =   5.1
colour:         min(2, remaining × 0.08) × 15.5 bits  =   2.5
```

The unrestricted ask wins, until the shopper runs out of things to say — at which point its
value drops to zero and the policy switches to typed questions on its own.

So the policy **derives** the right answer rather than assuming it. If the private set
changes how much a question reveals, the arithmetic re-derives a different answer without
anyone editing a constant.

---

## 4. The one hard rule: never stay silent

The contract allows `ask_attribute: null`. Using it is a trap.

The simulator answers a null question with *"Those options are not quite right yet. Ask me
about one specific attribute."* — a turn that reveals nothing whatsoever.

Our first version stopped asking once the candidate set got small, on the reasoning that
there was nothing left to learn. That reasoning is wrong: **a small candidate set is not
evidence that the right product is in it.** It only means the requirements we happen to hold
intersect tightly — which is exactly what happens when we hold too few of them.

Measured cost: **6 of 200 conversations deadlocked to turn 10**, repeating the same wrong
list while the shopper repeated the same nudge.

The rule now is simply: always ask. A typed question that whiffs costs nothing more than
silence, and might land.

The escape hatch survives as `AskConfig.allow_null_ask`, default off.

---

## 5. Reading the shopper's replies

The policy also listens to what comes back, and records it in the session graph:

| The shopper says | We conclude |
|---|---|
| *"For that, what matters is: A; B."* | two new requirements |
| *"...what matters is: A; A."* (repeated) | they have run out — stop asking about it |
| *"I don't have an additional preference for X."* | attribute X is exhausted, never ask again |
| *"I don't have a preference for X; use your judgment."* | this shopper deflects — expect it again |
| *"Actually, ignore my earlier preference..."* | change of mind, handled separately |

The repeated-phrase case is a genuinely useful signal. When a product's listing is too thin,
the simulator pads its card by duplicating an earlier entry, so the reply repeats itself
word for word. That duplication is a reliable "there is nothing more" marker, and we detect
it rather than burning turns discovering it.

---

## 6. What it costs in turns

| Conversation type | Average turns to find the product |
|---|---|
| buying | 1.29 |
| browsing | 1.29 |
| boundary | 1.60 |
| intent_override | 3.63 |
| **overall** | **1.655** |

Change-of-mind conversations are slower for a reason outside our control: the evaluator
refuses to record a win until the new intent arrives, on turn 3 or 4. 3.63 is close to the
floor.
