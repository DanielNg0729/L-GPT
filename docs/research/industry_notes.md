# Industry Practice Notes — Conversational Commerce Search

Design-grounding notes: how production conversational shopping systems handle the
problems this track poses, and which of those practices the agent adopts or
deliberately departs from. These are working notes distilled from public engineering
practice, not a literature survey; no proprietary information.

## 1. Clarifying questions are budgeted, not free

Production shopping assistants treat every clarifying question as spent user patience:
ask too little and results stay broad, ask too much and the shopper leaves. The common
pattern is to ask only while the candidate pool is too wide to commit, and to prefer
open questions early (they harvest the most information per turn) before narrowing to
specific facets. **Adopted:** the ask policy asks the open `other` probe while it pays,
then the highest-yield facets (`feature`, `material`) — a policy the data profile
(§6) derives from measured expected value per question, and the MTTC term in the
score mirrors the engagement cost real systems manage.

## 2. Dialogue state is slots plus history, and overrides must reset it

Commercial dialogue systems keep an explicit structured state (facet slots, hard
constraints, rejected items) rather than re-deriving intent from raw history each
turn; and a change of mind must invalidate stale constraints or the assistant argues
with the customer. **Adopted:** the session ledger (evidence with tiers, rejection
history) and the override rule that wipes rejections when intent changes — the
measured failure of missing that rule (retaining stale evidence) is a hit-rate loss,
the same failure mode CRM teams describe for slot-carryover bugs.

## 3. Popularity and sales priors carry cold-start ranking

E-commerce ranking stacks lean heavily on engagement/sales priors (best-seller rank,
review volume) whenever query evidence underdetermines the answer — most retail
queries are broad, and demand is extremely long-tailed. **Adopted, with its cost
measured:** the `log1p(rating_number)` prior; the inverse-popularity population fold
(0.868 vs 0.954) quantifies exactly how much of the score rides on the demand
distribution, which is the number a deployment against different demand would need to
revisit.

## 4. Lexical-first retrieval, semantic assist — not the other way round

Despite the dense-retrieval literature, production product search remains anchored on
inverted-index lexical matching with strict attribute filters, because product queries
are dominated by exact tokens (brand, model, material, size) and because a lexical
core is cheap, debuggable, and never hallucinates. Semantic components are added where
lexical fails: query rewriting, synonym/attribute normalization, and reformulating
vague requests. **Adopted:** FTS5 conjunctive retrieval as the core; the LLM appears
only as a normalizer (deparaphraser: reworded value → catalogue term, verified against
the catalogue before use) — the same "rewrite into the index's vocabulary" role query
rewriters play in production, with a stricter guarantee (the catalogue attests every
admitted term).

## 5. Assistants must not invent inventory

The hard guardrail in every deployed shopping assistant: never show a product that is
not in the catalogue, never assert an attribute the listing does not carry.
**Adopted as the system invariant:** "the model proposes, the catalogue disposes" —
no layer, learned or hosted, can inject a phrase or a product the frozen catalogue
does not contain.

## 6. Where the benchmark and the real product diverge

Two deliberate departures, both documented in the report: sequential disclosure
optimises the benchmark's MRR and is not how a real surface would page results (the
`full` mode exists for that, and the demo uses it); and questions real shoppers value
(`budget`, `brand`) are dead turns against this simulator (measured 0% payout), so the
scored agent never asks them while the demo re-enables them. Keeping both modes in
the same codebase, switched by configuration, is how the team reconciled scoring
against the benchmark with demonstrating a credible product.
