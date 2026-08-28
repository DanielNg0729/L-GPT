# Internal Robustness Benchmark

This analysis is separate from the submitted agent. It adds no runtime cost, model dependency, or network requirement to the shipped deterministic path.

## What it measures

- **Population variance:** 5,000 stratified bootstrap resamples of the released 200 sessions. TechnicalScore: `0.967550`; 90% interval: `0.957550`–`0.974900`; SD: `0.005399`.
- **Organizer uncertainty:** existing counterfactual probes and contract analysis. It does not claim to measure performance on a truly different catalogue or intent generator—no such dataset is available locally.

Risk scale: **1** = invariant/confirmed; **2** = degrades but remains viable; **3** = can become detrimental or relies on an unconfirmed population assumption. Lower is better.

## Ranked component ledger

| Component | Population risk | Organizer risk | Overall | Evidence |
|---|---:|---:|---:|---|
| No profile prior (W_PROFILE = 0) | 1 | 1 | 1 | Profile overlap was harmful on held-out data, so the shipped agent deliberately does not use it. |
| Rejection-feedback demotion | 1 | 1 | 1 | A later turn proves the prior list did not contain the target; override handling clears stale negatives. |
| Safety envelope and valid-ID fallback | 1 | 1 | 1 | respond() cannot raise; invalid API/network output falls back to the previous local ranking. |
| Session ledger | 1 | 1 | 1 | The interface confirms messages are turn-local; maintaining history is required, not population-tuned. |
| Catalog-grounded mining | 2 | 2 | 2 | Dictionary segmentation is stable under template changes, but still needs catalogue-attested lexical fragments. |
| Category evidence weight | 2 | 2 | 2 | Broad optimum in the weight sweep, but category text and its mapping are simulator/catalogue dependent. |
| FTS5 multi-rung recall ladder | 2 | 2 | 2 | POOL, DF cap and rung order are tuned, but graceful AND→OR backoff survives exact-match misses. |
| Fuzzy longest-attested-substring resolution | 2 | 2 | 2 | Held-out improvement (+0.0081) fixes formatting/synthetic-field drift; it cannot recover semantic paraphrases. |
| IDF-weighted coverage ranker | 2 | 2 | 2 | Weights were tuned but survive held-out evaluation; it loses discriminative power when wording becomes semantic. |
| Information-ordered probe policy | 2 | 2 | 2 | Order and reachable attributes were measured on this simulator; typed rotation remains a viable fallback. |
| Template extraction channel | 1 | 3 | 3 | No population learning, but template-only extraction fell from 0.8097 clean to 0.3513 under light paraphrase; hybrid fallback contains the damage. |
| Exact phrase/provenance thesis | 3 | 3 | 3 | Core retrieval assumption: customer constraints derive verbatim from the target document. Strong on this harness, brittle to intent-card paraphrase. |
| Optional Groq tie-breaker (OFF by default) | 3 | 3 | 3 | +0.000124 on public data, but needs network, credentials and a population-specific tie set. It is excluded from shipped cost and official fallback. |
| Popularity prior (W_POP = 0.35) | 3 | 3 | 3 | Largest gain on this purchase-derived population; relies on rating_number tracking purchase likelihood in private data. |
| Sequential disclosure (top-1 through turn 9) | 3 | 3 | 3 | Current source returns one recommendation until turn 10. It relies on the evaluator ending a session on any hit and changes the stated Top-10 shopping behaviour. |

## Measured uncertainty probes

| Condition | TechnicalScore | Interpretation |
|---|---:|---|
| Current deterministic reference | 0.967550 | Current source with network/LLM disabled. |
| Typed rotation; `other` unavailable | 0.828452* | Earlier pipeline probe: the dialogue strategy remains viable without the `other` quirk. |
| Hybrid extraction, light paraphrase | 0.659230* | Substantial degradation, but the catalogue-grounded fallback retains a meaningful floor. |
| Template-only, light paraphrase | 0.351322* | Adversarial phrasing makes the template channel unsafe as a standalone design. |
| Optional Groq tie-breaker | 0.910643* | Measured before the current sequential-disclosure change; excluded from the shipped scoring path. |

* These values are from earlier-stage robustness probes or a prior disclosure policy, so they establish failure direction rather than a final-agent score. Re-running these perturbations against the final agent is the next empirical priority.

## Bottom line

The pipeline is statistically stable against resampling of the released population, but that is not the same as cross-dataset generalization. The main non-transfer risks are the exact-provenance premise, the popularity prior, and the current sequential-disclosure policy. Keep the deterministic offline core as the shipped agent; treat Groq and any new learned/population-derived feature as experimental until it clears a separate held-out or external-catalogue benchmark.
