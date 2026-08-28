# Competition slides — interpretation and implications

## Executive summary

The challenge evaluates one locally imported Python agent on multi-turn product search. The agent must infer a hidden target from gradually disclosed preferences, maintain conversation state, handle preference overrides, ask clarifying questions, and return up to ten exact catalogue `parent_asin` values. Performance is measured on 800 private sessions using coverage, rank quality, and time to first hit.

The strongest implication for our system is that the catalogue is stable while the evaluated users and targets are not. Public and private sessions share the same frozen 50,000-product catalogue, but have no user or target overlap. Catalogue-derived indexes and representations should therefore transfer; assumptions fitted to the 200 public targets may not.

## What is fixed

- One frozen catalogue of 50,000 Amazon Clothing products.
- A local `Agent` interface with `reset` and `respond`.
- Up to ten turns and ten scored unique recommendations per turn.
- Exact `parent_asin` matching; generated explanations do not determine correctness.
- Four scenario proportions: 40% buying, 40% browsing, 15% intent override, and 5% boundary.
- Public/private separation by both user and target.
- A private evaluation set of 800 sessions.

## What the scoring rewards

TechnicalScore assigns 50% to Hit Rate@10, 30% to reciprocal rank, and 20% to conversational efficiency. Consequently:

1. Missing the target is the largest error.
2. Among successful sessions, ranking the target first matters substantially.
3. Extra turns have a smaller but still real cost.

This supports an agent that returns its highest-confidence candidate early, remembers rejected recommendations, and progressively explores the ranking while continuing to gather evidence. The policy should be explained as confidence-guided sequential retrieval, not merely as an output-width trick.

## What should generalize to private evaluation

Because the product catalogue is frozen, the following offline work is reusable:

- lexical and phrase indexes;
- catalogue document frequencies;
- product-field representations;
- category normalization;
- precomputed sparse or dense embeddings, if used;
- catalogue-grounded span validation.

The following are exposed to population shift and deserve stricter controls:

- popularity or review-count priors;
- constants selected only on the 200 public targets;
- learned rankers trained on public target IDs;
- assumptions about which product categories dominate evaluation.

Training data generated from non-public-target catalogue products is safer than fitting directly to the 200 public labels, provided final model selection is evaluated on target-disjoint folds.

## Dialogue requirements

The intended agent behavior is explicitly stateful:

- accumulate active constraints across turns;
- distinguish missing information from an expressed lack of preference;
- replace superseded constraints on intent override rather than concatenate contradictions;
- rerank after every useful answer;
- preserve safe, limited use of the anonymous aggregate profile.

The actual evaluator and API contract remain the authority for implementation details. The slides describe the intended behavior at a higher level and do not override executable simulator behavior.

## Recommended architecture story

Our clearest judge-facing description is:

> A grounded conversational retrieval agent that converts gradual dialogue into an auditable evidence state, retrieves from the frozen catalogue, updates its ranking after every answer or rejection, and uses guarded semantic extraction only when normal structured understanding fails.

That story maps naturally to four capabilities requested by the deck:

| Requirement | System capability |
|---|---|
| Search | Catalogue indexing, phrase retrieval, candidate backoff, and coverage ranking |
| Ask | Clarification selected to obtain new discriminative evidence |
| Remember | Per-session evidence, questions, overrides, and rejected products |
| Rank | Evidence-grounded reranking followed by confidence-guided disclosure |

## ML and LLM interpretation

The slides place keyword, dense, hybrid, query rewriting, and semantic reranking in scope, but do not require full-model training. They also state that the solution includes LLM semantic ranking and that teams bear external-service credentials, limits, and cost.

For this benchmark, semantic models should be guarded additions rather than ungrounded decision makers:

- keep exact catalogue evidence as the correctness anchor;
- use a recognition gate so clean simulator messages stay on the deterministic path;
- allow a local token tagger or optional LLM to recover product spans from unfamiliar wording;
- reject model-generated spans that are not copied from the visible message and attested in the catalogue;
- retain a deterministic fallback when the model, network, or credential is unavailable.

This combines a strong technical score with a credible innovation story without making external availability a single point of failure.

## Important limits on inference from the slides

- The slides guarantee a frozen catalogue and user/target-disjoint private sessions; they do not guarantee that public and private targets have identical popularity or category distributions.
- They say dialogue is simulated but do not state that private message templates must be byte-for-byte identical to public templates.
- They do not promise that paraphrasing will occur. Paraphrase experiments are robustness probes, not official-score estimates.
- “No hosted service is required” means the evaluator imports the agent locally. It does not guarantee outbound network availability.
- The high-level override example should be checked against the executable simulator before changing state-reset behavior.

## Judging strategy

Only 35% of event-level judging is Technical Execution. The remaining 65% rewards insight, impact, feasibility, and communication. A strong submission should therefore present:

- the large measured improvement over the starter baseline;
- ablations identifying which components actually matter;
- target-disjoint and population-shift robustness tests;
- failure analysis, including rejected ML approaches;
- deterministic fallback and external-cost controls;
- a concise explanation of why the design fits this benchmark rather than imitating a generic chatbot.
