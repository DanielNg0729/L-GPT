# Experiment-by-Experiment Decision Log

This is the frozen decision trail for the submitted agent. Each row corresponds to one
versioned experiment program. Results are taken from the committed scripts, their recorded
JSON outputs, and the detailed findings ledger. No experiment was rerun while preparing
this document.

Status uses the following meanings:

- **Adopted**: the mechanism remains in the final pipeline.
- **Rejected**: measured evidence did not justify including the mechanism.
- **Supported**: validates a shipped choice without adding a separate mechanism.
- **Superseded**: informed a later configuration or protocol but is not the final form.
- **Invalidated**: retained because a later audit found a methodological defect.
- **Infrastructure**: creates a reproducibility, training, optimization, or audit asset.
- **Diagnostic**: explains the task or a failure mode and guides a later choice.

Historical scores describe the configuration at the time of that experiment. They are not
claims about the final trial 38 agent unless the row explicitly says so.

## A. Task structure and lexical retrieval

| Experiment | Recorded result | Ruling | Effect on final design |
|---|---|---|---|
| [`01_catalog_and_leak.py`](scripts/01_catalog_and_leak.py) | Established that customer constraints are generated from the target's own catalogue metadata. | Diagnostic | Framed the problem as provenance recovery, making exact grounded matching the primary approach. |
| [`02_information_budget.py`](scripts/02_information_budget.py) | Enumerated the simulator's disclosure channel and found `other` exposes the most useful remaining evidence. | Adopted | Set the clarification strategy and removed speculation about opaque dialogue behavior. |
| [`03_retrieval_ceiling.py`](scripts/03_retrieval_ceiling.py) | Showed that a small set of literal constraints can retrieve the target at high depth; residual error is mostly ranking. | Supported | Directed effort toward evidence quality and reranking rather than broad semantic retrieval. |
| [`04_ablation.py`](scripts/04_ablation.py) | Measured separate gains from state, probing, phrase queries, and coverage ranking on the official evaluator. | Adopted | Established the deterministic core: stateful dialogue, phrase retrieval, and coverage scoring. |
| [`05_failure_dense_robustness.py`](scripts/05_failure_dense_robustness.py) | Dense bi-encoder fusion reduced tuning TechnicalScore by 0.047 and template-only extraction was brittle. | Rejected | Excluded dense fusion; motivated catalogue-grounded mining as the non-template fallback. |
| [`06_grounded_mining.py`](scripts/06_grounded_mining.py) | Catalogue-attested n-gram mining and the category-channel correction improved robustness and nominal behavior. | Adopted | Added the principal deterministic fallback evidence channel. |
| [`07_rank_refinement.py`](scripts/07_rank_refinement.py) | Coordinate ascent improved the then-current public score by 0.037. | Superseded | Identified useful ranking dimensions, later replaced by joint trial 38 optimization. |
| [`08_closing_the_gaps.py`](scripts/08_closing_the_gaps.py) | Fuzzy phrase resolution added 0.003; per-field BM25 tuning regressed held-out performance by 0.013. | Partly adopted | Kept bounded phrase resolution and rejected per-field-only tuning. |
| [`09_disclosure_policy.py`](scripts/09_disclosure_policy.py) | Rejection feedback improved the score by 0.010; narrow disclosure exposed a legitimate policy and metric tradeoff. | Partly adopted | Led to rejection demotion and later sequential-disclosure analysis. |
| [`10_retrieval_structure.py`](scripts/10_retrieval_structure.py) | Field restrictions and full category paths were null; sparse rank fusion slightly regressed. | Rejected | Kept a simple global lexical index and avoided extra retrieval branches. |
| [`11_cross_encoder.py`](scripts/11_cross_encoder.py) | Local cross-encoder reranking reduced held-out score by 0.030. | Rejected | Excluded semantic reranking from the critical path. |
| [`12_rejection_decomposition.py`](scripts/12_rejection_decomposition.py) | Showed rejection feedback is valuable through ordering, not by shrinking the recommendation list. | Adopted | Demotes rejected candidates instead of deleting them. |
| [`13_rag_techniques.py`](scripts/13_rag_techniques.py) | Structure-aware chunking was null and RM3 pseudo-relevance feedback regressed by 0.004. | Rejected | Avoided unnecessary RAG-style query expansion and chunking. |
| [`14_target_prior.py`](scripts/14_target_prior.py) | Measured a large review-count target bias; alternative prior shapes gained only 0.003 within noise. | Diagnostic | Established popularity as a useful but population-sensitive tie signal. |
| [`15_adaptive_prior.py`](scripts/15_adaptive_prior.py) | Turn-adaptive prior decay regressed held-out performance by 0.005. | Rejected | Kept a non-decaying prior form. |
| [`16_generative_verification.py`](scripts/16_generative_verification.py) | Exact inversion of the generator did not improve ranking and exposed irreducible evidence ties. | Rejected | Focused tie analysis on observable candidate signals instead of generator simulation. |
| [`17_everything_else.py`](scripts/17_everything_else.py) | Length, rating, slot, stemming, prefix, proximity, ensemble, and negative-evidence variants did not generalize. | Rejected | Kept the scoring function compact and recorded the negative search space. |
| [`18_learning_to_rank.py`](scripts/18_learning_to_rank.py) | Tree learning-to-rank improved on its training half but regressed held-out score by 0.040. | Rejected | Ruled out public-set supervised ranking as a final mechanism. |

## B. LLM and learned tie-breaking alternatives

| Experiment | Recorded result | Ruling | Effect on final design |
|---|---|---|---|
| [`19_llm_rerank.py`](scripts/19_llm_rerank.py) | Groq listwise reranking within top coverage ties reduced end-to-end score by approximately 0.027. | Rejected | LLM reranking is disabled in the release and retained only for documented comparison. |
| [`20_tiebreak_accuracy.py`](scripts/20_tiebreak_accuracy.py) | LLM target-first accuracy was 41.2% within ties versus 57.4% for popularity. | Rejected | Confirmed the rejection was signal quality, not merely end-to-end variance. |
| [`20_robustness_benchmark.py`](scripts/20_robustness_benchmark.py) | Introduced the first component-level population and organizer-uncertainty grading framework. | Superseded | Seeded the later, stricter robustness audit and proxy construction. |
| [`21_synthetic_ltr.py`](scripts/21_synthetic_ltr.py) | Synthetic-session LTR tune and holdout effects disagreed and remained near zero. | Rejected | Did not reopen learned ranking despite much larger synthetic training data. |
| [`22_ml_tiebreak_accuracy.py`](scripts/22_ml_tiebreak_accuracy.py) | Learned models did not consistently beat popularity inside exact evidence ties. | Rejected | Kept one-dimensional popularity tie-breaking. |
| [`23_tie_crossencoder.py`](scripts/23_tie_crossencoder.py) | A task-specific fine-tuned tie cross-encoder did not generalize with available independent queries. | Rejected | Closed the fine-tuned cross-encoder direction. |
| [`24_tie_features.py`](scripts/24_tie_features.py) | Adding within-group normalized features progressively reduced held-out tie accuracy. | Rejected | Avoided feature-heavy learned tie models. |

## C. Dialogue policy, recall repair, and robustness

| Experiment | Recorded result | Ruling | Effect on final design |
|---|---|---|---|
| [`25_disclosure_policy_solver.py`](scripts/25_disclosure_policy_solver.py) | Solved the published reward and found one recommendation per early turn optimal under the contract. | Adopted | Defined the sequential disclosure schedule. |
| [`26_disclosure_validate.py`](scripts/26_disclosure_validate.py) | Confirmed the schedule on separate tuning and held-out halves; adaptive widening lost. | Adopted | Locked in one candidate on turns 1 through 9, then ten at turn 10. |
| [`27_learned_retrieval.py`](scripts/27_learned_retrieval.py) | Learned retrieval under the width-one objective did not improve reliable recall. | Rejected | Kept deterministic FTS5 candidate generation. |
| [`28_miss_autopsy.py`](scripts/28_miss_autopsy.py) | Traced the last public misses to synthesized or non-contiguous evidence, not a shallow candidate pool. | Diagnostic | Identified the specific category-evidence repair target. |
| [`29_hitrate_repair.py`](scripts/29_hitrate_repair.py) | Repaired non-contiguous category evidence, raising public HitRate from 0.990 to 0.995. | Adopted | Added component-aware category handling. |
| [`30_robustness_benchmark.py`](scripts/30_robustness_benchmark.py) | Measured four unseen-target scores from 0.9437 to 0.9554 and created shift and ablation tests. | Supported | Made unseen and population stress a selection criterion, not an afterthought. |
| [`31_paraphrase_stress.py`](scripts/31_paraphrase_stress.py) | Quantified controlled message and constraint rewrite failure modes. | Diagnostic | Justified a gated unfamiliar-wording fallback while keeping paraphrase outside the official score claim. |
| [`32_conditional_attribution.py`](scripts/32_conditional_attribution.py) | Isolated population dependence to the popularity coefficient; mining became essential under paraphrase. | Supported | Preserved mining despite negligible nominal ablation value and focused calibration on `W_POP`. |
| [`32_local_bert_phrase_verifier.py`](scripts/32_local_bert_phrase_verifier.py) | Tested a local MiniLM verifier after template failure. | Superseded | Informed the later scaffolding-tagger approach rather than a phrase-verification filter. |
| [`33_finetuned_bert_phrase_verifier.py`](scripts/33_finetuned_bert_phrase_verifier.py) | Fine-tuned phrase verification did not produce stable end-to-end improvement. | Rejected | Did not ship a learned phrase accept-or-reject classifier. |
| [`33_retune_validation.py`](scripts/33_retune_validation.py) | Retuned population-sensitive constants over public, unseen, and stress conditions. | Superseded | Provided robust starting regions for later joint optimization. |
| [`34_finalist_selection.py`](scripts/34_finalist_selection.py) | Applied a seven-condition no-regression rule to select a robust pre-Optuna configuration. | Superseded | Established the conservative selection discipline used again after v2 optimization. |
| [`35_override_safety.py`](scripts/35_override_safety.py) | Bounded worst rejection-feedback failure at 0.040 while preserving final HitRate. | Supported | Kept rejection feedback with explicit override guards. |
| [`36_disclosure_risk_adjudication.py`](scripts/36_disclosure_risk_adjudication.py) | Independently re-evaluated disclosure risk and corrected its population grade from P1 to P2. | Supported | Tempered the sequential-disclosure robustness claim without removing the policy. |
| [`37_population_shift_hardening.py`](scripts/37_population_shift_hardening.py) | Tie-only, percentile, capped, gated, and fusion priors improved adversarial populations by at most 0.003. | Rejected | Showed that changing the prior shape does not solve a directionally wrong population prior. |
| [`38_bayesian_prior_calibration.py`](scripts/38_bayesian_prior_calibration.py) | Explore-then-commit selection could not estimate the better prior arm within a private run. | Rejected | Excluded outcome-feedback bandits and other label-dependent adaptation. |
| [`39_population_detector.py`](scripts/39_population_detector.py) | Corrected reward alignment and estimated approximately 8,500 sessions per arm for reliable feedback selection. | Diagnostic | Motivated aggregate, label-free detection instead of arm exploration. |
| [`40_detector_driven_prior.py`](scripts/40_detector_driven_prior.py) | Retrieved-pool popularity calibration added 0.034 on inverse-population stress without labels. | Adopted | Added the self-calibrating popularity-prior scale. |

## D. Extraction robustness, BERT fallback, and optional APIs

| Experiment | Recorded result | Ruling | Effect on final design |
|---|---|---|---|
| [`21_llm_extraction_robustness.py`](scripts/21_llm_extraction_robustness.py) | Compared LLM extraction and grounded mining under controlled paraphrase levels. | Supported | Established the measurement framework for optional extraction, not the default path. |
| [`41_recognition_gate.py`](scripts/41_recognition_gate.py) | All official message forms bypassed optional extraction; altered wording reached the fallback gate. | Adopted | Guarantees BERT and LLM extraction cannot affect a clean official-form run. |
| [`42_mining_paraphrase_floor.py`](scripts/42_mining_paraphrase_floor.py) | Tuned mining under clean and transformed messages; later joint search superseded the constants. | Superseded | Retained the mining mechanism and informed its final parameter range. |
| [`43_llm_extraction_hybrid.py`](scripts/43_llm_extraction_hybrid.py) | Gated hybrid preserved clean score with zero calls and recovered 21.6% to 68.8% of paraphrase loss. | Supported, optional | Retained optional Groq extraction behind explicit flags and hard span validation. |
| [`44_llm_failure_modes.py`](scripts/44_llm_failure_modes.py) | Verified 16 API failure modes and fixed time-budget and zero-yield circuit-breaker defects. | Adopted, optional | Ensures optional API failure falls back to deterministic ranking. |
| [`45_ml_constraint_likeness.py`](scripts/45_ml_constraint_likeness.py) | Learned constraint filtering reached 0.637 held-out accuracy and degraded as influence increased. | Rejected | Kept mining unfiltered by a learned phrase score. |
| [`46_ml_soft_and_probe.py`](scripts/46_ml_soft_and_probe.py) | Soft learned weighting and a learned probe policy both regressed, including MTTC. | Rejected | Kept deterministic phrase weights and fixed probe order. |
| [`47_local_paraphrase_extractor.py`](scripts/47_local_paraphrase_extractor.py) | Local synthetic paraphrase extraction overfit transforms and reduced T1 score by 0.098. | Rejected | Rejected sequence-style local extraction trained only on synthetic paraphrases. |
| [`49_bert_scaffolding_tagger.py`](scripts/49_bert_scaffolding_tagger.py) | Pretrained BERT token tagging generalized better than the linear extractor; catalogue MLM adaptation added no value. | Adopted | Created the BERT scaffolding fallback that only removes filler before deterministic mining. |
| [`50_train_and_save_tagger.py`](scripts/50_train_and_save_tagger.py) | Trained and exported the selected tagger as a versioned model asset. | Infrastructure | Made the optional local fallback reproducible. |
| [`51_encoder_bakeoff.py`](scripts/51_encoder_bakeoff.py) | No alternate pretrained encoder delivered enough practical advantage to replace the selected tagger. | Supported | Kept the DistilBERT tagger as the local fallback. |
| [`52_tie_feature_dependency.py`](scripts/52_tie_feature_dependency.py) | The initial popularity-conditioning design was too coarse to support its conclusion. | Invalidated | Prevented a false claim that non-popularity features had been controlled correctly. |
| [`53_complementary_signal.py`](scripts/53_complementary_signal.py) | On ties where popularity fails, none of 16 visible catalogue features beat the permutation noise band. | Supported | Closed the search for an additional visible-field tie signal. |

## E. Joint optimization and independent validation

| Experiment | Recorded result | Ruling | Effect on final design |
|---|---|---|---|
| [`48_optuna_coarse.py`](scripts/48_optuna_coarse.py) | Broad search mapped joint sensitivity over public, proxy, and paraphrase objectives. | Superseded | Identified candidate regions but was not used for final selection after paraphrasing was ruled out. |
| [`54_optuna_validate.py`](scripts/54_optuna_validate.py) | Reweighted the v1 objective after the organizer clarification, but candidate selection included pruned trials. | Invalidated | Its candidate ranking was excluded from final selection. |
| [`55_optuna_official_v2.py`](scripts/55_optuna_official_v2.py) | Optimized completed trials on fixed official 200 plus fixed stratified private-like 800, without paraphrase in the objective. | Infrastructure | Produced reproducible conservative, balanced, and aggressive final candidates. |
| [`55_optuna_dashboard_v2.py`](scripts/55_optuna_dashboard_v2.py) | Exposed the isolated v2 study through a local dashboard. | Infrastructure | Made optimization state inspectable; it makes no performance claim. |
| [`56_optuna_exploit_v2.py`](scripts/56_optuna_exploit_v2.py) | Ran a separate 50/50 public-proxy reward to expose the exploration versus exploitation tradeoff. | Infrastructure | Prevented the two objectives from being silently mixed during selection. |
| [`57_independent_validation.py`](scripts/57_independent_validation.py) | Frozen candidates were tested on four untouched same-population folds and controlled population disturbances; trial 38 had a `+0.001213` mean delta. | Adopted | Selected balanced trial 38 and consumed these folds as validation evidence, not further tuning data. |

## F. Post-selection override robustness probe

| Experiment | Recorded result | Ruling | Effect on final design |
|---|---|---|---|
| [`58_override_replacement.py`](scripts/58_override_replacement.py) | Category-preserving evidence reset reduced intent-override HitRate from 1.000 to 0.933 on Official200 and from 0.983 to 0.858 on Unseen800. A broader cue catches more rewordings but also falsely fires on “Actually, I need cotton.” | Rejected | Keep accumulated target-derived evidence and clear only rejection state. Do not claim support for true semantic preference replacement. |
| [`59_override_opening_evidence.py`](scripts/59_override_opening_evidence.py) | Organizer source establishes that the intent-override opening’s second slot is `soft_preferences[-1]` from the target card. Giving it ordinary constraint weight produced 0.969600 on Official200 and 0.943250 on Unseen800, identical to the prior shipped score; weak weighting reduced Unseen800 to 0.942794. | Adopted | Parse the source-confirmed old-value slot as normal grounded evidence. It adds no API or model cost and cannot alter the measured clean scores. |
| [`60_override_focus_contradiction.py`](scripts/60_override_focus_contradiction.py) | On 800 source-faithful override sessions, full opening evidence scored 0.918631. On 800 counterfactual sessions where every old value was a catalogue-attested material absent from the target and never re-disclosed, it fell to 0.859962. Removing opening evidence scored 0.892825 under contradiction but 0.918156 under source-faithful data; neutralising only unconfirmed openings scored 0.892125 and 0.915956. | Supported diagnostic | Retain full opening evidence for the released simulator. The evidence is target-derived there. Do not ship a universal reset: it sacrifices source-faithful performance and cannot distinguish a real contradiction without a semantic conflict detector. |
| [`61_catalog_conflict_guard.py`](scripts/61_catalog_conflict_guard.py) | Tested the proposed literal pair search: same high-confidence family, both values attested, and zero catalogue co-occurrence. It preserved source-faithful score exactly, but removed 0/800 deliberately incompatible material values because another catalogue product supported every pair. | Rejected | Catalogue-wide co-occurrence measures product compatibility, not conversational replacement. It cannot identify which value a customer withdrew. |
| [`62_same_family_override_replace.py`](scripts/62_same_family_override_replace.py) | Replaced only unconfirmed opening evidence in the same high-confidence material, colour, or closure family as the later override. It removed 589/800 counterfactual collisions and improved 0.859962 to 0.888956, while changing source-faithful OverrideFocus800 from 0.918631 to 0.917956 and Official200 from 0.969600 to 0.969500. | Rejected for shipment | The selective rule is the strongest deterministic recovery tested, but it still deletes 56 legitimate source-faithful opening values and violates the no-regression selection rule. It defines the target behavior for a future semantic conflict detector. |
| [`63_gated_paraphrase_override.py`](scripts/63_gated_paraphrase_override.py) | A replacement cue on an unrecognised message was permitted to clear non-category evidence and add the stated new value. Its gate was blocked on all 464 Official200 and all 2,262 Unseen800 recognised messages. On a fixed paraphrased OverrideFocus800, however, it changed compatible score from 0.917181 to 0.798482 and contradictory score from 0.852800 to 0.761347. | Rejected for shipment | The no-public/no-Unseen trigger guarantee works, but deleting accumulated evidence is not a viable repair. Retain the experiment as a negative result and do not add this controller to the agent. |

## Final decision synthesis

The final agent is not a collection of individually optimized features. It is the smallest
combined pipeline that survived the following decision rules:

1. prefer verified literal provenance over semantic similarity when the simulator exposes
   target-derived strings;
2. keep a mechanism only if it improves the relevant condition or protects a known failure
   mode without harming official behavior;
3. keep unfamiliar-wording models behind a recognition gate so they cannot perturb official
   messages;
4. reject any learned ranker, dense retriever, or LLM reranker that failed on held-out or
   within-tie measurements;
5. localize population dependence to one calibrated coefficient rather than inventing
   private eligibility labels;
6. freeze final candidates before independent validation and do not retune on those folds.

The final pipeline is documented in the [root README](../README.md) and in
[`docs/design/architecture.md`](../docs/design/architecture.md).
