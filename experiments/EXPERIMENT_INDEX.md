# Experiment Registry

This registry is the compact navigator for every versioned experiment program. The complete
one-by-one result, ruling, and final-design impact is in
[`EXPERIMENT_DECISION_LOG.md`](EXPERIMENT_DECISION_LOG.md). Detailed measurements and
methodological corrections are retained in [`EXPERIMENT_FINDINGS.md`](EXPERIMENT_FINDINGS.md);
raw outputs are in [`results/`](results/).

Status definitions:

- **Shipped**: the resulting mechanism is present in the final agent.
- **Supported**: the experiment validates a shipped decision or documented claim.
- **Rejected**: measured evidence did not justify adoption.
- **Superseded**: useful at the time, but replaced by a later experiment or configuration.
- **Diagnostic**: characterizes the task or a failure mode without directly selecting code.
- **Infrastructure**: supports reproducibility, training, optimization, or monitoring.
- **Invalidated**: retained because a methodological defect was found and corrected later.

## Task characterization and deterministic retrieval

| Program | Question and primary result | Decision |
|---|---|---|
| [`01_catalog_and_leak.py`](scripts/01_catalog_and_leak.py) | Characterized catalogue fields and proved that simulator constraints originate from target metadata. | Diagnostic |
| [`02_information_budget.py`](scripts/02_information_budget.py) | Measured the disclosure channel and found that the `other` probe exposes the useful constraints most efficiently. | Shipped |
| [`03_retrieval_ceiling.py`](scripts/03_retrieval_ceiling.py) | Established that exact phrase retrieval over disclosed evidence has a high ceiling and that ranking dominates residual error. | Supported |
| [`04_ablation.py`](scripts/04_ablation.py) | Isolated gains from session state, probing, phrase queries, and coverage ranking against the official evaluator. | Shipped |
| [`05_failure_dense_robustness.py`](scripts/05_failure_dense_robustness.py) | Diagnosed template brittleness; dense bi-encoder fusion reduced the tuning score by 0.047. | Rejected |
| [`06_grounded_mining.py`](scripts/06_grounded_mining.py) | Added catalogue-grounded n-gram mining and corrected the category channel, improving nominal and paraphrase recovery. | Shipped |
| [`07_rank_refinement.py`](scripts/07_rank_refinement.py) | Coordinate-ascent reranker tuning improved the then-current public score by 0.037. | Superseded |
| [`08_closing_the_gaps.py`](scripts/08_closing_the_gaps.py) | Fuzzy phrase resolution added 0.003; per-field BM25 weights regressed held-out performance by 0.013. | Partly shipped |
| [`09_disclosure_policy.py`](scripts/09_disclosure_policy.py) | Rejection feedback improved the score by 0.010; narrow disclosure exposed a scoring and policy tradeoff. | Partly shipped |
| [`10_retrieval_structure.py`](scripts/10_retrieval_structure.py) | Field restrictions and full category paths were null; sparse rank fusion slightly regressed. | Rejected |
| [`11_cross_encoder.py`](scripts/11_cross_encoder.py) | Local cross-encoder reranking reduced held-out score by 0.030 and reduced lexical precision. | Rejected |
| [`12_rejection_decomposition.py`](scripts/12_rejection_decomposition.py) | Separated rejection-ordering value from list-length effects and supported demotion rather than deletion. | Shipped |
| [`13_rag_techniques.py`](scripts/13_rag_techniques.py) | Structure-aware chunking was null and RM3 pseudo-relevance feedback regressed by 0.004. | Rejected |
| [`14_target_prior.py`](scripts/14_target_prior.py) | Measured a strong popularity bias in target selection, but alternative prior shapes gained only 0.003 within noise. | Diagnostic |
| [`15_adaptive_prior.py`](scripts/15_adaptive_prior.py) | Turn-adaptive prior decay regressed held-out performance by 0.005. | Rejected |
| [`16_generative_verification.py`](scripts/16_generative_verification.py) | Inverting the intent generator produced no meaningful gain and exposed evidence ties as the main residual limit. | Rejected |
| [`17_everything_else.py`](scripts/17_everything_else.py) | Tested length, rating, slot, stemming, prefix, proximity, ensemble, and negative-evidence variants; none generalized. | Rejected |
| [`18_learning_to_rank.py`](scripts/18_learning_to_rank.py) | Tree-based learning-to-rank improved its training half but regressed held-out score by 0.040. | Rejected |

## LLM and learned ranking alternatives

| Program | Question and primary result | Decision |
|---|---|---|
| [`19_llm_rerank.py`](scripts/19_llm_rerank.py) | Tested listwise Groq reranking only within top coverage ties; end-to-end score fell by approximately 0.027. | Rejected |
| [`20_tiebreak_accuracy.py`](scripts/20_tiebreak_accuracy.py) | Measured the tie decision directly: LLM target-first accuracy was 41.2% versus 57.4% for popularity. | Rejected |
| [`20_robustness_benchmark.py`](scripts/20_robustness_benchmark.py) | Introduced the first component-level population and organizer-uncertainty audit. | Superseded |
| [`21_synthetic_ltr.py`](scripts/21_synthetic_ltr.py) | Trained rankers on simulator-minted sessions; tune and holdout effects disagreed and remained near zero. | Rejected |
| [`22_ml_tiebreak_accuracy.py`](scripts/22_ml_tiebreak_accuracy.py) | Evaluated learned models inside exact evidence ties and found no consistent advantage over popularity. | Rejected |
| [`23_tie_crossencoder.py`](scripts/23_tie_crossencoder.py) | Fine-tuned a task-specific cross-encoder for tie groups; capacity and sample limits prevented generalization. | Rejected |
| [`24_tie_features.py`](scripts/24_tie_features.py) | Tested within-group normalized feature models; adding features progressively reduced held-out tie accuracy. | Rejected |

## Dialogue policy, recall repair, and robustness

| Program | Question and primary result | Decision |
|---|---|---|
| [`25_disclosure_policy_solver.py`](scripts/25_disclosure_policy_solver.py) | Solved disclosure width against the published reward and identified sequential top-one disclosure as optimal. | Shipped |
| [`26_disclosure_validate.py`](scripts/26_disclosure_validate.py) | Confirmed the disclosure schedule on separate tuning and held-out halves; adaptive widening lost. | Shipped |
| [`27_learned_retrieval.py`](scripts/27_learned_retrieval.py) | Tested learned retrieval under the width-one objective; it did not improve target recall reliably. | Rejected |
| [`28_miss_autopsy.py`](scripts/28_miss_autopsy.py) | Traced both remaining public misses to synthesized or non-contiguous evidence rather than candidate-pool exhaustion. | Diagnostic |
| [`29_hitrate_repair.py`](scripts/29_hitrate_repair.py) | Repaired non-contiguous category evidence, raising public HitRate from 0.990 to 0.995 and improving unseen samples. | Shipped |
| [`30_robustness_benchmark.py`](scripts/30_robustness_benchmark.py) | Built unseen-target, population-shift, and component ablation tests; measured four unseen folds from 0.9437 to 0.9554. | Supported |
| [`31_paraphrase_stress.py`](scripts/31_paraphrase_stress.py) | Quantified degradation under controlled message and constraint rewrites. | Diagnostic |
| [`32_conditional_attribution.py`](scripts/32_conditional_attribution.py) | Isolated population exposure to the popularity coefficient and showed mining becomes essential under paraphrase. | Supported |
| [`32_local_bert_phrase_verifier.py`](scripts/32_local_bert_phrase_verifier.py) | Tested a local MiniLM phrase verifier gated behind template failure. | Superseded |
| [`33_finetuned_bert_phrase_verifier.py`](scripts/33_finetuned_bert_phrase_verifier.py) | Fine-tuned and optimized a narrow phrase classifier; it did not deliver stable end-to-end improvement. | Rejected |
| [`33_retune_validation.py`](scripts/33_retune_validation.py) | Retuned population-sensitive constants across public, unseen, and stress conditions. | Superseded |
| [`34_finalist_selection.py`](scripts/34_finalist_selection.py) | Applied a no-regression rule across seven conditions to select the pre-Optuna robust configuration. | Superseded |
| [`35_override_safety.py`](scripts/35_override_safety.py) | Defeated both override guards to bound the worst rejection-feedback failure at 0.040 while preserving HitRate. | Supported |
| [`36_disclosure_risk_adjudication.py`](scripts/36_disclosure_risk_adjudication.py) | Independently re-evaluated disclosure risk and corrected its population grade from P1 to P2. | Supported |
| [`37_population_shift_hardening.py`](scripts/37_population_shift_hardening.py) | Tie-only, percentile, capped, gated, and fusion priors improved adversarial populations by at most 0.003. | Rejected |
| [`38_bayesian_prior_calibration.py`](scripts/38_bayesian_prior_calibration.py) | Explore-then-commit prior selection lacked enough sessions to estimate the better arm reliably. | Rejected |
| [`39_population_detector.py`](scripts/39_population_detector.py) | Corrected the reward definition and showed outcome-based arm selection needs approximately 8,500 sessions per arm. | Diagnostic |
| [`40_detector_driven_prior.py`](scripts/40_detector_driven_prior.py) | Used aggregate retrieved-pool popularity to calibrate the prior without labels, adding 0.034 on inverse-popularity stress. | Shipped |

## Extraction robustness and model integration

| Program | Question and primary result | Decision |
|---|---|---|
| [`21_llm_extraction_robustness.py`](scripts/21_llm_extraction_robustness.py) | Compared deterministic mining and LLM extraction across controlled paraphrase levels. | Supported, optional |
| [`41_recognition_gate.py`](scripts/41_recognition_gate.py) | Proved that all official message forms bypass optional extraction and that altered forms enter it. | Shipped |
| [`42_mining_paraphrase_floor.py`](scripts/42_mining_paraphrase_floor.py) | Tuned deterministic mining under clean and paraphrased inputs; later joint optimization superseded the constants. | Superseded |
| [`43_llm_extraction_hybrid.py`](scripts/43_llm_extraction_hybrid.py) | The gated hybrid preserved clean score with zero calls and recovered 21.6% to 68.8% of paraphrase loss. | Supported, optional |
| [`44_llm_failure_modes.py`](scripts/44_llm_failure_modes.py) | Verified 16 API failure modes and corrected time-budget and zero-yield circuit-breaker defects. | Shipped, optional |
| [`45_ml_constraint_likeness.py`](scripts/45_ml_constraint_likeness.py) | A learned constraint filter reached 0.637 held-out accuracy and degraded as its influence increased. | Rejected |
| [`46_ml_soft_and_probe.py`](scripts/46_ml_soft_and_probe.py) | Soft constraint weighting and learned probing both regressed; learned probing worsened MTTC. | Rejected |
| [`47_local_paraphrase_extractor.py`](scripts/47_local_paraphrase_extractor.py) | A local synthetic paraphrase extractor overfit the transform family and reduced T1 score by 0.098. | Rejected |
| [`49_bert_scaffolding_tagger.py`](scripts/49_bert_scaffolding_tagger.py) | A pretrained BERT token tagger generalized better than linear extraction; catalogue MLM adaptation added no value. | Shipped |
| [`50_train_and_save_tagger.py`](scripts/50_train_and_save_tagger.py) | Trained and exported the gated local tagger as a versioned submission asset. | Infrastructure |
| [`V2 semantic node protocol`](../robustness/v2/README.md) | Freezes semantic splits, constructs 22,550 catalogue-grounded equivalence negatives, and requires isolated node metrics before ranking integration. | Infrastructure |
| [`V2.01 pretrained attribute baseline`](../robustness/v2/pretrained_attribute_baseline.py) | Frozen all-MiniLM-L6-v2 retrieval over 7,922 catalogue attributes reached Recall@5 0.1334 and MRR 0.0486 on 712 development atoms. | Diagnostic baseline |
| [`V2.02 route and dialogue-act audit`](../robustness/v2/audit_route_node.py) | All 1,943 V2 development messages matched their generated route with no unmatched messages or dialogue-act confusions. | Supported |
| [`V2.03 template span audit`](../robustness/v2/audit_span_node.py) | Recovered every exposed semantic-value span exactly in 460 preserved-format development template cases. | Supported |
| [`V2.04 pretrained family baseline`](../robustness/v2/pretrained_family_baseline.py) | Frozen all-MiniLM family routing reached top-one accuracy 0.5351 but top-two recall 0.9522 on 712 development atoms. | Diagnostic baseline |
| [`V2.05 pretrained equivalence baseline`](../robustness/v2/pretrained_equivalence_baseline.py) | Frozen all-MiniLM achieved AUROC 0.7758 on a frozen 40-positive and 94-hard-negative canonical equivalence split. | Diagnostic baseline |
| [`V2.06 semantic guardrail audit`](../robustness/v2/audit_semantic_guardrails.py) | Current soft-provenance gate produced zero nonzero semantic contributions on Official200 and Unseen800 canonical traffic. | Supported |
| [`V2.07 preliminary MiniLM fine-tune`](../robustness/v2/train_attribute_encoder.py) | Pre-audit 186-pair run raised Recall@10 from 0.1334 to 0.1545 and verifier AUROC from 0.7758 to 0.7819, while Recall@5 stayed 0.1334. | Invalidated for selection |
| [`V2.08 synonym corpus audit`](../robustness/v2/audit_synonym_corpus.py) | Found 20 ambiguous surface forms spanning 41 accepted pairs; the training pipeline now excludes them and frozen verifier anchors. | Infrastructure |
| [`V2.09 filtered hard-negative MiniLM`](../robustness/v2/train_attribute_encoder.py) | With 178 filtered triplets, Recall@5 remained 0.1334, Recall@10 reached 0.1545, MRR 0.0605, and verifier AUROC 0.7798. | Rejected |
| [`V2.10 RouteShift fallback`](../robustness/v2/evaluate_route_fallback.py) | Strict recognition covered none of 2,400 altered-wrapper cases; a hand-authored fallback matched its predeclared cue bank. | Control-flow only |
| [`V2.11 route classifier`](../robustness/v2/train_route_classifier.py) | TF-IDF word and character route classifier reached 0.5092 on a cue-disjoint RouteShift template bank. | Inconclusive |
| [`V2.12 abstract route classifier`](../robustness/v2/train_pretrained_route_classifier.py) | Pretrained DistilBERT reached 0.9071 on an abstract six-class route taxonomy that merged V1 opening branches. | Invalidated for integration |
| [`V2.13 V1-action route classifier`](../robustness/v2/train_v1_route_action_classifier.py) | Initial 0.9604 held-out result was not reproduced: saved rerun scored 0.7696 because the classifier head was initialized before seeding. | Invalidated |
| [`51_encoder_bakeoff.py`](scripts/51_encoder_bakeoff.py) | Compared pretrained encoders and found no replacement with sufficient practical advantage. | Supported |
| [`52_tie_feature_dependency.py`](scripts/52_tie_feature_dependency.py) | Measured feature dependency inside ties, then found its popularity conditioning was too coarse. | Invalidated |
| [`53_complementary_signal.py`](scripts/53_complementary_signal.py) | On the exact subset where popularity fails, none of 16 catalogue features exceeded the permutation noise band. | Supported |

## Joint optimization and independent selection

| Program | Question and primary result | Decision |
|---|---|---|
| [`48_optuna_coarse.py`](scripts/48_optuna_coarse.py) | Mapped a broad joint parameter space using public, proxy, and paraphrase objectives. | Superseded |
| [`54_optuna_validate.py`](scripts/54_optuna_validate.py) | Reweighted v1 after paraphrasing was ruled out, but candidate selection included pruned trials. | Invalidated |
| [`55_optuna_official_v2.py`](scripts/55_optuna_official_v2.py) | Ran fixed-data Optuna optimization over the official 200 and stratified private-like 800 with complete-trial filtering. | Infrastructure |
| [`55_optuna_dashboard_v2.py`](scripts/55_optuna_dashboard_v2.py) | Exposed the isolated v2 study through a local inspection dashboard. | Infrastructure |
| [`56_optuna_exploit_v2.py`](scripts/56_optuna_exploit_v2.py) | Ran a separate 50/50 public-proxy objective to expose the exploration versus exploitation tradeoff. | Infrastructure |
| [`57_independent_validation.py`](scripts/57_independent_validation.py) | Evaluated frozen Pareto candidates on four untouched folds and controlled population disturbances; trial 38 was selected. | Shipped |

## Post-selection robustness probes

| Program | Question and primary result | Decision |
|---|---|---|
| [`58_override_replacement.py`](scripts/58_override_replacement.py) | Tested category-preserving prior-evidence reset on an override. It reduced Official200 intent-override HitRate from 1.000 to 0.933 and Unseen800 from 0.983 to 0.858. | Rejected |
| [`59_override_opening_evidence.py`](scripts/59_override_opening_evidence.py) | Recovered the organizer-confirmed target-derived old-value slot from intent-override openings. Full constraint treatment was exactly neutral on Official200 and Unseen800; weaker treatment regressed Unseen800 by 0.000456. | Shipped |
| [`60_override_focus_contradiction.py`](scripts/60_override_focus_contradiction.py) | Built OverrideFocus800 and saturated its initial old-value slot with incompatible, catalogue-attested materials. Retaining stale evidence lost 0.058669 versus source-faithful overrides; no safe semantic conflict detector is available from released wording alone. | Supported diagnostic |
| [`61_catalog_conflict_guard.py`](scripts/61_catalog_conflict_guard.py) | Required same attribute family and zero catalogue co-occurrence before removal. It made zero removals in all 800 adversarial material collisions because incompatible materials commonly co-occur in unrelated products. | Rejected |
| [`62_same_family_override_replace.py`](scripts/62_same_family_override_replace.py) | Replaced only unconfirmed opening values when the later override named the same high-confidence family. It removed 589 of 800 adversarial values and recovered 0.028994, but regressed Official200 by 0.000100. | Rejected for shipment; retained diagnostic |
| [`63_gated_paraphrase_override.py`](scripts/63_gated_paraphrase_override.py) | Tested a strict unfamiliar-wording replacement gate. It triggered 0 times on all 464 Official200 and all 2,262 Unseen800 recognised messages, but an aggressive evidence reset reduced compatible and contradictory paraphrase scores. | Rejected |

## V2 semantic-node protocols

| Program | Question and primary result | Decision |
|---|---|---|
| [`build_v1_route_template_bank.py`](../robustness/v2/build_v1_route_template_bank.py) | Built 72 paraphrased train and 48 independently authored held-out template families for the six exact V1 fallback actions. It also derives all 2,305 distinct messages obtainable from actual Official200 cards and allowed reply actions for training only. Attributes remain verbatim; paraphrased template overlap is zero. | Protocol ready |
| [`train_v1_route_template_bank.py`](../robustness/v2/train_v1_route_template_bank.py) | CUDA DistilBERT achieved 0.859896 on the 9,600-row held-out bank, but buying openings scored only 0.4300 and constraint updates 0.7325. | Rejected for integration |
| [`train_v1_turn_gated_classifiers.py`](../robustness/v2/train_v1_turn_gated_classifiers.py) | Separate turn-gated CUDA classifiers reached 0.795625 on a newly authored final set. The follow-up override-update class scored 0.446250. | Rejected for integration |
| [`64_node1_route_integration.py`](scripts/64_node1_route_integration.py) | Strict-gated integration check for the accepted six-route classifier. The promoted V2 runtime is metric-identical on Official200 and Unseen800, with zero model loads and zero inferences. | Node 1 complete in V2 |
| [`evaluate_node2_scaffolding_template_test.py`](../robustness/v2/evaluate_node2_scaffolding_template_test.py) | Fixed template-disjoint Node 2 test of the existing BERT cleanup model. It retains nearly all slots but cannot overcome V1 mining’s four-token minimum for short canonical values. | BERT retained; Node 2 incomplete |
| [`evaluate_node2_short_span_dictionary.py`](../robustness/v2/evaluate_node2_short_span_dictionary.py) | Fixed-test exact short-span dictionary ablation after BERT cleanup. Category masking preserves 99.25% constraint recall while reducing extra candidates to 0.152 per message. | Node 2 integration candidate |
| [`run_official_template_paraphrase.py`](../robustness/v2/run_official_template_paraphrase.py) | Replays Official200 unchanged except for held-out TemplateParaphrase9600 Test wrappers. Exact catalogue span recovery reaches 0.939900, compared with 0.720525 raw and 0.738198 route-only. | Node 2 complete for wrapper paraphrase |
| [`65_candidate_information_gain_probe.py`](scripts/65_candidate_information_gain_probe.py) | Uniform candidate-partition clarification is neutral to slightly positive on Official200, review-weighted, and inverse populations, and slightly negative on uniform targets. Static review weighting regresses Official200. | Uniform controller adopted; review prior rejected |
| [`66_expected_rank_gain_probe.py`](scripts/66_expected_rank_gain_probe.py) | Top-five counterfactual rank-gain clarification predicted positive local gain but reduced Official200 score by 0.000600 through MTTC. | Rejected |
| [`67_information_gain_runtime_benchmark.py`](scripts/67_information_gain_runtime_benchmark.py) | Five-run Official200 timing comparison: integer signatures preserve information-gain metrics at 69.57 ms/session, compared with 59.49 ms fixed V1. | Accepted implementation |
| [`evaluate_node6_gate_baseline.py`](../robustness/v2/evaluate_node6_gate_baseline.py) | CUDA pre-augmentation feasibility audit. Frozen top-one canonical retrieval is 0 of 712, so no similarity or margin threshold can produce an accepted correct mapping. | Node 6 thresholding blocked pending Nodes 3 to 5 |
| [`train_attribute_encoder.py`](../robustness/v2/train_attribute_encoder.py) | CUDA low-data fine-tune on 213 verified train pairs. Retrieval Recall@5 regressed from 0.133427 to 0.073034 despite a small verifier AUROC gain. | Rejected; retained as low-data baseline |
| [`train_attribute_encoder.py`](../robustness/v2/train_attribute_encoder.py) | CUDA mixed strict plus lexical-overlap augmentation on 158 trainable pairs. Recall@5 recovered to the frozen baseline while Recall@10, MRR, and verifier AUROC improved. | Promising data policy; not integrated |

## Final selection rule

The final agent uses balanced Optuna trial 38. Selection followed this order:

1. optimize only on the released public set and one fixed, disclosed-statistics proxy;
2. freeze conservative, balanced, and aggressive candidates;
3. stop both optimizers;
4. evaluate four untouched same-population folds;
5. evaluate controlled popularity-distribution disturbances;
6. prefer trial 38 because it preserved the public score, improved the independent mean,
   and matched the aggressive candidate without its public regression.

These independent folds are consumed validation evidence and must not be reused for tuning.
