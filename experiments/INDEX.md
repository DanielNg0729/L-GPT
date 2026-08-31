# Experiment Registry

This registry is the compact navigator for every versioned experiment program. The complete
one-by-one result, ruling, and final-design impact is in
[`DECISION_LOG.md`](DECISION_LOG.md). Detailed measurements and
methodological corrections are retained in [`FINDINGS.md`](FINDINGS.md);
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
| [`01_catalog_and_leak.py`](log/01_catalog_and_leak.py) | Characterized catalogue fields and proved that simulator constraints originate from target metadata. | Diagnostic |
| [`02_information_budget.py`](log/02_information_budget.py) | Measured the disclosure channel and found that the `other` probe exposes the useful constraints most efficiently. | Shipped |
| [`03_retrieval_ceiling.py`](log/03_retrieval_ceiling.py) | Established that exact phrase retrieval over disclosed evidence has a high ceiling and that ranking dominates residual error. | Supported |
| [`04_ablation.py`](log/04_ablation.py) | Isolated gains from session state, probing, phrase queries, and coverage ranking against the official evaluator. | Shipped |
| [`05_failure_dense_robustness.py`](log/05_failure_dense_robustness.py) | Diagnosed template brittleness; dense bi-encoder fusion reduced the tuning score by 0.047. | Rejected |
| [`06_grounded_mining.py`](log/06_grounded_mining.py) | Added catalogue-grounded n-gram mining and corrected the category channel, improving nominal and paraphrase recovery. | Shipped |
| [`07_rank_refinement.py`](log/07_rank_refinement.py) | Coordinate-ascent reranker tuning improved the then-current public score by 0.037. | Superseded |
| [`08_closing_the_gaps.py`](log/08_closing_the_gaps.py) | Fuzzy phrase resolution added 0.003; per-field BM25 weights regressed held-out performance by 0.013. | Partly shipped |
| [`09_disclosure_policy.py`](log/09_disclosure_policy.py) | Rejection feedback improved the score by 0.010; narrow disclosure exposed a scoring and policy tradeoff. | Partly shipped |
| [`10_retrieval_structure.py`](log/10_retrieval_structure.py) | Field restrictions and full category paths were null; sparse rank fusion slightly regressed. | Rejected |
| [`11_cross_encoder.py`](log/11_cross_encoder.py) | Local cross-encoder reranking reduced held-out score by 0.030 and reduced lexical precision. | Rejected |
| [`12_rejection_decomposition.py`](log/12_rejection_decomposition.py) | Separated rejection-ordering value from list-length effects and supported demotion rather than deletion. | Shipped |
| [`13_rag_techniques.py`](log/13_rag_techniques.py) | Structure-aware chunking was null and RM3 pseudo-relevance feedback regressed by 0.004. | Rejected |
| [`14_target_prior.py`](log/14_target_prior.py) | Measured a strong popularity bias in target selection, but alternative prior shapes gained only 0.003 within noise. | Diagnostic |
| [`15_adaptive_prior.py`](log/15_adaptive_prior.py) | Turn-adaptive prior decay regressed held-out performance by 0.005. | Rejected |
| [`16_generative_verification.py`](log/16_generative_verification.py) | Inverting the intent generator produced no meaningful gain and exposed evidence ties as the main residual limit. | Rejected |
| [`17_everything_else.py`](log/17_everything_else.py) | Tested length, rating, slot, stemming, prefix, proximity, ensemble, and negative-evidence variants; none generalized. | Rejected |
| [`18_learning_to_rank.py`](log/18_learning_to_rank.py) | Tree-based learning-to-rank improved its training half but regressed held-out score by 0.040. | Rejected |

## LLM and learned ranking alternatives

| Program | Question and primary result | Decision |
|---|---|---|
| [`19_llm_rerank.py`](log/19_llm_rerank.py) | Tested listwise Groq reranking only within top coverage ties; end-to-end score fell by approximately 0.027. | Rejected |
| [`20_tiebreak_accuracy.py`](log/20_tiebreak_accuracy.py) | Measured the tie decision directly: LLM target-first accuracy was 41.2% versus 57.4% for popularity. | Rejected |
| [`20_robustness_benchmark.py`](log/20_robustness_benchmark.py) | Introduced the first component-level population and organizer-uncertainty audit. | Superseded |
| [`21_synthetic_ltr.py`](log/21_synthetic_ltr.py) | Trained rankers on simulator-minted sessions; tune and holdout effects disagreed and remained near zero. | Rejected |
| [`22_ml_tiebreak_accuracy.py`](log/22_ml_tiebreak_accuracy.py) | Evaluated learned models inside exact evidence ties and found no consistent advantage over popularity. | Rejected |
| [`23_tie_crossencoder.py`](log/23_tie_crossencoder.py) | Fine-tuned a task-specific cross-encoder for tie groups; capacity and sample limits prevented generalization. | Rejected |
| [`24_tie_features.py`](log/24_tie_features.py) | Tested within-group normalized feature models; adding features progressively reduced held-out tie accuracy. | Rejected |

## Dialogue policy, recall repair, and robustness

| Program | Question and primary result | Decision |
|---|---|---|
| [`25_disclosure_policy_solver.py`](log/25_disclosure_policy_solver.py) | Solved disclosure width against the published reward and identified sequential top-one disclosure as optimal. | Shipped |
| [`26_disclosure_validate.py`](log/26_disclosure_validate.py) | Confirmed the disclosure schedule on separate tuning and held-out halves; adaptive widening lost. | Shipped |
| [`27_learned_retrieval.py`](log/27_learned_retrieval.py) | Tested learned retrieval under the width-one objective; it did not improve target recall reliably. | Rejected |
| [`28_miss_autopsy.py`](log/28_miss_autopsy.py) | Traced both remaining public misses to synthesized or non-contiguous evidence rather than candidate-pool exhaustion. | Diagnostic |
| [`29_hitrate_repair.py`](log/29_hitrate_repair.py) | Repaired non-contiguous category evidence, raising public HitRate from 0.990 to 0.995 and improving unseen samples. | Shipped |
| [`30_robustness_benchmark.py`](log/30_robustness_benchmark.py) | Built unseen-target, population-shift, and component ablation tests; measured four unseen folds from 0.9437 to 0.9554. | Supported |
| [`31_paraphrase_stress.py`](log/31_paraphrase_stress.py) | Quantified degradation under controlled message and constraint rewrites. | Diagnostic |
| [`32_conditional_attribution.py`](log/32_conditional_attribution.py) | Isolated population exposure to the popularity coefficient and showed mining becomes essential under paraphrase. | Supported |
| [`32_local_bert_phrase_verifier.py`](log/32_local_bert_phrase_verifier.py) | Tested a local MiniLM phrase verifier gated behind template failure. | Superseded |
| [`33_finetuned_bert_phrase_verifier.py`](log/33_finetuned_bert_phrase_verifier.py) | Fine-tuned and optimized a narrow phrase classifier; it did not deliver stable end-to-end improvement. | Rejected |
| [`33_retune_validation.py`](log/33_retune_validation.py) | Retuned population-sensitive constants across public, unseen, and stress conditions. | Superseded |
| [`34_finalist_selection.py`](log/34_finalist_selection.py) | Applied a no-regression rule across seven conditions to select the pre-Optuna robust configuration. | Superseded |
| [`35_override_safety.py`](log/35_override_safety.py) | Defeated both override guards to bound the worst rejection-feedback failure at 0.040 while preserving HitRate. | Supported |
| [`36_disclosure_risk_adjudication.py`](log/36_disclosure_risk_adjudication.py) | Independently re-evaluated disclosure risk and corrected its population grade from P1 to P2. | Supported |
| [`37_population_shift_hardening.py`](log/37_population_shift_hardening.py) | Tie-only, percentile, capped, gated, and fusion priors improved adversarial populations by at most 0.003. | Rejected |
| [`38_bayesian_prior_calibration.py`](log/38_bayesian_prior_calibration.py) | Explore-then-commit prior selection lacked enough sessions to estimate the better arm reliably. | Rejected |
| [`39_population_detector.py`](log/39_population_detector.py) | Corrected the reward definition and showed outcome-based arm selection needs approximately 8,500 sessions per arm. | Diagnostic |
| [`40_detector_driven_prior.py`](log/40_detector_driven_prior.py) | Used aggregate retrieved-pool popularity to calibrate the prior without labels, adding 0.034 on inverse-popularity stress. | Shipped |

## Extraction robustness and model integration

| Program | Question and primary result | Decision |
|---|---|---|
| [`21_llm_extraction_robustness.py`](log/21_llm_extraction_robustness.py) | Compared deterministic mining and LLM extraction across controlled paraphrase levels. | Supported, optional |
| [`41_recognition_gate.py`](log/41_recognition_gate.py) | Proved that all official message forms bypass optional extraction and that altered forms enter it. | Shipped |
| [`42_mining_paraphrase_floor.py`](log/42_mining_paraphrase_floor.py) | Tuned deterministic mining under clean and paraphrased inputs; later joint optimization superseded the constants. | Superseded |
| [`43_llm_extraction_hybrid.py`](log/43_llm_extraction_hybrid.py) | The gated hybrid preserved clean score with zero calls and recovered 21.6% to 68.8% of paraphrase loss. | Supported, optional |
| [`44_llm_failure_modes.py`](log/44_llm_failure_modes.py) | Verified 16 API failure modes and corrected time-budget and zero-yield circuit-breaker defects. | Shipped, optional |
| [`45_ml_constraint_likeness.py`](log/45_ml_constraint_likeness.py) | A learned constraint filter reached 0.637 held-out accuracy and degraded as its influence increased. | Rejected |
| [`46_ml_soft_and_probe.py`](log/46_ml_soft_and_probe.py) | Soft constraint weighting and learned probing both regressed; learned probing worsened MTTC. | Rejected |
| [`47_local_paraphrase_extractor.py`](log/47_local_paraphrase_extractor.py) | A local synthetic paraphrase extractor overfit the transform family and reduced T1 score by 0.098. | Rejected |
| [`49_bert_scaffolding_tagger.py`](log/49_bert_scaffolding_tagger.py) | A pretrained BERT token tagger generalized better than linear extraction; catalogue MLM adaptation added no value. | Shipped |
| [`50_train_and_save_tagger.py`](log/50_train_and_save_tagger.py) | Trained and exported the gated local tagger as a versioned submission asset. | Infrastructure |
| [`V2 semantic node protocol`](../experiments/studies/README.md) | Freezes semantic splits, constructs 22,550 catalogue-grounded equivalence negatives, and requires isolated node metrics before ranking integration. | Infrastructure |
| [`V2.01 pretrained attribute baseline`](../experiments/studies/pretrained_attribute_baseline.py) | Frozen all-MiniLM-L6-v2 retrieval over 7,922 catalogue attributes reached Recall@5 0.1334 and MRR 0.0486 on 712 development atoms. | Diagnostic baseline |
| [`V2.02 route and dialogue-act audit`](../experiments/studies/audit_route_node.py) | All 1,943 V2 development messages matched their generated route with no unmatched messages or dialogue-act confusions. | Supported |
| [`V2.03 template span audit`](../experiments/studies/audit_span_node.py) | Recovered every exposed semantic-value span exactly in 460 preserved-format development template cases. | Supported |
| [`V2.04 pretrained family baseline`](../experiments/studies/pretrained_family_baseline.py) | Frozen all-MiniLM family routing reached top-one accuracy 0.5351 but top-two recall 0.9522 on 712 development atoms. | Diagnostic baseline |
| [`V2.05 pretrained equivalence baseline`](../experiments/studies/pretrained_equivalence_baseline.py) | Frozen all-MiniLM achieved AUROC 0.7758 on a frozen 40-positive and 94-hard-negative canonical equivalence split. | Diagnostic baseline |
| [`V2.06 semantic guardrail audit`](../experiments/studies/audit_semantic_guardrails.py) | Current soft-provenance gate produced zero nonzero semantic contributions on Official200 and Unseen800 canonical traffic. | Supported |
| [`V2.07 preliminary MiniLM fine-tune`](../experiments/studies/train_attribute_encoder.py) | Pre-audit 186-pair run raised Recall@10 from 0.1334 to 0.1545 and verifier AUROC from 0.7758 to 0.7819, while Recall@5 stayed 0.1334. | Invalidated for selection |
| [`V2.08 synonym corpus audit`](../experiments/studies/audit_synonym_corpus.py) | Found 20 ambiguous surface forms spanning 41 accepted pairs; the training pipeline now excludes them and frozen verifier anchors. | Infrastructure |
| [`V2.09 filtered hard-negative MiniLM`](../experiments/studies/train_attribute_encoder.py) | With 178 filtered triplets, Recall@5 remained 0.1334, Recall@10 reached 0.1545, MRR 0.0605, and verifier AUROC 0.7798. | Rejected |
| [`V2.10 RouteShift fallback`](../experiments/studies/evaluate_route_fallback.py) | Strict recognition covered none of 2,400 altered-wrapper cases; a hand-authored fallback matched its predeclared cue bank. | Control-flow only |
| [`V2.11 route classifier`](../experiments/studies/train_route_classifier.py) | TF-IDF word and character route classifier reached 0.5092 on a cue-disjoint RouteShift template bank. | Inconclusive |
| [`V2.12 abstract route classifier`](../experiments/studies/train_pretrained_route_classifier.py) | Pretrained DistilBERT reached 0.9071 on an abstract six-class route taxonomy that merged V1 opening branches. | Invalidated for integration |
| [`V2.13 V1-action route classifier`](../experiments/studies/train_route_action_classifier.py) | Initial 0.9604 held-out result was not reproduced: saved rerun scored 0.7696 because the classifier head was initialized before seeding. | Invalidated |
| [`51_encoder_bakeoff.py`](log/51_encoder_bakeoff.py) | Compared pretrained encoders and found no replacement with sufficient practical advantage. | Supported |
| [`52_tie_feature_dependency.py`](log/52_tie_feature_dependency.py) | Measured feature dependency inside ties, then found its popularity conditioning was too coarse. | Invalidated |
| [`53_complementary_signal.py`](log/53_complementary_signal.py) | On the exact subset where popularity fails, none of 16 catalogue features exceeded the permutation noise band. | Supported |

## Joint optimization and independent selection

| Program | Question and primary result | Decision |
|---|---|---|
| [`48_optuna_coarse.py`](log/48_optuna_coarse.py) | Mapped a broad joint parameter space using public, proxy, and paraphrase objectives. | Superseded |
| [`54_optuna_validate.py`](log/54_optuna_validate.py) | Reweighted v1 after paraphrasing was ruled out, but candidate selection included pruned trials. | Invalidated |
| [`55_optuna_official.py`](log/55_optuna_official.py) | Ran fixed-data Optuna optimization over the official 200 and stratified private-like 800 with complete-trial filtering. | Infrastructure |
| [`55_optuna_dashboard.py`](log/55_optuna_dashboard.py) | Exposed the isolated v2 study through a local inspection dashboard. | Infrastructure |
| [`56_optuna_exploit.py`](log/56_optuna_exploit.py) | Ran a separate 50/50 public-proxy objective to expose the exploration versus exploitation tradeoff. | Infrastructure |
| [`57_independent_validation.py`](log/57_independent_validation.py) | Evaluated frozen Pareto candidates on four untouched folds and controlled population disturbances; trial 38 was selected. | Shipped |

## Post-selection robustness probes

| Program | Question and primary result | Decision |
|---|---|---|
| [`58_override_replacement.py`](log/58_override_replacement.py) | Tested category-preserving prior-evidence reset on an override. It reduced Official200 intent-override HitRate from 1.000 to 0.933 and Unseen800 from 0.983 to 0.858. | Rejected |
| [`59_override_opening_evidence.py`](log/59_override_opening_evidence.py) | Recovered the organizer-confirmed target-derived old-value slot from intent-override openings. Full constraint treatment was exactly neutral on Official200 and Unseen800; weaker treatment regressed Unseen800 by 0.000456. | Shipped |
| [`60_override_focus_contradiction.py`](log/60_override_focus_contradiction.py) | Built OverrideFocus800 and saturated its initial old-value slot with incompatible, catalogue-attested materials. Retaining stale evidence lost 0.058669 versus source-faithful overrides; no safe semantic conflict detector is available from released wording alone. | Supported diagnostic |
| [`61_catalog_conflict_guard.py`](log/61_catalog_conflict_guard.py) | Required same attribute family and zero catalogue co-occurrence before removal. It made zero removals in all 800 adversarial material collisions because incompatible materials commonly co-occur in unrelated products. | Rejected |
| [`62_same_family_override_replace.py`](log/62_same_family_override_replace.py) | Replaced only unconfirmed opening values when the later override named the same high-confidence family. It removed 589 of 800 adversarial values and recovered 0.028994, but regressed Official200 by 0.000100. | Rejected for shipment; retained diagnostic |
| [`63_gated_paraphrase_override.py`](log/63_gated_paraphrase_override.py) | Tested a strict unfamiliar-wording replacement gate. It triggered 0 times on all 464 Official200 and all 2,262 Unseen800 recognised messages, but an aggressive evidence reset reduced compatible and contradictory paraphrase scores. | Rejected |

## V2 semantic-node protocols

| Program | Question and primary result | Decision |
|---|---|---|
| [`build_route_template_bank.py`](../experiments/studies/build_route_template_bank.py) | Built 72 paraphrased train and 48 independently authored held-out template families for the six exact V1 fallback actions. It also derives all 2,305 distinct messages obtainable from actual Official200 cards and allowed reply actions for training only. Attributes remain verbatim; paraphrased template overlap is zero. | Protocol ready |
| [`train_route_template_bank.py`](../experiments/studies/train_route_template_bank.py) | CUDA DistilBERT achieved 0.859896 on the 9,600-row held-out bank, but buying openings scored only 0.4300 and constraint updates 0.7325. | Rejected for integration |
| [`train_turn_gated_classifiers.py`](../experiments/studies/train_turn_gated_classifiers.py) | Separate turn-gated CUDA classifiers reached 0.795625 on a newly authored final set. The follow-up override-update class scored 0.446250. | Rejected for integration |
| [`64_node1_route_integration.py`](log/64_node1_route_integration.py) | Strict-gated integration check for the accepted six-route classifier. The promoted V2 runtime is metric-identical on Official200 and Unseen800, with zero model loads and zero inferences. | Node 1 complete in V2 |
| [`evaluate_node2_scaffolding_template_test.py`](../experiments/studies/evaluate_node2_scaffolding_template_test.py) | Fixed template-disjoint Node 2 test of the existing BERT cleanup model. It retains nearly all slots but cannot overcome V1 mining’s four-token minimum for short canonical values. | BERT retained; Node 2 incomplete |
| [`evaluate_node2_short_span_dictionary.py`](../experiments/studies/evaluate_node2_short_span_dictionary.py) | Fixed-test exact short-span dictionary ablation after BERT cleanup. Category masking preserves 99.25% constraint recall while reducing extra candidates to 0.152 per message. | Node 2 integration candidate |
| [`run_official_template_paraphrase.py`](../experiments/studies/run_official_template_paraphrase.py) | Replays Official200 unchanged except for held-out TemplateParaphrase9600 Test wrappers. Exact catalogue span recovery reaches 0.939900, compared with 0.720525 raw and 0.738198 route-only. | Node 2 complete for wrapper paraphrase |
| [`65_candidate_information_gain_probe.py`](log/65_candidate_information_gain_probe.py) | Uniform candidate-partition clarification is neutral to slightly positive on Official200, review-weighted, and inverse populations, and slightly negative on uniform targets. Static review weighting regresses Official200. | Uniform controller adopted; review prior rejected |
| [`66_expected_rank_gain_probe.py`](log/66_expected_rank_gain_probe.py) | Top-five counterfactual rank-gain clarification predicted positive local gain but reduced Official200 score by 0.000600 through MTTC. | Rejected |
| [`67_information_gain_runtime_benchmark.py`](log/67_information_gain_runtime_benchmark.py) | Five-run Official200 timing comparison: integer signatures preserve information-gain metrics at 69.57 ms/session, compared with 59.49 ms fixed V1. | Accepted implementation |
| [`evaluate_node6_gate_baseline.py`](../experiments/studies/evaluate_node6_gate_baseline.py) | CUDA pre-augmentation feasibility audit. Frozen top-one canonical retrieval is 0 of 712, so no similarity or margin threshold can produce an accepted correct mapping. | Node 6 thresholding blocked pending Nodes 3 to 5 |
| [`train_attribute_encoder.py`](../experiments/studies/train_attribute_encoder.py) | CUDA low-data fine-tune on 213 verified train pairs. Retrieval Recall@5 regressed from 0.133427 to 0.073034 despite a small verifier AUROC gain. | Rejected; retained as low-data baseline |
| [`train_attribute_encoder.py`](../experiments/studies/train_attribute_encoder.py) | CUDA mixed strict plus lexical-overlap augmentation on 158 trainable pairs. Recall@5 recovered to the frozen baseline while Recall@10, MRR, and verifier AUROC improved. | Promising data policy; not integrated |

## Suppression, deparaphrase, and the attribute-paraphrase programme

| Experiment | Result | Status |
|---|---|---|
| [`68_dfcap_recheck.py`](log/68_dfcap_recheck.py) | Re-swept `DF_CAP` at the frozen trial-38 configuration rather than the one it was originally tuned on. No value improved every decision criterion. | Shipped value retained |
| [`69_suppression_gate.py`](log/69_suppression_gate.py) | Dropping `_resolve`'s substring and single-token fallbacks leaves official200, org-proxy, review800 and uniform byte-identical, moves inverse by −0.000200, and gains +0.056000 on attribute paraphrase. The fallback's original +0.0081 had decayed to 0.000 once the category part-split subsumed it. | Shipped |
| [`evaluate_oracle_decomposition.py`](../experiments/studies/evaluate_oracle_decomposition.py) | Of a 0.1931 attribute-paraphrase gap, a perfect resolver recovers +0.0979 while merely DELETING the unresolvable clause recovers +0.0467. A quarter of the loss was self-inflicted. | Diagnostic; motivated suppression |
| [`probe_llm_deparaphrase.py`](../experiments/studies/probe_llm_deparaphrase.py) | Generate-then-verify feasibility. First run reported 27 abstentions that were in fact 27 HTTP 403s, and scored the model by exact string equality against raw catalogue strings, marking `cotton` wrong against `100 cotton size 3t 2 3 4t 3 4 5 5`. | Invalidated; superseded by end-to-end scoring |
| [`evaluate_llm_resolver_end_to_end.py`](../experiments/studies/evaluate_llm_resolver_end_to_end.py) | Attenuation is the mechanism: proposals at CONSTRAINT weight reach 81.5% of the oracle, at the weak tier ~96%. The three weak weights span 0.0046 non-monotonically, so the finding is insensitivity, not an optimum. | Shipped (`W_SEM = 0.15`) |
| [`evaluate_recall_at_k_ceiling.py`](../experiments/studies/evaluate_recall_at_k_ceiling.py) | The encoders were retired on the wrong statistic. Recall does not flatten past k=10: the correct canonical sits at median rank 56–96, so a candidate list has a ~0.76 ceiling, not ~0.15. Qwen3-Embedding-0.6B loses to all-MiniLM-L6-v2 at R@100. | Diagnostic; reopened then closed `choose` |
| [`evaluate_choose_vs_generate.py`](../experiments/studies/evaluate_choose_vs_generate.py) | Retrieval augmentation is rejected. `choose` scores BELOW the no-model floor (−0.0050); told candidates were optional hints, the model still answered off-list only 2 times in 21. | Rejected |
| [`evaluate_node5_verifies_llm_proposal.py`](../experiments/studies/evaluate_node5_verifies_llm_proposal.py) | Node 5 separates correct proposals from competing attested values at 0.8349 AUROC, but no threshold transfers: calibrated on synonym pairs it discards 76% of correct proposals. | Rejected pending train-only calibration |
| [`build_open_vocabulary_suite.py`](../experiments/studies/build_open_vocabulary_suite.py) | 204 independently generated paraphrases over targets disjoint from Official200, generator (Claude Haiku) distinct from solver (gpt-oss-120b). Six rows excluded because the generation prompt had illustrated the task with examples copied from the prior suite. | Infrastructure |
| [`evaluate_open_vocabulary.py`](../experiments/studies/evaluate_open_vocabulary.py) | On open vocabulary the resolver recovers 17.2% of the suppression→canonical gap against 27.6% on the 27-phrase suite — the same quantity, and the prior "96%" was a fraction of the ORACLE, a different denominator. | Supported, weaker than advertised |
| [`evaluate_llm_resolver_populations.py`](../experiments/studies/evaluate_llm_resolver_populations.py) | Node 7 clearance. All five decision criteria move by +0.000000 while the resolver is reached 26 times, so safety is measured on live invocations rather than on absence of them. | Shipped (Node 7 cleared) |
| [`evaluate_open_vocab_oracle.py`](../experiments/studies/evaluate_open_vocab_oracle.py) | The attribute ceiling is 95.1%, not the 50.7% the prior suite implied. ORACLE-DROP is exactly 0.000000 because suppression already deletes the clause. | Diagnostic |
| [`evaluate_attr_accuracy_vs_weight.py`](../experiments/studies/evaluate_attr_accuracy_vs_weight.py) | Oracle and resolver curves run in OPPOSITE directions across weights: perfect knowledge wants full weight, the real resolver collapses at it. The shortfall is knowledge (−0.0603 at equal weight), not integration. | Diagnostic; closes the weight lever |
| [`evaluate_corroboration_filter.py`](../experiments/studies/evaluate_corroboration_filter.py) | Encoder corroboration fails INVERTED, AUROC 0.3885: the encoder ranks harmful proposals better. The two mechanisms are opposed, not independent — the LLM adds value exactly where surface similarity misleads. | Rejected |

## Node 1, exact span recovery, and the category finding

| Experiment | Result | Status |
|---|---|---|
| [`audit_span_node_precision.py`](../experiments/studies/audit_span_node_precision.py) | The span node self-routes without an action label: value recall ~1.0 on all four value-bearing actions, category on exactly the three that can carry one, 0.000 spurious on no-evidence. Facet names had to be excluded from the dictionary first — every one of 1,600 reworded no-preference rows had yielded exactly one spurious attribute. | Shipped (span node) |
| [`audit_node1_vs_regex.py`](../experiments/studies/audit_node1_vs_regex.py) | Lexical cues recover override state at 37.5% and no-evidence at 0.0% held out; the trained router reaches 100% on both at 0/8000 dangerous false positives. Semantic classification is the only mechanism that transfers across vocabulary shift. | Shipped (Node 1) |
| [`audit_template_residual.py`](../experiments/studies/audit_template_residual.py) | Counting partial recovery, 99.66% of 3,190 constraint values reach the ledger and true loss is 0.34%. A better tagger or span-extraction model could claim 0.16%. | Diagnostic; closes template extraction |
| [`build_hostile_suite.py`](../experiments/studies/build_hostile_suite.py) / [`evaluate_hostile.py`](../experiments/studies/evaluate_hostile.py) | With the category neutralised and every remaining value paraphrased, the CEILING itself collapses to 0.280362 (HR@10 0.3183) from 0.947263. The category carries roughly two thirds of the achievable score. | Diagnostic; the central task insight |
| [`build_span_bert_data.py`](../experiments/studies/build_span_bert_data.py) / [`train_span_bert.py`](../experiments/studies/train_span_bert.py) / [`evaluate_span_bert_heldout.py`](../experiments/studies/evaluate_span_bert_heldout.py) | A value-span extractor trained with the slot vocabulary deliberately randomised transfers to unseen paraphrases (VALUE F1 0.6916 → 0.8647, recall 0.9595) with zero false spans on value-bearing actions. Not shipped: after the hostile finding its target is ~0.007 absolute for 254 MB. | Rejected on cost, retained |
| [`audit_scenario_under_paraphrase.py`](../experiments/studies/audit_scenario_under_paraphrase.py) | Browsing is the MOST damaged scenario under paraphrase, not the least: it starts with a category and acquires its constraints through probing, so a larger share of its evidence travels the channel paraphrase attacks. | Diagnostic |
| [`70_category_leverage.py`](log/70_category_leverage.py) | Ablation and weight sweep for the category channel across all five decision criteria, following the hostile finding. | See decision log |
| [`70_category_leverage.py`](log/70_category_leverage.py) | Ablated the category channel on a hostile condition. With perfect canonical values but the category removed, the achievable score collapses from 0.945 to 0.280 -- the category carries roughly two thirds of the task. | Accepted as a finding |
| [`benchmark_pipeline.py`](../tools/benchmark_pipeline.py) | Runtime by configuration, median of five runs. Clean traffic is unchanged by the optional layers because they are never constructed; the reworded column is what they cost when every message reaches them. | Characterisation |
| [`tune_deparaphrase_prompt.py`](studies/tune_deparaphrase_prompt.py) | Three deparaphrase prompt arms (value only / value + message / value + category + transcript) plus a 20b-vs-120b model comparison. Models are score-equivalent on 800 sessions (0.870525 against 0.868219, inside the +/-0.0027 band) and 20b is 3.6x faster, so the smaller model ships. Arms B and C remain unmeasured: two attempts died on provider quota. | Model switched; prompt arms open |
| [`audit_ml_reachability.py`](studies/audit_ml_reachability.py) | Per-suite reachability of every learned component. Across official200 and unseen800 -- 2,716 messages -- zero are unrecognised, so both DistilBERTs record 0 model loads and 0 inferences. The deparaphraser is gated per value rather than per message and is reachable 0 times on the public set and twice in 800 unseen sessions. | Accepted as a finding |
| [`audit_proposal_gate_noise.py`](studies/audit_proposal_gate_noise.py) | Tested tighter admission gates for model-proposed evidence. A document-frequency FLOOR raises proposal precision from 0.754 to 0.871 (the ceiling hypothesis was inverted: rare proposals are the hallucinations), but end-to-end it is worth +0.000813, inside noise. Removing 41 confidently wrong proposals changed the score by exactly 0.000000. | Rejected; confirms the attenuated weight already absorbs wrong evidence |
| [`simulate_rescue_token_budget.py`](studies/simulate_rescue_token_budget.py) | Sized the transcript rescue's token budget locally with the model's own tokenizer, no provider calls. Prompt ~504 tokens, structured output 152 at p95, so 95% of the configured 3072 is reasoning headroom -- and an 84-call arm reserves 300k tokens against a 200k/day limit. | Characterisation |
| [`evaluate_rescue_weight.py`](studies/evaluate_rescue_weight.py) | Full-strength against attenuated weight for rescue-recovered requirements. Inconclusive: two runs failed 14/17 and 16/17 calls on a tokens-per-day ceiling, so neither arm exercised the comparison. No delta reported. | Unresolved |
| [`evaluate_substring_fallback_weight.py`](studies/evaluate_substring_fallback_weight.py) | Re-tested the removed substring/token fallback at attenuated weights, which experiment 69 never did -- it compared fragments at CONSTRAINT against no fragments at all. Attenuation removes most of the damage on attribute paraphrase (-0.0228 at 1.00, -0.0025 at 0.48, -0.0047 at 0.15) and the five decision criteria stay flat, but the sign never turns and the curve is non-monotonic, so 0.48 is an optimum rather than a trend. | Rejected |

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
