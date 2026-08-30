# V2 Semantic Roadmap and Experiment Contract

## Governing rule

Every semantic change is an isolated experiment before it can influence the submission.
No semantic component may replace exact catalogue evidence, modify V1 behavior by default,
or be accepted solely because it improves a synthetic development set.

**Continuity checkpoint.** At the start of any resumed or compacted working context,
re-read this roadmap before selecting the next semantic experiment. The current-node table
is the source of truth for what is complete, blocked, or still only a baseline.

## Non-interference contract

Every candidate must report all of the following before adoption:

| Suite | Required outcome |
|---|---|
| Official200 canonical replay | No score regression; semantic contribution must be zero or empirically harmless |
| Unseen800 canonical population | No score regression; semantic contribution must be zero or empirically harmless |
| Public value-only semantic development | Improvement over the declared baseline |
| Public value-only semantic holdout | Improvement without post-hoc tuning |
| Legacy target-disjoint semantic shift | Reported as secondary transfer evidence, never used as the sole adoption criterion |
| Override stress suite | No regression when no true semantic contradiction exists |

## Gate taxonomy and node assignment

The labels below describe the gate required before a node can alter V1 behaviour.
They do not describe model quality. A model may execute for diagnostics while its
effective ranking weight remains zero.

| Gate class | Meaning |
|---|---|
| **G1: exact exclusion** | A deterministic predicate proves that the message remains on the released literal path. The semantic node is unreachable on that traffic. |
| **G2: pessimistic hard signal** | The predicate is not a logical proof, but a deliberately conservative threshold can make activation empirically zero on Official200 and Unseen800. |
| **G3: soft attenuation** | The node may run on canonical traffic. A calibrated confidence and separate semantic weight must make its effective contribution zero, or empirically harmless, there. |
| **G4: learned decision required** | No safe structural predicate identifies when the action is correct. A learned semantic decision, evaluated on dedicated counterfactual data, is required before it can change V1 state. |

| Node | Required gate class | Exact reason and operational rule |
|---|---|---|
| 1. Route | **G1** | `recognised(message)` matches every released organizer wrapper. When true, the classifier is unreachable. V2.23 measured zero model loads and zero inferences on Official200 and Unseen800. |
| 2. Span extraction | **G1** | The same literal wrapper gate selects the existing template parser. With preserved wrappers, V2.03 recovered 460 of 460 exposed values verbatim. A learned span extractor is only eligible after wrapper recognition fails. |
| 3. Attribute family | **G3** | Family prediction is not evidence by itself and top-one routing is only 0.5351. It may narrow a semantic candidate search, but must retain a global fallback and may not delete literal evidence or hard-prune V1 candidates. |
| 4. Canonical attribute retrieval | **G3** | Absence of a useful literal resolution is a necessary entry condition, not proof that a semantic nearest neighbour is correct. The retrieved phrase therefore needs calibrated weak evidence, not a hard replacement. |
| 5. Equivalence verification | **G3** | Similarity and margin are continuous, and the pretrained verifier AUROC is 0.7758 rather than a proof. It can attenuate or reject a semantic candidate, but cannot be the sole hard gate. |
| 6. Calibration and provenance | **G3** | This is the explicit soft gate. Canonical provenance must yield zero effective semantic contribution; unfamiliar wording may receive only confidence-weighted contribution after all predecessor nodes pass. |
| 7. Semantic evidence integration | **G3** | A separate weak tier is required. Its weight is tuned pessimistically under Official200 and Unseen800 non-regression constraints, never substituted for literal evidence. |
| 8. Direct product retrieval | **Not active** | Dense product RAG is rejected. If reconsidered, it would require at least G3 because a dense product score has no hard semantic correctness signal. It must not be introduced through a G2 threshold alone. |
| 9. Semantic override | **G4** | An explicit replacement cue identifies that a change occurred, but does not identify which earlier value was withdrawn or whether two values are truly incompatible. The failed literal conflict rules demonstrate that no safe structural gate exists. A learned semantic contradiction decision plus compatible and contradictory override tests is required. |
| 10. Clarification policy | **G1 for current V1; G3 for any semantic extension** | Current V1 uses a fixed, catalogue-grounded probe order, so no learned gate executes. A future semantic question policy may use calibrated candidate uncertainty, but must not replace the fixed policy unless it is non-regressing on Hit Rate and MTTC. It is not currently a G4 problem because the decision can be grounded in measured candidate uncertainty rather than opaque dialogue intent. |

There is intentionally no active G2 node. A threshold that merely appears safe on a
small canonical replay is weaker than either a true G1 exclusion or a G3 contribution
that is explicitly calibrated to zero. G2 may be considered only when a future node
has a stable, independently measured hard signal and a documented pessimistic threshold.

### Existing V1 BERT within this roadmap

The shipped `ScaffoldingTagger` is an existing **Node 2 preprocessor**, not a new
semantic resolver. After the G1 wrapper-recognition gate fails, it labels message tokens
as scaffolding or product-content and returns a reduced text string. The inherited V1
catalogue miner then performs the actual lexical phrase recovery. It has no route labels,
does not choose an attribute family, does not retrieve a canonical synonym, does not
verify equivalence, and does not alter ranking weights.

Its relationship to the pipeline is therefore:

```text
unknown wrapper -> Node 1 route -> V1 BERT scaffolding removal (Node 2 preprocessor)
                -> literal catalogue mining -> later V2 semantic nodes only if literal recovery fails
```

For a recognized organizer wrapper, neither Node 1 nor the V1 BERT can execute. The
deterministic template parser extracts spans directly. The V1 BERT remains a retained,
already-shipped fallback within Node 2, while a future V2 semantic span extractor must
be measured against it rather than silently replacing it.

## Semantic nodes

### Approved resolver architecture

Nodes 3 through 6 are implemented and evaluated as one **semantic attribute resolver**,
not as independently activated runtime features. The deterministic G1 wrapper gate remains
separate and always has precedence. For an unknown wrapper, the resolver may run only when
exact catalogue recovery has not already resolved the relevant value.

```text
unknown wrapper and unresolved literal value
  -> Node 3: broad family distribution, retaining a global fallback
  -> Node 4: canonical top-k attribute retrieval
  -> Node 5: pairwise equivalence and non-equivalence verification
  -> Node 6: one calibrated acceptance gate
  -> accepted canonical attribute plus confidence and provenance, or abstain
```

Node 5 is necessary because nearest-neighbour retrieval alone cannot distinguish a true
synonym from a related attribute, a subtype, a broader term, a sibling mechanism, or an
inferred product property. Node 6 is the sole learned semantic gate: it consumes retrieval
rank, similarity margin, verifier score, family support, and provenance. Node 4 supplies
the top five canonical candidates; Node 5 evaluates each candidate pair; Node 6 can admit
only the highest verified candidate within that top-five set. It must abstain
when those signals are insufficient. Node 7 remains separate because correctness of an
accepted semantic mapping and the amount of ranking influence assigned to that mapping are
different decisions. It is used only if a nonzero, pessimistically calibrated weak-evidence
weight is required after the resolver itself passes holdout evaluation.

```mermaid
flowchart LR
  M[Message] --> R[1. Route and dialogue act]
  R --> S[2. Constraint span extraction]
  S --> F[3. Attribute family distribution]
  F --> C[4. Canonical attribute retrieval]
  C --> V[5. Equivalence verification]
  V --> P[6. Provenance and confidence calibration]
  P --> E[7. Semantic evidence tier]
  E --> K[8. Existing V1 candidate ladder and ranking]
  K --> O[9. Semantic override update]
  K --> Q[10. Clarification policy]
```

| Node | Candidate approaches | Required experiment |
|---|---|---|
| 1. Route | Regex primary, local dialogue-act fallback, no runtime rewriting | Format-paraphrase routing accuracy and canonical reach rate |
| 2. Span extraction | Existing BERT tagger, QA extraction, offline generative labels | Span precision/recall and held-out paraphrase family |
| 3. Family | Coarse multi-label route plus global retrieval | Family accuracy, top-two recall, failure isolation |
| 4. Canonical retrieval | Fine-tuned bi-encoder, late interaction, top-k cross-encoder | Canonical recall@k and semantic end-to-end score |
| 5. Verification | Similarity and margin, NLI/cross-encoder, catalogue rules | Equivalence precision against sibling, subtype, and mechanism negatives |
| 6. Calibration | Soft provenance, selective prediction, conformal abstention | Canonical contribution distribution and held-out semantic precision |
| 7. Integration | Separate weak semantic evidence tier only | Pessimistic weight sweep under non-interference constraints |
| 8. Product retrieval | Existing lexical ladder first; sparse or late interaction only later | Compare only after attribute resolver passes holdout |
| 9. Override | Explicit cue plus same-family high-confidence incompatibility | Compatible and contradictory semantic override suites |
| 10. Clarification | Fixed question set plus candidate entropy or expected value | MTTC and hit-rate experiment, no generated question text |
| Multilingual metadata | English veto, Unicode index, separate multilingual model | Separate corpus and benchmark; never mixed into English branch silently |

## Accepted design decisions

- Semantic models retrieve canonical attributes, never products directly.
- Dense product RAG is rejected for this task because it retrieves generic product semantics rather than a precise catalogue attribute.
- A flat 7,922-way attribute classifier is rejected. Use family routing and candidate retrieval instead.
- Runtime LLM rewriting and runtime LLM reranking are out of scope for the V2 core.
- The soft provenance gate routes canonical traffic but gives complete lexical provenance zero semantic weight.
- Empty English normalisation is a hard veto for the current English local encoder.

## Literature map and retained rationale

The roadmap is intentionally staged. The cited work supports evaluating each node
independently before it can affect retrieval or ranking.

| Node | Relevant literature | Retained implication |
|---|---|---|
| Dialogue state and route | [TripPy](https://aclanthology.org/2020.sigdial-1.4/), [Dual Slot Selector](https://aclanthology.org/2021.acl-long.12/) | Track update and carry-over independently. Do not infer an override from lexical novelty alone. |
| E-commerce extraction | [E-commerce NER versus QA](https://aclanthology.org/2023.emnlp-industry.16/), [Explicit Attribute Extraction](https://aclanthology.org/2024.ecnlp-1.13/) | Measure extraction spans separately from downstream ranking. |
| Attribute retrieval | [Sentence-BERT](https://arxiv.org/abs/1908.10084), [SimCSE](https://arxiv.org/abs/2104.08821) | Train a phrase-to-canonical-attribute encoder, then report recall at k. |
| Equivalence verification | [MultiNLI](https://aclanthology.org/N18-1101/) | Treat synonymy, subtype relations, and incompatible mechanisms as separate outcomes. |
| Retrieval alternatives | [ColBERTv2](https://aclanthology.org/2022.naacl-main.272.pdf), [GPL](https://arxiv.org/abs/2112.07577) | Consider late interaction only after the attribute resolver succeeds. Product-level dense RAG remains rejected by existing evidence. |
| Abstention and calibration | [Selective classification](https://proceedings.mlr.press/v130/gangrade21a.html), [Conformal limited false positives](https://proceedings.mlr.press/v162/fisch22a.html) | Calibrate semantic weight and abstention against canonical non-interference, not semantic development score alone. |
| Clarification policy | [Clarification EVPI](https://aclanthology.org/P18-1255/), [Content-grounded questions](https://aclanthology.org/2022.dialdoc-1.7/), [Conversational query reformulation](https://aclanthology.org/2023.acl-long.274/) | Evaluate questions by hit rate and MTTC, with a fixed, catalogue-grounded question set. |

## Correction log

### V2.32 / V2.33 -- two measurement defects invalidated the Node 4 and Node 5 conclusions

**Defect 1: the metric was not the contracted one.** `V2_NODES_3_TO_5_DATA_CONTRACT.md`
requires cluster-aware scoring ("counts a hit when its top-k contains any member of the
correct cluster"). `pretrained_attribute_baseline.py` scored a single exact string, so
`light weight` was counted wrong when the target was `lightweight`. Frozen rescore with the
contracted metric: R@1 0.0000 -> 0.0590, MRR 0.0486 -> 0.0880.

**Defect 2: the benchmark has 67 concepts, not 712 examples.** The 712 rows collapse to 67
distinct (paraphrase, canonical) atoms, frequency-weighted by session recurrence -- "made
overseas" alone is 100 rows, 14% of the reported sample. Every Node 3/4/5 metric therefore
has a resolution of roughly 1/67, and no configuration selection is possible on it.

**Defect 3: the fine-tunes never ran.** 158-178 pairs at batch 16 for 3 epochs is ~30
optimizer steps; the saved checkpoints differ from frozen by `max|delta|` = 2.3e-4 on word
embeddings, and two differently-trained checkpoints produced byte-identical Recall@1/3/5/10.
V2.33 re-ran with 600 steps and in-batch negatives (loss 0.596 -> 0.025) and moved MRR by
+0.0001. So "fine-tuning does not help" is now a measured result rather than an artifact --
and it points at data, not optimisation.

### V2.34 -- half the retrieval index can never be a correct answer

The candidate dictionary was built from phrases ATTESTED IN CATALOGUE TEXT. The only
phrases a customer can ever say are the ones `intent_card()` emits, which is a different
set. Replaying the generator's constraint construction over all 50,000 products:

| quantity | value |
|---|---|
| distinct phrases the simulator can ever emit | 58,801 |
| dictionary entries that are emittable | 3,873 |
| dictionary entries that are NEVER emittable | **4,049 (51.1%)** |
| emission mass the dictionary covers | 63.5% |

Slightly over half the index consists of candidates that cannot be correct under any
session while still competing for top-k slots. Restricting the index to emittable phrases,
same frozen encoder, same cluster-aware metric, like-for-like on the 56 concepts
representable in both indexes:

| metric | full (7,922) | emittable (3,873) | delta |
|---|---|---|---|
| R@1 | 0.0536 | 0.0893 | +0.0357 |
| R@10 | 0.1429 | 0.2321 | +0.0893 |
| MRR | 0.0878 | 0.1332 | +0.0454 |

Free: no training, no data, and the index halves at inference. This is an index
construction error, not a modelling problem, and no amount of training would have fixed it.

Caveat: with 56 concepts, R@1 0.0536 -> 0.0893 is three concepts becoming five. The
*mechanism* is what justifies the change -- removing candidates that can never be correct
cannot hurt -- rather than the size of the measured delta.

Also exposed: the benchmark's colour canonicals are unreachable by construction. The
simulator emits `color: green`, never bare `green`, so benchmark rows whose canonical is
`Green`/`Beige`/`Brown`/`Gray` can never be matched. Regenerate those with the emission
convention.

### V2.35 -- the emittable inventory is a SOFT PRIOR, not an index filter (supersedes V2.34's form)

V2.34's substance holds: for the RELEASED simulator, every constraint is `intent_card()`
output, so the canonical behind any paraphrase is emittable by construction. Its *form* was
wrong in two ways.

**It broke this roadmap's own gate discipline.** Node 3/4 are G3: a semantic narrowing
"must retain a global fallback and may not hard-prune". A hard index filter is a G1 action
taken on a G3 signal, and the private simulator is precisely the case we cannot verify.

**It already destroyed correct answers.** Eleven of 67 benchmark concepts became
unreachable under the filter -- `Green`, `Beige`, `Brown`, `Gray` -- because the simulator
emits `color: green` and never bare `green`. There the filter removed the truth, not a
distractor.

Re-run as an additive similarity bonus with every phrase still reachable:

| bonus | R@1 | R@10 | MRR | unreachable |
|---|---|---|---|---|
| 0.00 (baseline) | 0.0448 | 0.1194 | 0.0761 | 0 |
| 0.05 | 0.0746 | 0.1493 | 0.1043 | 0 |
| 0.20 | 0.0746 | 0.1940 | 0.1109 | 0 |
| 0.50 | 0.0746 | 0.1940 | 0.1114 | 0 |

| | MRR gain | concepts scored | unreachable |
|---|---|---|---|
| hard filter (V2.34) | +0.0454 | 56 | **11** |
| soft prior (V2.35) | +0.0353 | **67** | **0** |

The prior captures ~78% of the gain, destroys nothing, and degrades to the baseline if our
reconstruction of the generator is wrong.

**TRAINING MUST NOT RELY ON THIS.** The prior is an inference-time device. Training targets
span the FULL catalogue inventory, not the emittable subset: an encoder trained only on
emittable targets never learns the other 4,049 phrases, so if the private simulator emits
any of them the model is blind and there is no signal that would tell us. Training against
the harder distribution and applying a provable prior at inference also keeps the two
separable, so the prior's contribution stays independently measurable and independently
removable.

### V2.36 / V2.37 -- the relation was wrong, and fixing it solved Node 5 outright

**The synonym surface is small, and we measured it twice.** ConceptNet supplies a genuine
non-morphological synonym for 8.1% of the dictionary. An LLM generation pass over 1,382
sampled catalogue phrases produced usable non-degenerate paraphrases for 53 of them (3.8%)
and declined 46% outright as having no natural spoken form; 94% of what it did produce
contained the canonical verbatim ("polyurethane sole" -> "polyurethane sole"). By family:
style 10.2%, feature 4.4%, material 1.2%, colour 0%, size 0%. Most catalogue attributes
simply have no synonym, and no amount of generation changes that.

**But synonymy was the wrong relation.** At runtime the resolver asks "does this catalogue
attribute SATISFY the customer's requirement", which is asymmetric and far larger:
`leather` <- `genuine leather`; `synthetic` <- `polyester`; `warm` <- `fleece lined`. An
equivalence verifier rejects all three. And symmetric similarity is not merely weaker, it is
unsafe: `cotton` and `polyester` are close in embedding space while being mutually
exclusive, so any similarity threshold that accepts synonyms also accepts those. Directional
entailment cannot make that error.

**Entailment also dissolves the data problem.** Synonym corpora must be generated and cap
out at a few hundred concepts. Entailment has ~400k human-annotated public pairs
(MultiNLI/SNLI) and off-the-shelf zero-shot cross-encoders.

Measured on the frozen 134-row verifier test, same rows and metric as the fine-tuned runs,
using `cross-encoder/nli-deberta-v3-small` with **no training at all**:

| scoring | AUROC |
|---|---|
| entailment A->B | **0.9726** |
| mutual entailment (min) | 0.9585 |
| entail_max | 0.8801 |
| V2.05 frozen bi-encoder | 0.7758 |
| V2.30 fine-tuned | 0.7840 |
| V2.31 fine-tuned | 0.7782 |

**Zero-shot entailment beats every fine-tuned equivalence encoder by +0.17 to +0.19 AUROC.**

**Consequence for the architecture.** Node 5 is no longer the blocker; Node 4 is. With a
0.97-AUROC verifier, the pipeline becomes retrieve-then-verify: Node 4 needs adequate
Recall@k rather than precision at rank 1, and Node 5 adjudicates. That is a different and
much easier requirement than the one Node 4 has been failing.

Cost note: the NLI cross-encoder is ~1.1 GB. It runs only behind the G1 wrapper gate on
unrecognised messages, and V2 contributes zero weight on canonical traffic, so this is a
demo/robustness dependency rather than a scored-path one.

**Consequence.** Nodes 5, 6 and 7 remain blocked, but for a corrected reason: not "the
models underperform" but "the corpus covers 2.2% of the candidate index and the benchmark
cannot resolve differences". Data expansion is the critical path.

## Experiment record requirement

Every attempted alternative, including negative results, must receive a dated entry in
the experiment log with: hypothesis, implementation scope, fixed data split, metrics,
canonical non-interference outcome, decision, and the reason for acceptance or
rejection. A result cannot be promoted from exploratory evidence to the submitted
pipeline without this record and the required suite results above.

## Pretrained comparison requirement

Every fine-tuned semantic result must be accompanied by its frozen-pretrained equivalent.
The comparison must use the same candidate dictionary, dataset split, query construction,
metrics, confidence gates, and end-to-end integration weight. Report the absolute delta,
not only the fine-tuned score. A fine-tuned model that cannot beat this matched baseline on
development and retain that benefit on the sealed holdout is rejected.

## Sequencing

1. Complete and audit verified synonym corpus generation.
2. Fine-tune the local attribute encoder.
3. Evaluate attribute retrieval and equivalence verification independently.
4. Integrate the calibrated semantic evidence tier.
5. Tune only among configurations satisfying canonical non-interference.
6. Freeze the public value-only holdout before final selection.
7. Evaluate semantic override, then clarification policy.
8. Consider product-level neural retrieval only if the grounded attribute path plateaus.

## Current node status

| Node | Current evidence | Status and next condition |
|---|---|---|
| 1. Route | V2.02: 1,943 of 1,943 messages correctly recognised. V2.22 route classifier: 0.990938 fixed-test masked accuracy. V2.23: exact Official200 and Unseen800 identity, with zero model loads and inferences. | **Complete and wired into the V2 runtime.** `RouteOnlyV2Agent` applies the strict literal gate, lazy shared six-route classifier, and deterministic turn mask. V1 remains unchanged. |
| 2. Span | V2.03: 460 of 460 exposed values recovered verbatim under preserved wrappers. On the **Test split of TemplateParaphrase9600**, V2.24 BERT retained 99.25% of canonical constraint slots but V1 mining recovered 0.00% of short constraints. V2.25 exact lookup recovered 99.25%; category masking reduced extras to 0.152. The authoritative V2.27 replay used Official200's original cards, behavior, and scoring while swapping only held-out wrappers: raw 0.720525, route-only 0.738198, route plus exact category and short spans 0.939900, with HR@10 0.970000 and MTTC 2.680000. | **Complete for wrapper paraphrase.** Exact catalogue-attested category and short-span recovery is accepted behind G1 for unknown wrappers. V1 remains unchanged. Attribute-value paraphrase is deliberately out of this node's scope and proceeds to semantic resolution. |
| 3. Family | V2.04 frozen encoder: top-one 0.5351, top-two 0.9522 | Baseline only. Use soft top-two routing with global fallback; compare any trained router. |
| 4. Canonical retrieval | **V2.32 correction:** the earlier numbers were produced by a metric the data contract does not sanction (single exact string, not cluster-aware) on a benchmark whose 712 rows contain only **67 distinct concepts**. Cluster-aware frozen rescore: R@1 0.0000 -> 0.0590, MRR 0.0486 -> 0.0880 (weighted); per-concept R@5 0.1194, MRR 0.0761. **V2.33:** the V2.30/V2.31 fine-tunes ran ~30 optimizer steps and moved weights by 2.3e-4, so fine-tuning was never actually tested. A real 600-step run with in-batch negatives drove loss 0.596 -> 0.025 and changed **MRR by +0.0001** (0.0761 -> 0.0762). | Training is no longer the missing ingredient; **data is**. Concept coverage is ~175 of 7,922 (2.2%) and the benchmark resolves only 67 concepts. Required next: a concept-expanded train-only corpus and a concept-split benchmark. Do not select configurations on the 67-concept set. |
| 5. Verification | V2.05 frozen encoder: AUROC 0.7758. V2.30 strict-only: 0.7840. V2.31 overlap-augmented: 0.7782 on the same fixed 134-row verifier test. **These deltas inherit the V2.33 defect: the models they compare received ~30 optimizer steps, so they are three readings of approximately the same frozen encoder.** | Do not treat the 0.7758/0.7840/0.7782 spread as a fine-tuning result. Re-run after Node 4 has a real corpus. Frozen verifier anchors remain excluded from all training. |
| 6. Calibration | V2.06 proved only lexical-provenance non-interference: zero nonzero contribution on Official200 and Unseen800. The former audit used a fixed control prediction, not actual resolver outputs. V2.29 CUDA feasibility baseline found 0 of 712 correct frozen top-one canonical candidates, so every threshold grid cell had zero correct accepted examples. | **Scaffold complete; calibration blocked.** Node 6 now has a fail-closed contract that consumes family, retrieval similarity and rank, verifier score, dictionary attestation, and provenance. A threshold is meaningless until Nodes 3 to 5 supply correct candidate mappings. |
| 7. Integration | The explicit `SemanticIntegrationPolicy` defaults to maximum weight 0.0 and is a strict score identity transformation. | **Scaffold complete; nonzero weight blocked.** Select one monotonic schedule only after Node 6 is calibrated and both Official200 and Unseen800 show no ranking regression. |
| 8. Product retrieval | Existing dense product RAG result rejected | Closed unless grounded attribute resolution plateaus after tuning. |
| 9. Override | Existing literal conflict experiments reject universal reset | Await a semantic contradiction detector with its own counterfactual suite. |
| 10. Clarification | V1.65 uniform candidate-partition controller: Official200 0.969600 to 0.970100 with unchanged HR@10; review-weighted Unseen800 0.956313 to 0.957038; uniform population 0.882869 to 0.881229; inverse population 0.895937 to 0.899631. Integer signatures cost 69.57 ms/session versus 59.49 ms fixed V1. | **Complete for the current simulator contract.** V1 uses the uniform, catalogue-grounded controller with a fixed-order fail-safe. Do not add a semantic question generator unless it clears the same non-interference and runtime bar. |

## V2.45 / V2.46 — the LLM resolver, and what redirected the programme

### Suppression first (shipped)

The V2.43 oracle decomposition showed that of the 0.1931 attribute-paraphrase gap, a
perfect resolver recovers +0.0979 while merely DELETING the unresolvable clause recovers
+0.0467. A quarter of the loss was self-inflicted: `_resolve`'s substring and single-token
fallbacks were contributing debris (`soft`, `plant` from "made from a soft plant fibre")
at full constraint weight. Deleting both fallbacks is now shipped in V1, measured on the
shipped path across all six suites:

| suite | before | after | delta |
|---|---|---|---|
| official200 | 0.970100 | 0.970100 | +0.000000 |
| org-proxy | 0.952788 | 0.952788 | +0.000000 |
| review800 | 0.945125 | 0.945125 | +0.000000 |
| uniform | 0.882763 | 0.882763 | +0.000000 |
| inverse | 0.866262 | 0.866062 | **-0.000200** |
| attr-para | 0.777000 | 0.833000 | **+0.056000** |

The `inverse` move is stated as a regression rather than rounded to a wash. It is an order
of magnitude inside the suite's bootstrap noise, and `inverse` is an adversarial bound.

**0.8330 is now the bar.** Any semantic resolver has to beat "delete the clause", not
"keep the debris" — a materially higher bar, because a confidently wrong canonical is
worse than an absent one.

### Naming correction: this is deparaphrasing, not RAG

V2.45/V2.46 were first labelled an "LLM RAG resolver". That is wrong, and the distinction
is a design question rather than a wording preference.

| | where the catalogue enters | what the model sees |
|---|---|---|
| what was built | **after** generation, as a provenance filter (`df > 0`) | the paraphrase alone |
| what RAG would be | **before** generation, as prompt context | k retrieved candidates |

The measured arm is **generate-then-verify** — paraphrase inversion from parametric
knowledge, gated on provenance. The model never sees a catalogue string.

The genuinely retrieval-augmented arm (`choose`: retrieve k candidates, model picks one or
answers NONE) was described in V2.45's plan and **never built**. It remains open, and it is
the more interesting branch for two reasons. It reuses the failed encoders honestly — their
top-1 was 0/27, but a candidate list needs RECALL@k, which was never measured. And it is
the arm that could convert abstentions into answers, since a model that declines to name a
value unprompted may recognise it in a list. V2.46's result says nothing about it.

### V2.45 — the probe, and a metric that was wrong in the model's favour

Scored by string equality against the suite's canonicals, the LLM resolved 3/27. That
number is wrong: the canonicals are RAW CATALOGUE STRINGS, so equality demanded the model
reproduce the whole thing.

| paraphrase | model | "canonical" | scored |
|---|---|---|---|
| made from a soft plant fibre | `cotton` | `100 cotton size 3t 2 3 4t 3 4 5 5` | wrong |
| in the darkest colour | `black` | `color black` | wrong |
| slips on without separate fasteners | `pull on` | `pull on closure` | wrong |
| made from heavy woven cloth | `canvas` | `canvas upper` | wrong |
| keeps water from penetrating | `waterproof` | `100 waterproof women s ...` | wrong |

The correction was NOT to loosen the string metric — a similarity threshold here is a free
parameter that could be tuned until the answer looked good. It was to stop scoring the
intermediate representation and score end to end.

The probe also corrected itself in a second way: its first run reported "27 abstentions"
that were in fact 27 HTTP 403s (Groq's edge rejects Python's default `urllib` identity).
An empty completion and a declined answer are now counted separately.

### V2.46 — end to end, against the suppression floor

| arm | attr-para | vs floor | % of the V2.43 oracle |
|---|---|---|---|
| suppression (shipped, no ML) | 0.833000 | — | 57.2% |
| LLM @ CONSTRAINT weight | 0.856800 | +0.0238 | 81.5% |
| LLM @ weak tier w=0.15 | 0.870800 | +0.0378 | 95.8% |
| LLM @ weak tier w=0.30 | 0.867300 | +0.0343 | 93.5% |
| LLM @ weak tier w=0.45 | 0.871900 | +0.0389 | 96.9% |
| ORACLE-RESOLVE (perfect) | 0.874925 | +0.0979 | 100.0% |

This is the first learned component to beat the no-ML baseline on attribute paraphrase,
after seven failures from encoder-based directions. It succeeds for the reason those
failed: the task is world knowledge over a closed answer set, not ranking. Bi-encoders
retrieved antonyms because cosine similarity does not encode negation; the LLM correctly
ABSTAINS on the negation case (`made overseas` -> NONE) rather than inverting it.

**Attenuation is the mechanism.** Arms 2 and 3 have identical knowledge and differ only in
what a wrong proposal costs. G3 soft attenuation beats G2 pessimistic-hard by 15 points of
oracle, on its own terms.

**The weight is not tuned and must not be.** The three weak-tier weights span 0.0046 and
are non-monotone (0.45 > 0.15 > 0.30) — noise on a 200-session suite. Reporting w=0.45 as
"best" would manufacture a sharp optimum where the data shows a flat one. The finding is
that the weak tier is worth ~+0.037 and is insensitive to its weight across 0.15–0.45.

### Guards, and one claim that had to be withdrawn

* PROVENANCE: `df(proposal) > 0` before anything becomes evidence. The model proposes, the
  catalogue disposes. 0 proposals were rejected here, but the gate is what makes a
  hallucinated phrase impossible rather than merely unlikely.
* NO EMITTABILITY FILTER: restricting proposals to values `intent_card()` can emit would
  score better and would be gaming — the suite's canonicals are emittable by construction.
* REACHABILITY, **corrected**: this was first claimed as "unreachable on clean traffic by
  construction", since the recognition gate matches 463/463 clean messages. Measuring it
  showed the claim is false. The gate governs MESSAGES, not VALUES, and `intent_card()`
  truncates long feature bullets mid-word — `"All daughters love their mom, but sometimes
  we just forget to sa."` is genuine catalogue prose whose truncation breaks the phrase.
  It reaches the resolver once in 463 clean messages at exactly 0.000000 cost on
  official200. The guarantee is empirical, not structural.

### Blocking condition before any of this can ship

The suite carries **27 distinct paraphrases**, and the resolver was consulted 25 times.
One phrase resolving differently moves the rate by 3.7 points. That is a weakness of the
suite, not a property of the problem — a real customer population has an open-ended
paraphrase vocabulary.

So: nothing may be keyed, sized, cached or tuned to those 27 phrases (the runtime cache is
gitignored for exactly this reason), and **96% of oracle is not a shippable claim until it
is reproduced on an open-vocabulary suite whose paraphrases were generated independently
of these.** Until then V2.46 establishes sign and mechanism, not effect size.

## V2.47 — recall@k, and a correction to "the encoders are hopeless"

### The claim that was wrong

V2.41 measured encoder recall only to k=10, where dev200 sat at 0.0896–0.1493. That was
read as "retrieval has no signal, so a candidate-list arm is capped around 15%". Extending
the curve to the full 7,922-canonical index shows the opposite: it does not flatten, it
climbs steeply past k=10.

| encoder | R@10 | R@50 | R@100 | R@200 | median rank |
|---|---|---|---|---|---|
| MiniLM-L6 (22M, incumbent) | 0.1194 | 0.3284 | **0.7612** | — | 70 |
| all-mpnet-base-v2 | 0.1045 | 0.4478 | 0.7164 | — | 56 |
| bge-small-en-v1.5 | 0.1493 | 0.4030 | 0.6269 | 0.8657 | 72 |
| e5-base-v2 | 0.0896 | 0.3134 | 0.5373 | — | 96 |
| Qwen3-Embedding-0.6B | 0.1642 | 0.3284 | 0.6119 | — | 82 |

The correct canonical sits at **median rank 56–96**. The encoders are not clueless; their
signal is DIFFUSE. Every prior conclusion about them was drawn from top-1 or k≤10, which
is the wrong statistic for a candidate list — a list needs recall, and recall was never
measured past 10. `choose` therefore has a real ceiling near 0.76, not near 0.15.

### The LLM-based embedder does not help

The question "can we use the LLM's own encoder" has no answer at the API — `gpt-oss-120b`
is decoder-only and Groq serves no embedding endpoint. Open-weight LLM embedders are the
usable form of the idea, and they lose: **Qwen3-Embedding-0.6B (600M, 2025) is beaten at
R@100 by all-MiniLM-L6-v2 (22M, 2021)**, on both benchmarks. `gte-Qwen2-1.5B-instruct`
failed to load (`Qwen2Config has no attribute rope_theta`; its remote code needs a newer
transformers than this environment pins) and was not pursued, since Qwen3 already tested
the hypothesis. Scale and recency are not the missing ingredient.

### Selection protocol, and its visible cost

dev200 shares paraphrase vocabulary with the suite that scores the end-to-end arms, so
selecting a retriever or a k there would be selecting on the evaluation data. **Selection
happens on the train-only synonymy corpus only.**

| encoder | corpus R@20 | R@50 | R@100 | R@200 |
|---|---|---|---|---|
| MiniLM-L6 | 0.7418 | 0.8242 | 0.8681 | 0.8956 |
| all-mpnet-base-v2 | 0.7912 | 0.8571 | 0.8791 | 0.9231 |
| bge-small-en-v1.5 | 0.7363 | 0.7912 | 0.8407 | 0.8791 |
| **e5-base-v2** | 0.8022 | 0.8846 | **0.9396** | 0.9560 |
| Qwen3-Embedding-0.6B | 0.5604 | 0.7033 | 0.7747 | 0.8352 |

The corpus selects **e5-base-v2 at k=100**. That model is the WORST of the six on dev200
(R@100 0.5373 against MiniLM's 0.7612) — so honouring the protocol hands the `choose` arm
a materially weaker retriever than selecting on the evaluation surface would have. The
handicap is the evidence that the protocol is real rather than decorative, and it is
recorded here so the arm's result is not later re-read as if the best retriever had been
used.

k is likewise fixed at 100 from the corpus curve (0.8846 → 0.9396 → 0.9560 for k = 50 →
100 → 200, i.e. near saturation), never from the end-to-end score.

## V2.48 — retrieval augmentation is rejected, on measurement

| arm | attr-para | vs floor | vs generate |
|---|---|---|---|
| suppression (floor, no LLM) | 0.833000 | — | −0.0372 |
| **generate** (no candidates) | **0.870200** | **+0.0372** | — |
| choose (must pick from k=100) | 0.828000 | **−0.0050** | −0.0422 |
| hybrid (candidates as hints) | 0.860400 | +0.0274 | −0.0098 |

| arm | calls | accepted | abstained | off-list |
|---|---|---|---|---|
| generate | 23 | 20 | 3 | — |
| choose | 23 | 17 | 6 | — |
| hybrid | 26 | 21 | 2 | **2** |

**`choose` scores below the no-ML floor.** Constraining the model to a retrieved list is
worse than contributing nothing at all: it accepted 17 proposals and they were wrong often
enough to cost more than silence. This is despite a genuine 0.54 ceiling (V2.47), so it is
not a recall failure — the list contained the answer far more often than the model picked
it.

**The mechanism is visible in `off-list`.** Hybrid was told explicitly that candidates are
hints, often wrong, and that it should ignore them and answer freely when none fit. It
answered off-list **2 times out of 21**. Unaided, the same model answers freely 23 times
out of 23. So presenting a list captured the model's answer ~90% of the time regardless of
the instruction not to defer to it. That is anchoring, and it is why hybrid — which had a
strict superset of generate's information — still lost.

**Read the magnitudes differently.** `choose` −0.0422 is large and directional. The
hybrid−generate gap of −0.0098 is about one or two phrases out of 23 and should not be
over-read on its own; it is the `off-list` count, not the score gap, that carries the
anchoring finding. Run-to-run variance is also nonzero: generate scored 0.870200 here
against 0.870800 in V2.46 (Δ0.0006) despite temperature 0, so anything below ~0.001 is
noise.

### Verdict

**Retrieval augmentation is rejected for this task, by measurement rather than assumption.**
The shape that works is `generate` — deparaphrasing from parametric knowledge with a
`df > 0` provenance gate and no catalogue context. This retroactively vindicates the V2.45
naming correction: the mechanism is not RAG, and RAG is actively worse here.

The encoder question is now closed from both directions. Top-1 was the wrong statistic
(V2.47), but fixing that and giving the encoder its best legitimate role — supplying
candidates for an LLM to filter — still loses to using no encoder at all.

## V2.50 — Node 5 verifies the LLM's proposal, and works

### Why the shape changed

Node 5 scored 0.9726 AUROC zero-shot and was still unusable, because it was being asked to
pick a winner out of ~100 retrieved candidates and pairwise AUROC is not selection
precision. In the `generate` design the LLM emits ONE proposal and node 5 answers one
pairwise question — which is what 0.9726 actually measured. The LLM absorbed the selection
problem; the verifier got its real competence back.

### The invalid first probe, withdrawn

The first adversarial probe mined negatives as the bi-encoder's nearest neighbours OF THE
PROPOSAL. For "made from animal hide" → `leather` it produced `leather fabric`,
`leather material`, `leather suede`, `canvas leather` — all of which SATISFY the
paraphrase. They were labelled negative while being true positives, so its **AUROC 0.5760
is void** and is withdrawn rather than deleted.

The valid construction is within-family cross-pairing: the proposal resolved for one
paraphrase, tested against a DIFFERENT paraphrase in the same family. `leather` against
"made from a soft plant fibre" is attested, hard, and certainly false — competing answers
to the same question. The label follows from the construction, so there is nothing to tune.

### Result

| measurement | value |
|---|---|
| AUROC, correct proposal vs competing same-family value | **0.8349** |
| mean entailment, correct | 0.7010 |
| mean entailment, competing | **0.1966** |
| competing values rejected at threshold | 68/76 (89.5%) |
| competing values the `df>0` gate rejects | **0/76** |

**The semantic hole is real and node 5 closes most of it.** All 76 competing values are
catalogue-attested and pass the provenance gate untouched; entailment separates them from
correct proposals by 0.50 of mean score.

### The threshold does not transfer, and must not be fixed by tuning

Set at 0.938 on the frozen 134-row set (precision 0.943 there), it rejects **9 of 20
correct proposals** at runtime. Correct proposals average 0.7010, far below 0.938. The
frozen set is SYNONYM pairs, which entail strongly; runtime is PARAPHRASE → CANONICAL,
which entails weakly. This is the same distribution-shift failure that made 0.9726
misleading, one level up.

End to end on the attribute-paraphrase suite:

| arm | attr-para | vs floor | vs generate |
|---|---|---|---|
| suppression (floor) | 0.833000 | — | −0.0378 |
| generate (`df>0` only) | 0.870800 | +0.0378 | — |
| generate + node 5 verifier | 0.864800 | +0.0318 | **−0.0060** |

Exactly the pre-registered expectation: on this suite the verifier can only reject correct
answers, because the LLM rarely proposes a competing value here. The insurance is real and
currently unused.

**The obvious fix is the forbidden one.** Lowering the threshold until attr-para improves
would be tuning on the evaluation suite. The legal path is to recalibrate on TRAIN-ONLY
data using the same within-family construction — hard negatives mined as the nearest
canonical that the corpus does not list as a synonym — and apply the result unchanged.
Until that is done, node 5 stays measured but unintegrated.

### Status

Node 5 is **validated as a mechanism and blocked on calibration**. It is the first node in
the 3/4/5 cluster to survive contact with the runtime distribution.

## V2.49 — Node 7 clears, and "unreachable" was wrong everywhere

Node 7's clearance condition asks for no ranking regression on Official200 and Unseen800.
V2.46 had measured only Official200 and the paraphrase suite, so node 7 had never actually
been cleared.

| suite | suppression | LLM arm | delta | reaches | calls |
|---|---|---|---|---|---|
| official200 | 0.970100 | 0.970100 | **+0.000000** | 1 | 0 |
| org-proxy | 0.952788 | 0.952788 | **+0.000000** | 2 | 0 |
| review800 | 0.945125 | 0.945125 | **+0.000000** | 7 | 0 |
| uniform | 0.882763 | 0.882763 | **+0.000000** | 9 | 0 |
| inverse | 0.866062 | 0.866062 | **+0.000000** | 7 | 0 |
| attr-para | 0.833000 | 0.870800 | +0.037800 | 304 | 0 |

**Node 7 clears**: worst decision-criterion delta +0.000000.

### REACHES is not CALLS, and conflating them produced a false claim

The first version of this run reported `calls` and concluded "official200 triggered zero
calls, therefore byte-identical by construction". Both halves were wrong.

`calls` counts cache MISSES. The resolver caches, so the same run showed the
attribute-paraphrase suite at 0 calls while it plainly resolved hundreds of phrases. A
cached resolution is still a resolution.

The scoring arm also hooks `_observe`/`_extract_templated`, while `_resolve` is
additionally called from `_seed_from_override_opening` — which is the path by which
Official200's one unattested clause (an `intent_card()` bullet truncated mid-word) arrives.
Absence of calls was never evidence of absence of reach.

With a probe that counts reaches across all call sites: **every decision suite reaches the
resolver.** None is structurally untouched. That makes the result stronger, not weaker —
the resolver fires 26 times across the five decision suites and moves the score by exactly
zero. Safety by measurement on live invocations, rather than safety by never being invoked.

This is the third time in this programme that a number measured on the wrong quantity
looked like a verdict: top-1 for the encoders, string equality for V2.45, and
calls-for-reaches here.

## V2.51 — the open-vocabulary suite

Built to replace the 27-phrase suite that blocks every effect-size claim in this document.

| property | prior suite | open-vocabulary suite |
|---|---|---|
| distinct paraphrases | 27 | **204** |
| targets | Official200 | review800, **overlap 0** |
| generator | 27 hand-written rules | Claude Haiku (Anthropic) |
| solver under test | gpt-oss-120b | gpt-oss-120b (Groq) |
| generator independent of solver | n/a | **yes** |
| atoms rewritten | 551 across 192 sessions | 1,280 across 710 sessions |

**Generator independence is the assumption that would invalidate everything.** A model that
both writes and solves the paraphrases is inverting its own encoding, and the result would
be inflated by an unknown amount. Haiku writes, Groq's gpt-oss-120b solves.

### Filter results — 78.5% pass, against a prior attempt's 94% failure

| rejected | n | example |
|---|---|---|
| shared content stem | 33 | `color black` -> `in the darkest colour` |
| SKIP (product codes, measurements) | 16 | `platform measures approximately 1` |
| already catalogue-attested | 1 | `man made` -> `created in a laboratory` |
| **present in the prior 27-rule suite** | **6** | `cotton` -> `made from a soft plant fibre` |

The last row is a mistake made while building this suite and then caught. The generation
prompt illustrated the task with seven worked examples, **all copied from the old suite**,
and the generator reproduced them verbatim. Those are the only phrases a resolver could
plausibly have been measured on before — precisely the contamination this suite exists to
remove. Excluding them drops high-frequency atoms (`cotton`, `imported`, `leather`) and
cuts coverage from 2,092 atoms to 1,280. Shrinking the gap is the correct price.

Function words are excluded from the stem-overlap test. Comparing them rejected good rows
for the wrong reason (`pull on closure` vs `slips on without separate fasteners` collides
only on "on") and, since such collisions are commoner in multi-word atoms, would have
skewed the surviving suite toward single-word atoms. Content words are still compared in
full, so `hand wash only` -> `wash by hand` is still caught.

The suite ships with a canonical control materialised from the same sessions, because a
paraphrase score is only interpretable against the same base materialised the same way.
Results are compared as FRACTIONS of the gap, never as raw scores against Official200's.
