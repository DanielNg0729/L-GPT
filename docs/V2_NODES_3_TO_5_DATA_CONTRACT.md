# V2 Nodes 3 to 5 data contract

This document fixes the exact supervision, split boundaries, and evaluation role for the
semantic resolver. A catalogue-attested alias and an unfamiliar paraphrase are different
training signals and must not be silently substituted for one another.

## Fixed objects

| Object | Meaning | May train? | May tune? |
|---|---|---:|---:|
| `catalogue_attribute_dictionary.jsonl` | 7,922 distinct catalogue-attested phrase strings. It is a phrase inventory, not a semantic ontology. | Candidate index only | No |
| `catalogue_equivalence_pairs_measurement_checked.jsonl` | 109 proposed, measurement-consistent direct aliases whose endpoints both appear in the catalogue. | Node 5 train only after cluster split | No direct runtime use |
| `cluster_level_paraphrases_train_only.jsonl` | Future unfamiliar phrases generated against a known alias cluster and rejected if already catalogue-attested. | Nodes 3, 4, and 5 | Internal split only |
| `catalogue_synonym_train_only_merged.jsonl` | Existing Groq/external unfamiliar phrase pairs. Existing aliases are removed. | Nodes 3, 4, and 5 | Internal split only |
| `catalogue_synonym_broad_overlap_train_only.jsonl` | Future non-identical paraphrases that retain one or more canonical words, for example `pure cotton`. | Nodes 3 and 4 representation training only | Never semantic-fallback acceptance or Node 5 ground truth |
| `semantic_attribute_development_200.jsonl` | Existing, repeatedly inspected semantic benchmark. | Never | Report-only comparison; not selection evidence |
| `semantic_attribute_holdout_800.jsonl` | Legacy target-disjoint semantic transfer suite. | Never | Secondary transfer report only |
| `frozen_equivalence_verification.jsonl` | Fixed 134-row pair-verification test. Its canonical anchors are excluded from all training. | Never | Node 5 final comparison only |

## Concept representation

An alias cluster is one concept with one selected representative and one or more valid
catalogue spellings.

```text
cluster 12
  representative: zipper closure
  members: [zip closure, zipper closure]
  future unfamiliar phrase: sliding toothed fastener
```

At inference, a successful semantic mapping returns the **cluster**, not merely the
representative. Any subsequent catalogue lookup expands to all cluster members. This avoids
penalising a correct semantic mapping merely because the target catalogue product uses a
different member spelling.

## Node 3: attribute-family routing

**Input:** one extracted unfamiliar attribute phrase, for example `sliding toothed
fastener`.

**Label:** the coarse family of its representative, produced by the existing deterministic
family mapping. The family model is a soft route only: it supplies a ranked family list and
the global candidate search remains available.

**Training rows:**

1. every accepted unfamiliar phrase from the future cluster-level generator, labelled by
   its cluster representative family;
2. every accepted unfamiliar non-alias phrase from the merged train-only corpus, labelled
   by its canonical family;
3. optional canonical phrase anchors, labelled by the same deterministic family, only as
   auxiliary regularisation and never as the sole semantic evidence.

**Not used as Node 3 training:** catalogue-to-catalogue alias pairs alone. They teach spelling
equivalence but provide no unfamiliar wording.

**Metric:** family top-one and top-two recall. A wrong family is not allowed to hard-prune
Node 4, because the frozen baseline already demonstrates that top-one routing is weak while
top-two recall is much stronger.

## Node 4: canonical-cluster retrieval

**Input:** the same unfamiliar phrase.

**Positive target:** a cluster identifier when one exists; otherwise the one canonical
phrase from a non-alias pair.

**Training tuple:**

```text
(unfamiliar phrase, positive cluster representative, hard negative catalogue phrase)
```

The hard negative is selected from the same broad family or shared lexical head, using the
existing deterministic negative bank. The model is trained to retrieve the representative,
but evaluation counts a hit when its top-k contains **any member** of the correct cluster.

**Training rows:** unfamiliar generated phrases only. Direct aliases are not one-to-one
Node 4 labels because both endpoints are valid retrieval targets.

Non-identical lexical-overlap augmentations may also be included in Node 4 representation
training, but they are tagged separately and cannot be cited as evidence that unknown
wording is resolved. Sampling must prevent this larger, easier tier from crowding out strict
unfamiliar phrases.

**Metrics:** cluster Recall@1, Recall@3, Recall@5, and MRR over the full 7,922-phrase index.
The exact frozen pretrained baseline is always run with the same cluster-aware metric.

## Node 5: pairwise equivalence verification

**Input:** `(unfamiliar phrase or candidate alias, retrieved catalogue phrase)`.

**Positive rows:**

1. the 109 measurement-checked proposed catalogue alias pairs, split by whole cluster;
2. future unfamiliar cluster phrase paired with every member of its cluster, with one
   sampled positive per cluster per batch to avoid giving large clusters excess weight;
3. existing unfamiliar non-alias pair matched to its canonical phrase.

**Negative rows:** the existing catalogue-grounded negative bank: shared-head, same-family,
and unrelated non-equivalent pairs. For unfamiliar phrases, negatives are drawn from the
Node 4 top-k competitors, especially sibling mechanisms and compatible-but-not-equivalent
properties.

**Split:** no alias cluster may cross train, internal evaluation, or the frozen verifier
test. The fixed 134-row test remains untouched and its anchor canonicals remain excluded
from all training inputs.

**Metrics:** AUROC, precision-recall curve, and false-positive rates by negative type. AUROC
alone does not approve Node 5: the resulting score must improve Node 4 top-k selection.

## Split policy for future generated phrases

The split key is the semantic concept, not individual pairs. It is deterministic:

| Split | Use | Rule |
|---|---|---|
| Train | Model gradients | 80% of canonical or cluster IDs by stable hash |
| Internal evaluation | Epoch and configuration selection | 10% by stable hash |
| Generated test | One-time model comparison | 10% by stable hash; no model selection after opening |
| Fixed verifier test | Node 5 final comparison | Existing frozen file, never training |
| Legacy semantic suites | Transfer reporting | Existing development and holdout files, never training |

The generator may create candidates for all concepts, but the encoder training code must
load only the train partition. The evaluation and generated-test records must be physically
separate files before model selection begins.

## Nodes 6 and 7 dependency

Node 6 receives only actual Node 3 family support, Node 4 rank/similarity/margin, and Node 5
verifier score. It evaluates acceptance on the generated test split and reports precision
versus coverage. Node 7 remains exact zero weight until Node 6 has a viable precision-coverage
region and the full pipeline is non-regressing on Official200 and Unseen800.
