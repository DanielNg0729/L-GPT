# V2.02 Provenance Gate, Dense RAG, and Attribute Tags

## Gate audit

The full provenance gate opens only when a complete phrase is absent, lexical span provenance
is weak, the phrase is not a known evaluator construction, and it is not a long or formatted
literal source continuation.

| Suite | Eligible semantic values | Total values |
|---|---:|---:|
| Official200 canonical | 0 | 800 |
| Unseen800 canonical | 0 | 3,197 |
| Official200 value-only paraphrase | 354 | 800 |
| SemanticShift-Dev200 | 483 | 712 |

Actual Unseen800 evaluation with the tag agent is unchanged: 0.943250, with 2,123 blocked
constraint observations and zero semantic activations.

## Candidate comparison on Official200 value-only development

| Candidate | Score | Canonical replay |
|---|---:|---:|
| Literal V1 | 0.765800 | 0.969600 |
| Dense product-passage RAG | 0.765800 | 0.969600 |
| Attribute tag prediction | 0.882300 | 0.969600 |
| Attribute tag plus dense RAG | 0.882300 | 0.969600 |

Dense product-passage RAG is rejected. It retrieved the target in top 60 for only 1.8 percent
of queried sessions. Attribute tagging is retained as the V2 leading candidate. It predicts a
small semantic attribute rule, then adds only the corresponding catalogue-attested canonical
phrase. Dense RAG adds no measurable benefit after tagging.
