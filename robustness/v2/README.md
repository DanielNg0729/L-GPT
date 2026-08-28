# V2 Semantic Attribute Benchmark

This benchmark preserves the released evaluator's customer-message formats and changes only
target-derived attribute values. It tests semantic attribute understanding without mixing in
format variation or an estimate of the organizer's private target distribution.

The data builder selects target-disjoint catalogue products that have at least three visible
attributes from a manually curated semantic rewrite bank. Each card stores the original
catalogue phrase, its attribute bucket, and only the paraphrase active for that split.
The runner must use development rows for design and tuning. It must not derive a resolver
from the holdout rows before the final frozen evaluation.

The eventual V2 runner keeps the released message wrappers verbatim. It substitutes only
the value represented by `<attribute paraphrase>` below:

| Scenario | Preserved wrapper |
|---|---|
| Buying | `I'm looking for <category>. A key requirement is: <attribute paraphrase>.` |
| Browsing follow-up | `For that, what matters is: <attribute paraphrase>; <attribute paraphrase>.` |
| Intent override | `Actually, ignore my earlier preference. What I need is: <attribute paraphrase>.` |

Examples:

| Catalogue fact | Development family | Holdout family |
|---|---|---|
| `Imported` | `made overseas` | `sourced from another country` |
| `Buckle closure` | `fastens using a metal clasp` | `secured with an adjustable clasp` |
| `Water resistant` | `repels light moisture` | `handles rain without soaking through` |

No semantic-retrieval model is used to create the set. The transformation bank is explicit,
versioned, and auditable. It is a fixed synthetic benchmark, not a claim that private users
will use these exact phrases.

Build the data from the released catalogue:

```bash
python -m robustness.v2.build_semantic_attribute_sets
```

The generated files are placed in `robustness/v2/sets/` and record their checksums in a
manifest. The development and holdout rewrite families are disjoint by construction.

Run the V2 harness with its lexical control:

```bash
python -m robustness.v2.run_semantic_attribute --candidate literal
```

The first semantic candidate is intentionally transparent and development-only. It maps
a development paraphrase to an attested catalogue phrase only after the complete
normalised paraphrase has zero FTS5 matches. This is a control-flow test, not the final
semantic system:

```bash
python -m robustness.v2.run_semantic_attribute --candidate development-lexicon --public-control
```

`semantic_gate_after_public.semantic_triggered` must remain zero. This verifies that the
semantic path is unreachable when released public constraints already have literal
catalogue evidence.
