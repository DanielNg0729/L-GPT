# Official200 Value-Only Semantic Perturbation

This internal development suite is a deterministic perturbation of the released Official200
sessions. It exists to measure semantic attribute loss without changing target population,
popularity prior, dialogue policy, templates, or score calculation.

## Invariants

- The same 200 public targets, profiles, and scenario labels are retained.
- The local evaluator materialises the same target intent card and uses the same deterministic
  scenario seed and turn timing.
- Category text, message wrappers, boundary replies, ranking, candidate generation, and the V1
  population prior are unchanged.
- The only intervention is replacement of supported target-derived constraint values with a
  fixed semantic paraphrase.

## Files

| File | Purpose |
|---|---|
| `official200_canonical_replay.jsonl` | Materialised evaluator cards with original values. It must reproduce 0.969600 with V1. |
| `official200_attribute_paraphrase_dev.jsonl` | The same sessions, with known attribute atoms paraphrased. |
| `manifest.json` | Checksums, rewrite coverage, rule counts, and invariants. |

The development perturbation rewrites 551 attribute atoms across 192 sessions. Eight sessions
have no supported rule and therefore remain textually identical. This is intentional and is
recorded rather than silently dropped.

## Baselines

| Candidate | Score |
|---|---:|
| V1 canonical replay | 0.969600 |
| Literal V1 on paraphrases | 0.765800 |
| Strict semantic-feature baseline | 0.765800 |

The strict semantic-feature candidate activates zero times. It is a measured ML baseline, not a
candidate success. Future RAG, tag-guess, and combined candidates use this exact suite before
any target-disjoint semantic holdout is opened.
