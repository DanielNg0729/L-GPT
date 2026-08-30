# Train-only semantic augmentation contract

Use a separate generator only for candidate training phrases. It may receive a list of
canonical attribute phrases from the visible frozen catalogue. It must not receive or read
any file under `robustness/v2/sets/`, any evaluation result, public session data, target
labels, or the current synonym corpus.

For each canonical phrase, produce zero to two short alternatives that are exact lexical
synonyms in a product-search constraint. Do not provide explanations, benefits, inferred
properties, causes, sources, broader or narrower terms, sibling materials, related colours,
or different closure mechanisms. Return JSONL only:

```json
{"canonical":"buckle closure","candidates":["clasp fastening"]}
```

The external output is untrusted. Before it can be used for training, import it with:

```powershell
& .\.venv-v2\Scripts\python.exe -m robustness.v2.import_train_only_synonyms path\to\external_pairs.jsonl
```

The importer accepts only catalogue-attested canonicals and rejects candidates overlapping
the frozen semantic development or holdout wording. The output is train-only. It cannot be
used for model selection, threshold tuning, or test construction.

## Catalogue-alias pass

For a separate catalogue-alias pass, the generator may receive the same canonical list and
return only equivalence links whose two endpoints both appear in that list:

```json
{"canonical":"zip closure","equivalents":["zipper closure"]}
```

These are catalogue-grounded **proposals**, not official semantic labels. Import them with:

```powershell
& .\.venv-v2\Scripts\python.exe -m robustness.v2.import_catalogue_equivalence_pairs path\to\catalogue_alias_pairs.jsonl
```

The importer rejects non-catalogue endpoints and emits auditable pair and cluster files.
Use these only as Node 5 verifier data until independent evaluation accepts an integration.
There is no fixed cap on direct equivalents in this alias pass: return every phrase that is
strictly interchangeable, and return an empty list when none are defensible.
