# Model Artifact and Checkpoint Policy

Every trainable V2 experiment must preserve artifacts before any evaluation or integration.

1. Save the immutable base-model reference and its local path or revision.
2. Save a run manifest with the script path, data manifests, seeds, hyperparameters, and
   training-start timestamp.
3. Save the selected best development checkpoint or the final checkpoint, including
   tokenizer and classifier head.
4. Never overwrite a prior selected run. Use a new versioned run directory for each rerun.
5. Do not select a checkpoint using a sealed holdout.
6. Record the selected artifact path and its measured result in the experiment log.

This permits exact regression comparison and recovery even when a later experiment fails.
