"""Fine-tune a local semantic attribute encoder from verified synonym pairs only."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from torch.utils.data import DataLoader
from sentence_transformers import InputExample, SentenceTransformer, losses

from robustness.v2.semantic_grounding import MODEL_CACHE, MODEL_NAME

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "robustness" / "v2" / "catalogue_synonym_train_only_merged.jsonl"
VERIFICATION = ROOT / "robustness" / "v2" / "sets" / "frozen_equivalence_verification.jsonl"
NEGATIVES = ROOT / "robustness" / "v2" / "sets" / "canonical_verification_negatives.jsonl"
OUT = MODEL_CACHE / "attribute_encoder"
REPORT = ROOT / "robustness" / "v2" / "results" / "attribute_encoder_training.json"


def held_out_canonicals() -> set[str]:
    if not VERIFICATION.exists():
        return set()
    return {
        str(row["canonical"])
        for row in (json.loads(line) for line in VERIFICATION.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def contextualise(value: str, key: str) -> str:
    """A deterministic wrapper adds language context without inventing a semantic label."""
    templates = ("{}", "I need a product with {}", "looking for {}", "a product that is {}")
    slot = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % len(templates)
    return templates[slot].format(value)


def load_examples(training: Path) -> tuple[list[InputExample], int, int]:
    rows = [json.loads(line) for line in training.read_text(encoding="utf-8").splitlines()]
    excluded = held_out_canonicals()
    candidate_to_canonicals: dict[str, set[str]] = {}
    for row in rows:
        canonical = str(row["canonical"])
        for candidate in row.get("synonyms", []):
            key = " ".join(str(candidate).lower().split())
            candidate_to_canonicals.setdefault(key, set()).add(canonical)
    ambiguous = {key for key, values in candidate_to_canonicals.items() if len(values) > 1}
    negatives: dict[str, list[str]] = {}
    if NEGATIVES.exists():
        for line in NEGATIVES.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            negatives.setdefault(str(row["canonical"]), []).append(str(row["candidate"]))
    examples = [
        InputExample(texts=[
            contextualise(candidate, f"query:{row['canonical']}:{candidate}"),
            contextualise(str(row["canonical"]), f"positive:{row['canonical']}:{candidate}"),
            contextualise(
                negatives[str(row["canonical"])][int(hashlib.sha256(str(candidate).encode("utf-8")).hexdigest(), 16) % len(negatives[str(row["canonical"])])],
                f"negative:{row['canonical']}:{candidate}",
            ),
        ])
        for row in rows if str(row["canonical"]) not in excluded and row.get("split", "train") == "train"
        for candidate in row.get("synonyms", [])
        if " ".join(str(candidate).lower().split()) not in ambiguous
        if negatives.get(str(row["canonical"]))
    ]
    random.Random(20260902).shuffle(examples)
    return examples, len(excluded), len(ambiguous)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-pairs", type=int, default=500)
    parser.add_argument("--training", type=Path, default=TRAINING)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    examples, held_out, ambiguous = load_examples(args.training)
    if len(examples) < args.min_pairs:
        raise RuntimeError(f"Need at least {args.min_pairs} verified pairs, found {len(examples)}")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[attribute-encoder-training] device={device}", flush=True)
    model = SentenceTransformer(MODEL_NAME, cache_folder=str(MODEL_CACHE), device=device, local_files_only=True)
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = max(1, int(len(loader) * args.epochs * 0.10))
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=warmup,
        show_progress_bar=True,
        output_path=str(args.output),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "base_model": MODEL_NAME,
        "training_corpus": str(args.training),
        "device": device,
        "verified_pairs": len(examples),
        "held_out_equivalence_canonicals": held_out,
        "excluded_ambiguous_surface_forms": ambiguous,
        "training_objective": "contrastive verified pair with one deterministic catalogue-grounded hard negative",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "warmup_steps": warmup,
        "output": str(args.output),
    }, indent=2) + "\n", encoding="utf-8")
    print(args.report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
