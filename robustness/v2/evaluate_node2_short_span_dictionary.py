"""V2.25: fixed-test ablation of a BERT-cleaned short-span dictionary path.

The candidate path is intentionally simple and lexical:
  BERT CONTENT tokens -> every contiguous 1/2/3-token span -> exact membership in
  the frozen 50k-catalogue attribute dictionary.

It neither trains a model nor uses semantic similarity.  This is an offline decision
test only; it does not modify V2 runtime until recall and false-candidate rates are known.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: E402

from evaluator.local_evaluator import catalog_index, coarse_category, load_jsonl  # noqa: E402
from submission.agent import raw_toks  # noqa: E402
from submission.bert_extract import KEEP_THRESHOLD, MAX_WORDS, MODEL_DIR  # noqa: E402

TEST = ROOT / "robustness" / "v2" / "v1_turn_gated_bank" / "final_test.jsonl"
DATASET = "TemplateParaphrase9600"
SPLIT = "Test"
DICTIONARY = ROOT / "robustness" / "v2" / "catalogue_attribute_dictionary.jsonl"
OUT = ROOT / "robustness" / "v2" / "results" / "v2_25_node2_short_span_dictionary_final_test.json"
CATALOG = ROOT / "data" / "catalog.jsonl"
BATCH = 64


def constraint_values(row: dict) -> list[str]:
    slots = row["slots"]
    return {
        "buying_opening": [slots["a"]],
        "override_opening": [slots["b"]],
        "constraint_update": [slots["a"], slots["b"]],
        "override_update": [slots["a"]],
    }.get(row["action"], [])


def clean_batch(model, tokenizer, messages: list[str], device: torch.device) -> list[list[str]]:
    words = [raw_toks(message)[:MAX_WORDS] for message in messages]
    encoded = tokenizer(words, is_split_into_words=True, padding=True, truncation=True,
                        max_length=MAX_WORDS, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in encoded.items()
              if key in {"input_ids", "attention_mask"}}
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, -1)[:, :, 1].cpu()
    result: list[list[str]] = []
    for batch_index, source in enumerate(words):
        kept, previous = [], None
        for position, word_index in enumerate(encoded.word_ids(batch_index)):
            if word_index is None or word_index == previous:
                continue
            previous = word_index
            if word_index < len(source) and float(probs[batch_index, position]) >= KEEP_THRESHOLD:
                kept.append(source[word_index])
        result.append(kept)
    return result


def candidates(tokens: list[str], dictionary: frozenset[str]) -> set[str]:
    return {
        " ".join(tokens[start:start + width])
        for width in (1, 2, 3)
        for start in range(0, len(tokens) - width + 1)
        if " ".join(tokens[start:start + width]) in dictionary
    }


def maximal(candidates_: set[str]) -> set[str]:
    """Keep only candidates not strictly contained in another exact candidate."""
    tokenized = {candidate: raw_toks(candidate) for candidate in candidates_}
    return {
        candidate for candidate, phrase in tokenized.items()
        if not any(
            len(other) > len(phrase)
            and any(other[start:start + len(phrase)] == phrase
                    for start in range(len(other) - len(phrase) + 1))
            for name, other in tokenized.items() if name != candidate
        )
    }


def extract_longest_category(tokens: list[str], category_patterns: list[list[str]]) -> tuple[list[str], str | None]:
    """Remove the single longest exact coarse-category mention from an opening.

    This uses only the visible frozen catalogue.  It is not a semantic category model.
    A constraint update has no category slot and never calls this function.
    """
    for pattern in category_patterns:
        width = len(pattern)
        for start in range(0, len(tokens) - width + 1):
            if tokens[start:start + width] == pattern:
                return tokens[:start] + tokens[start + width:], " ".join(pattern)
    return tokens, None


def main() -> None:
    assert torch.cuda.is_available(), "This fixed 9,600-row BERT evaluation requires CUDA."
    rows = load_jsonl(TEST)
    dictionary = frozenset(json.loads(line)["canonical"] for line in DICTIONARY.read_text(encoding="utf-8").splitlines() if line)
    assert len(rows) == 9600 and dictionary
    _, categories, _ = catalog_index(CATALOG)
    category_patterns = sorted({tuple(raw_toks(coarse_category(path))) for path in categories.values()
                                if len(raw_toks(coarse_category(path))) >= 2}, key=len, reverse=True)
    device = torch.device("cuda:0")
    print(f"[{time.strftime('%H:%M:%S')}] loading V1 BERT on {torch.cuda.get_device_name(device)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(device)
    model.eval()
    cleaned: list[list[str]] = []
    for start in range(0, len(rows), BATCH):
        cleaned.extend(clean_batch(model, tokenizer, [row["message"] for row in rows[start:start + BATCH]], device))
        if (start // BATCH + 1) % 25 == 0 or start + BATCH >= len(rows):
            print(f"[{time.strftime('%H:%M:%S')}] {min(start + BATCH, len(rows))}/{len(rows)} messages", flush=True)

    totals = Counter()
    by_action: dict[str, Counter] = defaultdict(Counter)
    examples: list[dict] = []
    for row, tokens in zip(rows, cleaned):
        expected = {" ".join(raw_toks(value)) for value in constraint_values(row)}
        output = candidates(tokens, dictionary) if expected else set()
        category_removed, category_match = extract_longest_category(tokens, [list(p) for p in category_patterns]) \
            if row["action"] in {"buying_opening", "override_opening", "plain_opening"} else (tokens, None)
        masked_output = candidates(category_removed, dictionary) if expected else set()
        maximal_output = maximal(masked_output)
        action = row["action"]
        for value in expected:
            totals["expected"] += 1
            by_action[action]["expected"] += 1
            if value in dictionary:
                totals["dictionary_covered"] += 1
                by_action[action]["dictionary_covered"] += 1
            if value in output:
                totals["recovered"] += 1
                by_action[action]["recovered"] += 1
            if value in masked_output:
                totals["masked_recovered"] += 1
                by_action[action]["masked_recovered"] += 1
            if value in maximal_output:
                totals["maximal_recovered"] += 1
                by_action[action]["maximal_recovered"] += 1
        totals["candidate_count"] += len(output)
        totals["extra_count"] += len(output - expected)
        totals["masked_candidate_count"] += len(masked_output)
        totals["masked_extra_count"] += len(masked_output - expected)
        totals["maximal_candidate_count"] += len(maximal_output)
        totals["maximal_extra_count"] += len(maximal_output - expected)
        totals["messages_with_expected"] += int(bool(expected))
        totals["messages_with_any_candidate"] += int(bool(output))
        totals["no_evidence_emitted"] += int(row["action"] == "no_evidence" and bool(output))
        if row["action"] in {"buying_opening", "override_opening", "plain_opening"}:
            expected_category = " ".join(raw_toks(row["slots"]["category"]))
            totals["category_expected"] += 1
            totals["category_exact_recovered"] += int(category_match == expected_category)
        if expected and (not expected.issubset(maximal_output) or maximal_output - expected) and len(examples) < 5:
            examples.append({"action": action, "message": row["message"], "expected": sorted(expected),
                             "bert_tokens": tokens, "after_category_mask": category_removed,
                             "dictionary_candidates": sorted(output), "masked_dictionary_candidates": sorted(masked_output),
                             "maximal_dictionary_candidates": sorted(maximal_output),
                             "missing": sorted(expected - maximal_output), "extra": sorted(maximal_output - expected)})

    per_action = {}
    for action, c in sorted(by_action.items()):
        per_action[action] = {
            "expected_values": c["expected"],
            "dictionary_coverage": round(c["dictionary_covered"] / c["expected"], 6),
            "recovery": round(c["recovered"] / c["expected"], 6),
            "category_masked_recovery": round(c["masked_recovered"] / c["expected"], 6),
            "maximal_recovery": round(c["maximal_recovered"] / c["expected"], 6),
        }
    result = {
        "experiment": "V2.25 BERT-cleaned exact 1-3 token dictionary candidate path",
        "dataset_name": DATASET,
        "split": SPLIT,
        "test_rows": len(rows), "test_split": str(TEST), "dictionary": str(DICTIONARY),
        "inference": {"device": str(device), "batch_size": BATCH, "keep_threshold": KEEP_THRESHOLD,
                      "candidate_rule": "contiguous 1-3 token exact dictionary membership"},
        "constraint_values": totals["expected"],
        "dictionary_coverage": round(totals["dictionary_covered"] / totals["expected"], 6),
        "constraint_recall": round(totals["recovered"] / totals["expected"], 6),
        "mean_candidates_per_constraint_message": round(totals["candidate_count"] / totals["messages_with_expected"], 6),
        "mean_extra_candidates_per_constraint_message": round(totals["extra_count"] / totals["messages_with_expected"], 6),
        "category_masked": {
            "constraint_recall": round(totals["masked_recovered"] / totals["expected"], 6),
            "mean_candidates_per_constraint_message": round(totals["masked_candidate_count"] / totals["messages_with_expected"], 6),
            "mean_extra_candidates_per_constraint_message": round(totals["masked_extra_count"] / totals["messages_with_expected"], 6),
            "category_patterns": len(category_patterns),
            "rule": "remove longest exact visible coarse-category mention on opening routes only",
        },
        "category_masked_maximal": {
            "constraint_recall": round(totals["maximal_recovered"] / totals["expected"], 6),
            "mean_candidates_per_constraint_message": round(totals["maximal_candidate_count"] / totals["messages_with_expected"], 6),
            "mean_extra_candidates_per_constraint_message": round(totals["maximal_extra_count"] / totals["messages_with_expected"], 6),
            "rule": "category mask followed by removal of strict nested dictionary spans",
        },
        "messages_with_any_candidate": round(totals["messages_with_any_candidate"] / totals["messages_with_expected"], 6),
        "no_evidence_candidates": totals["no_evidence_emitted"],
        "category_exact_catalogue_recovery": round(totals["category_exact_recovered"] / totals["category_expected"], 6),
        "per_action": per_action, "five_diagnostic_examples": examples,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"per_action", "five_diagnostic_examples"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
