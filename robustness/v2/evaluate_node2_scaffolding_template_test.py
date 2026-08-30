"""V2.24: test the existing V1 BERT as Node 2 on the fixed wrapper test split.

This uses the untouched V2.18 9,600-row template-disjoint test.  Slots remain
canonical catalogue text; only conversational wrappers vary.  The experiment asks
whether BERT scaffolding removal preserves known value spans and improves the
catalogue-grounded miner over raw unfamiliar wrapper text.

It does not train, tune, or alter the shipped tagger.  Batched CUDA inference mirrors
``ScaffoldingTagger.strip`` exactly: same model, tokenization, CONTENT threshold,
96-word cap, and first-subtoken word score.
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: E402

from evaluator.local_evaluator import load_jsonl  # noqa: E402
from submission.agent import Agent, raw_toks  # noqa: E402
from submission.bert_extract import KEEP_THRESHOLD, MAX_WORDS, MODEL_DIR  # noqa: E402

TEST = ROOT / "robustness" / "v2" / "v1_turn_gated_bank" / "final_test.jsonl"
DATASET = "TemplateParaphrase9600"
SPLIT = "Test"
OUT = ROOT / "robustness" / "v2" / "results" / "v2_24_node2_scaffolding_final_test.json"
BATCH = 64
SEED = 20260831


def expected_slots(row: dict) -> dict[str, list[str]]:
    slots = row["slots"]
    action = row["action"]
    if action == "buying_opening":
        return {"category": [slots["category"]], "constraint": [slots["a"]]}
    if action == "plain_opening":
        return {"category": [slots["category"]], "constraint": []}
    if action == "override_opening":
        return {"category": [slots["category"]], "constraint": [slots["b"]]}
    if action == "constraint_update":
        return {"category": [], "constraint": [slots["a"], slots["b"]]}
    if action == "override_update":
        return {"category": [], "constraint": [slots["a"]]}
    return {"category": [], "constraint": []}


def contiguous(haystack: list[str], needle: list[str]) -> bool:
    return bool(needle) and any(haystack[i:i + len(needle)] == needle
                                for i in range(len(haystack) - len(needle) + 1))


def strip_batch(model, tokenizer, messages: list[str], device: torch.device) -> list[str]:
    words = [raw_toks(message)[:MAX_WORDS] for message in messages]
    encoded = tokenizer(words, is_split_into_words=True, padding=True, truncation=True,
                        max_length=MAX_WORDS, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in encoded.items()
              if key in {"input_ids", "attention_mask"}}
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits, -1)[:, :, 1].cpu()
    cleaned: list[str] = []
    for row_index, original_words in enumerate(words):
        kept, previous = [], None
        for position, word_index in enumerate(encoded.word_ids(row_index)):
            if word_index is None or word_index == previous:
                continue
            previous = word_index
            if word_index < len(original_words) and float(probabilities[row_index, position]) >= KEEP_THRESHOLD:
                kept.append(original_words[word_index])
        cleaned.append(" ".join(kept))
    return cleaned


def recovered(mined: set[str], target: str, agent: Agent) -> bool:
    canonical = agent._resolve(target)
    return any(phrase in mined for phrase in canonical)


def main() -> None:
    assert torch.cuda.is_available(), "This fixed 9,600-row BERT evaluation requires CUDA."
    rows = load_jsonl(TEST)
    assert len(rows) == 9600
    device = torch.device("cuda:0")
    print(f"[{time.strftime('%H:%M:%S')}] loading existing V1 tagger on {torch.cuda.get_device_name(device)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(device)
    model.eval()
    agent = Agent(ROOT / "data" / "catalog.jsonl")
    cleaned: list[str] = []
    for start in range(0, len(rows), BATCH):
        cleaned.extend(strip_batch(model, tokenizer, [row["message"] for row in rows[start:start + BATCH]], device))
        if (start // BATCH + 1) % 25 == 0 or start + BATCH >= len(rows):
            print(f"[{time.strftime('%H:%M:%S')}] {min(start + BATCH, len(rows))}/{len(rows)} messages", flush=True)

    counters: Counter = Counter()
    action_metrics: dict[str, Counter] = defaultdict(Counter)
    examples: list[dict] = []
    for row, bert_text in zip(rows, cleaned):
        action = row["action"]
        required = expected_slots(row)
        raw_text = " ".join(raw_toks(row["message"]))
        raw_mined = {phrase for phrase, _ in agent.ix.mine(raw_text)}
        bert_mined = {phrase for phrase, _ in agent.ix.mine(bert_text)}
        for kind, values in required.items():
            for value in values:
                value_tokens = raw_toks(value)
                key = f"{kind}_slot"
                counters[key + "_total"] += 1
                action_metrics[action][key + "_total"] += 1
                if contiguous(raw_toks(bert_text), value_tokens):
                    counters[key + "_bert_retained"] += 1
                    action_metrics[action][key + "_bert_retained"] += 1
                if recovered(raw_mined, value, agent):
                    counters[key + "_raw_recovered"] += 1
                    action_metrics[action][key + "_raw_recovered"] += 1
                if recovered(bert_mined, value, agent):
                    counters[key + "_bert_recovered"] += 1
                    action_metrics[action][key + "_bert_recovered"] += 1
        if action == "no_evidence":
            counters["no_evidence_rows"] += 1
            counters["no_evidence_raw_mined_phrases"] += len(raw_mined)
            counters["no_evidence_bert_mined_phrases"] += len(bert_mined)
        if len(examples) < 5 and required["constraint"] and any(
            not contiguous(raw_toks(bert_text), raw_toks(value)) for value in required["constraint"]
        ):
            examples.append({"action": action, "message": row["message"], "expected": required,
                             "bert_cleaned": bert_text, "raw_mined": sorted(raw_mined),
                             "bert_mined": sorted(bert_mined)})

    def rate(numerator: str, denominator: str) -> float:
        return round(counters[numerator] / counters[denominator], 6) if counters[denominator] else 0.0

    per_action = {}
    for action, c in sorted(action_metrics.items()):
        row = {}
        for kind in ("constraint", "category"):
            total = c[f"{kind}_slot_total"]
            if total:
                row[f"{kind}_retention_rate"] = round(c[f"{kind}_slot_bert_retained"] / total, 6)
                row[f"{kind}_raw_recovery_rate"] = round(c[f"{kind}_slot_raw_recovered"] / total, 6)
                row[f"{kind}_bert_recovery_rate"] = round(c[f"{kind}_slot_bert_recovered"] / total, 6)
        per_action[action] = row
    result = {
        "experiment": "V2.24 existing V1 BERT Node 2 on fixed template-disjoint test",
        "dataset_name": DATASET,
        "split": SPLIT,
        "test_rows": len(rows),
        "test_split": str(TEST),
        "model": str(MODEL_DIR),
        "inference": {"device": str(device), "batch_size": BATCH, "keep_threshold": KEEP_THRESHOLD,
                      "max_words": MAX_WORDS, "implementation": "exact ScaffoldingTagger strip semantics, batched"},
        "constraint_slot_retention": rate("constraint_slot_bert_retained", "constraint_slot_total"),
        "constraint_mining_recovery": {"raw": rate("constraint_slot_raw_recovered", "constraint_slot_total"),
                                        "bert": rate("constraint_slot_bert_recovered", "constraint_slot_total")},
        "category_slot_retention": rate("category_slot_bert_retained", "category_slot_total"),
        "category_mining_recovery": {"raw": rate("category_slot_raw_recovered", "category_slot_total"),
                                      "bert": rate("category_slot_bert_recovered", "category_slot_total")},
        "no_evidence": {"rows": counters["no_evidence_rows"],
                        "raw_mean_mined_phrases": round(counters["no_evidence_raw_mined_phrases"] / max(1, counters["no_evidence_rows"]), 6),
                        "bert_mean_mined_phrases": round(counters["no_evidence_bert_mined_phrases"] / max(1, counters["no_evidence_rows"]), 6),
                        "node1_interpretation": "The completed route node suppresses mining for correctly routed no-evidence messages; these counts are diagnostic only."},
        "per_action": per_action,
        "five_constraint_retention_failures": examples,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"per_action", "five_constraint_retention_failures"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
