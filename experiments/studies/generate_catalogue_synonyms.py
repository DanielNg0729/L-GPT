"""Generate verified, synonym-only V2 training pairs for the full catalogue dictionary.

The job is resumable.  It uses Groq only to propose and independently adjudicate lexical
equivalence; all accepted labels are existing, visible catalogue phrases.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

from experiments.studies.build_semantic_attribute_sets import RULES, norm

ROOT = Path(__file__).resolve().parents[2]
DICTIONARY = ROOT / "experiments" / "datasets" / "catalogue_attribute_dictionary.jsonl"
OUT = ROOT / "experiments" / "datasets" / "catalogue_synonym_training.jsonl"
MODEL = "openai/gpt-oss-20b"
GENERIC = {"closure", "material", "colour", "color", "feature", "fastening"}


def load_env() -> None:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def call(prompt: dict, max_tokens: int) -> dict:
    model = os.environ.get("GROQ_AUGMENT_MODEL", MODEL)
    reasoning_effort = "none" if model.startswith("qwen/") else "low"
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}", "Content-Type": "application/json"},
        json={"model": model, "temperature": 0.1,
              "max_completion_tokens": max_tokens, "reasoning_effort": reasoning_effort,
              "messages": [{"role": "user", "content": json.dumps(prompt)}]},
        timeout=60,
    )
    if response.status_code == 429:
        message = response.json().get("error", {}).get("message", "")
        match = re.search(r"try again in\s+([0-9.]+)\s*(ms|s|m)?", message, flags=re.I)
        value, unit = (float(match.group(1)), (match.group(2) or "s").lower()) if match else (10.0, "s")
        delay = value / 1000.0 if unit == "ms" else value * 60.0 if unit == "m" else value
        time.sleep(max(5.0, delay + 1.0))
        return call(prompt, max_tokens)
    if not response.ok:
        raise RuntimeError(f"Groq request failed ({response.status_code}): {response.text[:300]}")
    message = response.json()["choices"][0]["message"]
    content = (message.get("content") or "").strip()
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError(f"Groq returned no parseable JSON content: {content[:300]!r}; fields={sorted(message)}")
    try:
        return json.loads(content[start:end + 1])
    except json.JSONDecodeError:
        # Model output is untrusted training input.  A malformed response contributes no
        # examples and the resumable caller records the batch without accepting anything.
        return {}


def banned_phrases() -> set[str]:
    return {norm(phrase) for _, _, development, holdout in RULES for phrase in (development, holdout)}


def shares_ngram(left: str, right: str, n: int = 3) -> bool:
    def grams(text: str) -> set[str]:
        tokens = text.split()
        return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}
    return bool(grams(left) & grams(right))


def valid(candidate: str, canonical: str, forbidden: set[str]) -> bool:
    text, label = norm(candidate), norm(canonical)
    if not text or text == label or text in forbidden or len(text.split()) > 8:
        return False
    distinctive = set(label.split()) - GENERIC
    if distinctive & set(text.split()):
        return False
    return not any(shares_ngram(text, phrase) for phrase in forbidden)


def generate(batch: list[dict], forbidden: set[str]) -> dict[str, list[str]]:
    prompt = {
        "task": "Generate lexical semantic-normalization alternatives for product attributes.",
        "rules": [
            "Each alternative must be interchangeable in: I want a product with [alternative].",
            "Return only an exact synonymous word or short noun/adjective phrase, never a full sentence.",
            "Do not explain mechanism, consequence, quality, provenance, or any inferred property.",
            "Never replace a closure with a different closure type, a material with its source/treatment, or a colour with a related colour.",
            "Use no distinctive word from the canonical attribute. Generic heads such as closure or material may remain.",
            "Return one or two conventional catalogue synonyms when one exists. The downstream verifier will reject approximate terms.",
            "Use an empty list only when no conventional lexical alternative exists.",
            "Return JSON only: {\"items\":[{\"id\":0,\"candidates\":[...]}]}.",
        ],
        "items": [{"id": index, "canonical": row["canonical"]} for index, row in enumerate(batch)],
        "forbidden_test_phrases": sorted(forbidden),
    }
    raw = call(prompt, max_tokens=700).get("items", [])
    output: dict[str, list[str]] = {}
    for entry in raw:
        index = entry.get("id")
        if not isinstance(index, int) or not 0 <= index < len(batch):
            continue
        values = entry.get("candidates", [])
        output[batch[index]["canonical"]] = [value for value in values if isinstance(value, str) and valid(value, batch[index]["canonical"], forbidden)]
    return output


def verify(pairs: list[dict]) -> set[int]:
    if not pairs:
        return set()
    prompt = {
        "task": "Independently verify lexical equivalence for product attributes.",
        "rule": "Accept only exact synonyms. Reject explanations, consequences, broader/narrower terms, sibling materials or closure mechanisms, inferred quality/source/treatment, or uncertainty.",
        "pairs": [{"id": index, "canonical": row["canonical"], "candidate": row["candidate"]} for index, row in enumerate(pairs)],
        "response": "Return JSON only: {\"accepted_ids\":[integer,...]}.",
    }
    result = call(prompt, max_tokens=400)
    return {index for index in result.get("accepted_ids", []) if isinstance(index, int) and 0 <= index < len(pairs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--pause-seconds", type=float, default=6.0)
    args = parser.parse_args()
    load_env()
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is required in .env")
    dictionary = [json.loads(line) for line in DICTIONARY.read_text(encoding="utf-8").splitlines()]
    existing = {json.loads(line)["canonical"] for line in OUT.read_text(encoding="utf-8").splitlines()} if OUT.exists() else set()
    todo = [row for row in dictionary if row["canonical"] not in existing]
    if args.limit is not None:
        todo = todo[:args.limit]
    forbidden = banned_phrases()
    with OUT.open("a", encoding="utf-8") as handle:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            proposed = generate(batch, forbidden)
            pairs = [{"canonical": canonical, "candidate": candidate}
                     for canonical, candidates in proposed.items() for candidate in candidates]
            accepted = verify(pairs)
            accepted_by_canonical: dict[str, list[str]] = {row["canonical"]: [] for row in batch}
            for index in accepted:
                pair = pairs[index]
                accepted_by_canonical[pair["canonical"]].append(pair["candidate"])
            for row in batch:
                canonical = row["canonical"]
                handle.write(json.dumps({
                    **row,
                    "proposed": proposed.get(canonical, []),
                    "synonyms": accepted_by_canonical[canonical],
                }) + "\n")
            handle.flush()
            print(json.dumps({"completed": start + len(batch), "total": len(todo), "accepted_pairs": len(accepted)}, separators=(",", ":")), flush=True)
            if start + len(batch) < len(todo):
                time.sleep(args.pause_seconds)


if __name__ == "__main__":
    main()
