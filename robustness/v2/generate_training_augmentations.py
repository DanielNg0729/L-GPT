"""Generate training-only semantic paraphrases with Groq and strict leakage checks."""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import requests

from robustness.v2.build_semantic_attribute_sets import RULES, norm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "robustness" / "v2" / "training_augmentations.jsonl"


def load_env() -> None:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def request(value: str, family: str, banned: list[str]) -> list[str]:
    prompt = {
        "task": "Produce lexical semantic-normalization examples, not product descriptions.",
        "canonical_value": value,
        "attribute_family": family,
        "rules": [
            "Each result must be interchangeable with the canonical value in: 'I want a product with [result].'",
            "Return only a short synonymous noun, adjective, or noun phrase, with no sentence framing.",
            "You may retain a generic grammatical head such as 'closure' or 'material', but replace the distinctive attribute word.",
            "Do not explain how the attribute works, give a consequence, or name a broader or different mechanism.",
            "Do not infer additional properties, quality, materials, adjustability, performance, or provenance.",
            "For a closure, retain the identical closure type. For materials, do not infer source, treatment, grade, or finish.",
            "If no exact synonym exists, return an empty list. Precision matters more than producing five items.",
            "Do not use any token from the canonical value.",
            "Do not copy or closely imitate any banned phrase.",
            "Return JSON only: {\"paraphrases\":[...]}",
        ],
        "banned_phrases": banned,
    }
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}", "Content-Type": "application/json"},
        json={"model": os.environ.get("GROQ_AUGMENT_MODEL", "openai/gpt-oss-20b"), "temperature": 0.2,
              "max_completion_tokens": 300,
              "messages": [{"role": "user", "content": json.dumps(prompt)}]},
        timeout=45,
    )
    if not response.ok:
        raise RuntimeError(f"Groq request failed ({response.status_code}): {response.text[:500]}")
    content = response.json()["choices"][0]["message"]["content"].strip()
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        return []
    return json.loads(content[start:end + 1]).get("paraphrases", [])


def has_shared_ngram(left: str, right: str, n: int = 3) -> bool:
    """Reject renderer leakage while permitting the same underlying meaning."""
    left_tokens, right_tokens = left.split(), right.split()
    left_ngrams = {" ".join(left_tokens[index:index + n]) for index in range(len(left_tokens) - n + 1)}
    right_ngrams = {" ".join(right_tokens[index:index + n]) for index in range(len(right_tokens) - n + 1)}
    return bool(left_ngrams & right_ngrams)


def valid(text: str, canonical: str, banned: set[str]) -> bool:
    phrase = norm(text)
    if not phrase or phrase in banned or len(phrase.split()) < 3:
        return False
    generic_heads = {"closure", "material", "colour", "color", "feature", "fastening"}
    distinctive_tokens = set(norm(canonical).split()) - generic_heads
    if distinctive_tokens & set(phrase.split()):
        return False
    return not any(has_shared_ngram(phrase, excluded) for excluded in banned)


def main() -> None:
    load_env()
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is required in .env")
    banned = {norm(dev) for _, _, dev, holdout in RULES for dev in (dev, holdout)}
    rows = []
    canonical_forms = {
        "pull_on": "pull on closure", "water_resistant": "water resistant",
        "machine_washable": "machine washable", "hand_wash": "hand wash",
        "slip_resistant": "slip resistant", "buckle": "buckle closure",
        "zipper": "zipper closure", "lace": "lace up closure",
        "button": "button closure",
    }
    for name, pattern, dev, holdout in random.Random(20260901).sample(list(RULES), 5):
        canonical = canonical_forms.get(name, name)
        outputs = request(canonical, "catalogue_attribute", [dev, holdout])
        kept = [item for item in outputs if isinstance(item, str) and valid(item, canonical, banned)]
        rows.append({"rule": name, "canonical": canonical, "generated": kept, "banned": [dev, holdout]})
    OUT.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
