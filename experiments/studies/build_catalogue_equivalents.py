import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "experiments" / "studies"

source = V2 / "external_train_only_canonicals.jsonl"
destination = V2 / "catalogue_direct_equivalents.jsonl"

number_words = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

word_replacements = {
    "grey": "gray",
    "colour": "color",
    "fibre": "fiber",
    "jewellery": "jewelry",
    "moulded": "molded",
    "syntethic": "synthetic",
    "syntheric": "synthetic",
    "leathe": "leather",
    "preshrunk": "pre shrunk",
}

def normalized(phrase: str) -> str:
    text = phrase.lower()
    text = re.sub(r"(?<=\d)pcs\b", " pcs", text)
    text = re.sub(r"\bpcs?\b", "piece", text)
    text = re.sub(r"\bpieces\b", "piece", text)
    text = re.sub(r"\bwomens\b", "women s", text)
    text = re.sub(r"\bmens\b", "men s", text)
    text = re.sub(r"\b100 percent\b", "100", text)
    text = re.sub(r"\binches\b", "inch", text)
    text = re.sub(r"\byears\b", "year", text)
    text = re.sub(r"\bsleeves\b", "sleeve", text)
    text = re.sub(r"\b3 4 length sleeve\b", "3 4 sleeve", text)
    for before, after in word_replacements.items():
        text = re.sub(rf"\b{re.escape(before)}\b", after, text)
    for word, digit in number_words.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    return re.sub(r"\s+", " ", text).strip()

explicit_groups = (
    (
        "zip closure", "zipper closure",
    ),
    (
        "zip front closure", "zipper front closure",
    ),
    (
        "zip top closure", "zipper top closure",
    ),
    (
        "velcro", "hook and loop",
    ),
    (
        "velcro closure", "hook and loop closure",
    ),
    (
        "water resistant 100m",
        "water resistant to 100 meters",
        "water resistant to 100 m 330 feet",
        "water resistant to 100 m 330 ft",
        "water resistant to 330 feet",
        "water resistant to 330 feet 100 m",
    ),
    (
        "water resistant to 165 feet",
        "water resistant to 165 feet 50 m",
        "water resistant to 165 feet 50m",
        "water resistant to 165 ft",
        "water resistant to 50 m 165 feet",
        "water resistant to 50 m 165 ft",
        "water resistant to 50 meters 165 feet",
        "water resistant to 50m 165 feet",
    ),
    (
        "water resistant to 30 m 100 feet",
        "water resistant to 30 m 99 feet",
        "water resistant to 30 m 99 ft",
        "water resistant to 33 feet",
        "water resistant to 99 feet",
        "water resistant to 99 feet 30 m",
        "water resistant up to 30 meters",
    ),
)

with source.open("r", encoding="utf-8") as incoming:
    phrases = [json.loads(line)["canonical"] for line in incoming]

present = set(phrases)
groups = defaultdict(set)
for phrase in phrases:
    groups[normalized(phrase)].add(phrase)

for group in explicit_groups:
    attested = set(group) & present
    if len(attested) > 1:
        key = "explicit:" + "|".join(sorted(attested))
        groups[key].update(attested)

equivalents = defaultdict(set)
for group in groups.values():
    if len(group) > 1:
        for phrase in group:
            equivalents[phrase].update(group - {phrase})

with destination.open("w", encoding="utf-8", newline="\n") as outgoing:
    for phrase in phrases:
        outgoing.write(json.dumps({"canonical": phrase, "equivalents": sorted(equivalents[phrase])}, ensure_ascii=False) + "\n")
