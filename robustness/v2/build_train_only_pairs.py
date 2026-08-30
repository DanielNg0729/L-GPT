import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "robustness" / "v2"

source = V2 / "external_train_only_canonicals.jsonl"
destination = V2 / "external_train_only_pairs.jsonl"

exact = {
    "velcro": ["hook and loop"],
    "velcro closure": ["hook and loop closure"],
    "zip closure": ["zipper closure"],
    "zipper closure": ["zip closure"],
    "zip front closure": ["zipper front closure"],
    "zipper front closure": ["zip front closure"],
    "zip top closure": ["zipper top closure"],
    "zipper top closure": ["zip top closure"],
    "v neck": ["v-neck"],
    "v neckline": ["v-neck"],
    "t shirt": ["t-shirt"],
    "t shirts": ["t-shirts"],
    "crew neck": ["crew-neck"],
    "mock neck": ["mock-neck"],
    "scoop neck": ["scoop-neck"],
    "water resistant": ["water-resistant"],
    "water repellent": ["water-repellent"],
    "machine washable": ["machine-washable"],
    "wrinkle resistant": ["wrinkle-resistant"],
    "wrinkle free": ["wrinkle-free"],
    "wire free": ["wire-free"],
    "two piece set": ["2 piece set"],
    "2 piece set": ["two piece set"],
    "two pockets": ["2 pockets"],
    "2 pockets": ["two pockets"],
    "two side pockets": ["2 side pockets"],
    "2 side pockets": ["two side pockets"],
    "two hand pockets": ["2 hand pockets"],
    "2 hand pockets": ["two hand pockets"],
    "two front pockets": ["2 front pockets"],
    "two front pockets": ["2 front pockets"],
    "two inside pockets": ["2 inside pockets"],
    "2 inside pockets": ["two inside pockets"],
    "two chest pockets": ["2 chest pockets"],
    "2 chest pockets": ["two chest pockets"],
}

spelling_pairs = (
    ("grey", "gray"),
    ("colour", "color"),
    ("fibre", "fiber"),
    ("jewellery", "jewelry"),
    ("moulded", "molded"),
)

def variants(canonical: str) -> list[str]:
    candidates = list(exact.get(canonical, []))
    padded = f" {canonical} "
    for left, right in spelling_pairs:
        if f" {left} " in padded:
            candidates.append(padded.replace(f" {left} ", f" {right} ").strip())
        elif f" {right} " in padded:
            candidates.append(padded.replace(f" {right} ", f" {left} ").strip())
    return list(dict.fromkeys(candidate for candidate in candidates if candidate != canonical))[:2]

with source.open("r", encoding="utf-8") as incoming, destination.open("w", encoding="utf-8", newline="\n") as outgoing:
    for line in incoming:
        row = json.loads(line)
        canonical = row["canonical"]
        outgoing.write(json.dumps({"canonical": canonical, "candidates": variants(canonical)}, ensure_ascii=False) + "\n")
