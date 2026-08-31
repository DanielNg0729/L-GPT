"""Build fixed-format, semantic-attribute paraphrase data for V2.

Messages retain the evaluator's original wrappers. Only target-derived attribute spans are
replaced by manually curated semantic rewrites that avoid the original lexical anchors.
The sets are target-disjoint from the public release and from each other; they are internal
semantic robustness data, not an estimate of organizer-private performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def flat(value: object) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value or "")


def norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coarse_category(values: list[str]) -> str:
    excluded = {"clothing", "clothing shoes jewelry"}
    parts: list[str] = []
    for value in values:
        for part in str(value).split(","):
            cleaned = norm(part)
            if cleaned and cleaned not in excluded:
                parts.append(part.strip())
    return " ".join(parts[-2:]) if parts else "clothing item"


def classify(phrase: str) -> str:
    text = norm(phrase)
    if any(word in text for word in ("cotton", "leather", "polyester", "nylon", "rubber", "suede", "canvas", "wool")):
        return "material"
    if any(word in text for word in ("black", "white", "blue", "red", "pink", "green", "brown", "beige", "gray", "grey", "purple")):
        return "color"
    if any(word in text for word in ("running", "hiking", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


# Each rule supplies two disjoint paraphrase families. Neither output intentionally
# contains the matched canonical phrase or a direct token from it.
RULES = (
    ("cotton", r"\bcotton\b", "made from a soft plant fibre", "crafted from a natural fibre textile"),
    ("leather", r"\bleather\b", "made from animal hide", "crafted from tanned hide material"),
    ("polyester", r"\bpolyester\b", "made from a man made textile", "built with a synthetic woven fibre"),
    ("nylon", r"\bnylon\b", "made from a durable synthetic polyamide", "constructed with a tough artificial fibre"),
    ("rubber", r"\brubber\b", "made with a flexible elastic polymer", "built from a resilient synthetic compound"),
    ("suede", r"\bsuede\b", "finished with brushed hide", "made from soft napped hide"),
    ("canvas", r"\bcanvas\b", "made from heavy woven cloth", "constructed with sturdy plain weave fabric"),
    ("wool", r"\bwool\b", "made from animal fleece fibre", "constructed with a natural fleece textile"),
    ("black", r"\bblack\b", "in the darkest colour", "in a deep ink toned shade"),
    ("white", r"\bwhite\b", "in a light neutral colour", "in a pale snow toned shade"),
    ("blue", r"\bblue\b", "in an ocean toned shade", "in a cool sky coloured hue"),
    ("red", r"\bred\b", "in a warm crimson shade", "in a vivid scarlet tone"),
    ("pink", r"\bpink\b", "in a rosy shade", "in a blush toned colour"),
    ("green", r"\bgreen\b", "in a leaf toned shade", "in an emerald coloured hue"),
    ("brown", r"\bbrown\b", "in an earthy tan shade", "in a chestnut toned colour"),
    ("beige", r"\bbeige\b", "in a sand toned shade", "in a pale khaki colour"),
    ("gray", r"\b(?:gray|grey)\b", "in a neutral ash shade", "in a muted slate tone"),
    ("imported", r"\bimported\b", "made overseas", "sourced from another country"),
    ("buckle", r"\bbuckle closure\b", "fastens using a metal clasp", "secured with an adjustable clasp"),
    ("pull_on", r"\bpull on closure\b", "slips on without separate fasteners", "wears without needing a fastening"),
    ("zipper", r"\bzipper closure\b", "closes with interlocking teeth", "fastens using a sliding toothed track"),
    ("lace", r"\blace up closure\b", "secured by tied cords", "fastens with laced ties"),
    ("button", r"\bbutton closure\b", "fastens with round closures", "secured with small fastening discs"),
    ("water_resistant", r"\bwater resistant\b", "repels light moisture", "handles rain without soaking through"),
    ("waterproof", r"\bwaterproof\b", "keeps water from penetrating", "forms a barrier against wet weather"),
    ("machine_washable", r"\bmachine washable\b", "safe to clean in a washing machine", "can be laundered by an appliance"),
    ("hand_wash", r"\bhand wash(?: only)?\b", "requires manual cleaning", "must be cleaned by hand"),
    ("lightweight", r"\blightweight\b", "built to add very little weight", "designed to feel scarcely heavy"),
    ("breathable", r"\bbreathable\b", "allows air to circulate", "keeps airflow moving through the material"),
    ("slip_resistant", r"\bslip resistant\b", "maintains grip on slick surfaces", "helps prevent sliding on smooth ground"),
    ("adjustable", r"\badjustable\b", "can be resized for fit", "allows the fit to be changed"),
    ("elastic", r"\belastic\b", "stretches to conform", "expands and returns to shape"),
)


def atoms(product: dict) -> list[dict]:
    text = " ".join(flat(product.get(field)) for field in (
        "title", "features", "details", "description", "categories"
    ))
    found: list[dict] = []
    used: set[str] = set()
    for name, pattern, dev, holdout in RULES:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        canonical = match.group(0)
        key = norm(canonical)
        if not key or key in used:
            continue
        if key in norm(dev) or key in norm(holdout):
            raise AssertionError(f"rewrite leaks canonical anchor: {canonical}")
        used.add(key)
        found.append({
            "rule": name,
            "canonical": canonical,
            "attribute": classify(canonical),
            "development_paraphrase": dev,
            "holdout_paraphrase": holdout,
        })
    return found


def choose_card(product: dict) -> dict | None:
    choices = atoms(product)
    if len(choices) < 3:
        return None
    # Prefer distinct attribute buckets, then a deterministic rule order. The resulting
    # card has two hard and two soft constraints whenever available.
    selected: list[dict] = []
    seen_attributes: set[str] = set()
    for item in choices:
        if item["attribute"] not in seen_attributes:
            selected.append(item)
            seen_attributes.add(item["attribute"])
        if len(selected) == 4:
            break
    for item in choices:
        if item not in selected:
            selected.append(item)
        if len(selected) == 4:
            break
    if len(selected) < 3:
        return None
    return {"hard_constraints": selected[:2], "soft_preferences": selected[2:4]}


def scenario_sequence(n: int, seed: int) -> list[str]:
    counts = {
        "buying": int(n * 0.40),
        "browsing": int(n * 0.40),
        "intent_override": int(n * 0.15),
    }
    counts["boundary"] = n - sum(counts.values())
    rows = [scenario for scenario, count in counts.items() for _ in range(count)]
    random.Random(seed).shuffle(rows)
    return rows


def write_set(name: str, targets: list[tuple[str, dict, dict]], profiles: list[dict],
              family: str, out: Path, seed: int) -> dict:
    scenarios = scenario_sequence(len(targets), seed + 91)
    rows = []
    for index, ((asin, product, card), scenario) in enumerate(zip(targets, scenarios), start=1):
        active_card = {
            group: [
                {
                    "rule": atom["rule"],
                    "canonical": atom["canonical"],
                    "attribute": atom["attribute"],
                    "paraphrase": atom[f"{family}_paraphrase"],
                }
                for atom in values
            ]
            for group, values in card.items()
        }
        rows.append({
            "sample_id": f"{name}_{index:04d}",
            "scenario_type": scenario,
            "ground_truth": {"parent_asin": asin},
            "user_profile": profiles[(index - 1) % len(profiles)],
            "category": coarse_category([str(value) for value in product.get("categories") or []]),
            "semantic_card": active_card,
            "paraphrase_family": family,
        })
    path = out / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "path": path.name,
        "rows": len(rows),
        "distinct_targets": len({row["ground_truth"]["parent_asin"] for row in rows}),
        "scenario_counts": {name: sum(row["scenario_type"] == name for row in rows)
                            for name in ("buying", "browsing", "intent_override", "boundary")},
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--public", type=Path, default=ROOT / "data" / "public_set.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "datasets" / "sets")
    parser.add_argument("--development-size", type=int, default=200)
    parser.add_argument("--holdout-size", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    public = [json.loads(line) for line in args.public.read_text(encoding="utf-8").splitlines() if line]
    public_targets = {str(row["ground_truth"]["parent_asin"]) for row in public}
    profiles = [row["user_profile"] for row in public]
    eligible: list[tuple[str, dict, dict]] = []
    with args.catalog.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            asin = str(product["parent_asin"])
            if asin in public_targets:
                continue
            card = choose_card(product)
            if card is not None:
                eligible.append((asin, product, card))
    needed = args.development_size + args.holdout_size
    if len(eligible) < needed:
        raise RuntimeError(f"only {len(eligible)} eligible products for {needed} requested rows")
    rng = random.Random(args.seed)
    selected = rng.sample(eligible, needed)
    args.out.mkdir(parents=True, exist_ok=True)
    dev = write_set("semantic_attribute_development_200", selected[:args.development_size], profiles,
                    "development", args.out, args.seed)
    holdout = write_set("semantic_attribute_holdout_800", selected[args.development_size:], profiles,
                        "holdout", args.out, args.seed + 1)
    manifest = {
        "schema_version": 1,
        "truth_status": "target-disjoint internal semantic robustness data; not organizer private data",
        "format": "released evaluator message templates with attribute values semantically rephrased",
        "selection": "uniform sample without replacement from catalogue products with at least three manually mapped semantic attributes",
        "public_target_overlap": 0,
        "development_holdout_target_overlap": 0,
        "eligible_products": len(eligible),
        "sets": {"SemanticShift-Dev200": dev, "SemanticShift-Holdout800": holdout},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
