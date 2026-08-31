import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "experiments" / "studies"

source = V2 / "external_train_only_canonicals.jsonl"
destination = V2 / "broad_semantic_paraphrases.jsonl"

rewrites = {
    "100 cotton": ["pure cotton"],
    "100 percent cotton": ["pure cotton"],
    "100 organic cotton": ["pure organic cotton"],
    "100 percent organic cotton": ["pure organic cotton"],
    "100 polyester": ["pure polyester"],
    "100 percent polyester": ["pure polyester"],
    "100 synthetic": ["entirely synthetic"],
    "100 synthetic nylon": ["entirely synthetic nylon"],
    "synthetic": ["man-made material"],
    "synthetic material": ["man-made material"],
    "faux leather": ["imitation leather"],
    "faux fur": ["imitation fur"],
    "faux suede": ["imitation suede"],
    "polyurethane": ["PU"],
    "100 polyurethane": ["100 PU"],
    "100 pu": ["100 polyurethane"],
    "polyvinyl chloride": ["PVC"],
    "100 polyvinyl chloride": ["100 PVC"],
    "100 pvc": ["100 polyvinyl chloride"],
    "ethylene vinyl acetate": ["EVA"],
    "100 ethylene vinyl acetate": ["100 EVA"],
    "100 eva": ["100 ethylene vinyl acetate"],
    "sterling silver": ["925 silver"],
    "solid sterling silver": ["solid 925 silver"],
    "cubic zirconia": ["CZ"],
    "14k gold": ["14 karat gold"],
    "10k yellow gold": ["10 karat yellow gold"],
    "14k yellow gold": ["14 karat yellow gold"],
    "18k yellow gold": ["18 karat yellow gold"],
    "hook and loop": ["touch fastener"],
    "hook and loop closure": ["touch fastener closure"],
    "velcro": ["hook and loop"],
    "velcro closure": ["hook and loop closure"],
    "snap closure": ["press stud closure"],
    "press stud closure": ["snap closure"],
    "button closure": ["button fastening"],
    "button fastening": ["button closure"],
    "buckle closure": ["buckle fastening"],
    "buckle fastening": ["buckle closure"],
    "magnetic closure": ["magnet closure"],
    "magnet closure": ["magnetic closure"],
    "turn lock closure": ["twist lock closure"],
    "twist lock closure": ["turn lock closure"],
    "seamless": ["without seams"],
    "without seams": ["seamless"],
    "hooded": ["with a hood"],
    "with a hood": ["hooded"],
    "collared": ["with a collar"],
    "with a collar": ["collared"],
    "sleeveless": ["without sleeves"],
    "without sleeves": ["sleeveless"],
    "turtleneck": ["polo neck"],
    "polo neck": ["turtleneck"],
    "high waist": ["high rise"],
    "high rise": ["high waist"],
    "high waisted": ["high rise"],
    "quick dry": ["fast drying"],
    "fast drying": ["quick dry"],
    "waterproof": ["watertight"],
    "watertight": ["waterproof"],
    "slip resistant": ["non-slip"],
    "non-slip": ["slip resistant"],
    "uv protection": ["UV blocking"],
    "uv blocking": ["UV protection"],
}

def constraint_type(phrase: str) -> str:
    p = phrase.lower()
    if any(x in p for x in ("cotton", "polyester", "leather", "suede", "nylon", "wool", "silk", "fabric", "textile", "synthetic", "rubber", "plastic", "polyurethane", "polyvinyl", "eva", "cubic zirconia", "gold", "silver", "metal", "cashmere", "fleece", "linen", "rayon", "viscose", "acrylic", "spandex", "material")):
        return "material"
    if any(x in p for x in ("closure", "fastening", "zipper", "zip ", "buckle", "button", "snap", "velcro", "hook and loop", "magnetic", "magnet", "clasp")):
        return "closure"
    if any(x in p for x in ("inch", "mm", "cm", "meter", "feet", "foot", "liter", "gauge", "diameter", "width", "height", "inseam", "drop", "size", "count", "pack", "piece", "pair", "year")):
        return "measurement"
    if any(x in p for x in ("water", "wind", "uv", "upf", "breathable", "wicking", "dry", "resistant", "proof", "thermal", "insulation", "antimicrobial", "odor")):
        return "performance"
    if any(x in p for x in ("sleeve", "neck", "waist", "fit", "lined", "hood", "collar", "pocket", "heel", "toe", "sole", "strap")):
        return "construction"
    return "other"

with source.open("r", encoding="utf-8") as incoming, destination.open("w", encoding="utf-8", newline="\n") as outgoing:
    for line in incoming:
        phrase = json.loads(line)["canonical"]
        candidates = [c for c in rewrites.get(phrase, []) if c.lower() != phrase.lower()][:4]
        outgoing.write(json.dumps({"canonical": phrase, "constraint_type": constraint_type(phrase), "candidates": candidates}, ensure_ascii=False) + "\n")
