"""V2.62: training data for a value-span extractor, with the vocabulary deliberately destroyed.

WHAT THE MODEL MUST LEARN, AND WHAT IT MUST NOT
------------------------------------------------
The span node is exact lookup against catalogue vocabulary, so it recovers a value only
when the value is still IN that vocabulary. Paraphrase moves it out, and no amount of
lookup gets it back. A span extractor's job is to answer a different question -- WHERE is
the value in this message -- which does not depend on the value being anything in
particular.

That distinction is the whole design risk. Trained naively on the template bank, a model
sees `Cotton`, `Leather`, `Imported` in the value slot ten thousand times and learns
"tokens that look like catalogue attributes are values". That is lexical memorisation
wearing a span extractor's clothes, and it fails on exactly the paraphrased values it was
built for -- while scoring beautifully on any canonical-value test, which is the trap this
project already fell into once by measuring template recovery with values held constant.

BOTH SLOT VOCABULARIES ARE DESTROYED, NOT JUST THE VALUE'S -- a correction. The first
version randomised only the VALUE slot and left CATEGORY always holding real taxonomy. That
gave the two classes different distributions, so the model could partly key on APPEARANCE:
taxonomy-shaped text meant category. Since 45% of randomised values were catalogue prose,
which is also taxonomy-shaped, the classes collided. Measured on held-out templates, the
model invented a VALUE span on 51.3% of `plain_opening` rows -- turns that carry a category
and no value at all -- and the rate was identical on canonical and paraphrased values,
confirming it was structural rather than vocabulary-driven.

Both slots are now drawn from the SAME pool, and a deliberate share of examples puts the
very same phrase in a category slot in one example and a value slot in another. That makes
appearance useless by construction and leaves position as the only learnable signal, which
is what the extractor is for.

SO THE SLOT VOCABULARIES ARE DESTROYED AT TRAINING TIME. Each slot is filled from a mixed
pool, so the only signal that survives across examples is the SLOT'S POSITION IN THE
WRAPPER:

  canonical atoms      the real thing, kept at a minority share so the easy case still
                       trains
  dictionary atoms     other catalogue attribute values, 1-3 tokens
  catalogue prose      random 3-8 token spans lifted from product `features` text. These
                       are multi-word descriptive phrases with no attribute-like shape,
                       which is what a paraphrased value looks like
  nonsense             random token sequences, so the model cannot fall back on English

The prose pool is drawn from `data/catalog.jsonl` only. It deliberately does NOT use the
204 evaluation paraphrases: training on the phrasings we score against would be the same
contamination that made a regex report a fitted 100% earlier today.

LABELS. BIO over two classes.
  VALUE     the {a} and {b} slots -- the constraint values
  CATEGORY  the {category} slot, included so the model does not conflate two adjacent
            slots rather than because we need category spans
  O         everything else, INCLUDING the {attribute} slot. That slot appears only in
            no-evidence templates and holds a facet NAME ("material", "color"), never a
            value. Labelling it O is what teaches the model to emit nothing on a turn that
            carries no requirement -- the same failure the span node had until facet names
            were excluded from its dictionary.

TRAIN ONLY. Source is `v1_route_template_bank/train.jsonl`, whose templates are disjoint
from the held-out test bank (verified: zero shared templates). Evaluation happens there,
with PARAPHRASED values substituted in, never here.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/studies/build_span_bert_data.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "experiments" / "studies"
TRAIN = V2 / "route_template_bank" / "train.jsonl"
OUT = V2 / "span_bert_v2" / "train.jsonl"
DEV = V2 / "span_bert_v2" / "dev.jsonl"

SLOT_RE = re.compile(r"\{(\w+)\}")
VALUE_SLOTS = {"a", "b"}
CATEGORY_SLOTS = {"category"}
SEED = 20260830          # literal, never hash() -- that is per-process randomised
DEV_FRACTION = 0.08


def render_with_spans(template: str, values: dict) -> tuple[str, list]:
    """Fill a template and return the text plus (start, end, slot) character spans."""
    out, spans, pos = [], [], 0
    for m in SLOT_RE.finditer(template):
        out.append(template[pos:m.start()])
        prefix = "".join(out)
        slot = m.group(1)
        text = str(values.get(slot, ""))
        spans.append((len(prefix), len(prefix) + len(text), slot))
        out.append(text)
        pos = m.end()
    out.append(template[pos:])
    return "".join(out), spans


def bio(text: str, spans: list) -> tuple[list, list]:
    """Word-level tokens and BIO labels, aligned by character offset."""
    words, offsets = [], []
    for m in re.finditer(r"\S+", text):
        words.append(m.group())
        offsets.append((m.start(), m.end()))
    labels = ["O"] * len(words)
    for start, end, slot in spans:
        if slot in VALUE_SLOTS:
            tag = "VALUE"
        elif slot in CATEGORY_SLOTS:
            tag = "CATEGORY"
        else:
            continue                      # {attribute} is a facet name, deliberately O
        first = True
        for i, (ws, we) in enumerate(offsets):
            if ws < end and we > start:   # any character overlap
                labels[i] = ("B-" if first else "I-") + tag
                first = False
    return words, labels


def main() -> None:
    from evaluator.local_evaluator import coarse_category
    rng = random.Random(SEED)
    rows = [json.loads(l) for l in TRAIN.open(encoding="utf-8") if l.strip()]
    print(f"train bank: {len(rows)} rows")

    dictionary = [json.loads(l)["canonical"] for l in
                  (V2 / "catalogue_attribute_dictionary.jsonl").open(encoding="utf-8")
                  if l.strip()]
    print(f"dictionary atoms: {len(dictionary):,}")

    # Catalogue prose: multi-word descriptive spans, the SHAPE a paraphrased value takes.
    prose: list[str] = []
    with (ROOT / "data" / "catalog.jsonl").open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if len(prose) >= 60000:
                break
            if i % 3:                     # thin the stream; 50k products is plenty
                continue
            try:
                product = json.loads(line)
            except Exception:
                continue
            for feature in (product.get("features") or [])[:2]:
                toks = str(feature).split()
                if len(toks) < 4:
                    continue
                w = rng.randint(3, 8)
                s = rng.randint(0, max(len(toks) - w, 0))
                phrase = " ".join(toks[s:s + w]).strip(" .,;:")
                if 2 < len(phrase) < 80:
                    prose.append(phrase)
    print(f"catalogue prose spans: {len(prose):,}")

    alphabet = sorted({w.lower() for p in prose[:4000] for w in p.split() if w.isalpha()})

    def nonsense() -> str:
        return " ".join(rng.choice(alphabet) for _ in range(rng.randint(2, 6)))

    # Real coarse categories, so the category slot can also be filled with something
    # other than the product's own taxonomy string.
    categories: list[str] = []
    seen_cat: set[str] = set()
    with (ROOT / "data" / "catalog.jsonl").open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if len(categories) >= 4000:
                break
            if i % 7:
                continue
            try:
                product = json.loads(line)
            except Exception:
                continue
            vals = [str(x) for x in (product.get("categories") or [])]
            phrase = coarse_category(vals)
            if phrase and phrase not in seen_cat:
                seen_cat.add(phrase)
                categories.append(phrase)
    print(f"coarse categories: {len(categories):,}")

    # THE SHARED POOL. Both slots draw from this, so no distribution difference exists for
    # the model to exploit, and the same phrase appears in both roles across examples.
    shared = dictionary + prose + categories

    def sample_value(canonical: str) -> str:
        # Mixture chosen so the majority of training examples carry a value the model has
        # no lexical reason to recognise. Canonical is kept as a minority so the easy case
        # is still represented rather than trained away.
        r = rng.random()
        if r < 0.15:
            return canonical
        if r < 0.85:
            return rng.choice(shared)     # atoms, prose AND categories, all one pool
        return nonsense()

    def sample_category(canonical: str) -> str:
        """Randomised on the SAME pool as values, for the reason in the docstring."""
        r = rng.random()
        if r < 0.35:
            return canonical              # the real taxonomy string, kept as a minority
        if r < 0.90:
            return rng.choice(shared)
        return nonsense()

    built, skipped = [], 0
    for row in rows:
        template = str(row.get("template", ""))
        slots = dict(row.get("slots") or {})
        if not SLOT_RE.search(template):
            skipped += 1                  # 'literal organizer generation' and friends
            continue
        filled = dict(slots)
        for slot in list(filled):
            if slot in VALUE_SLOTS:
                filled[slot] = sample_value(str(slots[slot]))
            elif slot in CATEGORY_SLOTS:
                filled[slot] = sample_category(str(slots[slot]))
        text, spans = render_with_spans(template, filled)
        words, labels = bio(text, spans)
        if not words:
            continue
        built.append({"tokens": words, "labels": labels, "action": row.get("action"),
                      "template": template})

    rng.shuffle(built)
    cut = int(len(built) * (1 - DEV_FRACTION))
    train, dev = built[:cut], built[cut:]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for path, data in ((OUT, train), (DEV, dev)):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data),
                        encoding="utf-8")
        print(f"[saved] {path.name}  {len(data)} examples")
    print(f"skipped (no slots in template): {skipped}")

    from collections import Counter
    lab = Counter(l for r in built for l in r["labels"])
    print(f"\nlabel distribution: {dict(lab)}")
    no_value = sum(1 for r in built if not any(l.endswith("VALUE") for l in r["labels"]))
    print(f"examples with NO value span: {no_value} ({no_value/len(built):.1%})"
          f"   <- the no-evidence turns the model must stay silent on")
    print("\nsample:")
    for r in built[:3]:
        pairs = " ".join(f"{t}/{l}" for t, l in zip(r["tokens"], r["labels"]))
        print(f"  {pairs[:150]}")


if __name__ == "__main__":
    main()
