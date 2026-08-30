"""V2.51a: extract paraphrasable attribute atoms from TARGET-DISJOINT sessions.

WHAT IS WRONG WITH THE SUITE WE HAVE
------------------------------------
`official200_attribute_paraphrase_dev.jsonl` rewrites 551 atoms, but through 27 hand-written
RULES -- one fixed paraphrase per concept. So the whole suite exercises 27 DISTINCT
paraphrases, over the same 200 targets as every other measurement.

Two independent weaknesses follow, and both cap what any result on it can mean:

  VOCABULARY. 27 phrases is not a vocabulary, it is a list. One phrase resolving
  differently moves the rate by 3.7 points, and nothing measured there can distinguish
  "the model understands paraphrase" from "the model knows these 27".

  TARGETS. The same 200 targets carry every result in this project, so a paraphrase result
  there is entangled with whatever is specific to those 200 products.

This builder fixes the second and prepares the first. Its output is the INPUT to
generation, not the suite -- V2.51b generates, V2.51c materialises.

TARGET DISJOINTNESS. Atoms are drawn from `catalog_review_distinct_800`, whose targets are
disjoint from Official200 by construction. The disjointness is asserted here rather than
assumed, and the run fails loudly if it does not hold.

WHAT COUNTS AS PARAPHRASABLE. Only atoms that are short enough to be a single attribute
value and are catalogue-attested. Long prose bullets are excluded: the simulator truncates
them mid-word, which produces the unattested-but-not-paraphrased case that suppression
already handles, and mixing that in would confound the measurement.

WHAT THIS DELIBERATELY DOES NOT DO. It does not filter atoms to concepts the resolver is
likely to get right, does not exclude anything for being hard, and does not consult any
model. Selection is purely structural, so the suite cannot be shaped toward a result.

Run:  PYTHONIOENCODING=utf-8 python -u robustness/v2/build_open_vocabulary_paraphrase_input.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "robustness" / "v2"
OUT = V2 / "open_vocabulary_generation_input.jsonl"

MIN_TOKENS, MAX_TOKENS = 1, 4      # a value, not a prose bullet
MAX_ATOMS = 260                    # generation budget; see the sampling note below


def main() -> None:
    from evaluator.local_evaluator import catalog_index, intent_card, load_jsonl
    from submission.agent import Agent, raw_toks

    _cid, _cats, products = catalog_index(ROOT / "data" / "catalog.jsonl")
    official = load_jsonl(ROOT / "data" / "public_set.jsonl")
    base = load_jsonl(ROOT / "robustness" / "sets" / "catalog_review_distinct_800.jsonl")

    off_targets = {str(s["ground_truth"]["parent_asin"]) for s in official}
    new_targets = {str(s["ground_truth"]["parent_asin"]) for s in base}
    overlap = off_targets & new_targets
    print(f"official200 targets {len(off_targets)}, review800 targets {len(new_targets)}")
    print(f"overlap: {len(overlap)}")
    assert not overlap, f"targets are NOT disjoint ({len(overlap)} shared) -- suite invalid"

    agent = Agent(ROOT / "data" / "catalog.jsonl")
    freq: Counter[str] = Counter()
    for sample in base:
        card = intent_card(products[str(sample["ground_truth"]["parent_asin"])])
        for group in ("hard_constraints", "soft_preferences"):
            for raw in card[group]:
                toks = raw_toks(str(raw))
                if not (MIN_TOKENS <= len(toks) <= MAX_TOKENS):
                    continue                       # prose bullet, not an attribute value
                atom = " ".join(toks)
                if agent.ix.df(atom) > 0:          # must be catalogue-attested
                    freq[atom] += 1

    print(f"\nparaphrasable atoms: {len(freq)} distinct, "
          f"{sum(freq.values())} occurrences")

    # SAMPLING. Ordered by how often the atom actually appears, then truncated. This favours
    # atoms that carry weight in real sessions rather than long-tail curiosities, and it is
    # deterministic. It is NOT a difficulty filter: nothing is dropped for being hard, and
    # the frequency order is a property of the session base, not of any model's behaviour.
    chosen = [a for a, _ in freq.most_common(MAX_ATOMS)]
    covered = sum(freq[a] for a in chosen)
    print(f"selected {len(chosen)} atoms covering {covered} occurrences "
          f"({covered / max(sum(freq.values()), 1):.1%} of the atom mass)")
    print(f"\ncompare: the existing suite exercises 27 distinct paraphrases. "
          f"This is {len(chosen)}.")
    print(f"\nmost frequent:  " + ", ".join(chosen[:12]))
    print(f"least frequent: " + ", ".join(chosen[-12:]))

    OUT.write_text("".join(
        json.dumps({"atom": a, "occurrences": freq[a], "df": agent.ix.df(a)},
                   ensure_ascii=False) + "\n" for a in chosen), encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
