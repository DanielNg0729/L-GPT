"""Was the substring fallback wrong, or was its WEIGHT wrong?

WHAT WAS REMOVED. `_resolve` used to try the whole clause, then the longest contiguous
substring the catalogue attests, then individual attested tokens. Experiment 69 deleted it:
four decision criteria were byte-identical without it, inverse moved -0.000200, and the
attribute-paraphrase suite GAINED +0.056000. The diagnosis was that on a clause the agent
cannot parse -- "made from a soft plant fibre" -- the fallback contributed `soft` and
`plant` as if the customer had said them.

WHAT WAS NEVER TESTED. Those fragments entered at CONSTRAINT weight, 1.0, the same as a
phrase the customer literally spoke. The experiment compared "fragments at full strength"
against "no fragments at all" and never tried the obvious third option.

We already know that distinction matters more than the knowledge itself. The deparaphraser
recovers 81.5% of a perfect resolver when its proposals enter at CONSTRAINT and roughly 96%
when they are attenuated -- same proposals, different weight. A separate audit found the
attenuated weight so effective that deleting 41 confidently wrong proposals changed the
score by exactly 0.000000. A substring fragment is epistemically the same kind of thing: not
what the customer said, but the part of it the catalogue recognises.

So this runs the original fallback at four weights, unchanged in every other respect.

WHAT THIS CANNOT FIX, and it is worth being precise because the two are easily confused.
The fallback fires only when the whole clause is unattested. The one miss on the public set
is the opposite case: `color: grey` resolves to `color grey`, which 52 products carry and
the target does not, so the clause IS attested and the fallback never engages. That needs
the synthesised-colour handling, not this.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/studies/evaluate_substring_fallback_weight.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ["LLM_RERANK"] = "0"
os.environ["LLM_EXTRACT"] = "0"
os.environ["LLM_RESOLVE"] = "0"
os.environ["V2_ROUTE"] = "1"
os.environ["BERT_EXTRACT"] = "1"

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import (_SYNTHESISED_COLOUR, CONSTRAINT, MINED, SEM, Agent,  # noqa: E402
                              raw_toks)

SETS = ROOT / "experiments" / "datasets" / "sets"
OV = ROOT / "experiments" / "datasets" / "open_vocabulary"
OUT = ROOT / "experiments" / "results" / "out_77_substring_fallback_weight.json"

SUITES = (
    ("official200", ROOT / "data" / "public_set.jsonl"),
    ("org_proxy_800", SETS / "organizer_proxy_800.jsonl"),
    ("review800", SETS / "catalog_review_distinct_800.jsonl"),
    ("uniform800", SETS / "catalog_uniform_800.jsonl"),
    ("inverse800", SETS / "catalog_inverse_800.jsonl"),
    ("attr_paraphrase800", OV / "review800_open_vocab_paraphrase.jsonl"),
)


def make(weight):
    """`weight` is None for the shipped suppression, else the weight fragments carry."""

    class Arm(Agent):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._fragment = set()          # phrases that came from the fallback

        def _resolve(self, text, cap=None):
            # REPRODUCE THE SHIPPED PREAMBLE, or this measures a different agent. An
            # earlier version of this override reimplemented `_resolve` from the whole
            # phrase onward and silently skipped the synthesised-colour handling that
            # ships ahead of it, so every arm -- including the baseline -- ran without a
            # fix worth +0.03 on two criteria. The baseline printed 0.970500 where the
            # shipped agent scores 0.971500, which is how it was caught.
            colour = _SYNTHESISED_COLOUR.match(str(text))
            if colour:
                text = colour.group(1).strip()

            t = raw_toks(text)[:self.RESOLVE_CAP if cap is None else cap]
            if not t:
                return []
            whole = " ".join(t)
            if self.ix.df(whole) > 0:
                return [whole]
            if weight is None:
                return []                   # shipped behaviour: suppress
            # The original fallback, restored verbatim: longest attested window, then
            # attested single tokens, at most two of either.
            for n in range(len(t) - 1, 1, -1):
                hits = [" ".join(t[i:i + n]) for i in range(0, len(t) - n + 1)
                        if self.ix.df(" ".join(t[i:i + n])) > 0]
                if hits:
                    self._fragment.update(hits[:2])
                    return hits[:2]
            single = [x for x in t if self.ix.df(x) > 0][:2]
            self._fragment.update(single)
            return single

        def _weight(self, phrase, df, tier):
            # A fragment is not what the customer said; it is the part of it the catalogue
            # recognises. Score it as such regardless of the tier its clause carried.
            # The CONSTRAINT arm must leave the tier alone -- that is the behaviour
            # experiment 69 rejected. An earlier version of this file mapped it to MINED
            # like the others, so two arms silently ran the same configuration and
            # reported identical numbers.
            if phrase in self._fragment and weight in ("mined", "sem"):
                return super()._weight(phrase, df, SEM if weight == "sem" else MINED)
            return super()._weight(phrase, df, tier)

    return Arm


def main() -> None:
    ids, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    loaded = [(n, load_jsonl(p)) for n, p in SUITES if p.exists()]
    arms = (("shipped (suppress)", None),
            ("fragments @ CONSTRAINT", "constraint"),
            ("fragments @ MINED 0.48", "mined"),
            ("fragments @ SEM 0.15", "sem"))

    print(f"{'arm':<26}" + "".join(f"{n[:17]:>19}" for n, _ in loaded))
    print("-" * (26 + 19 * len(loaded)))
    rows, base = {}, {}
    for label, w in arms:
        cells, rows[label] = [], {}
        cls = make(None if w is None else w)
        for name, samples in loaded:
            r = evaluate(cls(ROOT / "data" / "catalog.jsonl"), samples, ids, cats, prods)
            s = r["recommended_technical_score"]
            rows[label][name] = round(s, 6)
            if w is None:
                base[name] = s
                cells.append(f"{s:>19.6f}")
            else:
                cells.append(f"{s:>12.6f}{s - base[name]:>+7.4f}")
        print(f"{label:<26}" + "".join(cells), flush=True)

    print("\n  A fragment arm is worth adopting only if it is non-negative on all five")
    print("  decision criteria AND positive on attribute paraphrase. The shipped")
    print("  suppression already gained +0.056 there by removing fragments at full weight.")
    OUT.write_text(json.dumps({
        "experiment": "substring fallback re-tested at attenuated weights",
        "baseline": "shipped suppression",
        "arms": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
