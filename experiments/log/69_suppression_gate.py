"""EDA pass 56: suppress unverifiable constraint clauses instead of mining junk from them.

THE MECHANISM, FROM THE ORACLE DECOMPOSITION (V2.43)
-----------------------------------------------------
On the attribute-paraphrase suite, V1 scores 0.777000 against 0.970100 canonical. Of that
0.1931 gap:

    a PERFECT semantic resolver would recover   +0.0979  (50.7%)
    merely DELETING the paraphrased clause      +0.0467  (24.2%)

Deleting recovers a quarter of the gap with no model at all, which means the paraphrased
text is not merely uninformative -- it is ACTIVELY HARMFUL. The reason is `_resolve`:

    whole phrase attested?  -> use it
    else                    -> longest attested SUBSTRING, else up to 2 single TOKENS

So "made from a soft plant fibre" contributes tokens like `soft` and `plant` at full
CONSTRAINT weight. Those are not the customer's requirement; they are debris from a phrase
we failed to understand, and they pull ranking toward the wrong products.

WHAT THIS SHIPS
---------------
A conservative predicate: if the whole clause is not catalogue-attested, do not fall back
to arbitrarily short fragments of it. Three strengths are measured, because the fallback is
not worthless in general -- on CANONICAL traffic a long soft-preference may legitimately
resolve through a substring.

    off          current behaviour
    tokens       drop the single-token fallback only; keep multi-token substrings
    substrings   drop both fallbacks; an unattested clause contributes nothing

NON-REGRESSION IS THE POINT. Official200 and the unseen population suites must not move.
The paraphrase suite is where a gain is expected, but it is characterisation -- the
organizer confirmed no paraphrasing -- so it cannot justify a regression elsewhere.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/scripts/69_suppression_gate.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent, raw_toks  # noqa: E402


def make(mode: str):
    class Suppressed(Agent):
        SUPPRESS = mode

        def _resolve(self, text, cap=12):
            t = raw_toks(text)[:cap]
            if not t:
                return []
            whole = " ".join(t)
            if self.ix.df(whole) > 0:
                return [whole]                      # attested: unchanged behaviour
            if self.SUPPRESS == "substrings":
                return []                           # unverifiable -> contribute nothing
            for n in range(len(t) - 1, 1, -1):      # multi-token substrings still allowed
                hits = [" ".join(t[i:i + n]) for i in range(0, len(t) - n + 1)
                        if self.ix.df(" ".join(t[i:i + n])) > 0]
                if hits:
                    return hits[:2]
            if self.SUPPRESS == "tokens":
                return []                           # no single-token debris
            return [x for x in t if self.ix.df(x) > 0][:2]
    return Suppressed


def main() -> None:
    t0 = time.time()
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    rs = ROOT / "robustness" / "sets"
    pvo = ROOT / "robustness" / "v2" / "public_value_only"
    sets = {
        "official200": samples,
        "org-proxy": load_jsonl(rs / "organizer_proxy_800.jsonl"),
        "review800": load_jsonl(rs / "catalog_review_distinct_800.jsonl"),
        "uniform": load_jsonl(rs / "catalog_uniform_800.jsonl"),
        "inverse": load_jsonl(rs / "catalog_inverse_800.jsonl"),
        "attr-para": load_jsonl(pvo / "official200_attribute_paraphrase_dev.jsonl"),
    }
    COLS = list(sets)

    def run(cls):
        r = {}
        for name, sub in sets.items():
            a = object.__new__(cls)
            a.ix, a.sessions = base.ix, {}
            a.llm = a.llm_extract = a.tagger = None
            r[name] = round(evaluate(a, sub, cid, cats, prods)[
                "recommended_technical_score"], 6)
        return r

    print(f"{'mode':<14}" + "".join(f"{c:>13}" for c in COLS))
    print("-" * (14 + 13 * len(COLS)))
    out = {}
    for mode in ("off", "tokens", "substrings"):
        r = run(Agent if mode == "off" else make(mode))
        out[mode] = r
        print(f"{mode:<14}" + "".join(f"{r[c]:>13.6f}" for c in COLS), flush=True)

    ref = out["off"]
    print(f"\n{'mode':<14}" + "".join(f"{c:>13}" for c in COLS) + "   verdict")
    print("-" * (14 + 13 * len(COLS) + 12))
    decisive = ["official200", "org-proxy", "review800", "uniform", "inverse"]
    for mode, r in out.items():
        d = {c: r[c] - ref[c] for c in COLS}
        worst = min(d[c] for c in decisive)
        verdict = ("reference" if mode == "off" else
                   "ADOPT -- no regression on decision criteria" if worst >= -1e-9 else
                   "inside noise" if worst > -0.005 else "REJECT")
        print(f"{mode:<14}" + "".join(f"{d[c]:>+13.6f}" for c in COLS) + f"   {verdict}")

    print("\n  Decision criteria are official200 + the four population suites. attr-para is")
    print("  characterisation: the organizer confirmed no paraphrasing, so a gain there")
    print("  cannot justify a regression anywhere else.")
    (ROOT / "experiments" / "results" / "out_69_suppression_gate.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] experiments/results/out_69_suppression_gate.json   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
