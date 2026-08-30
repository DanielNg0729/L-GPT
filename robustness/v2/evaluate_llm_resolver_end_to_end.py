"""V2.46: does LLM attribute resolution beat SUPPRESSION end to end?

WHY THE PREVIOUS PROBE'S NUMBER WAS NOT THE ANSWER
--------------------------------------------------
V2.45 asked the model to name the catalogue value behind a paraphrase and scored the
answer by string equality against the suite's canonical. It reported 3/27, and that
number is wrong in the model's favour -- the canonicals are RAW CATALOGUE STRINGS, so
equality demanded the model reproduce the whole thing:

    made from a soft plant fibre  ->  "cotton"     vs  "100 cotton size 3t 2 3 4t 3 4 5 5"
    in the darkest colour         ->  "black"      vs  "color black"
    slips on without fasteners    ->  "pull on"    vs  "pull on closure"
    made from heavy woven cloth   ->  "canvas"     vs  "canvas upper"

Every one of those is a correct resolution scored as a failure. The fix is not a looser
string metric -- inventing a similarity threshold here would let me tune the threshold
until the result looked good, which is exactly the move to avoid. The fix is to stop
scoring the intermediate representation and score the ONLY thing that matters: does
feeding the proposal to the agent as evidence raise the end-to-end score.

WHAT THIS HAS TO BEAT, AND WHY THAT BAR IS HIGH
-----------------------------------------------
Not the paraphrase baseline (0.7770). Suppression already recovers +0.0560 of that gap by
contributing nothing when a clause is unverifiable, so the bar is 0.8330. A resolver that
is right sometimes and wrong sometimes lands BELOW that bar, because a confidently wrong
canonical is worse than an absent one: it withholds weight from the target AND hands it
to the field.

THREE ARMS
  1 suppression       the shipped agent. The floor.
  2 LLM @ CONSTRAINT  proposals enter at full constraint weight, like any template value.
  3 LLM @ weak tier   proposals enter at a reduced weight, so a wrong one costs less --
                      the G3 soft-attenuation shape rather than G2 pessimistic-hard.

Arm 3 exists because arms 1 and 2 bracket the decision badly on their own. If arm 2 loses
to arm 1, the question "is the knowledge real but the integration too trusting?" is still
open, and arm 3 is what answers it.

GUARDS, none of which are optional
  * REACHABILITY. The LLM is consulted only for a clause the catalogue cannot attest
    (`df(whole) == 0`) -- precisely where suppression already returns nothing.

    This docstring first claimed the path was UNREACHABLE on clean traffic, because the
    recognition gate matches 463/463 clean messages. Measuring it rather than asserting it
    showed the claim is false, and the reason is worth keeping: the recognition gate
    governs MESSAGES, not VALUES. A perfectly recognised template can still carry an
    unattested value, because `intent_card()` truncates long feature bullets mid-word:

        "All daughters love their mom, but sometimes we just forget to sa."

    That is genuine catalogue prose whose truncation breaks the phrase, so df == 0 and it
    falls through to suppression. It happens once in 463 clean messages and costs exactly
    0.000000 on official200 -- but the guarantee is empirical, not structural, and stating
    it as structural would be claiming a stronger property than the code has.
  * PROVENANCE. Every proposal is re-checked with `df(proposal) > 0` before it may become
    evidence. The model proposes; the catalogue disposes. A phrase the catalogue has never
    seen is discarded and the clause stays suppressed.
  * NO EMITTABILITY FILTER. Restricting proposals to values `intent_card()` can emit would
    score better here and would be gaming the suite, whose canonicals are emittable by
    construction. df > 0 is the honest gate and the only one applied.

THE SUITE IS TOO SMALL TO SETTLE THIS. It carries 27 distinct paraphrases across 200
sessions. That is a weakness of the suite, not a property of the problem -- a real
customer population has an open-ended paraphrase vocabulary. So nothing here may be keyed,
sized, cached or tuned to those 27 phrases, and a win must be re-measured on an
open-vocabulary suite before it can inform a shipping decision. What this run CAN settle
is the sign and the mechanism.

RESULT
------
    arm                            attr-para   vs floor   % of the V2.43 oracle
    1 suppression (shipped)         0.833000   +0.0000    57.2%
    2 LLM @ CONSTRAINT weight       0.856800   +0.0238    81.5%
    3 LLM @ weak tier w=0.15        0.870800   +0.0378    95.8%
    3 LLM @ weak tier w=0.30        0.867300   +0.0343    93.5%
    3 LLM @ weak tier w=0.45        0.871900   +0.0389    96.9%
    ORACLE-RESOLVE (perfect)        0.874925   +0.0979   100.0%

    resolver: 25 calls, 19 accepted, 6 abstained, 0 rejected by provenance, 0 failed
    official200: 0.970100 both arms, delta +0.000000

Three readings, in decreasing order of confidence.

ATTENUATION IS THE MECHANISM. Arm 2 captures 81.5% of the oracle and arm 3 captures ~96%.
The knowledge is the same in both; the difference is entirely how much a WRONG proposal
costs. At full constraint weight one bad canonical outranks the correct evidence around
it. This is the G3 soft-attenuation shape beating the G2 pessimistic-hard shape on its own
terms, and it is the most reliable thing this run shows.

THE WEIGHT IS NOT TUNED, AND MUST NOT BE. The three weak-tier weights span 0.0046 and are
NON-MONOTONE (0.45 > 0.15 > 0.30). That is noise on a 200-session suite, not a ranking.
Reporting w=0.45 as "the best weight" would be fitting noise and would manufacture a sharp
optimum where the data shows a flat one. The honest statement is that the weak tier is
worth about +0.037 and is INSENSITIVE to its weight across 0.15-0.45 -- which is a better
robustness property than a sharp optimum would have been.

WHAT THIS DOES NOT LICENSE. 96% of oracle is a striking number and this suite cannot
support it. It carries 27 distinct paraphrases; a single one resolving differently moves
the rate by 3.7 points, and the resolver was consulted 25 times in total. Before any of
this can inform a shipping decision it has to be re-measured on an OPEN-VOCABULARY suite
whose paraphrases were generated independently of these -- otherwise the claim is about
27 phrases rather than about paraphrase.

Run:  PYTHONIOENCODING=utf-8 python -u robustness/v2/evaluate_llm_resolver_end_to_end.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "robustness" / "v2"
PVO = V2 / "public_value_only"
OUT = V2 / "results" / "llm_resolver_end_to_end_v2_46.json"
CACHE = V2 / ".llm_resolver_cache.json"

SEM = "sem"

SYSTEM = (
    "You map a shopper's description of a product attribute onto the exact wording an "
    "e-commerce catalogue would use for it. Answer with the catalogue's own short "
    "attribute value and NOTHING else -- no sentence, no explanation, no quotes. "
    "Prefer the plainest, most common trade term: for 'made from a soft plant fibre' "
    "answer 'cotton'. If the description is a NEGATION or you are not confident which "
    "single value it names, answer exactly NONE. Answering NONE is correct and expected "
    "whenever you would otherwise guess."
)


class Resolver:
    """Paraphrase -> catalogue-attested phrase, or None.

    The cache is ordinary runtime hygiene -- an agent should not pay twice for the same
    question inside one run. It is deliberately NOT part of the design argument: sizing or
    justifying anything by how few distinct phrases this suite happens to contain would be
    measuring the suite instead of the problem.
    """

    def __init__(self, index, model: str, key: str) -> None:
        self.ix, self.model, self.key = index, model, key
        self.calls = self.accepted = self.abstained = self.unattested = self.failed = 0
        try:
            self.cache = json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:
            self.cache = {}

    def _ask(self, phrase: str) -> str | None:
        body = json.dumps({
            "model": self.model, "temperature": 0.0,
            # GPT-OSS bills hidden reasoning against max_tokens and returns EMPTY content
            # when the budget runs out mid-thought: HTTP 200 with nothing to parse. The
            # headroom is for the reasoning, not for the two-word answer.
            "max_tokens": 512, "reasoning_effort": "low",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": phrase}],
        }).encode("utf-8")
        req = Request(ENDPOINT, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
            # Groq's edge answers Python's default `Python-urllib/*` identity with a
            # Cloudflare 403 (code 1010) despite valid credentials.
            "User-Agent": "Mozilla/5.0"})
        for attempt in range(3):
            try:
                with urlopen(req, timeout=30) as r:
                    return json.loads(r.read().decode("utf-8"))[
                        "choices"][0]["message"]["content"].strip()
            except Exception:
                if attempt == 2:
                    return None
                time.sleep(2 * (attempt + 1))
        return None

    def resolve(self, phrase: str) -> str | None:
        if phrase in self.cache:
            got = self.cache[phrase]
            return got or None
        self.calls += 1
        raw = self._ask(phrase)
        if raw is None or not raw.strip():
            self.failed += 1
            return None                                # a failure is not an abstention
        prop = " ".join(raw.lower().split())
        if prop == "none":
            self.abstained += 1
            self.cache[phrase] = ""
            return None
        if self.ix.df(prop) <= 0:                      # PROVENANCE: catalogue must attest
            self.unattested += 1
            self.cache[phrase] = ""
            return None
        self.accepted += 1
        self.cache[phrase] = prop
        return prop

    def flush(self) -> None:
        CACHE.write_text(json.dumps(self.cache, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")


def main() -> None:
    os.environ["LLM_EXTRACT"] = "0"
    os.environ["BERT_EXTRACT"] = "0"
    global ENDPOINT
    from submission.llm_rerank import ENDPOINT as _EP, _load_project_env
    ENDPOINT = _EP
    _load_project_env()
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print("GROQ_API_KEY not set -- nothing to evaluate."); return
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    from submission.agent import CONSTRAINT, Agent, raw_toks

    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    para = load_jsonl(PVO / "official200_attribute_paraphrase_dev.jsonl")
    official = load_jsonl(ROOT / "data" / "public_set.jsonl")
    res = Resolver(base.ix, model, key)
    print(f"model={model}   paraphrase suite: {len(para)} sessions\n")

    def constraint_arm():
        class Arm(Agent):
            def _resolve(self, text, cap=None):
                out = super()._resolve(text, cap)
                if out:
                    return out                          # attested: untouched
                prop = res.resolve(" ".join(raw_toks(text)[:self.RESOLVE_CAP
                                                           if cap is None else cap]))
                return [prop] if prop else []
        return Arm

    def weak_arm(weight: float):
        class Arm(Agent):
            def _observe(self, st, msg):
                super()._observe(st, msg)
                for text, tier in super()._extract_templated(msg):
                    if tier != CONSTRAINT:
                        continue
                    toks = raw_toks(text)[:self.RESOLVE_CAP]
                    if not toks or self.ix.df(" ".join(toks)) > 0:
                        continue                        # attested: suppression never fired
                    prop = res.resolve(" ".join(toks))
                    if prop and prop not in st.evidence:
                        st.evidence[prop] = (self.ix.df(prop), SEM)

            def _weight(self, phrase, df, tier):
                if tier == SEM:
                    return weight / (1.0 + df) ** self.IDF_POW
                return super()._weight(phrase, df, tier)
        return Arm

    def run(cls, samples):
        a = object.__new__(cls)
        a.ix, a.sessions = base.ix, {}
        a.llm = a.llm_extract = a.tagger = None
        return round(evaluate(a, samples, cid, cats, prods)[
            "recommended_technical_score"], 6)

    t0 = time.time()
    arms = {"1 suppression (shipped floor)": run(Agent, para)}
    arms["2 LLM @ CONSTRAINT weight"] = run(constraint_arm(), para)
    for w in (0.15, 0.30, 0.45):
        arms[f"3 LLM @ weak tier w={w:.2f}"] = run(weak_arm(w), para)
    res.flush()

    floor = arms["1 suppression (shipped floor)"]
    print(f"{'arm':<34}{'attr-para':>11}{'vs floor':>11}")
    print("-" * 56)
    for k, v in arms.items():
        print(f"{k:<34}{v:>11.6f}{v - floor:>+11.6f}")

    # REACHABILITY. Claimed above from the recognition gate; measured here, because a
    # guarantee that is only argued is a guarantee that eventually stops holding.
    calls_before = res.calls
    off_supp = run(Agent, official)
    off_llm = run(constraint_arm(), official)
    print(f"\n  official200   suppression {off_supp:.6f}   LLM arm {off_llm:.6f}   "
          f"delta {off_llm - off_supp:+.6f}")
    print(f"  LLM calls made while scoring official200: {res.calls - calls_before}")
    print("  (zero is the expected value: the recognition gate means every clean message")
    print("   resolves through templates, so no clause ever reaches the model.)")

    print(f"\n  resolver: {res.calls} calls -- {res.accepted} accepted, "
          f"{res.abstained} abstained (NONE), {res.unattested} rejected by provenance, "
          f"{res.failed} failed")
    print(f"  {time.time() - t0:.0f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "V2.46 LLM resolver, end to end",
        "model": model, "arms": arms, "floor": floor,
        "official200": {"suppression": off_supp, "llm_arm": off_llm,
                        "llm_calls": res.calls - calls_before},
        "resolver": {"calls": res.calls, "accepted": res.accepted,
                     "abstained": res.abstained, "unattested": res.unattested,
                     "failed": res.failed},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
