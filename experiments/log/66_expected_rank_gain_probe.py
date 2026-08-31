"""Experiment 66: choose clarification by expected one-turn rank improvement.

This is not target-labelled optimization.  At a live session state, the current V1
ranking supplies the five most plausible target hypotheses, weighted proportional to
1/rank.  For every admissible question and every distinct reply those hypotheses imply,
the experiment forks only visible-catalogue state, applies that reply as ordinary
constraint evidence, and reranks.  It chooses the question with largest expected
reciprocal-rank gain for the hypotheses:

  sum_i p_i * (1/rank_after(i, reply_i) - 1/rank_before(i)).

The real target is never read by the policy.  This Official200-only pass is a feasibility
test before any unseen-population comparison.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, classify_constraint, evaluate, intent_card, load_jsonl
from submission.agent import Agent, CONSTRAINT, DEAD_ATTRIBUTES, SessionState

ATTRIBUTES = ("feature", "material", "color", "style", "size", "use_case", "other")
HYPOTHESIS_CAP = 5
OUT = ROOT / "experiments" / "results" / "out_66_expected_rank_gain_probe_official200.json"


class ExpectedRankGain(Agent):
    signatures_by_asin: dict[str, dict[str, tuple[str, ...]]]

    @staticmethod
    def _key(st: SessionState) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((phrase, tier) for phrase, (_, tier) in st.evidence.items()))

    def _candidates(self, st: SessionState, message: str) -> list[str]:
        cached = self._pool_cache.get(st.sid)
        if cached and cached[0] == self._key(st):
            del self._pool_cache[st.sid]
            return list(cached[1])
        return super()._candidates(st, message)

    def _reply(self, asin: str, attribute: str, disclosed: set[str]) -> tuple[str, ...]:
        return tuple(value for value in self.signatures_by_asin[asin][attribute]
                     if value not in disclosed)[:2]

    @staticmethod
    def _rank_map(ranked: list[str]) -> dict[str, int]:
        return {asin: index + 1 for index, asin in enumerate(ranked)}

    def _shadow(self, st: SessionState, reply: tuple[str, ...], attribute: str) -> SessionState:
        shadow = SessionState(list(st.tags), None)  # sid=None prevents population-calibration writes
        shadow.evidence = dict(st.evidence)
        shadow.asked = [*st.asked, attribute]
        shadow.turn = st.turn
        shadow.last_rank = list(st.last_rank)
        shadow.rejected = set(st.rejected)
        for value in reply:
            if value not in shadow.evidence:
                df = self.ix.df(value)
                shadow.evidence[value] = (df if df > 0 else self.ix.DF_CAP * 2, CONSTRAINT)
        return shadow

    def _next_probe(self, st: SessionState) -> str:
        options = [a for a in ATTRIBUTES if a not in st.asked and a not in DEAD_ATTRIBUTES]
        if not options:
            return "other"
        pool = super()._candidates(st, "")
        self._pool_cache[st.sid] = (self._key(st), tuple(pool))
        if len(pool) < 2:
            return super()._next_probe(st)

        before = super()._rank(st, pool, len(pool))
        before_rank = self._rank_map(before)
        hypotheses = before[:HYPOTHESIS_CAP]
        norm = sum(1.0 / (i + 1) for i in range(len(hypotheses)))
        weights = {asin: (1.0 / (index + 1)) / norm for index, asin in enumerate(hypotheses)}
        disclosed = {phrase for phrase, (_, tier) in st.evidence.items() if tier != "cat"}

        expected_gain: dict[str, float] = {}
        for attribute in options:
            grouped: dict[tuple[str, ...], list[str]] = defaultdict(list)
            for asin in hypotheses:
                grouped[self._reply(asin, attribute, disclosed)].append(asin)
            gain = 0.0
            for reply, members in grouped.items():
                shadow = self._shadow(st, reply, attribute)
                after_pool = super()._candidates(shadow, "")
                after_rank = self._rank_map(super()._rank(shadow, after_pool, len(after_pool)))
                for asin in members:
                    old = before_rank.get(asin, len(before) + 1)
                    new = after_rank.get(asin, len(after_pool) + 1)
                    gain += weights[asin] * (1.0 / new - 1.0 / old)
            expected_gain[attribute] = gain

        selected = max(options, key=lambda a: (expected_gain[a], -ATTRIBUTES.index(a)))
        self.rank_gain_trace.append({"selected": selected, "expected_gain": expected_gain[selected]})
        return selected


def shared(base: Agent, signatures: dict[str, dict[str, tuple[str, ...]]]) -> ExpectedRankGain:
    agent = object.__new__(ExpectedRankGain)
    agent.ix, agent.sessions = base.ix, {}
    agent.llm, agent.llm_extract, agent.tagger = None, None, None
    agent.signatures_by_asin = signatures
    agent._pool_cache, agent.rank_gain_trace = {}, []
    return agent


def summary(result: dict) -> dict:
    return {key: result[key] for key in ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score")}


def main() -> None:
    public = load_jsonl(ROOT / "data" / "public_set.jsonl")
    ids, categories, products = catalog_index(ROOT / "data" / "catalog.jsonl")
    signatures: dict[str, dict[str, tuple[str, ...]]] = {}
    for asin, product in products.items():
        constraints = tuple(dict.fromkeys([
            *(str(v) for v in intent_card(product).get("hard_constraints", [])),
            *(str(v) for v in intent_card(product).get("soft_preferences", [])),
        ]))
        signatures[asin] = {attribute: (constraints if attribute == "other" else tuple(
            value for value in constraints if classify_constraint(value) == attribute
        )) for attribute in ATTRIBUTES}

    base = Agent(ROOT / "data" / "catalog.jsonl")
    fixed = object.__new__(Agent)
    fixed.ix, fixed.sessions = base.ix, {}
    fixed.llm, fixed.llm_extract, fixed.tagger = None, None, None
    policy = shared(base, signatures)
    reference, candidate = evaluate(fixed, public, ids, categories, products), evaluate(policy, public, ids, categories, products)
    trace = policy.rank_gain_trace
    output = {
        "hypothesis_cap": HYPOTHESIS_CAP,
        "hypothesis_prior": "p(rank=i) proportional to 1/i for i in current top five",
        "objective": "expected reciprocal-rank gain after simulated visible-catalogue reply",
        "fixed_v1": summary(reference), "expected_rank_gain": summary(candidate),
        "delta_score": candidate["recommended_technical_score"] - reference["recommended_technical_score"],
        "question_counts": dict(Counter(row["selected"] for row in trace)),
        "mean_selected_expected_gain": sum(row["expected_gain"] for row in trace) / len(trace) if trace else 0.0,
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
