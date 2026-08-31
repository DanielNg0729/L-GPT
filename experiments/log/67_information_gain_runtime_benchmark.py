"""Five-repetition Official200 latency comparison for clarification policies.

Measures only the session loop after the immutable catalogue index is loaded.  Each
repetition gets fresh session and population-calibration state.  The order is fixed by
request: current string-signature policy, integer-signature equivalent, then fixed V1.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, classify_constraint, evaluate, intent_card, load_jsonl
from submission.agent import Agent, DEAD_ATTRIBUTES

ATTRIBUTES = ("feature", "material", "color", "style", "size", "use_case", "other")
REPETITIONS = 5
OUT = ROOT / "experiments" / "results" / "out_67_information_gain_runtime_benchmark.json"


def load_current_module():
    path = ROOT / "experiments" / "log" / "65_candidate_information_gain_probe.py"
    spec = importlib.util.spec_from_file_location("information_gain_current", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IntegerSignatureInformationGain(Agent):
    """Exact uniform expected-elimination policy with integer response codes."""

    values_by_asin: dict[str, dict[str, tuple[int, ...]]]
    phrase_to_id: dict[str, int]
    code_base: int

    @staticmethod
    def _evidence_key(st):  # type: ignore[no-untyped-def]
        return tuple(sorted((phrase, tier) for phrase, (_, tier) in st.evidence.items()))

    def _candidates(self, st, message: str):  # type: ignore[no-untyped-def]
        cached = self._ig_pool_cache.get(st.sid)
        if cached and cached[0] == self._evidence_key(st):
            del self._ig_pool_cache[st.sid]
            return list(cached[1])
        return super()._candidates(st, message)

    def _reply_code(self, asin: str, attribute: str, disclosed: set[int]) -> int:
        values = self.values_by_asin[asin][attribute]
        first = next((value for value in values if value not in disclosed), 0)
        if not first:
            return 0
        second = next((value for value in values if value != first and value not in disclosed), 0)
        return first * self.code_base + second

    def _next_probe(self, st):  # type: ignore[no-untyped-def]
        options = [a for a in ATTRIBUTES if a not in st.asked and a not in DEAD_ATTRIBUTES]
        if not options:
            return "other"
        pool = super()._candidates(st, "")
        self._ig_pool_cache[st.sid] = (self._evidence_key(st), tuple(pool))
        if len(pool) < 2:
            return super()._next_probe(st)
        disclosed = {self.phrase_to_id[phrase] for phrase, (_, tier) in st.evidence.items()
                     if tier != "cat" and phrase in self.phrase_to_id}
        expected: dict[str, float] = {}
        for attribute in options:
            groups = Counter(self._reply_code(asin, attribute, disclosed) for asin in pool)
            expected[attribute] = sum(size * size for size in groups.values()) / len(pool)
        selected = min(options, key=lambda a: (expected[a], ATTRIBUTES.index(a)))
        self.information_gain_trace.append(selected)
        return selected


def fixed_agent(base: Agent) -> Agent:
    agent = object.__new__(Agent)
    agent.ix, agent.sessions = base.ix, {}
    agent.llm, agent.llm_extract, agent.tagger = None, None, None
    return agent


def current_agent(module, base: Agent, signatures):  # type: ignore[no-untyped-def]
    agent = object.__new__(module.CandidateInformationGain)
    agent.ix, agent.sessions = base.ix, {}
    agent.llm, agent.llm_extract, agent.tagger = None, None, None
    agent.signatures_by_asin = signatures
    agent.prior_by_asin = {asin: 1.0 for asin in signatures}
    agent.information_gain_trace, agent._ig_pool_cache = [], {}
    return agent


def integer_agent(base: Agent, values, phrase_to_id):  # type: ignore[no-untyped-def]
    agent = object.__new__(IntegerSignatureInformationGain)
    agent.ix, agent.sessions = base.ix, {}
    agent.llm, agent.llm_extract, agent.tagger = None, None, None
    agent.values_by_asin, agent.phrase_to_id = values, phrase_to_id
    agent.code_base = len(phrase_to_id) + 1
    agent.information_gain_trace, agent._ig_pool_cache = [], {}
    return agent


def compact(result: dict) -> dict:
    return {key: result[key] for key in ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score")}


def main() -> None:
    rows = load_jsonl(ROOT / "data" / "public_set.jsonl")
    ids, categories, products = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    signatures: dict[str, dict[str, tuple[str, ...]]] = {}
    all_values: set[str] = set()
    for asin, product in products.items():
        constraints = tuple(dict.fromkeys([
            *(str(v) for v in intent_card(product).get("hard_constraints", [])),
            *(str(v) for v in intent_card(product).get("soft_preferences", [])),
        ]))
        signatures[asin] = {attribute: (constraints if attribute == "other" else tuple(
            value for value in constraints if classify_constraint(value) == attribute
        )) for attribute in ATTRIBUTES}
        all_values.update(constraints)
    phrase_to_id = {value: index + 1 for index, value in enumerate(sorted(all_values))}
    integer_values = {
        asin: {attribute: tuple(phrase_to_id[value] for value in values)
               for attribute, values in by_attribute.items()}
        for asin, by_attribute in signatures.items()
    }
    current = load_current_module()
    variants = [
        ("current_string_signature", lambda: current_agent(current, base, signatures)),
        ("integer_signature", lambda: integer_agent(base, integer_values, phrase_to_id)),
        ("fixed_v1", lambda: fixed_agent(base)),
    ]
    output = {"scope": "Official200 session loop after one shared catalogue index load", "repetitions": REPETITIONS, "variants": {}}
    for name, factory in variants:
        records = []
        for rep in range(1, REPETITIONS + 1):
            agent = factory()
            started = time.perf_counter()
            result = evaluate(agent, rows, ids, categories, products)
            elapsed = time.perf_counter() - started
            record = {"seconds": elapsed, "milliseconds_per_session": elapsed * 1000 / len(rows), **compact(result)}
            records.append(record)
            print(json.dumps({"variant": name, "rep": rep, **record}, separators=(",", ":")), flush=True)
        output["variants"][name] = {
            "runs": records,
            "mean_seconds": sum(row["seconds"] for row in records) / len(records),
            "mean_ms_per_session": sum(row["milliseconds_per_session"] for row in records) / len(records),
        }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
