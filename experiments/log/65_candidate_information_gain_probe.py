"""Experiment 65: candidate-conditioned expected-elimination clarification policy.

For each admissible ``ask_attribute``, reconstruct only the *visible-catalogue*
intent-card constraints for every candidate currently returned by V1 retrieval.  Group
candidates by the answer that question would produce and choose the question minimizing
the expected number of candidates that survive:

    E[remaining | question] = sum_r |G_r|^2 / |C|

where C is the current candidate pool and G_r is the group with response r.  This is
equivalent to maximizing expected eliminated candidates under a uniform posterior over C.
No target label, hidden private card, profile, or population prior is used to choose a
question.  It changes only ``_next_probe`` for this experiment.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import argparse
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index, classify_constraint, evaluate, intent_card, load_jsonl,
)
from submission.agent import Agent, DEAD_ATTRIBUTES  # noqa: E402

ATTRIBUTES = ("feature", "material", "color", "style", "size", "use_case", "other")
OUT = ROOT / "experiments" / "results" / "out_65_candidate_information_gain_probe.json"


def load_minter():
    path = ROOT / "experiments" / "log" / "30_robustness_benchmark.py"
    spec = importlib.util.spec_from_file_location("robustness30", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CandidateInformationGain(Agent):
    """Choose the question with the greatest expected candidate elimination."""

    signatures_by_asin: dict[str, dict[str, tuple[str, ...]]]
    prior_by_asin: dict[str, float]

    def _reply_signature(self, asin: str, attribute: str, disclosed: set[str]) -> tuple[str, ...]:
        # Every possible reply is pre-indexed from the frozen visible catalogue.  The
        # remaining dynamic operation is only exclusion of values already disclosed.
        return tuple(value for value in self.signatures_by_asin[asin][attribute]
                     if value not in disclosed)[:2]

    @staticmethod
    def _evidence_key(st) -> tuple[tuple[str, str], ...]:  # type: ignore[no-untyped-def]
        return tuple(sorted((phrase, tier) for phrase, (_, tier) in st.evidence.items()))

    def _candidates(self, st, message: str):  # type: ignore[no-untyped-def]
        """Reuse the pool built for this turn's question selection exactly once.

        Agent.respond calls _next_probe before its normal candidate-and-ranking call.
        Without this cache, an information-gain probe executes the identical FTS query
        twice.  The cache is keyed by immutable evidence and consumed immediately.
        """
        cached = self._ig_pool_cache.get(st.sid)
        if cached and cached[0] == self._evidence_key(st):
            del self._ig_pool_cache[st.sid]
            return list(cached[1])
        return super()._candidates(st, message)

    def _next_probe(self, st):  # type: ignore[no-untyped-def]
        options = [a for a in ATTRIBUTES if a not in st.asked and a not in DEAD_ATTRIBUTES]
        if not options:
            return "other"

        # The normal call to _candidates follows immediately in Agent.respond.  Calling it
        # here sees the identical evidence state and performs only visible-catalogue search.
        pool = super()._candidates(st, "")
        self._ig_pool_cache[st.sid] = (self._evidence_key(st), tuple(pool))
        if len(pool) < 2:
            return super()._next_probe(st)

        disclosed = {phrase for phrase, (_, tier) in st.evidence.items() if tier != "cat"}
        expected_remaining: dict[str, float] = {}
        for attribute in options:
            groups: dict[tuple[str, ...], tuple[int, float]] = {}
            for asin in pool:
                reply = self._reply_signature(asin, attribute, disclosed)
                count, mass = groups.get(reply, (0, 0.0))
                groups[reply] = (count + 1, mass + self.prior_by_asin[asin])
            total_mass = sum(mass for _, mass in groups.values())
            expected_remaining[attribute] = sum(count * mass for count, mass in groups.values()) / total_mass

        selected = min(options, key=lambda a: (expected_remaining[a], ATTRIBUTES.index(a)))
        stats = getattr(self, "information_gain_trace", None)
        if stats is not None:
            stats.append({
                "pool": len(pool), "selected": selected,
                "expected_remaining": expected_remaining[selected],
                "expected_eliminated": len(pool) - expected_remaining[selected],
            })
        return selected


def shared(base: Agent, signatures: dict[str, dict[str, tuple[str, ...]]],
           prior: dict[str, float]) -> CandidateInformationGain:
    agent = object.__new__(CandidateInformationGain)
    agent.ix, agent.sessions = base.ix, {}
    agent.llm, agent.llm_extract, agent.tagger = None, None, None
    agent.signatures_by_asin = signatures
    agent.prior_by_asin = prior
    agent.information_gain_trace = []
    agent._ig_pool_cache = {}
    return agent


def summary(result: dict) -> dict:
    return {key: result[key] for key in ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior", choices=("uniform", "reviews"), default="uniform")
    parser.add_argument("--all-populations", action="store_true")
    args = parser.parse_args()
    public = load_jsonl(ROOT / "data" / "public_set.jsonl")
    ids, categories, products = catalog_index(ROOT / "data" / "catalog.jsonl")
    signatures: dict[str, dict[str, tuple[str, ...]]] = {}
    for asin, product in products.items():
        constraints = tuple(dict.fromkeys([
            *(str(v) for v in intent_card(product).get("hard_constraints", [])),
            *(str(v) for v in intent_card(product).get("soft_preferences", [])),
        ]))
        signatures[asin] = {
            attribute: (constraints if attribute == "other" else tuple(
                value for value in constraints if classify_constraint(value) == attribute
            ))
            for attribute in ATTRIBUTES
        }
    base = Agent(ROOT / "data" / "catalog.jsonl")
    minter = load_minter()
    public_targets = {str(row["ground_truth"]["parent_asin"]) for row in public}
    profiles = [row["user_profile"] for row in public]
    suites = {"Official200": public, "Unseen800_review_weighted": minter.mint(
        products, public_targets, profiles, "reviews", 800, seed=minter.SEEDS["reviews"]
    )}
    if args.all_populations:
        suites["Unseen800_uniform"] = minter.mint(
            products, public_targets, profiles, "uniform", 800, seed=minter.SEEDS["uniform"]
        )
        suites["Unseen800_inverse"] = minter.mint(
            products, public_targets, profiles, "inverse", 800, seed=minter.SEEDS["inverse"]
        )
    prior = {asin: 1.0 for asin in products}
    if args.posterior == "reviews":
        prior = {asin: 1.0 + max(0.0, float(product.get("rating_number") or 0.0))
                 for asin, product in products.items()}
    output: dict[str, dict] = {
        "posterior": args.posterior,
        "formula": "E_remaining=sum_r P(r|q)*|G_r|; uniform prior reduces to sum_r |G_r|^2/|C|",
        "suites": {},
    }
    for name, rows in suites.items():
        # Share the loaded index across variants.  Canonical messages never invoke BERT.
        fixed_agent = object.__new__(Agent)
        fixed_agent.ix, fixed_agent.sessions = base.ix, {}
        fixed_agent.llm, fixed_agent.llm_extract, fixed_agent.tagger = None, None, None
        info_agent = shared(base, signatures, prior)
        fixed_result = evaluate(fixed_agent, rows, ids, categories, products)
        info_result = evaluate(info_agent, rows, ids, categories, products)
        trace = info_agent.information_gain_trace
        output["suites"][name] = {
            "fixed_v1": summary(fixed_result),
            "candidate_information_gain": summary(info_result),
            "delta_score": info_result["recommended_technical_score"] - fixed_result["recommended_technical_score"],
            "question_counts": dict(Counter(item["selected"] for item in trace)),
            "mean_pool_size": sum(item["pool"] for item in trace) / len(trace) if trace else 0.0,
            "mean_expected_eliminated": sum(item["expected_eliminated"] for item in trace) / len(trace) if trace else 0.0,
        }
        print(name, json.dumps(output["suites"][name], sort_keys=True), flush=True)

    out = OUT if args.posterior == "uniform" else ROOT / "experiments" / "results" / "out_66_review_weighted_information_gain_probe.json"
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
