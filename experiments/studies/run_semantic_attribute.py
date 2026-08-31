"""Evaluate semantic-attribute robustness without changing evaluator message formats.

This runner is deliberately separate from the submission entry point.  It materialises
the V2 cards into the released evaluator's templates, then evaluates a frozen Agent or
an analysis-only semantic fallback.  The fallback is guarded at the phrase level: it may
translate a phrase only when that complete normalised phrase has zero catalogue matches.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# Keep the V2 evaluation deterministic and cost-free. These settings are read when the
# optional components are constructed by the submitted Agent.
os.environ.setdefault("LLM_RERANK", "0")
os.environ.setdefault("LLM_EXTRACT", "0")

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    TOP_K,
    behavior_for,
    catalog_index,
    classify_constraint,
    coarse_category,
    evaluate,
    load_jsonl,
    metric_summary,
    normalize_recommendations,
)
from submission.agent import Agent, raw_toks  # noqa: E402
from experiments.studies.semantic_grounding import SemanticFeatureGrounder  # noqa: E402
from experiments.studies.route_node import RouteAndSpanV2Agent, RouteOnlyV2Agent  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def normalise(text: str) -> str:
    """The same token normalisation used by the FTS5-backed submitted agent."""
    return " ".join(raw_toks(text))


def values(sample: dict, value_key: str = "paraphrase") -> tuple[list[str], list[str]]:
    card = sample["semantic_card"]
    hard = [str(atom[value_key]) for atom in card.get("hard_constraints", [])]
    soft = [str(atom[value_key]) for atom in card.get("soft_preferences", [])]
    return hard, soft


def v2_initial_message(sample: dict, category: str, disclosed: set[str], value_key: str) -> str:
    hard, soft = values(sample, value_key)
    if sample["scenario_type"] == "buying" and hard:
        disclosed.add(hard[0])
        return f"I'm looking for {category}. A key requirement is: {hard[0]}."
    if sample["scenario_type"] == "intent_override":
        old_value = soft[-1] if soft else "I prefer a different style."
        return f"I'm looking for {category}. {old_value}"
    return f"I'm looking for {category}, but I'm still exploring."


def v2_customer_reply(sample: dict, ask_attribute: object, disclosed: set[str],
                      boundary_used: bool, value_key: str) -> tuple[str, bool]:
    """Released reply wrappers, with only semantic attribute values replaced.

    Attribute routing intentionally uses the released evaluator classifier on the
    paraphrased value. This preserves the dialogue consequence of changing an attribute's
    surface form instead of using the synthetic card's hidden attribute label.
    """
    attribute = ask_attribute if isinstance(ask_attribute, str) else None
    if sample["scenario_type"] == "boundary" and not boundary_used and attribute:
        return f"I don't have a preference for {attribute}; please use your judgment.", True
    if not attribute:
        return "Those options are not quite right yet. Ask me about one specific attribute.", boundary_used
    hard, soft = values(sample, value_key)
    constraints = [*hard, *soft]
    matches = [
        value for value in constraints
        if value not in disclosed and (attribute == "other" or classify_constraint(value) == attribute)
    ][:2]
    if not matches:
        return f"I don't have an additional preference for {attribute}.", boundary_used
    disclosed.update(matches)
    return "For that, what matters is: " + "; ".join(matches) + ".", boundary_used


def v2_behavior(sample: dict, value_key: str) -> dict:
    hard, soft = values(sample, value_key)
    # `behavior_for` chooses the released deterministic override turn. Replace its two
    # values only; the message wrapper and timing are otherwise identical.
    seed = f"{sample.get('sample_id', '')}\0{sample.get('scenario_type', '')}"
    behavior = behavior_for(str(sample["scenario_type"]), {
        "hard_constraints": hard,
        "soft_preferences": soft,
    }, random.Random(seed))
    return behavior


class ExactAbsentLexiconAgent(Agent):
    """Analysis candidate: translate known semantic phrases only after an exact miss.

    This is intentionally a transparent development-only candidate, not a proposed final
    model. It proves the control-flow contract needed by later semantic models: literal
    catalogue evidence wins whenever present, and semantic translation is unreachable on
    that path.
    """

    def __init__(self, catalog_path: str | Path, lexicon: dict[str, str]) -> None:
        super().__init__(catalog_path)
        self.lexicon = lexicon
        self.semantic_gate = defaultdict(int)

    def _resolve(self, text: str, cap: int | None = None) -> list[str]:
        phrase = normalise(text)
        if not phrase:
            return []
        if self.ix.df(phrase) > 0:
            self.semantic_gate["literal_present"] += 1
            return super()._resolve(text, cap)
        self.semantic_gate["literal_absent"] += 1
        canonical = self.lexicon.get(phrase)
        if canonical is None:
            self.semantic_gate["unmapped_absent"] += 1
            return super()._resolve(text, cap)
        self.semantic_gate["semantic_triggered"] += 1
        return super()._resolve(canonical, cap)


class SemanticFeatureAgent(Agent):
    """V2-only semantic grounding node, guarded after complete literal absence."""

    def __init__(self, catalog_path: str | Path) -> None:
        super().__init__(catalog_path)
        self.grounder = SemanticFeatureGrounder(Path(catalog_path), self.ix.df)

    def _resolve(self, text: str, cap: int | None = None) -> list[str]:
        phrase = normalise(text)
        # The existing resolver may safely recover an attested substring even when the
        # complete surface phrase is synthetic (for example, a generated colour label).
        # That lexical result is stronger provenance than a semantic neighbour, so it is
        # an independent hard gate for this node.
        lexical = super()._resolve(text, cap)
        if lexical:
            return lexical
        if not phrase:
            return lexical
        grounded = self.grounder.resolve(text)
        if grounded is None:
            return lexical
        return super()._resolve(grounded, cap)


def development_lexicon(rows: Iterable[dict]) -> dict[str, str]:
    """Build a resolver from development data only, rejecting inconsistent mappings."""
    candidates: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for group in row["semantic_card"].values():
            for atom in group:
                candidates[normalise(str(atom["paraphrase"]))].add(normalise(str(atom["canonical"])))
    return {phrase: next(iter(mapped)) for phrase, mapped in candidates.items() if len(mapped) == 1}


def compact(result: dict) -> dict:
    keys = ("sample_count", "hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score")
    return {key: result[key] for key in keys}


def evaluate_v2(agent: Agent, samples: list[dict], catalog_ids: set[str],
                categories: dict[str, list[str]], value_key: str = "paraphrase") -> dict:
    sessions: list[dict] = []
    for sample in samples:
        session_id = f"semantic_v2_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = v2_initial_message(sample, str(sample["category"]), disclosed, value_key)
        behavior = v2_behavior(sample, value_key)
        hit_turn: int | None = None
        best_rank: int | None = None
        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            if not isinstance(response, dict):
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break
            override = behavior.get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                user_message, boundary_used = v2_customer_reply(
                    sample, response.get("ask_attribute"), disclosed, boundary_used, value_key
                )
        sessions.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })
    overall = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    score = 0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[session["scenario_type"]].append(session)
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(score, 6),
        "scenario_metrics": {name: metric_summary(grouped[name]) for name in sorted(grouped)},
        "sessions": sessions,
    }


def make_agent(candidate: str, catalog: Path, development_rows: list[dict]) -> Agent:
    if candidate == "literal":
        return Agent(catalog)
    if candidate == "route-only":
        return RouteOnlyV2Agent(catalog)
    if candidate == "route-span":
        return RouteAndSpanV2Agent(catalog)
    if candidate == "development-lexicon":
        return ExactAbsentLexiconAgent(catalog, development_lexicon(development_rows))
    if candidate == "semantic-feature":
        return SemanticFeatureAgent(catalog)
    raise ValueError(f"unknown candidate: {candidate}")


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 semantic-attribute robustness runner")
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "experiments" / "studies" / "sets" / "semantic_attribute_development_200.jsonl")
    parser.add_argument("--candidate", choices=("literal", "route-only", "route-span", "development-lexicon", "semantic-feature"), default="literal")
    parser.add_argument("--value-mode", choices=("paraphrase", "canonical"), default="paraphrase",
                        help="Use semantic rewrites or replay canonical values on the same rows.")
    parser.add_argument("--public-control", action="store_true",
                        help="also score the candidate on official public data to verify the semantic gate")
    parser.add_argument("--unseen-control", action="store_true",
                        help="also score the candidate on the frozen same-population Unseen800 control")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    development_rows = load_jsonl(ROOT / "experiments" / "studies" / "sets" / "semantic_attribute_development_200.jsonl")
    rows = load_jsonl(args.dataset)
    catalog = ROOT / "data" / "catalog.jsonl"
    ids, categories, products = catalog_index(catalog)
    agent = make_agent(args.candidate, catalog, development_rows)
    result = {
        "candidate": args.candidate,
        "dataset": str(args.dataset),
        "value_mode": args.value_mode,
        "semantic_shift": compact(evaluate_v2(agent, rows, ids, categories, args.value_mode)),
    }
    if isinstance(agent, ExactAbsentLexiconAgent):
        result["semantic_gate_after_shift"] = dict(agent.semantic_gate)
    if isinstance(agent, SemanticFeatureAgent):
        result["semantic_grounding_after_shift"] = dict(agent.grounder.calls)
    if isinstance(agent, RouteOnlyV2Agent):
        result["route_node_after_shift"] = agent.route_node.stats()
    if args.public_control:
        public_agent = make_agent(args.candidate, catalog, development_rows)
        public = evaluate(public_agent, load_jsonl(ROOT / "data" / "public_set.jsonl"), ids, categories, products)
        result["official200"] = compact(public)
        if isinstance(public_agent, ExactAbsentLexiconAgent):
            result["semantic_gate_after_public"] = dict(public_agent.semantic_gate)
        if isinstance(public_agent, SemanticFeatureAgent):
            result["semantic_grounding_after_public"] = dict(public_agent.grounder.calls)
        if isinstance(public_agent, RouteOnlyV2Agent):
            result["route_node_after_public"] = public_agent.route_node.stats()
    if args.unseen_control:
        unseen_path = ROOT / "robustness" / "optuna_v2_sets" / "population_shift_01_800.jsonl"
        unseen_agent = make_agent(args.candidate, catalog, development_rows)
        unseen = evaluate(unseen_agent, load_jsonl(unseen_path), ids, categories, products)
        result["unseen800"] = compact(unseen)
        if isinstance(unseen_agent, RouteOnlyV2Agent):
            result["route_node_after_unseen800"] = unseen_agent.route_node.stats()
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
