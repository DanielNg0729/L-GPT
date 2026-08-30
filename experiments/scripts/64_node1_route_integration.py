"""V2.23: integrate only the trained Node 1 dialogue-act router behind V1's gate.

Scope is deliberately narrow.  The classifier identifies one of the six exact V1
message-level actions only after a message fails ``recognised()``.  It does not
extract category or constraint spans, resolve semantic attributes, change query
formation, alter ranking weights, or replace the lexical fallback.

The canonical Official200 and Unseen800 runs therefore test a strong invariant:
all traffic must remain on the literal V1 path, with no transformer model loaded
and no route inference performed.  This is an integration/non-interference test,
not evidence that Node 1 improves canonical ranking.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["LLM_EXTRACT"] = "0"
os.environ["LLM_RERANK"] = "0"

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent, recognised  # noqa: E402

OUT = ROOT / "experiments" / "results" / "out_64_node1_route_integration.json"
UNSEEN = ROOT / "robustness" / "optuna_v2_sets" / "population_shift_01_800.jsonl"
MODEL = ROOT / ".v2_model_cache" / "shared_sixway_phrase_augmented_cuda"
LABELS = (
    "buying_opening", "constraint_update", "no_evidence",
    "override_opening", "override_update", "plain_opening",
)
OPENING = frozenset({"buying_opening", "plain_opening", "override_opening"})


class Node1RouteOnlyAgent(Agent):
    """V1 plus a lazy, fail-closed dialogue-act route node for unknown wrappers.

    Its only non-literal effects are to preserve the existing no-evidence and
    override-reset controls when a semantically equivalent wrapper is unrecognised.
    Evidence extraction remains the inherited V1 parser/tagger/miner path.
    """

    def __init__(self, catalog_path: str | Path) -> None:
        super().__init__(catalog_path)
        self._route_model = None
        self._route_tokenizer = None
        self.route_model_loads = 0
        self.route_inferences = 0
        self.route_failures = 0
        self.route_actions: dict[str, int] = {}

    def _route_unknown(self, message: str, turn: int) -> str | None:
        """Return a turn-masked action, or None if this optional node is unavailable."""
        if recognised(message):
            return None
        try:
            if self._route_model is None:
                # Imported and loaded only after the strict literal gate has failed.
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                if not MODEL.is_dir():
                    return None
                self._route_tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
                self._route_model = AutoModelForSequenceClassification.from_pretrained(
                    MODEL, local_files_only=True
                )
                self._route_model.eval()
                self.route_model_loads += 1
            encoded = self._route_tokenizer(
                [message], padding=True, truncation=True, max_length=80, return_tensors="pt"
            )
            with torch.no_grad():
                logits = self._route_model(**encoded).logits[0]
            allowed = OPENING if turn == 1 else frozenset(LABELS).difference(OPENING)
            action = max(allowed, key=lambda label: float(logits[LABELS.index(label)]))
            self.route_inferences += 1
            self.route_actions[action] = self.route_actions.get(action, 0) + 1
            return action
        except Exception:
            self.route_failures += 1
            return None

    def _observe(self, state, message: str) -> None:
        action = self._route_unknown(message, state.turn)
        # These are the two V1 state transitions whose literal cues would otherwise
        # be absent under a paraphrased wrapper.  The other four route labels leave
        # extraction to the inherited V1 fallback unchanged.
        if action == "no_evidence":
            return
        if action == "override_update":
            state.rejected.clear()
        super()._observe(state, message)

    def route_diagnostics(self) -> dict:
        return {
            "model_loads": self.route_model_loads,
            "inferences": self.route_inferences,
            "failures": self.route_failures,
            "actions": self.route_actions,
        }


def shared_agent(cls: type[Agent], base: Agent) -> Agent:
    """Reuse the immutable index while isolating session and optional-model state."""
    candidate = object.__new__(cls)
    candidate.ix = base.ix
    candidate.sessions = {}
    candidate.llm = None
    candidate.llm_extract = None
    candidate.tagger = None
    candidate._route_model = None
    candidate._route_tokenizer = None
    candidate.route_model_loads = 0
    candidate.route_inferences = 0
    candidate.route_failures = 0
    candidate.route_actions = {}
    return candidate


def compact(result: dict) -> dict:
    return {
        "technical_score": result["recommended_technical_score"],
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "scenario_metrics": result["scenario_metrics"],
    }


def main() -> None:
    catalog = ROOT / "data" / "catalog.jsonl"
    ids, categories, products = catalog_index(catalog)
    base = Agent(catalog)
    datasets = {
        "Official200": load_jsonl(ROOT / "data" / "public_set.jsonl"),
        "Unseen800": load_jsonl(UNSEEN),
    }
    result: dict = {
        "experiment": "V2.23 Node 1 route-only V1 integration",
        "scope": "Strict-gated six-route classifier only. No semantic span, family, canonical retrieval, verifier, calibration, evidence-weight, retrieval, override-replacement, or question-policy change.",
        "model": str(MODEL),
        "datasets": {"Official200": "data/public_set.jsonl", "Unseen800": str(UNSEEN.relative_to(ROOT))},
        "variants": {},
        "acceptance_rule": "Technical score and every reported metric must be exactly identical; route model loads and inferences must both be zero on each canonical dataset.",
    }
    for label, rows in datasets.items():
        baseline = shared_agent(Agent, base)
        candidate = shared_agent(Node1RouteOnlyAgent, base)
        started = time.perf_counter()
        base_score = compact(evaluate(baseline, rows, ids, categories, products))
        candidate_score = compact(evaluate(candidate, rows, ids, categories, products))
        result["variants"][label] = {
            "baseline": base_score,
            "node1_route_only": candidate_score,
            "exact_metric_identity": base_score == candidate_score,
            "route_diagnostics": candidate.route_diagnostics(),
            "wall_seconds": round(time.perf_counter() - started, 3),
        }
    accepted = all(
        row["exact_metric_identity"]
        and row["route_diagnostics"]["model_loads"] == 0
        and row["route_diagnostics"]["inferences"] == 0
        for row in result["variants"].values()
    )
    result["decision"] = "Accepted as a non-interfering integration candidate." if accepted else "Rejected: canonical non-interference invariant failed."
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "variants": result["variants"]}, indent=2))


if __name__ == "__main__":
    main()
