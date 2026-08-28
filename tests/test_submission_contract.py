from __future__ import annotations

import unittest
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from submission.agent import Agent


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "catalog.jsonl"
PUBLIC = ROOT / "data" / "public_set.jsonl"
ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand", "budget",
    "feature", "use_case", "other", None,
}


class _ContractAuditor:
    def __init__(self, agent: Agent, catalog_ids: set[str]) -> None:
        self.agent = agent
        self.catalog_ids = catalog_ids
        self.responses = 0

    def reset(self, session_id: str, user_profile: dict) -> None:
        result = self.agent.reset(session_id, user_profile)
        if result is not None:
            raise AssertionError("reset must return None")

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        response = self.agent.respond(session_id, user_message, turn, top_k)
        self.responses += 1

        if set(response) != {"message", "ask_attribute", "recommendations", "usage"}:
            raise AssertionError(f"unexpected response fields: {set(response)}")
        if not isinstance(response["message"], str):
            raise AssertionError("message must be a string")
        if response["ask_attribute"] not in ALLOWED_ATTRIBUTES:
            raise AssertionError(f"invalid ask_attribute: {response['ask_attribute']!r}")

        recommendations = response["recommendations"]
        if not isinstance(recommendations, list) or len(recommendations) > 100:
            raise AssertionError("recommendations must be an array with at most 100 items")
        asins: list[str] = []
        for item in recommendations:
            if not isinstance(item, dict) or set(item) != {"parent_asin"}:
                raise AssertionError(f"invalid recommendation item: {item!r}")
            asin = item["parent_asin"]
            if not isinstance(asin, str) or not asin or asin not in self.catalog_ids:
                raise AssertionError(f"invalid catalogue identifier: {asin!r}")
            asins.append(asin)
        if len(asins) != len(set(asins)):
            raise AssertionError("recommendations must be unique")

        usage = response["usage"]
        if not isinstance(usage, dict) or set(usage) != {
            "prompt_tokens", "completion_tokens"
        }:
            raise AssertionError(f"invalid usage object: {usage!r}")
        for key in ("prompt_tokens", "completion_tokens"):
            if not isinstance(usage[key], int) or usage[key] < 0:
                raise AssertionError(f"invalid {key}: {usage[key]!r}")
        return response


class SubmissionContractTest(unittest.TestCase):
    @unittest.skipUnless(CATALOG.exists(), "download data/catalog.jsonl to run contract audit")
    def test_every_public_response_matches_the_contract(self) -> None:
        samples = load_jsonl(PUBLIC)
        catalog_ids, categories, products = catalog_index(CATALOG)
        auditor = _ContractAuditor(Agent(CATALOG), catalog_ids)
        result = evaluate(auditor, samples, catalog_ids, categories, products)
        self.assertGreater(auditor.responses, len(samples))
        self.assertEqual(result["recommended_technical_score"], 0.9696)
        self.assertEqual(result["reported_token_usage"]["total_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
