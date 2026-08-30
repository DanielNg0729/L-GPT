from __future__ import annotations

import os
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
    def setUp(self) -> None:
        # Pin the deterministic configuration. These layers are enabled by default and are
        # inert without credentials, but this test is about the offline guarantee, so it
        # states the configuration it is measuring instead of inheriting one.
        self._env = {k: os.environ.get(k)
                     for k in ("LLM_RESOLVE", "LLM_EXTRACT", "LLM_RERANK", "BERT_EXTRACT",
                               "V2_ROUTE")}
        for k in ("LLM_RESOLVE", "LLM_EXTRACT", "LLM_RERANK"):
            os.environ[k] = "0"

    def tearDown(self) -> None:
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    @unittest.skipUnless(CATALOG.exists(), "download data/catalog.jsonl to run contract audit")
    def test_every_public_response_matches_the_contract(self) -> None:
        samples = load_jsonl(PUBLIC)
        catalog_ids, categories, products = catalog_index(CATALOG)
        auditor = _ContractAuditor(Agent(CATALOG), catalog_ids)
        result = evaluate(auditor, samples, catalog_ids, categories, products)
        self.assertGreater(auditor.responses, len(samples))
        # A FLOOR, not an equality. Pinning the exact score makes every legitimate
        # improvement look like a failure -- this assertion already broke once when the
        # information-gain probe policy raised the score from 0.96960 to 0.97010. A floor
        # still catches what this test is for: a regression, or a silent break in the
        # contract that shows up as a collapsed score. Raise the floor deliberately when
        # a gain is confirmed, rather than editing it to match whatever today's number is.
        self.assertGreaterEqual(result["recommended_technical_score"], 0.9696)
        # THE DETERMINISTIC PATH MUST STAY OFFLINE, and this test pins that configuration
        # explicitly rather than depending on ambient defaults.
        #
        # The optional model-backed layers ship ENABLED, but each additionally requires a
        # credential the evaluator will not have, so in any normal environment they are
        # inert. On a developer machine that does hold a key, the deparaphraser can fire
        # once on Official200 -- `intent_card()` truncates one long feature bullet mid-word
        # and the result is not catalogue-attested -- and that single call would spend
        # tokens. Reading that as a contract violation would be wrong: it is the layer
        # working as designed.
        #
        # What this assertion is FOR is catching a layer that escapes its gate and runs on
        # traffic it was never meant to see. That question is only meaningful about the
        # deterministic configuration, so the configuration is set here rather than assumed.
        self.assertEqual(result["reported_token_usage"]["total_tokens"], 0)

    def test_starter_entry_point_is_the_shipped_agent(self) -> None:
        """`starter/agent.py` must re-export the implementation, never re-implement it.

        The harness loads `from starter.agent import Agent`. That file used to be a byte
        copy of `submission/agent.py` maintained by hand, alongside copies of the component
        modules -- and an audit found the copies had already drifted, with a stale default
        sitting in the scored entry point. Asserting object identity makes that class of
        divergence impossible rather than merely unlikely.
        """
        import starter.agent as entry
        import submission.agent as impl
        self.assertIs(entry.Agent, impl.Agent)


if __name__ == "__main__":
    unittest.main()
