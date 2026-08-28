from __future__ import annotations

import unittest

from submission.agent import Agent, CAT, CONSTRAINT


class _Index:
    def df(self, phrase: str) -> int:
        return 1


class OverrideOpeningEvidenceTest(unittest.TestCase):
    def test_released_override_opening_keeps_target_old_value(self) -> None:
        agent = object.__new__(Agent)
        agent.ix = _Index()
        extracted = agent._extract_templated(
            "I'm looking for Accessories Belts. Buckle closure"
        )
        self.assertIn(("Accessories Belts", CAT), extracted)
        self.assertIn(("Buckle closure", CONSTRAINT), extracted)

    def test_buying_opening_is_not_double_parsed(self) -> None:
        agent = object.__new__(Agent)
        agent.ix = _Index()
        extracted = agent._extract_templated(
            "I'm looking for Novelty Women. A key requirement is: cotton."
        )
        self.assertEqual(extracted.count(("cotton", CONSTRAINT)), 1)


if __name__ == "__main__":
    unittest.main()
