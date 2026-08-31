from __future__ import annotations

import unittest

from submission.agent import Agent, CAT, CONSTRAINT, SessionState


class _Index:
    def df(self, phrase: str) -> int:
        return 1


class OverrideOpeningEvidenceTest(unittest.TestCase):
    def test_released_override_opening_keeps_target_old_value(self) -> None:
        agent = object.__new__(Agent)
        agent.ix = _Index()
        message = (
            "I'm looking for Accessories Belts. Buckle closure"
        )
        extracted = agent._extract_templated(message)
        self.assertIn(("Accessories Belts", CAT), extracted)
        self.assertNotIn(("Buckle closure", CONSTRAINT), extracted)
        state = SessionState()
        agent._recover_override_opening(state, message)
        self.assertIn("buckle closure", state.evidence)
        self.assertEqual(state.evidence["buckle closure"][1], CONSTRAINT)

    def test_buying_opening_is_not_double_parsed(self) -> None:
        agent = object.__new__(Agent)
        agent.ix = _Index()
        extracted = agent._extract_templated(
            "I'm looking for Novelty Women. A key requirement is: cotton."
        )
        self.assertEqual(extracted.count(("cotton", CONSTRAINT)), 1)


if __name__ == "__main__":
    unittest.main()
