"""Unit tests for the optional contradiction-demotion filter.

No test here touches the network: the transport (`_call`) is replaced. What is tested
is the part that must be right regardless of any model — the gating, the windowed
walk-and-refill, the demote-not-drop ordering, and every fail-open path.
"""
import json
import os
import unittest

from submission.llm_filter import LLMRelevanceFilter


DOCS = {f"A{i:02d}": f"product {i} description" for i in range(30)}
RANKED = [f"A{i:02d}" for i in range(30)]
REQS = ["women's running shoes", "black"]
CHAT = ["I'm looking for women's running shoes.", "For that, black matters."]


def enabled_filter() -> LLMRelevanceFilter:
    os.environ["LLM_FILTER"] = "1"
    os.environ["GROQ_API_KEY"] = "test-key-never-used"
    return LLMRelevanceFilter().bind(DOCS.get)


class TestGating(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.get(k) for k in ("LLM_FILTER", "GROQ_API_KEY")}

    def tearDown(self):
        for key, value in self._env.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value

    def test_off_by_default_even_with_key(self):
        os.environ.pop("LLM_FILTER", None)
        os.environ["GROQ_API_KEY"] = "present"
        flt = LLMRelevanceFilter().bind(DOCS.get)
        self.assertFalse(flt.enabled)
        self.assertEqual(flt.rearrange(RANKED, REQS, CHAT, need=10), RANKED)
        self.assertEqual(flt.calls, 0)

    def test_off_without_key(self):
        os.environ["LLM_FILTER"] = "1"
        os.environ.pop("GROQ_API_KEY", None)
        self.assertFalse(LLMRelevanceFilter().bind(DOCS.get).enabled)

    def test_needs_requirements_and_a_choice(self):
        flt = enabled_filter()
        self.assertFalse(flt.should_fire(0, 20))    # nothing to contradict
        self.assertFalse(flt.should_fire(3, 1))     # nothing to reorder
        self.assertTrue(flt.should_fire(3, 20))


class TestWindowWalk(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.get(k) for k in ("LLM_FILTER", "GROQ_API_KEY")}
        self.flt = enabled_filter()

    def tearDown(self):
        for key, value in self._env.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value

    def test_demote_and_refill(self):
        """First window loses two candidates; survivors of the second window refill."""
        replies = iter(['{"remove": [1, 3]}', '{"remove": []}'])
        self.flt._call = lambda prompt: next(replies)
        out = self.flt.rearrange(RANKED, REQS, CHAT, need=10)
        self.assertEqual(sorted(out), sorted(RANKED))           # nothing lost or invented
        self.assertEqual(out[-2:], ["A01", "A03"])              # flagged land at the tail
        self.assertEqual(out[:10], ["A00", "A02", "A04", "A05", "A06", "A07",
                                    "A08", "A09", "A10", "A11"])
        self.assertEqual(out[10:28], RANKED[12:])               # unjudged tail keeps order

    def test_stops_once_enough_survive(self):
        calls = []
        self.flt._call = lambda prompt: calls.append(prompt) or '{"remove": []}'
        self.flt.rearrange(RANKED, REQS, CHAT, need=1)
        self.assertEqual(len(calls), 1)             # width-1 turn: one window is enough

    def test_call_budget(self):
        self.flt._call = lambda prompt: '{"remove": [0,1,2,3,4,5,6,7,8]}'
        out = self.flt.rearrange(RANKED, REQS, CHAT, need=10)
        self.assertEqual(self.flt.calls, self.flt.MAX_CALLS_PER_TURN)
        self.assertEqual(sorted(out), sorted(RANKED))

    def test_failed_call_keeps_window(self):
        self.flt._call = lambda prompt: None
        self.assertEqual(self.flt.rearrange(RANKED, REQS, CHAT, need=10), RANKED)

    def test_flagging_everything_is_ignored(self):
        self.flt._call = lambda prompt: json.dumps({"remove": list(range(10))})
        self.assertEqual(self.flt.rearrange(RANKED, REQS, CHAT, need=10), RANKED)

    def test_malformed_replies_keep_window(self):
        for raw in ("not json", '{"remove": "A01"}', '[]', '{"other": []}',
                    '{"remove": [99, -1, "x"]}'):
            flt = enabled_filter()
            flt._call = lambda prompt, raw=raw: raw
            out = flt.rearrange(RANKED, REQS, CHAT, need=10)
            self.assertEqual(out[:10], RANKED[:10], raw)

    def test_prompt_carries_context(self):
        seen = {}
        self.flt._call = lambda prompt: seen.setdefault("p", prompt) and None or None
        self.flt.rearrange(RANKED, REQS, CHAT, need=10)
        prompt = seen["p"]
        self.assertIn("women's running shoes", prompt)          # requirements
        self.assertIn("For that, black matters.", prompt)       # shopper's own words
        self.assertIn("[0] product 0 description", prompt)      # numbered candidates
        self.assertIn("[9] product 9 description", prompt)


if __name__ == "__main__":
    unittest.main()
