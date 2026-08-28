"""The optional LLM extraction layer must be unable to hurt the agent.

These assert the two properties the layer is sold on, so a future edit that breaks either
one fails the suite rather than the private run:

  1. THE RECOGNITION GATE. Every message shape the simulator can emit is recognised, so the
     LLM is unreachable at zero paraphrase and the clean score cannot move.
  2. TOTALITY OF `extract()`. No network state, credential state, or response body causes
     it to raise or to return anything but verbatim spans -- and it gives up quickly when
     the endpoint is sick, because a private run makes ~1,500 calls and an unbounded retry
     path would never finish.
"""
from __future__ import annotations

import io
import json
import socket
import sys
import unittest
from pathlib import Path
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import submission.llm_extract as LE
from submission.agent import recognised


def _resp(payload):
    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return R(raw)


def _ok(content):
    return _resp({"choices": [{"message": {"content": content}}],
                  "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


class RecognitionGate(unittest.TestCase):
    """Anything the simulator can say must be recognised; paraphrases must not be."""

    CLEAN = [
        "I'm looking for Novelty Women. A key requirement is: cotton.",
        "I'm looking for Boys Pants, but I'm still exploring.",
        "I'm looking for Jewelry Necklaces. I prefer a different style.",
        "For that, what matters is: Imported; Pull On closure.",
        "I don't have an additional preference for color.",
        "I don't have a preference for size; please use your judgment.",
        "Those options are not quite right yet. Ask me about one specific attribute.",
        "Actually, ignore my earlier preference. What I need is: cotton.",
        "Actually, please ignore my earlier preference.",
    ]
    PARAPHRASED = [
        "I want to find Novelty Women. It absolutely has to be cotton.",
        "I'm browsing around for Boys Pants at the moment, nothing fixed yet.",
        "Sure -- the thing that counts for me is Imported; Pull On closure.",
        "No strong feelings about color, honestly.",
        "Hmm, scratch all that. What I actually need is cotton.",
        "Novelty Women; cotton",
        "i m looking for novelty women  a key requirement is  cotton ",
    ]

    def test_every_clean_shape_is_recognised(self):
        for message in self.CLEAN:
            self.assertTrue(recognised(message), f"clean message leaked to LLM: {message!r}")

    def test_paraphrases_are_not_recognised(self):
        for message in self.PARAPHRASED:
            self.assertFalse(recognised(message), f"paraphrase wrongly recognised: {message!r}")


class ExtractorSafety(unittest.TestCase):
    def setUp(self):
        self._urlopen = LE.urlopen
        self._env = dict(LE.os.environ)
        LE.os.environ["LLM_EXTRACT"] = "1"
        LE.os.environ["GROQ_API_KEY"] = "test-key-not-real"
        self.cache = Path(__file__).with_name(".test_extract_cache.json")

    def tearDown(self):
        LE.urlopen = self._urlopen
        LE.os.environ.clear()
        LE.os.environ.update(self._env)
        self.cache.unlink(missing_ok=True)

    def _extractor(self):
        ex = LE.LLMExtractor(cache_path=self.cache)
        ex.cache = {}
        ex.limiter.rpm, ex.limiter.tpm = 10 ** 6, 10 ** 9   # test the breaker, not the throttle
        return ex

    def test_disabled_without_flag(self):
        LE.os.environ["LLM_EXTRACT"] = "0"
        self.assertFalse(self._extractor().enabled)

    def test_disabled_without_key(self):
        LE.os.environ.pop("GROQ_API_KEY")
        self.assertFalse(self._extractor().enabled)

    def test_never_raises_on_any_failure(self):
        def http(code):
            def f(*a, **k):
                raise HTTPError("https://x", code, "e", {}, None)
            return f

        failures = {
            "network": lambda *a, **k: (_ for _ in ()).throw(URLError("down")),
            "refused": lambda *a, **k: (_ for _ in ()).throw(ConnectionRefusedError()),
            "timeout": lambda *a, **k: (_ for _ in ()).throw(socket.timeout("timed out")),
            "http_401": http(401), "http_500": http(500), "http_429": http(429),
            "not_json": lambda *a, **k: _resp(b"<html>502</html>"),
            "bad_schema": lambda *a, **k: _resp({"nope": 1}),
        }
        for name, fake in failures.items():
            with self.subTest(mode=name):
                LE.urlopen = fake
                ex = self._extractor()
                self.assertIsNone(ex.extract("some paraphrased text about cotton"),
                                  f"{name} should yield None")

    def test_empty_completion_is_a_result_not_a_failure(self):
        """`None` and `[]` mean different things and the breaker depends on the difference.

        None  = the call did not work -> counts toward the consecutive-failure breaker.
        []    = the call worked and the message genuinely carries no requirement (common:
                "no strong feelings about colour") -> counts toward the ZERO-YIELD breaker.
        Collapsing them would either trip the failure breaker on healthy traffic or let a
        permanently-useless endpoint run for the whole session budget.
        """
        LE.urlopen = lambda *a, **k: _ok("")
        ex = self._extractor()
        result = ex.extract("some paraphrased text about cotton")
        self.assertEqual(result, [])
        self.assertEqual(ex.failures, 0, "an empty completion is not a transport failure")

    def test_persistent_zero_yield_trips_breaker(self):
        LE.urlopen = lambda *a, **k: _ok("")
        ex = self._extractor()
        ex.ZERO_YIELD_TRIP = 5
        for i in range(ex.ZERO_YIELD_TRIP + 3):
            ex.extract(f"paraphrased message number {i}")
        self.assertTrue(ex.circuit_open,
                        "a healthy endpoint returning nothing useful must still be given up on")

    def test_terminal_status_trips_breaker_immediately(self):
        def f(*a, **k):
            raise HTTPError("https://x", 401, "unauthorised", {}, None)
        LE.urlopen = f
        ex = self._extractor()
        ex.extract("a paraphrased message")
        self.assertTrue(ex.circuit_open)
        self.assertEqual(ex.calls, 1, "an invalid key must not be retried across messages")
        ex.extract("another message")
        self.assertEqual(ex.calls, 1, "no further calls once the breaker is open")

    def test_repeated_network_failure_trips_breaker(self):
        LE.urlopen = lambda *a, **k: (_ for _ in ()).throw(URLError("down"))
        ex = self._extractor()
        for i in range(ex.TRIP_AFTER + 5):
            ex.extract(f"paraphrased message number {i}")
        self.assertTrue(ex.circuit_open)
        self.assertLessEqual(ex.calls, ex.TRIP_AFTER,
                             "breaker must stop the run well before every message is tried")

    def test_hallucinated_spans_are_discarded(self):
        # A healthy endpoint returning real catalogue vocabulary the customer never said.
        LE.urlopen = lambda *a, **k: _ok("100% Cotton\nMachine Wash\nImported")
        ex = self._extractor()
        self.assertEqual(ex.extract("I want something soft for the winter"), [],
                         "spans absent from the message must never reach the ledger")

    def test_verbatim_spans_survive(self):
        LE.urlopen = lambda *a, **k: _ok("Jewelry Necklaces\nMaterial:alloy")
        ex = self._extractor()
        spans = ex.extract("Appreciate it. I want to find Jewelry Necklaces. "
                           "It absolutely has to be Material:alloy. Cheers.")
        self.assertEqual(spans, ["Jewelry Necklaces", "Material:alloy"])

    def test_only_validated_responses_are_cached(self):
        LE.urlopen = lambda *a, **k: (_ for _ in ()).throw(URLError("down"))
        ex = self._extractor()
        ex.extract("a paraphrased message")
        self.assertEqual(ex.cache, {}, "a failed call must not poison the cache")


if __name__ == "__main__":
    unittest.main()
