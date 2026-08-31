"""Optional LLM phrasing for the customer-facing `message`. DEFAULT OFF. Demo only.

WHAT THIS TOUCHES, AND WHY IT IS THE SAFEST PLACE IN THE SYSTEM FOR A MODEL.
The contract returns three things. `recommendations` decides HitRate and MRR.
`ask_attribute` decides what the simulator discloses next, and therefore MTTC. `message` is
read by nobody: `local_evaluator.customer_reply` takes `ask_attribute` and the intent card
and never inspects the text. So this layer cannot change the score by any route -- not
ranking, not disclosure, not turn count. It exists so a demo reads like a shopping
assistant instead of a template.

That is also why it is default OFF. A layer that cannot help the score should not be in the
scored path at all; it is presentation, and it is opt-in.

THE ONE HARD REQUIREMENT. Whatever the model writes must still ASK about the attribute the
policy chose. The prompt names the word that must appear, and the reply is then CHECKED for
it rather than trusted:

    required word absent  ->  discard, use the deterministic sentence

A model that drops the word has changed the question, and the question is the only part of
this response that does any work. Substring matching is deliberate -- "colour" inside "any
colour preference?" passes -- because the test is that the word survived, not that the
sentence matched a template.

`other` IS EXEMPT. Requiring the literal identifier bent the output around it -- "Is there
anything else you would like to add, OTHER than this item?" -- because `other` names no
property. It is the open "anything else" turn, so there is nothing whose absence would make
the sentence wrong. `use_case` is checked as "use" for the same reason.

EVERY FAILURE PATH RETURNS THE DETERMINISTIC SENTENCE -- disabled, no credential, network
error, timeout, empty completion, missing attribute word, over-long output, exception. The
caller cannot tell the difference except by the text being blander.

    LLM_MESSAGE=1   enable (also needs GROQ_API_KEY)
    LLM_MESSAGE=0   default
"""
from __future__ import annotations

import json
import os
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

try:
    from submission.llm_rerank import ENDPOINT, _load_project_env
except Exception:                                   # pragma: no cover - standalone import
    from llm_rerank import ENDPOINT, _load_project_env  # type: ignore

_load_project_env()

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

# (word to ASK for, substring to CHECK). The two differ only where a stem accepts more than
# one spelling: asking for "colo" produced the literal "Which colo would you prefer?", so
# the prompt names a real word and the check stays permissive.
#
# `use_case` is asked as "use" because the identifier is not English. `other` is absent
# deliberately -- it names no property, it is the open "anything else" turn, so there is no
# word whose absence would make the sentence wrong and requiring one only bent the phrasing.
_REQUIRED_WORD = {
    "feature": ("feature", "feature"),
    "material": ("material", "material"),
    "color": ("colour", "colo"),
    "style": ("style", "style"),
    "size": ("size", "size"),
    "use_case": ("use", "use"),
}

SYSTEM = (
    "You write one short line for a shopping assistant that has just shown a customer "
    "some products and now needs to ask a follow-up question.\n"
    "Rules:\n"
    "1. Write ONE sentence, at most 25 words, warm and plain. No greeting, no emoji, no "
    "list, no quotes around it.\n"
    "2. If you are told the sentence must contain a word, it must contain that word "
    "exactly. The sentence is discarded otherwise.\n"
    "3. Ask only what you are asked to ask. Do not invent product details, prices or "
    "names.\n"
    "4. If you were told only one product is shown, say so honestly rather than implying "
    "a list."
)


def _num(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


class LLMMessageWriter:
    """Rephrases the follow-up question. Never raises, never blocks past its budget."""

    MAX_TOKENS = int(_num("LLM_MESSAGE_MAX_TOKENS", 160))
    TIMEOUT = _num("LLM_MESSAGE_TIMEOUT", 10.0)
    TIME_BUDGET = _num("LLM_MESSAGE_TIME_BUDGET", 120.0)
    TRIP_AFTER = int(_num("LLM_MESSAGE_TRIP_AFTER", 3))
    MAX_WORDS = int(_num("LLM_MESSAGE_MAX_WORDS", 40))
    TERMINAL_STATUS = frozenset({400, 401, 403, 404, 413, 422})

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        flag = os.environ.get("LLM_MESSAGE", "0").strip().lower()
        self._flag = flag in {"1", "true", "yes", "on"}
        self.model = model
        self.calls = self.accepted = self.rejected_missing_word = 0
        self.failures = self.too_long = 0
        self._consecutive = 0
        self._spent = 0.0
        self._open = None

    @property
    def enabled(self) -> bool:
        return self._flag and bool(os.environ.get("GROQ_API_KEY", "").strip())

    def _call(self, prompt: str) -> str | None:
        if self._open is not None or self._spent >= self.TIME_BUDGET:
            return None
        payload = json.dumps({
            "model": self.model, "temperature": 0.7,
            "max_tokens": self.MAX_TOKENS, "reasoning_effort": "low",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}],
        }).encode("utf-8")
        request = Request(ENDPOINT, data=payload, method="POST", headers={
            "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0"})
        started = time.time()
        try:
            with urlopen(request, timeout=self.TIMEOUT) as response:
                body = json.loads(response.read().decode("utf-8"))
            self._spent += time.time() - started
            self._consecutive = 0
            return body["choices"][0]["message"]["content"]
        except HTTPError as exc:
            self._spent += time.time() - started
            self.failures += 1
            if exc.code in self.TERMINAL_STATUS:
                self._open = f"HTTP {exc.code}"
            else:
                self._consecutive += 1
        except Exception:
            self._spent += time.time() - started
            self.failures += 1
            self._consecutive += 1
        if self._consecutive >= self.TRIP_AFTER:
            self._open = f"{self._consecutive} consecutive failures"
        return None

    def write(self, attribute: str, fallback: str, *, narrow: bool,
              shown: int = 0) -> str:
        """Return a rephrased question, or `fallback` if anything at all is off."""
        if not self.enabled or not attribute:
            return fallback
        pair = _REQUIRED_WORD.get(attribute)
        ask_word, required = pair if pair else (None, None)
        if required is None:
            task = ("Ask an open-ended question about whether there is anything else "
                    "they need. Do not name a specific property.")
        else:
            task = (f'Ask about the {attribute}. The sentence must contain the word '
                    f'"{ask_word}".')
        prompt = (f'Products shown: {"one" if narrow else shown or "several"}\n{task}')
        self.calls += 1
        raw = self._call(prompt)
        if not raw or not raw.strip():
            return fallback
        text = " ".join(raw.split()).strip().strip('"').strip()

        # THE CHECK THE PROMPT ASKED FOR, ENFORCED RATHER THAN TRUSTED. A sentence that
        # dropped the attribute is asking a different question from the one the policy
        # chose, and the question is the only load-bearing part of this response.
        if required is not None and required not in text.lower():
            self.rejected_missing_word += 1
            return fallback
        if len(text.split()) > self.MAX_WORDS:
            self.too_long += 1
            return fallback
        self.accepted += 1
        return text

    def stats(self) -> dict:
        return {"enabled": self.enabled, "model": self.model, "calls": self.calls,
                "accepted": self.accepted,
                "rejected_missing_word": self.rejected_missing_word,
                "too_long": self.too_long, "failures": self.failures,
                "seconds": round(self._spent, 2), "circuit_reason": self._open}
