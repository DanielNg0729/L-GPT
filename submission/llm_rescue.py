"""Transcript rescue: re-read the whole conversation when the turn-by-turn parse has stalled.

WHAT IT IS FOR. The deparaphraser works one unattested VALUE at a time, and the span node
and tagger work one MESSAGE at a time. None of them can use the fact that turn 6 clarifies
something the agent misread on turn 2. This layer is the only one that sees the conversation
as a whole, and it runs only when the per-turn machinery has visibly failed: several turns
in, several candidates already rejected, and nothing new arriving.

The design is taken from a teammate's L-GPT branch, which measured it at +0.0092 on a
held-out wrapper-paraphrase suite. Two things are changed on the way in:

  WEIGHT. Their integration admits recovered requirements at full CONSTRAINT strength. Ours
  admits them at SEM, the same attenuated tier as a deparaphrase proposal, for the reason
  measured on that layer: identical proposals recover 81.5% of a perfect resolver at full
  weight and roughly 96% attenuated. The difference is entirely what a WRONG proposal costs,
  and on their own wrapper suite the rescue accepted 43 requirements while 41 more were
  rejected as unattested -- so roughly half of what it proposes is already known to be wrong
  before weighting is considered.

  ATTESTATION. Every recovered phrase must exist in the frozen catalogue (`df > 0`) before it
  can become evidence. The model proposes; the catalogue disposes. A phrase the catalogue has
  never seen cannot be a customer requirement, because every constraint the simulator speaks
  is a verbatim substring of the target document.

WHAT IT CANNOT DO. It never ranks, never chooses the question, never emits a product id, and
never removes evidence the agent already holds -- it can only ADD attested phrases. Every
failure path returns None and the turn proceeds exactly as it would have.

WHERE IT CAN AND CANNOT FIRE. The gate needs a stalled session, which on clean traffic does
not occur: the agent converges at MTTC 2.2 and the recognition gate matches every message, so
this layer is unreachable on the scored suites by control flow rather than by threshold. It
exists for wording the benchmark does not contain.

    LLM_RESCUE=0        disable outright
    LLM_RESCUE_TURN     first turn it may fire (default 5)
    LLM_RESCUE_REJECTS  rejected candidates required before it may fire (default 4)
"""
from __future__ import annotations

import json
import os
import re
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from submission.llm_client import ENDPOINT, load_project_env

load_project_env()

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

SYSTEM = (
    "You recover shopping requirements from a conversation, for a search engine that "
    "matches them against product listing text.\n"
    "Rules:\n"
    "1. Copy the shopper's OWN WORDING for each requirement, word for word. The engine "
    "matches these as literal text, so paraphrasing them breaks the match.\n"
    "2. One requirement per distinct thing they asked for. Split combined sentences.\n"
    "3. Do not invent attributes they never mentioned, and do not add filler like 'good "
    "quality' or 'comfortable' unless they said it.\n"
    "4. Reply with ONLY a JSON object, no prose and no code fence:\n"
    '   {"category": "", "requirements": [], "material": null, "color": null}'
)


def _num(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


class LLMTranscriptRescue:
    """Once per session, late, on a stall. Never raises."""

    MAX_TOKENS = int(_num("LLM_RESCUE_MAX_TOKENS", 768))
    TIMEOUT = _num("LLM_RESCUE_TIMEOUT", 20.0)
    TIME_BUDGET = _num("LLM_RESCUE_TIME_BUDGET", 600.0)
    RETRIES = int(_num("LLM_RESCUE_RETRIES", 3))
    BACKOFF = _num("LLM_RESCUE_BACKOFF", 2.0)
    TRIP_AFTER = int(_num("LLM_RESCUE_TRIP_AFTER", 6))
    MIN_TURN = int(_num("LLM_RESCUE_TURN", 5))
    MIN_REJECTS = int(_num("LLM_RESCUE_REJECTS", 4))
    TERMINAL_STATUS = frozenset({400, 401, 403, 404, 413, 422})

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        flag = os.environ.get("LLM_RESCUE", "1").strip().lower()
        self._flag = flag not in {"0", "false", "no", "off"}
        self.model = model
        self.reaches = self.calls = self.usable = 0
        self.accepted = self.unattested = self.failures = self.retries = 0
        # Reported through `Agent.respond()['usage']`. That accounting already polled
        # these two attributes; nothing ever assigned them, so a run that really did
        # spend tokens still disclosed zero.
        self.prompt_tokens = self.completion_tokens = 0
        self._spent = 0.0
        self._open: str | None = None
        self._done: set[str] = set()
        self._df = None

    def bind(self, df_fn):
        """Injected so this module never imports the agent back."""
        self._df = df_fn
        return self

    @property
    def enabled(self) -> bool:
        return self._flag and bool(os.environ.get("GROQ_API_KEY", "").strip())

    def should_fire(self, sid: str, turn: int, rejected: int) -> bool:
        return (self.enabled and self._open is None and sid not in self._done
                and turn >= self.MIN_TURN and rejected >= self.MIN_REJECTS)

    def _call(self, transcript: str) -> str | None:
        if self._spent >= self.TIME_BUDGET:
            self._open = "time budget exhausted"
            return None
        payload = json.dumps({
            "model": self.model, "temperature": 0.0,
            "max_tokens": self.MAX_TOKENS, "reasoning_effort": "low",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": transcript}],
        }).encode("utf-8")
        request = Request(ENDPOINT, data=payload, method="POST", headers={
            "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0"})
        for attempt in range(max(self.RETRIES, 1)):
            started = time.time()
            try:
                with urlopen(request, timeout=self.TIMEOUT) as response:
                    body = json.loads(response.read().decode("utf-8"))
                # What the API actually charged, straight from the body we already parse.
                used = body.get("usage") or {}
                self.prompt_tokens += max(0, int(used.get("prompt_tokens", 0) or 0))
                self.completion_tokens += max(0, int(used.get("completion_tokens", 0) or 0))
                self._spent += time.time() - started
                return body["choices"][0]["message"]["content"]
            except HTTPError as exc:
                self._spent += time.time() - started
                self.failures += 1
                if exc.code in self.TERMINAL_STATUS:
                    self._open = f"HTTP {exc.code}"
                    return None
            except Exception:
                self._spent += time.time() - started
                self.failures += 1
            if attempt + 1 < max(self.RETRIES, 1):
                self.retries += 1
                time.sleep(min(self.BACKOFF ** attempt, 30.0))
        return None

    def recover(self, sid: str, turns: list[str]) -> list[str]:
        """Return catalogue-attested phrases to ADD, or an empty list."""
        if not self.enabled or self._open is not None or sid in self._done:
            return []
        self._done.add(sid)              # once per session, whatever happens next
        self.reaches += 1
        transcript = "\n".join(f"Turn {i}, shopper: {t}" for i, t in enumerate(turns, 1))
        if not transcript.strip():
            return []
        self.calls += 1
        raw = self._call(transcript)
        if not raw:
            return []
        # The model is asked for bare JSON; a fenced block or a stray sentence around it is
        # common enough to be worth recovering from rather than discarding the whole call.
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            return []
        if not isinstance(parsed, dict):
            return []
        self.usable += 1

        values = [v for v in (parsed.get("requirements") or []) if isinstance(v, str)]
        for field in ("material", "color"):
            if isinstance(parsed.get(field), str) and parsed[field].strip():
                values.append(parsed[field])
        if isinstance(parsed.get("category"), str) and parsed["category"].strip():
            values.append(parsed["category"])

        out: list[str] = []
        for value in values:
            phrase = " ".join(str(value).lower().split()).strip(".,;:")
            if not phrase:
                continue
            try:
                ok = self._df is not None and self._df(phrase) > 0
            except Exception:
                ok = False
            if ok:
                self.accepted += 1
                out.append(phrase)
            else:
                self.unattested += 1
        return out

    def stats(self) -> dict:
        return {"enabled": self.enabled, "model": self.model, "reaches": self.reaches,
                "calls": self.calls, "usable": self.usable, "accepted": self.accepted,
                "unattested": self.unattested, "failures": self.failures,
                "retries": self.retries,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "seconds": round(self._spent, 2),
                "circuit_reason": self._open}
