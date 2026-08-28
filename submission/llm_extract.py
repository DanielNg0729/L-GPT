"""
Optional LLM extraction layer. Reachable ONLY on messages the agent does not recognise.

WHY THIS EXISTS
---------------
Evidence extraction has two channels: template regexes matched against the simulator's
literal format strings, and catalogue-grounded n-gram mining. Templates are precise and
carry most of the score (+0.154 public / +0.188 unseen), but they stop firing the instant
the organizer rewords anything -- and the specification explicitly reserves that right:

    "The simulator policy decides what information to reveal. If natural-language
     paraphrasing is added by the organizer, it cannot decide correctness."

Mining is what stops that being a collapse (0.838 instead of 0.164 at realistic
paraphrase), and it is already at its optimum: sweeping its two governing constants either
regresses the clean score (minn=2: clean -0.016) or makes paraphrase worse (minn=4: T1
-0.084). So a third channel is the only remaining lever, and an LLM is the natural fit --
the task is paraphrase INVERSION back into catalogue vocabulary, which is why a BERT-style
tagger reportedly failed at it while an LLM did not.

THE GUARANTEE THIS MODULE IS BUILT AROUND
-----------------------------------------
It must be impossible for this layer to change the score when the organizer ships clean
templates. That is not enforced by a confidence threshold -- any scorer fires sometimes --
but by the caller's RECOGNITION GATE in `agent.recognised()`: the simulator emits a closed
set of message shapes, and this module is only consulted for a message matching NONE of
them. Measured over a clean run: 463/463 messages recognised, so this file is unreachable
at zero paraphrase and the clean score is unchanged by construction, not by luck.

EXTRACT VERBATIM, NEVER GENERATE
--------------------------------
The whole agent rests on provenance: every constraint is a literal substring of the target
product's catalogue text, so an invented or reworded phrase is worse than no phrase -- it
withholds weight from the target and hands it to the field. The prompt therefore asks for
spans COPIED from the message, and the caller re-validates every returned span against the
catalogue before it may enter the evidence ledger.

Contract: `extract()` returns a list of phrases, or None. None means "caller proceeds with
templates and mining alone". It never raises, never blocks past the timeout, and is inert
unless BOTH `LLM_EXTRACT=1` and `GROQ_API_KEY` are set -- so the shipped agent is
byte-identical offline.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from submission.llm_rerank import ENDPOINT, RateLimiter, _load_project_env
except Exception:  # pragma: no cover - keeps this module importable standalone
    from llm_rerank import ENDPOINT, RateLimiter, _load_project_env  # type: ignore

_load_project_env()

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


def _num(name: str, default: float) -> float:
    """Read a tunable from the environment, falling back on a free-tier-safe default."""
    try:
        return float(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default
CACHE_PATH = Path(os.environ.get(
    "LLM_EXTRACT_CACHE", str(Path(__file__).resolve().parent / ".llm_extract_cache.json")))

SYSTEM = (
    "You extract shopping requirements from a customer message for a product search "
    "engine. Return ONLY the spans that describe the product -- materials, colours, "
    "features, closures, care instructions, product type, category words. "
    "COPY each span EXACTLY as it appears in the message, character for character. "
    "Do NOT paraphrase, translate, normalise, pluralise, or invent anything: the spans are "
    "matched literally against a product catalogue, so any rewording makes them useless. "
    "Ignore greetings, filler, politeness, and statements that the customer has no "
    "preference. Output one span per line, at most 6 lines, no numbering, no commentary. "
    "If the message states no product requirement at all, output the single word NONE."
)


class CircuitOpen(Exception):
    """Raised internally when the breaker has tripped; never escapes `extract()`."""


class LLMExtractor:
    MAX_TOKENS = 400
    MAX_PHRASES = 6
    MAX_MESSAGE_CHARS = 600
    MIN_PHRASE_CHARS = 2
    MAX_PHRASE_CHARS = 120
    TIMEOUT = 12.0
    MAX_RETRIES = 2

    # ---- circuit breaker -------------------------------------------------------------
    #
    # Without this, a dead network is not a degradation, it is a catastrophe. A private run
    # sends ~1,500 extraction calls; at 3 attempts x a 12 s timeout that is 36 s per
    # message, or roughly FIFTEEN HOURS of waiting to arrive at exactly the score the agent
    # would have produced offline in 14 seconds. The failure mode is not a wrong answer, it
    # is never finishing -- and the submission rules warn that timeouts "may count as a
    # miss".
    #
    # So the layer gives up, permanently and early, on evidence that the endpoint is not
    # going to work:
    #   * a terminal error (bad key, forbidden, model not found) trips it on the FIRST
    #     occurrence -- retrying an invalid credential cannot succeed;
    #   * TRIP_AFTER consecutive non-terminal failures trip it, covering a network that is
    #     down, a host that is unreachable, or a quota that is exhausted;
    #   * TIME_BUDGET caps total seconds spent waiting on the endpoint across the whole
    #     run, so even a slow-but-alive endpoint cannot dominate the wall clock.
    #
    # Once open the breaker never closes: every later call returns None immediately with no
    # socket opened, and the agent runs on templates plus mining -- its offline behaviour.
    TRIP_AFTER = 8
    # A separate breaker for the mode the failure harness exposed: HTTP 200, well-formed,
    # and USELESS -- an empty completion, a wall of prose, or spans the model invented that
    # the verbatim check then discards. None of those are "failures", so the consecutive-
    # failure counter never moves and only the time budget stops them. Some messages
    # legitimately yield nothing ("no strong feelings about colour"), so the threshold is
    # far higher than TRIP_AFTER: a run that extracts nothing from this many CONSECUTIVE
    # calls is broken, not merely unlucky.
    ZERO_YIELD_TRIP = 50
    # TIME_BUDGET must comfortably exceed the network time a HEALTHY full run needs, or the
    # breaker fires on success. A private run is ~1,500 calls at ~1.5 s of round trip each
    # = ~2,250 s, so the previous 2,400 s left almost no margin and a slightly slower
    # endpoint would have tripped it mid-run -- costing the benefit while looking like a
    # protection working correctly. 90 minutes of NETWORK time (throttle sleep excluded)
    # leaves real headroom while still bounding a pathologically slow endpoint.
    TIME_BUDGET = _num("LLM_TIME_BUDGET", 5400.0)
    TERMINAL_STATUS = frozenset({400, 401, 403, 404, 413, 422})

    def __init__(self, model: str = DEFAULT_MODEL, cache_path: Path = CACHE_PATH,
                 rpm: int | None = None, tpm: int | None = None) -> None:
        # Defaults are the Groq FREE tier (25 req/min, 5.5K tok/min). A paid key raises
        # both by orders of magnitude, and the throttle -- not the model -- is what makes a
        # full run take an hour, so make it configurable rather than a buried constant.
        rpm = int(_num("LLM_RPM", 25)) if rpm is None else rpm
        tpm = int(_num("LLM_TPM", 5500)) if tpm is None else tpm
        self.model = model
        self.cache_path = Path(cache_path)
        self.limiter = RateLimiter(rpm, tpm)
        self.cache: dict[str, list[str]] = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}
        self.calls = self.cache_hits = self.failures = 0
        self.prompt_tokens = self.completion_tokens = 0
        self._dirty = 0
        # Failure taxonomy. Kept separately from the total because the disclosure needs to
        # distinguish "the model could not do it" from "we were rate limited" -- on the
        # first measured run 35 of 110 apparent failures were quota, not quality, which
        # understated the layer.
        self.reasons: dict[str, int] = {}
        self._consecutive = 0
        self._zero_yield = 0
        self._spent = 0.0
        self._open_reason: str | None = None
        # ENABLED BY DEFAULT, but the credential is still mandatory: no key, no calls,
        # ever. That ordering is what makes defaulting on safe -- the layer cannot activate
        # itself in an environment that never opted in by providing a key, and
        # `submission_rules.md` forbids committing one, so the organizer's environment has
        # none unless they deliberately supply it.
        #
        # `LLM_EXTRACT=0` is an explicit opt-out for the case where a key happens to be
        # present in the environment for unrelated reasons and an online run is not wanted.
        #
        # The key's VALUE is never stored on the instance, logged, or sent anywhere but
        # this request's Authorization header.
        # Default OFF: the local BERT tagger (bert_extract.py) is now the primary
        # extraction channel for unrecognised messages, and beats this layer on the
        # hardest transform while needing no network. This stays as an alternate route --
        # set LLM_EXTRACT=1 to A/B it, or to try a stronger model.
        flag = os.environ.get("LLM_EXTRACT", "0").strip().lower()
        self._enabled = (flag not in {"0", "false", "no", "off"}
                         and bool(os.environ.get("GROQ_API_KEY")))

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def _norm(message: str) -> str:
        """Cache key. Whitespace and case carry no extraction signal, so collapsing them
        raises the hit rate without changing what is asked."""
        return " ".join(message.lower().split())

    def _flush(self, force: bool = False) -> None:
        self._dirty += 1
        if force or self._dirty >= 20:
            try:
                self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")
                self._dirty = 0
            except Exception:
                pass

    # ---- breaker plumbing ------------------------------------------------------------
    @property
    def circuit_open(self) -> bool:
        return self._open_reason is not None

    def _trip(self, reason: str) -> None:
        if self._open_reason is None:
            self._open_reason = reason

    def _note(self, reason: str, terminal: bool = False) -> None:
        """Record a failure and decide whether it should open the breaker."""
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        self._consecutive += 1
        if terminal:
            self._trip(f"terminal:{reason}")
        elif self._consecutive >= self.TRIP_AFTER:
            self._trip(f"{self.TRIP_AFTER} consecutive failures ({reason})")

    def _call(self, message: str) -> str | None:
        if self.circuit_open:
            raise CircuitOpen(self._open_reason)
        if self._spent >= self.TIME_BUDGET:
            self._trip(f"time budget {self.TIME_BUDGET:.0f}s exhausted")
            raise CircuitOpen(self._open_reason)
        prompt = f"Customer message:\n{message}\n\nSpans:"
        est = (len(SYSTEM) + len(prompt)) // 3 + 120
        started = time.time()
        for attempt in range(self.MAX_RETRIES + 1):
            # The rate limiter's sleep is deliberately NOT charged to TIME_BUDGET. At
            # 25 RPM a healthy 1,500-call run spends ~60 minutes waiting on our OWN
            # throttle, which would exhaust any sane budget and trip the breaker on a
            # perfectly working endpoint -- the budget exists to bound time lost to a SICK
            # endpoint, so it must measure only time spent waiting on the network.
            self.limiter.acquire(est)
            started = time.time()
            try:
                payload = json.dumps({
                    "model": self.model, "temperature": 0,
                    # Reproducibility: temperature 0 plus a fixed seed makes the endpoint
                    # as deterministic as it is willing to be, and the on-disk cache makes
                    # any warm re-run exactly reproducible regardless.
                    "seed": 0,
                    # GPT-OSS bills hidden reasoning against max_tokens and returns EMPTY
                    # content when the budget runs out mid-thought -- HTTP 200, nothing to
                    # parse. Extraction reasons less than listwise ranking does, but leave
                    # headroom rather than rediscover that failure.
                    "max_tokens": self.MAX_TOKENS,
                    "reasoning_effort": "low",
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": prompt}],
                }).encode("utf-8")
                request = Request(
                    ENDPOINT, data=payload,
                    headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
                             "Content-Type": "application/json",
                             # Groq's edge rejects Python's default `Python-urllib/*`
                             # identity with HTTP 403 despite valid credentials.
                             "User-Agent": "Mozilla/5.0"},
                    method="POST",
                )
                with urlopen(request, timeout=self.TIMEOUT) as response:
                    body = json.loads(response.read().decode("utf-8"))
                usage = body.get("usage") or {}
                self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                self.completion_tokens += int(usage.get("completion_tokens") or 0)
                content = body["choices"][0]["message"]["content"]
                self._spent += time.time() - started
                self._consecutive = 0                 # a success closes the failure streak
                return content
            except HTTPError as error:
                if error.code in self.TERMINAL_STATUS:
                    # A bad key, a forbidden request or a missing model will fail exactly
                    # the same way on every one of the next ~1,500 messages. Stop now.
                    self._spent += time.time() - started
                    self._note(f"http_{error.code}", terminal=True)
                    return None
                if error.code == 429:
                    if attempt >= self.MAX_RETRIES:
                        self._spent += time.time() - started
                        self._note("rate_limited")
                        return None
                    time.sleep(min(2.0 * (attempt + 1), 8.0))
                    continue
                if attempt >= self.MAX_RETRIES:
                    self._spent += time.time() - started
                    self._note(f"http_{error.code}")
                    return None
                time.sleep(1.0)
            except (URLError, OSError) as error:
                # Network-level: unreachable host, refused connection, DNS failure, or a
                # read timeout. Retrying a dead network only multiplies the wait, so these
                # get ONE retry rather than the full budget.
                reason = "timeout" if "timed out" in str(error).lower() else "network"
                if attempt >= min(1, self.MAX_RETRIES):
                    self._spent += time.time() - started
                    self._note(reason)
                    return None
                time.sleep(0.5)
            except (ValueError, KeyError, IndexError, TypeError):
                # Malformed body, unexpected schema, non-JSON payload.
                self._spent += time.time() - started
                self._note("malformed_response")
                return None
        self._spent += time.time() - started
        self._note("exhausted_retries")
        return None

    def _parse(self, raw: str | None, message: str) -> list[str] | None:
        """Turn the completion into spans, discarding anything not literally present.

        The substring check is the load-bearing guard. It is what makes a hallucinated or
        helpfully-normalised phrase impossible to act on: if the model did not COPY it
        from the message, it does not survive. Without this the layer could inject
        plausible-sounding catalogue text the customer never said.
        """
        if raw is None:
            return None
        haystack = " ".join(message.lower().split())
        out: list[str] = []
        for line in raw.splitlines():
            span = line.strip().strip("-*• \t").strip()
            span = re.sub(r"^\d+[.)]\s*", "", span)
            if not span or span.upper() == "NONE":
                continue
            if not (self.MIN_PHRASE_CHARS <= len(span) <= self.MAX_PHRASE_CHARS):
                continue
            if " ".join(span.lower().split()) not in haystack:
                continue                       # not copied from the message -> discard
            if span not in out:
                out.append(span)
            if len(out) >= self.MAX_PHRASES:
                break
        return out

    def extract(self, message: str) -> list[str] | None:
        """Spans copied out of `message`, or None if the caller should proceed without.

        Total function: it does not raise for any input, any network state, any credential
        state, or any response the endpoint can return. Every path out is either a list of
        verbatim spans or None, and None always means "carry on with templates and mining".
        """
        if not self._enabled:
            return None
        message = (message or "").strip()[:self.MAX_MESSAGE_CHARS]
        if not message:
            return None
        key = hashlib.sha256(f"{self.model}\0{self._norm(message)}".encode("utf-8")).hexdigest()

        # The cache is consulted even with the breaker open: a hit costs no network and
        # work already paid for should not be thrown away because a later call failed.
        if key in self.cache:
            self.cache_hits += 1
            return list(self.cache[key])
        if self.circuit_open:
            self.reasons["skipped_circuit_open"] = \
                self.reasons.get("skipped_circuit_open", 0) + 1
            return None

        try:
            raw = self._call(message)
        except CircuitOpen:
            self.reasons["skipped_circuit_open"] = \
                self.reasons.get("skipped_circuit_open", 0) + 1
            return None
        except Exception:
            # Belt and braces. Nothing this layer does may reach the agent as an exception.
            self._note("unexpected")
            self.calls += 1
            self.failures += 1
            return None
        self.calls += 1
        spans = self._parse(raw, message)
        if spans is None:
            self.failures += 1
            return None
        self._consecutive = 0
        if spans:
            self._zero_yield = 0
        else:
            self._zero_yield += 1
            self.reasons["zero_yield"] = self.reasons.get("zero_yield", 0) + 1
            if self._zero_yield >= self.ZERO_YIELD_TRIP:
                self._trip(f"{self.ZERO_YIELD_TRIP} consecutive calls yielded no spans")

        # Cache ONLY validated output. Caching before validation makes a transient empty
        # completion permanent -- the bad value is served from cache forever and the
        # failure can never be retried. That bug cost a full run once already.
        self.cache[key] = spans
        self._flush()
        return list(spans)

    def stats(self) -> dict:
        self._flush(force=True)
        return {"enabled": self._enabled, "model": self.model, "api_calls": self.calls,
                "cache_hits": self.cache_hits, "failures": self.failures,
                "failure_reasons": dict(self.reasons),
                "circuit_open": self.circuit_open, "circuit_reason": self._open_reason,
                "seconds_on_network": round(self._spent, 1),
                "consecutive_zero_yield": self._zero_yield,
                "cached_entries": len(self.cache),
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}
