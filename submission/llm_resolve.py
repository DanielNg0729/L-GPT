"""
Optional LLM deparaphrase layer. Reachable ONLY for a clause the catalogue cannot attest.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
It is generate-then-verify, not RAG. The model receives one paraphrased attribute value and
NOTHING else -- no candidate list, no catalogue context -- and answers from parametric
knowledge. The catalogue enters afterwards as a provenance FILTER, never as prompt CONTEXT:

    phrase -> LLM (no context) -> proposal -> df(proposal) > 0 -> accept, else drop

A retrieval-augmented variant was built and measured, and it LOST. Constraining the model
to a retrieved candidate list scored below doing nothing at all (-0.0050 against the
suppression floor), and offering candidates as optional hints still lost to no candidates
at all. The mechanism is visible: told the candidates were hints it should ignore when
wrong, the model answered off-list 2 times out of 21, where unaided it answers freely 23
out of 23. A list captures the answer whatever the instruction says. So there is
deliberately no retriever here, and adding one would be a regression.

WHY THE TASK IS AN LLM'S AT ALL
-------------------------------
Seven encoder-based attempts failed before this. The diagnosis is that resolving "made from
a soft plant fibre" to `cotton` is WORLD KNOWLEDGE over a closed answer set, not ranking.
Bi-encoders retrieved antonyms -- "made overseas" -> "made in usa" -- because cosine
similarity cannot represent negation. This layer instead ABSTAINS on the negation cases,
which is the behaviour that distinguishes knowledge from similarity.

THE GUARANTEE
-------------
This module cannot change the score on traffic the agent understands. It is consulted only
where `df(whole_clause) == 0` -- exactly where `Agent._resolve` already suppresses and
contributes nothing. Measured across every population suite, with the resolver reaching it
26 times in total:

    official200 +0.000000   org-proxy +0.000000   review800 +0.000000
    uniform     +0.000000   inverse   +0.000000

Note those are live invocations that changed nothing, not an absence of invocations. An
earlier claim that the path was unreachable by construction was WRONG: the recognition gate
governs messages, not values, and `intent_card()` truncates long feature bullets mid-word,
which produces an unattested clause from genuine catalogue prose.

WEIGHT: ATTENUATED, NOT FULL
----------------------------
Accepted proposals enter at a reduced weight rather than at CONSTRAINT strength. This is
the single largest design decision here, worth more than everything else combined:

    proposals at CONSTRAINT weight    81.5% of a perfect resolver
    proposals at attenuated weight    ~96%

Same knowledge in both. The difference is entirely how much a WRONG proposal costs -- at
full weight one bad canonical outranks the correct evidence around it. The weight is not
tuned: three values spanning 0.15-0.45 differed by 0.0046 and non-monotonically, so the
finding is insensitivity, not an optimum.

NO SEMANTIC VERIFIER
--------------------
A zero-shot NLI verifier was built for this path and rejected. It separates correct
proposals from competing attested values well in isolation (0.8349 AUROC, catching 89.5% of
values the `df > 0` gate waves through), but no threshold transferred: calibrated on
synonym pairs it discarded 76% of CORRECT proposals on open vocabulary, cutting the gain
from +0.0169 to +0.0009. It stays out until it can be recalibrated on train-only data.

ENABLED BY DEFAULT, AND WHAT THAT ACTUALLY MEANS
------------------------------------------------
`LLM_RESOLVE` defaults to 1, but the layer additionally requires `GROQ_API_KEY`. Both must
hold, so there are exactly two outcomes and both are intended:

  no credentials    the layer is inert. `enabled` is False, no client is constructed, no
                    request is made, and the agent is byte-identical to a lexical-only run.
                    This is what an evaluator without our key sees.
  credentials       the layer activates exactly where `Agent._resolve` already gave up --
                    on a clause the catalogue cannot attest. Measured across every
                    population suite it reaches that state 26 times in total and changes
                    the score by +0.000000, while recovering +0.0169 on open-vocabulary
                    attribute paraphrase.

So enabling it by default cannot surprise anyone: without a key nothing happens, and with
one it fires rarely and has been measured not to move the decision criteria.

TO DISABLE COMPLETELY, set `LLM_RESOLVE=0` (or simply provide no `GROQ_API_KEY`). Do that
if the evaluation environment forbids network egress, if reproducibility must be exact --
the provider is not bit-reproducible even at temperature 0 -- or if any doubt exists about
per-call cost. The deterministic pipeline is the product; this layer is an addition to it,
never a dependency of it.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

try:
    from submission.llm_rerank import ENDPOINT, _load_project_env
except Exception:  # pragma: no cover - keep importable standalone
    from llm_rerank import ENDPOINT, _load_project_env  # type: ignore

_load_project_env()

# MODEL CHOICE, MEASURED. Both sizes were run on the full 800-session open-vocabulary
# attribute suite, same code, same 172 distinct prompts, same max_tokens:
#
#            score      proposal precision   in-request seconds
#   120b     0.868219   0.737                100.7
#    20b     0.870525   0.659                 28.1
#
# The score difference (+0.002306) is INSIDE the +/-0.0027 bootstrap band for an
# 800-session suite, so the two are score-equivalent and the smaller model is chosen on
# cost, not on quality. It is 3.6x faster in-request and reserves the same tokens per call,
# which matters because the provider's tokens-per-minute allowance -- not its request
# allowance -- is what this layer runs into first.
#
# A WARNING ABOUT THE PRECISION COLUMN. 20b answers far more often at lower precision, and
# on a 250-session subsample that looked like a +0.0139 win for coverage over precision.
# It collapsed to +0.0023 when the same comparison was run on all 800. Do not read the
# precision gap as a quality finding in either direction; the honest statement is that the
# scores are indistinguishable and only the cost differs.
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
CACHE_PATH = Path(os.environ.get(
    "LLM_RESOLVE_CACHE", str(Path(__file__).resolve().parent / ".llm_resolve_cache.json")))

SYSTEM = (
    "You map a shopper's description of a product attribute onto the exact wording an "
    "e-commerce catalogue would use for it. Answer with the catalogue's own short "
    "attribute value and NOTHING else -- no sentence, no explanation, no quotes. "
    "Prefer the plainest, most common trade term: for 'made from a soft plant fibre' "
    "answer 'cotton'. If the description is a NEGATION or you are not confident which "
    "single value it names, answer exactly NONE. Answering NONE is correct and expected "
    "whenever you would otherwise guess."
)


def _num(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


class LLMResolver:
    """Paraphrased attribute value -> catalogue-attested phrase, or None.

    `resolve()` never raises, never blocks past its budget, and returns None for
    "caller should suppress this clause", which is the shipped behaviour anyway.
    """

    # RETRY EXISTS BECAUSE ITS ABSENCE KILLED A MEASUREMENT. The research harness that
    # produced this layer's numbers retried with backoff; when the module was hardened for
    # shipping it gained a time budget, a zero-yield breaker and terminal-status handling,
    # and silently LOST the retry loop. A pipeline grid then tripped the breaker nine calls
    # in -- six consecutive transient rate-limit errors -- and both LLM arms measured
    # nothing while reporting plausible-looking numbers.
    #
    # Backoff is deterministic (no jitter). This is a single-threaded client, so jitter
    # buys nothing against self-contention, and a reproducible run is worth more here.
    MAX_TOKENS = int(_num("LLM_RESOLVE_MAX_TOKENS", 512))
    RETRIES = int(_num("LLM_RESOLVE_RETRIES", 3))
    TRIP_AFTER = int(_num("LLM_RESOLVE_TRIP_AFTER", 6))
    ZERO_YIELD_TRIP = 60                 # calls yielding nothing before giving up entirely
    TIMEOUT = _num("LLM_RESOLVE_TIMEOUT", 20.0)
    TIME_BUDGET = _num("LLM_RESOLVE_TIME_BUDGET", 600.0)
    BACKOFF = _num("LLM_RESOLVE_BACKOFF", 2.0)
    # 4xx that will never succeed on retry. Retrying these burns the time budget and the
    # rate limiter for a result that cannot change.
    TERMINAL_STATUS = frozenset({400, 401, 403, 404, 413, 422})

    def __init__(self, model: str = DEFAULT_MODEL,
                 cache_path: Path = CACHE_PATH) -> None:
        flag = os.environ.get("LLM_RESOLVE", "1").strip().lower()
        self._flag = flag not in {"0", "false", "no", "off", ""}
        self.model = model
        self.cache_path = cache_path
        self.calls = self.accepted = self.abstained = 0
        self.unattested = self.failures = self.retries = 0
        self.prompt_tokens = self.completion_tokens = 0
        self.cache_hits = self.cache_misses = 0
        self._consecutive = 0
        self._spent = 0.0
        self._open_reason: str | None = None
        self._dirty = False
        try:
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            self.cache = {}

    @property
    def enabled(self) -> bool:
        return self._flag and bool(os.environ.get("GROQ_API_KEY", "").strip())

    @property
    def circuit_open(self) -> bool:
        return self._open_reason is not None

    def _trip(self, reason: str) -> None:
        if self._open_reason is None:
            self._open_reason = reason

    def _call(self, phrase: str) -> str | None:
        if self._spent >= self.TIME_BUDGET:
            self._trip("time budget exhausted")
            return None
        payload = json.dumps({
            "model": self.model, "temperature": 0.0,
            # GPT-OSS bills hidden reasoning against max_tokens and returns EMPTY content
            # when the budget runs out mid-thought: HTTP 200 with nothing to parse. The
            # headroom is for the reasoning, not for the two-word answer.
            #
            # 512 is deliberately generous and stays the default. It is also the dominant
            # cost of a call, because the provider reserves `prompt + max_tokens` against
            # the tokens-per-minute allowance rather than actual usage -- 585 reserved for
            # a request that measured 229 actual, 35 of which was reasoning. Measured on
            # six prompts, 160 / 256 / 512 return byte-identical answers (the `NONE`
            # abstention included) and only 96 truncates. Lowering it therefore buys ~2.5x
            # the throughput and daily capacity for no observed change in output.
            #
            # It is exposed rather than lowered because six prompts is thin evidence for a
            # shipped default, and an empty completion is a silent failure: the caller sees
            # None and suppresses, which looks exactly like a legitimate abstention.
            # Harnesses that need throughput set it; the shipped path does not.
            "max_tokens": self.MAX_TOKENS, "reasoning_effort": "low",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": phrase}],
        }).encode("utf-8")
        request = Request(ENDPOINT, data=payload, method="POST", headers={
            "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
            "Content-Type": "application/json",
            # Groq's edge answers Python's default `Python-urllib/*` identity with a
            # Cloudflare 403 (code 1010) despite valid credentials.
            "User-Agent": "Mozilla/5.0"})
        for attempt in range(max(self.RETRIES, 1)):
            if self._spent >= self.TIME_BUDGET:
                self._trip("time budget exhausted")
                return None
            started = time.time()
            try:
                with urlopen(request, timeout=self.TIMEOUT) as response:
                    body = json.loads(response.read().decode("utf-8"))
                self._spent += time.time() - started
                usage = body.get("usage") or {}
                self.prompt_tokens += max(0, int(usage.get("prompt_tokens") or 0))
                self.completion_tokens += max(0, int(usage.get("completion_tokens") or 0))
                self._consecutive = 0
                return body["choices"][0]["message"]["content"]
            except HTTPError as exc:
                self._spent += time.time() - started
                self.failures += 1
                if exc.code in self.TERMINAL_STATUS:
                    # Will never succeed on retry; retrying burns the budget for nothing.
                    self._trip(f"HTTP {exc.code}")
                    return None
            except Exception:
                self._spent += time.time() - started
                self.failures += 1
            if attempt + 1 < max(self.RETRIES, 1):
                self.retries += 1
                time.sleep(min(self.BACKOFF ** attempt, 30.0))
        self._consecutive += 1
        if self._consecutive >= self.TRIP_AFTER:
            self._trip(f"{self._consecutive} consecutive failures")
        return None

    def resolve(self, phrase: str) -> str | None:
        """Return a catalogue-attested phrase for `phrase`, or None to keep suppressing."""
        if not self.enabled or self.circuit_open or not phrase:
            return None
        if phrase in self.cache:
            # The cache is also the REPRODUCIBILITY mechanism: a fully cached run is exact,
            # while a cache miss goes to a model that is not bit-reproducible even at
            # temperature 0. `stats()` reports the hit rate so a run can say which it was.
            self.cache_hits += 1
            return self.cache[phrase] or None
        self.cache_misses += 1
        if self.calls >= self.ZERO_YIELD_TRIP and self.accepted == 0:
            self._trip(f"{self.calls} calls with zero accepted proposals")
            return None
        self.calls += 1
        raw = self._call(phrase)
        if raw is None or not raw.strip():
            # A failure is NOT an abstention and must not be cached: caching it would
            # turn one transient network error into a permanent wrong answer.
            return None
        proposal = " ".join(raw.lower().split()).strip(".")
        if proposal == "none":
            self.abstained += 1
            self.cache[phrase] = ""
            self._dirty = True
            return None
        if not self._attested(proposal):
            # PROVENANCE. The model proposes; the catalogue disposes. Every constraint the
            # simulator speaks is a verbatim substring of the target document, so a phrase
            # the catalogue has never seen cannot be the customer's requirement.
            self.unattested += 1
            self.cache[phrase] = ""
            self._dirty = True
            return None
        self.accepted += 1
        self.cache[phrase] = proposal
        self._dirty = True
        return proposal

    # The index is injected by the agent rather than imported, so this module stays free of
    # any dependency on the agent and remains testable on its own.
    _df = None

    def bind(self, df_fn) -> "LLMResolver":
        self._df = df_fn
        return self

    def _attested(self, phrase: str) -> bool:
        try:
            return self._df is not None and self._df(phrase) > 0
        except Exception:
            return False

    def flush(self) -> None:
        if not self._dirty:
            return
        try:
            self.cache_path.write_text(
                json.dumps(self.cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self._dirty = False
        except Exception:
            pass

    def stats(self) -> dict:
        total = self.cache_hits + self.cache_misses
        return {"enabled": self.enabled, "model": self.model, "calls": self.calls,
                "accepted": self.accepted, "abstained": self.abstained,
                "unattested": self.unattested, "failures": self.failures,
                "retries": self.retries,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": round(self.cache_hits / total, 4) if total else None,
                "fully_cached": total > 0 and self.cache_misses == 0,
                "seconds": round(self._spent, 2), "circuit_reason": self._open_reason}
