"""Relevance filter: push candidates that CONTRADICT an explicit requirement to the tail.

WHAT IT IS FOR. Exact matching cannot see a contradiction, only an absence. "Women's
running shoes" retrieves anything covering those tokens, and a men's shoe whose listing
happens to carry the phrase (a unisex size chart, a "not for women" line, a store name)
scores on coverage like any other row. The lexical layers have no notion of "this row
says the OPPOSITE of what the shopper asked". This node does exactly one thing: given
the shopper's stated requirements and a window of ranked candidates, it names the
candidates that clearly contradict a requirement, and those are demoted behind everyone
else.

HOW IT WALKS THE LIST. In windows, front to back: judge the first ten, keep the
survivors, refill the window from the next ranked candidates, and stop as soon as
enough candidates for this turn's disclosure width have survived (or the per-turn call
budget is spent). Unjudged candidates keep their order behind the survivors; flagged
candidates go last.

DEMOTE, NEVER DROP. The same rule the rejection ledger follows, for the same measured
reason: dropping shortens the returned list and quietly collects the MRR denominator
benefit this agent explicitly declines to exploit, and a false positive on a dropped
TARGET would be unrecoverable. Demotion with refill shows the identical top-N while
keeping every candidate reachable on the full-width final turn.

WHAT IT CANNOT DO. It never adds evidence, never promotes a candidate above the lexical
ranking's survivors, never chooses the question, and never emits a product id of its
own. A failed call, a malformed reply, or a reply that flags the ENTIRE window (a
filter that kills everything is broken, not discerning) keeps the window unchanged.
Every failure path leaves the ranking exactly as the lexical layers built it.

WHERE IT FIRES. Off by default, including when a key is present: on clean simulator
traffic the constraints are verbatim substrings of the target document, contradictions
of this kind do not occur, and the experiment record shows LLM relevance judgments
losing to the popularity prior there (41.2% vs 57.4% target-first on ties). This node
exists for real-shopper wording — demos, and any evaluator that paraphrases — where
"for my husband" defeats a substring matcher.

    LLM_FILTER=1            enable (also needs GROQ_API_KEY; default off)
    LLM_FILTER_CALLS        judged windows per turn (default 3)
    LLM_FILTER_BATCH        window size (default 10)
"""
from __future__ import annotations

import json
import os
import re
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

try:
    from submission.llm_rerank import ENDPOINT, RateLimiter, _load_project_env
except Exception:                                   # pragma: no cover - standalone import
    from llm_rerank import ENDPOINT, RateLimiter, _load_project_env  # type: ignore

_load_project_env()

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

SYSTEM = (
    "You review product candidates for a shopping search engine.\n"
    "You are given the shopper's own messages, their extracted requirements, and a "
    "numbered list of candidate products.\n"
    "Flag ONLY candidates that clearly CONTRADICT an explicit requirement — for "
    "example the shopper asked for women's items and the product is for men, they "
    "named a colour and the product is only sold in a different one, they set a price "
    "ceiling and the product is far above it.\n"
    "Rules:\n"
    "1. A missing detail is NOT a contradiction. If the listing simply does not "
    "mention the requirement, KEEP the candidate.\n"
    "2. When unsure, KEEP the candidate. Flagging a correct product is far worse than "
    "keeping a wrong one.\n"
    "3. Never flag for style, taste, or quality — only for direct conflict with what "
    "the shopper explicitly said.\n"
    '4. Reply with ONLY a JSON object, no prose and no code fence: {"remove": [0, 3]}\n'
    "   using the candidate numbers shown, or an empty list if nothing conflicts."
)


def _num(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


class LLMRelevanceFilter:
    """Windowed contradiction demotion over an already-ranked candidate list.

    Never raises; every failure leaves the input order unchanged.
    """

    MAX_TOKENS = int(_num("LLM_FILTER_MAX_TOKENS", 256))
    TIMEOUT = _num("LLM_FILTER_TIMEOUT", 15.0)
    TIME_BUDGET = _num("LLM_FILTER_TIME_BUDGET", 600.0)
    RETRIES = int(_num("LLM_FILTER_RETRIES", 2))
    BACKOFF = _num("LLM_FILTER_BACKOFF", 2.0)
    BATCH = max(1, int(_num("LLM_FILTER_BATCH", 10)))
    MAX_CALLS_PER_TURN = max(1, int(_num("LLM_FILTER_CALLS", 3)))
    TRANSCRIPT_TURNS = 8       # most recent shopper messages shown to the model
    SNIPPET = 240              # characters of listing context per candidate
    TERMINAL_STATUS = frozenset({400, 401, 403, 404, 413, 422})

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        flag = os.environ.get("LLM_FILTER", "0").strip().lower()
        self._flag = flag in {"1", "true", "yes", "on"}
        self.model = model
        self.calls = self.usable = self.flagged = self.failures = self.retries = 0
        self.prompt_tokens = self.completion_tokens = 0
        self._spent = 0.0
        self._open: str | None = None
        self._doc = None
        # Shared free-tier budget shape (requests/min and tokens/min), same defaults as
        # the other hosted layers so several of them cannot jointly exceed the account.
        self.limiter = RateLimiter(int(_num("LLM_RPM", 25)), int(_num("LLM_TPM", 5500)))

    def bind(self, doc_fn):
        """Injected so this module never imports the agent back."""
        self._doc = doc_fn
        return self

    @property
    def enabled(self) -> bool:
        return self._flag and bool(os.environ.get("GROQ_API_KEY", "").strip())

    def should_fire(self, evidence_count: int, pool_len: int) -> bool:
        """There must be requirements to contradict and more than one candidate."""
        return (self.enabled and self._open is None
                and evidence_count > 0 and pool_len > 1)

    # ------------------------------------------------------------------ transport
    def _call(self, prompt: str) -> str | None:
        if self._spent >= self.TIME_BUDGET:
            self._open = "time budget exhausted"
            return None
        self.limiter.acquire(len(prompt) // 4 + self.MAX_TOKENS)
        payload = json.dumps({
            "model": self.model, "temperature": 0.0,
            "max_tokens": self.MAX_TOKENS, "reasoning_effort": "low",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}],
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
                self._spent += time.time() - started
                usage = body.get("usage") or {}
                self.prompt_tokens += max(0, int(usage.get("prompt_tokens") or 0))
                self.completion_tokens += max(0, int(usage.get("completion_tokens") or 0))
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

    # ------------------------------------------------------------------ judgment
    def _judge(self, batch: list[str], requirements: list[str],
               transcript: list[str]) -> set[int] | None:
        """Window indices to demote, or None meaning "the call failed, keep the window"."""
        lines = ["Shopper messages:"]
        lines += [f"  {t}" for t in transcript[-self.TRANSCRIPT_TURNS:] if t.strip()]
        lines.append("Extracted requirements: "
                     + ("; ".join(requirements) if requirements else "(none)"))
        lines.append("Candidates:")
        for i, asin in enumerate(batch):
            doc = ""
            try:
                doc = str((self._doc(asin) if self._doc else "") or "")
            except Exception:
                pass
            lines.append(f"  [{i}] {doc[:self.SNIPPET]}")
        self.calls += 1
        raw = self._call("\n".join(lines))
        if not raw:
            return None
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            return None
        if not isinstance(parsed, dict) or not isinstance(parsed.get("remove"), list):
            return None
        out = {i for i in parsed["remove"] if isinstance(i, int) and 0 <= i < len(batch)}
        if len(out) >= len(batch):
            return None                 # flagged the whole window: broken, not discerning
        self.usable += 1
        self.flagged += len(out)
        return out

    def rearrange(self, ranked: list[str], requirements: list[str],
                  transcript: list[str], need: int) -> list[str]:
        """Return `ranked` reordered: survivors, then unjudged, then contradictions.

        Walks the head of the list in windows, refilling each window from the next
        ranked candidates ("judge ten, demote the wrong ones, pull the next ten"),
        until `need` candidates have survived or the per-turn call budget is spent.
        """
        if not self.should_fire(len(requirements), len(ranked)):
            return ranked
        kept: list[str] = []
        flagged: list[str] = []
        index, calls = 0, 0
        while (index < len(ranked) and len(kept) < max(1, need)
               and calls < self.MAX_CALLS_PER_TURN):
            batch = ranked[index:index + self.BATCH]
            index += len(batch)
            calls += 1
            removed = self._judge(batch, requirements, transcript)
            if removed is None:
                kept.extend(batch)      # fail open: an unjudged window is a kept window
                continue
            for i, asin in enumerate(batch):
                (flagged if i in removed else kept).append(asin)
        kept.extend(ranked[index:])     # unjudged tail keeps its lexical order
        return kept + flagged

    def stats(self) -> dict:
        return {"enabled": self.enabled, "model": self.model, "calls": self.calls,
                "usable": self.usable, "flagged": self.flagged,
                "failures": self.failures, "retries": self.retries,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "seconds": round(self._spent, 2), "circuit_reason": self._open}
