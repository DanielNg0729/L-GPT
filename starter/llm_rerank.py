"""
Optional LLM tie-break layer. Strictly off the critical path.

WHY THIS EXISTS
---------------
The irreducibility diagnostic showed 59% of rank>1 hits are cases where every product
ranked above the target covers *exactly the same evidence*. No coverage feature can break
those ties, because there is nothing left in the disclosed text to separate them. Breaking
them needs a signal ORTHOGONAL to the constraints -- e.g. knowing which of two
identically-matching garments a person would actually buy. That is the "content and
context knowledge" the zero-shot CRS literature credits LLMs with.

WHY IT IS GUARDED SO HEAVILY
----------------------------
  * submission_rules.md: "organizer policy may disable network access", and exceptions,
    invalid output and timeouts "may count as a miss". A network call on the scored path
    is a liability, so this layer must be unable to make anything worse.
  * InteRecAgent measured LLMs ranking BELOW random on Amazon data, by emitting
    out-of-scope item IDs -- which our harness silently discards. So the model is never
    allowed to invent an ID: its output must be a permutation of the candidates we handed
    it, verified, or the whole call is discarded.

Contract: `rerank()` returns a reordered list, or None. None means "caller keeps its own
ranking". It never raises, never blocks longer than the timeout, and is a no-op when
GROQ_API_KEY is unset -- so the agent is byte-identical offline.

Free-tier reality (30 RPM / ~6-12K TPM / ~1000 RPD) is handled by three things: calling
only on genuine ties, a persistent disk cache keyed by prompt hash, and a token-bucket
limiter that respects both requests and tokens per minute.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from pathlib import Path


def _load_project_env() -> None:
    """Load the local experiment file without overriding real environment variables."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_project_env()

ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
CACHE_PATH = Path(os.environ.get(
    "LLM_CACHE", str(Path(__file__).resolve().parent / ".llm_cache.json")))

SYSTEM = (
    "You rank e-commerce products. You are given a shopper's stated requirements and a "
    "numbered list of candidate products that ALL already match those requirements "
    "equally well. Your job is to decide which one the shopper most likely actually "
    "bought, using general knowledge about products and buying behaviour. "
    "Reply with ONLY the candidate numbers in order, best first, comma-separated. "
    "Use every number exactly once. No words, no explanation."
)


class RateLimiter:
    """Token bucket over both requests/min and tokens/min (free tier limits both)."""

    def __init__(self, rpm: int = 25, tpm: int = 5500) -> None:
        self.rpm, self.tpm = rpm, tpm
        self._events: list[tuple[float, int]] = []
        self._lock = threading.Lock()

    def acquire(self, est_tokens: int) -> None:
        while True:
            with self._lock:
                now = time.time()
                self._events = [(t, n) for t, n in self._events if now - t < 60.0]
                if (len(self._events) < self.rpm
                        and sum(n for _, n in self._events) + est_tokens <= self.tpm):
                    self._events.append((now, est_tokens))
                    return
                oldest = min(t for t, _ in self._events)
                wait = max(0.25, 60.0 - (now - oldest))
            time.sleep(min(wait, 5.0))


class LLMReranker:
    MAX_CANDIDATES = 8
    MAX_TOKENS = 900
    DESC_CHARS = 320
    TIMEOUT = 12.0
    MAX_RETRIES = 2

    def __init__(self, model: str = DEFAULT_MODEL, cache_path: Path = CACHE_PATH,
                 rpm: int = 25, tpm: int = 5500) -> None:
        self.model = model
        self.cache_path = Path(cache_path)
        self.limiter = RateLimiter(rpm, tpm)
        self.cache: dict[str, str] = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}
        self.calls = self.cache_hits = self.failures = 0
        self.prompt_tokens = self.completion_tokens = 0
        self._dirty = 0
        # Read once; the VALUE is never stored on the instance, logged, or transmitted
        # anywhere except the Authorization header of the request itself.
        # A key alone must not turn a normal evaluation into an online experiment.
        self._enabled = (os.environ.get("LLM_RERANK", "").lower() in {"1", "true", "yes"}
                         and bool(os.environ.get("GROQ_API_KEY")))

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _flush(self, force: bool = False) -> None:
        self._dirty += 1
        if force or self._dirty >= 20:
            try:
                self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")
                self._dirty = 0
            except Exception:
                pass

    def _prompt(self, requirements: list[str], docs: list[str]) -> str:
        req = "; ".join(r for r in requirements if r)[:400] or "(none stated)"
        lines = [f"{i+1}. {d[:self.DESC_CHARS]}" for i, d in enumerate(docs)]
        return (f"Shopper requirements: {req}\n\nCandidates:\n" + "\n".join(lines)
                + f"\n\nOrder all {len(docs)} numbers, best first:")

    def _call(self, prompt: str) -> str | None:
        est = (len(SYSTEM) + len(prompt)) // 3 + 120
        for attempt in range(self.MAX_RETRIES + 1):
            self.limiter.acquire(est)
            try:
                payload = json.dumps({
                    "model": self.model, "temperature": 0,
                    # GPT-OSS bills hidden reasoning against max_tokens and returns an
                    # EMPTY content string when the budget runs out mid-thought -- HTTP
                    # 200, finish_reason "stop", nothing to parse. At 128 tokens that hit
                    # 80% of real 8-candidate groups (synthetic prompts fit fine, which is
                    # why a smoke test on toy data hides it). Reasoning on a real group
                    # runs ~150-400 tokens, so leave clear headroom.
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
                             # identity with HTTP 403, despite valid credentials.
                             "User-Agent": "Mozilla/5.0"},
                    method="POST",
                )
                with urlopen(request, timeout=self.TIMEOUT) as response:
                    body = json.loads(response.read().decode("utf-8"))
                usage = body.get("usage") or {}
                self.prompt_tokens += max(0, int(usage.get("prompt_tokens") or 0))
                self.completion_tokens += max(0, int(usage.get("completion_tokens") or 0))
                return body["choices"][0]["message"]["content"]
            except HTTPError as error:
                if error.code == 429:
                    time.sleep(min(2.0 * (attempt + 1), 8.0))
                    continue
                return None
            except (OSError, URLError, ValueError, KeyError, IndexError, TypeError):
                if attempt >= self.MAX_RETRIES:
                    return None
                time.sleep(1.0)
        return None

    def rerank(self, requirements: list[str], asins: list[str],
               docs: list[str]) -> list[str] | None:
        """Reorder `asins`. Returns None whenever the caller should keep its own order."""
        if not self._enabled or len(asins) < 2:
            return None
        asins = asins[:self.MAX_CANDIDATES]
        docs = docs[:len(asins)]
        prompt = self._prompt(requirements, docs)
        key = hashlib.sha256(f"{self.model}\0{prompt}".encode("utf-8")).hexdigest()

        cached = key in self.cache
        if cached:
            self.cache_hits += 1
            raw = self.cache[key]
        else:
            raw = self._call(prompt)
            self.calls += 1
            if raw is None:
                self.failures += 1
                return None

        # The model may only PERMUTE what we gave it. Anything else is discarded whole.
        order = self._permutation(raw, len(asins))
        if order is None:
            self.failures += 1
            return None

        # Cache ONLY validated responses. Caching before validation (an earlier version
        # did) makes every transient empty completion permanent: the bad value is served
        # from cache forever, so the failure can never be retried or fixed by raising
        # max_tokens. Poisoned entries have to be purged by hand.
        if not cached:
            self.cache[key] = raw
            self._flush()
        return [asins[i] for i in order]

    @staticmethod
    def _permutation(raw: str | None, n: int) -> list[int] | None:
        seen, order = set(), []
        for token in re.findall(r"\d+", raw or ""):
            value = int(token)
            if 1 <= value <= n and value not in seen:
                seen.add(value)
                order.append(value - 1)
        return order if len(order) == n else None

    def stats(self) -> dict:
        self._flush(force=True)
        return {"enabled": self._enabled, "model": self.model, "api_calls": self.calls,
                "cache_hits": self.cache_hits, "failures": self.failures,
                "cached_entries": len(self.cache),
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}
