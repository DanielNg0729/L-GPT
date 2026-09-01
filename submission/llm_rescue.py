"""Transcript rescue: re-read the whole conversation when the turn-by-turn parse has stalled.

WHAT IT IS FOR. This is a TEMPLATE solver. It handles the case where the per-turn PARSE
failed -- wording the recognition gate does not know -- by re-reading the conversation as a
whole, which no other component does: the deparaphraser sees one unattested VALUE at a
time, and the span node and tagger see one MESSAGE at a time, so none of them can use the
fact that turn 6 clarifies something misread on turn 2.

WHAT IT IS NOT FOR. A reworded VALUE inside a wrapper that parsed correctly. That is the
deparaphraser's fault to fix and it is consulted at exactly that point. Keeping the two
apart is deliberate: an earlier gate treated both as "evidence lost" and the rescue then
fired 41 times on the attribute-paraphrase axis, competing for a fault it cannot fix
better and spending tokens to do it.

WHEN IT RUNS. Two conditions, both measured rather than assumed: the session has at least
one message the regex could not parse, and it has gone several consecutive turns without
learning anything new.

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

WHERE IT CAN AND CANNOT FIRE. Unreachable on the scored suites because the recognition gate
parses every official message, so the parse-failure count stays 0 -- control flow, not a
threshold. Measured reachability over 200 sessions per axis: clean 0, reworded values 0,
reworded wrappers 17, both reworded 65.

That first zero used to be 6. The gate was `turn >= 5 AND rejected >= 4`, which under
sequential disclosure collapses into "still going at turn 5", and it fired on clean traffic
where extraction had already captured every constraint verbatim -- 34 phrases accepted for
+0.000000. The same threshold then blocked all 97 eligible turns on the wrapper axis,
because `rejected` counts DISTINCT candidates shown and a stuck agent re-shows the same
few. It was unreachable exactly where it was needed.

    LLM_RESCUE=0        disable outright
    LLM_RESCUE_TURN     first turn it may fire (default 5)
    LLM_RESCUE_STALE    consecutive turns adding no evidence (default 2)
    LLM_RESCUE_REJECTS  retained for .env compatibility; no longer part of the gate
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from submission.llm_client import ENDPOINT, load_project_env

load_project_env()

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
# Keyed by transcript hash. Gitignored like the deparaphraser's, so a fresh clone
# starts empty and pays for what it uses; a repeated benchmark run does not.
CACHE_PATH = Path(os.environ.get(
    "LLM_RESCUE_CACHE", str(Path(__file__).resolve().parent / ".llm_rescue_cache.json")))

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
    MIN_REJECTS = int(_num("LLM_RESCUE_REJECTS", 4))   # retained: env-compatible, unused
    MIN_STALE = int(_num("LLM_RESCUE_STALE", 2))
    TERMINAL_STATUS = frozenset({400, 401, 403, 404, 413, 422})

    def __init__(self, model: str = DEFAULT_MODEL,
                 cache_path: Path = CACHE_PATH) -> None:
        flag = os.environ.get("LLM_RESCUE", "1").strip().lower()
        self._flag = flag not in {"0", "false", "no", "off"}
        self.model = model
        self.cache_path = cache_path
        self.reaches = self.calls = self.usable = 0
        self.accepted = self.unattested = self.failures = self.retries = 0
        self.cache_hits = self.cache_misses = 0
        self._dirty = False
        try:
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            self.cache = {}      # absent or corrupt: start empty, never fail to construct
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

    def should_fire(self, sid: str, turn: int, rejected: int, lost: int = 1,
                    stale: int = 99) -> bool:
        """May the rescue fire? `lost` counts EVIDENCE-LOSS EVENTS in this session.

        THE TURN AND REJECTION THRESHOLDS ARE A SYMPTOM, NOT A CAUSE, AND ALONE THEY GATE
        THE WRONG THING. Under sequential disclosure the agent shows exactly one candidate
        on turns 1-9, so a session still alive at turn 5 has necessarily accumulated four
        or five rejections: `turn >= 5` very nearly IMPLIES `rejected >= 4`, and the pair
        collapses into the single condition "still going at turn 5". Measured on the
        official public set, 7 of 200 sessions reach turn 5 and 6 of them opened this gate
        -- firing a whole-transcript re-read on traffic where template extraction had
        already captured every constraint verbatim. There was nothing to recover. Those
        sessions are slow for RANKING reasons, and the layer duly accepted 34 phrases for a
        score change of +0.000000.

        `lost` is the missing precondition: how many messages this session FAILED TO PARSE
        -- wording the recognition gate did not know, so template extraction never ran on
        it. Re-reading a transcript can only recover what the per-turn parse lost, so at
        `lost == 0` there is provably nothing to find and the call is pure cost.

        THIS COUNTS PARSE FAILURES ONLY, NOT UNATTESTED VALUES. A reworded VALUE inside a
        wrapper that parsed correctly belongs to the deparaphraser, which is consulted at
        exactly that point; this layer is a TEMPLATE solver. An earlier version of this
        gate counted suppressed value clauses as loss too, and the result was the rescue
        reaching 41 times on the attribute-paraphrase axis -- competing for the same fault
        with the layer that actually handles it, and spending tokens to do so.

        So reachability now matches purpose: zero on the exact-extraction benchmark, zero
        on reworded values alone, and open on reworded wrappers, where the parse is what
        broke.

        `stale` is the OTHER half, and it replaces `rejected` rather than joining it. The
        rejection count was meant to say "the session is not progressing", but it counts
        DISTINCT candidates already shown, and under paraphrase a stuck agent re-shows the
        same few items -- so it stops climbing exactly when the session is most stuck.
        Measured on the template axis: 97 turns reached turn 5 with evidence lost, and the
        rejection threshold blocked every one of them, because it never got past 3 while
        sessions ran to turn 10. A stall signal that saturates under stress is worse than
        none, because it silently makes the layer unreachable where it was needed.

        `stale` counts CONSECUTIVE TURNS THAT ADDED NO EVIDENCE, which is what "nothing
        new arriving" always meant. It rises precisely when the conversation continues and
        the agent learns nothing from it.

        Defaults keep any caller that passes neither on the old behaviour, and a caller
        that passes only `lost` still cannot fire without a stall.
        """
        return (self.enabled and self._open is None and sid not in self._done
                and turn >= self.MIN_TURN and lost > 0 and stale >= self.MIN_STALE)

    def _call(self, transcript: str) -> str | None:
        """The completion for this transcript, from cache when we have already paid for it.

        THE KEY IS THE TRANSCRIPT, which is the whole request: same system prompt, same
        model, same temperature 0, so the same conversation is the same question and the
        stored answer is the answer. Hashed rather than stored verbatim because transcripts
        are long, unlike the short values the deparaphraser caches.

        ONLY SUCCESSES ARE CACHED. A timeout, a 429 or a dropped connection is not an
        answer about the transcript, it is a fact about the network at that moment, and
        storing it would turn one bad minute into a permanently wrong result. An empty or
        unusable completion IS stored, because that is the model's answer.
        """
        if self._spent >= self.TIME_BUDGET:
            self._open = "time budget exhausted"
            return None
        key = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key] or None
        self.cache_misses += 1
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
                content = body["choices"][0]["message"]["content"]
                self.cache[key] = content or ""
                self._dirty = True
                return content
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

        # RETURN EVERY PROPOSAL, ATTESTED OR NOT. This used to drop whatever the catalogue
        # could not confirm -- 163 dropped against 150 kept on the compound axis, more than
        # half the output discarded. But an unattested proposal is not noise; it is a
        # reworded VALUE, which is the exact input the deparaphraser exists to resolve, and
        # binning it here meant the one layer that could rescue it never saw it.
        #
        # Attestation still happens, once, in `Agent._admit_value()`, together with the
        # deparaphrase fallback -- so this layer no longer decides admission on its own and
        # every candidate value in the agent takes the same path regardless of producer.
        # The df counters are kept for observability only: they no longer gate anything.
        out: list[str] = []
        for value in values:
            phrase = " ".join(str(value).lower().split()).strip(".,;:")
            if not phrase:
                continue
            try:
                if self._df is not None and self._df(phrase) > 0:
                    self.accepted += 1
                else:
                    self.unattested += 1
            except Exception:
                self.unattested += 1
            out.append(phrase)
        return out

    def stats(self) -> dict:
        return {"enabled": self.enabled, "model": self.model, "reaches": self.reaches,
                "calls": self.calls, "usable": self.usable, "accepted": self.accepted,
                "unattested": self.unattested, "failures": self.failures,
                "retries": self.retries,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "seconds": round(self._spent, 2),
                "circuit_reason": self._open}

    def flush(self) -> None:
        """Persist the cache. Callers opt in; the agent does not, on purpose.

        Writing files during a scored evaluation is a side effect, and the shipped agent
        stays free of it. Harnesses that replay the same suites call this so a second run
        costs nothing.
        """
        if not self._dirty:
            return
        try:
            self.cache_path.write_text(
                json.dumps(self.cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self._dirty = False
        except OSError:
            pass                     # a cache we cannot write is not a reason to fail
