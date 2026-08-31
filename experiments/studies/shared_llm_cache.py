"""One response cache shared by every experiment that does not change the request.

THE RULE. A cached response is still that model's capability. If an experiment changes only
how an answer is USED -- the weight it enters at, the gate that admits it, the tier it is
stored under, the ranking around it -- then replaying a stored answer measures exactly what
a fresh call would, for nothing. Per-experiment caches re-buy the same answers.

That is not a small cost here. In a single session the same deparaphrase answers were paid
for three times, one cache was destroyed by a directory move, and the provider's
tokens-per-DAY allowance was exhausted twice -- 199,841 of 200,000 -- which blocked unrelated
work for hours and invalidated two experiments that had nothing to do with the model.

WHAT GOES IN THE KEY. Everything that changes what the model RETURNS:

    model id, max_tokens, reasoning effort, and the exact prompt text

`max_tokens` earns its place: it decides whether a structured response is truncated
mid-write, so a 1024-era answer must never satisfy a 3072 run. Model id likewise -- a shared
key once let a 120b answer serve a 20b lookup and report the two models as identical.

WHAT STAYS OUT. Anything downstream of the answer. Weight, tier, admission threshold,
ranking, disclosure. Those are what this cache exists to let you vary cheaply.

FAILURES ARE NEVER CACHED. A transient error stored as an answer becomes a permanent,
silent absence of evidence, indistinguishable from the model having nothing to say. This
has bitten twice: once here, once in an imported harness that cached whatever came back.

ONE LIMITATION IN THE MIGRATED ENTRIES. The shipped resolver caches the POST-GATE result,
not the raw completion: an abstention and a proposal the catalogue refused are both stored
as "". So of the 596 entries folded in from earlier per-experiment caches, the 150 empty
ones cannot distinguish "the model said NONE" from "the model proposed something the
catalogue rejected". That is enough to replay weight and ranking experiments, and NOT
enough to replay an admission-gate experiment, which needs the rejected proposal itself.
New entries written through this module should store the raw completion so that limitation
does not propagate.

THE LIMIT. A frozen set of responses invites tuning a threshold until it happens to suit
those particular answers. That is overfitting to a sample of the model, not measuring the
layer. Sweep coarsely, prefer a value that is flat across neighbours, and treat a sharp
optimum on cached data as evidence of nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = Path(os.environ.get(
    "SHARED_LLM_CACHE",
    str(ROOT / "experiments" / "datasets" / "prompt_arm_caches" / "shared_llm_cache.json")))

_lock = threading.Lock()
_cache: dict[str, Any] | None = None
stats = {"hits": 0, "misses": 0, "stores": 0, "failures_not_cached": 0}


def _load() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            _cache = {}
    return _cache


def key(model: str, prompt: str, *, max_tokens: int, effort: str = "low",
        extra: str = "") -> str:
    """Hash of everything that can change the model's answer, and nothing else."""
    payload = f"{model}\n{max_tokens}\n{effort}\n{extra}\n{prompt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_or_call(model: str, prompt: str, call: Callable[[], Any], *,
                max_tokens: int, effort: str = "low", extra: str = "",
                valid: Callable[[Any], bool] = lambda v: v is not None) -> Any:
    """Return a cached answer, or make the call and store it if it is valid.

    `valid` decides what counts as an answer worth keeping. Its default rejects None; pass
    something stricter for structured output, e.g. `lambda v: isinstance(v, dict)`.
    """
    cache = _load()
    k = key(model, prompt, max_tokens=max_tokens, effort=effort, extra=extra)
    if k in cache:
        stats["hits"] += 1
        return cache[k]
    stats["misses"] += 1
    value = call()
    if not valid(value):
        stats["failures_not_cached"] += 1
        return value
    with _lock:
        cache[k] = value
        stats["stores"] += 1
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(CACHE_PATH)      # atomic, so an interrupted run cannot corrupt it
    return value


def summary() -> dict:
    total = stats["hits"] + stats["misses"]
    return {**stats, "entries": len(_load()),
            "hit_rate": round(stats["hits"] / total, 4) if total else None,
            "path": str(CACHE_PATH)}


def merge(*paths: Path) -> int:
    """Fold older per-experiment caches in. Entries already present are left alone."""
    cache = _load()
    added = 0
    for p in paths:
        try:
            other = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        for k, v in other.items():
            if k not in cache and v not in (None, "", {}):
                cache[k] = v
                added += 1
    if added:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n",
                              encoding="utf-8")
    return added
