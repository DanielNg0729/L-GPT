"""Isolated effort ablation of the actual shipped V2 agent.

This evaluator intentionally imports ``submission.agent.Agent`` rather than copying or
subclassing its extraction/ranking methods.  The tested execution path is therefore the
submitted path: template extraction, the eight-token deparaphrase limit, provenance gate,
weak semantic weight, span node, route node, and response loop are all untouched.

Only the outbound resolver payload differs: ``reasoning_effort='none'``.  The test suite
is attribute paraphrase only.  Canonical Official200 and Unseen800 are deliberately out
of scope because an exact catalogue lookup should handle them without a resolver request.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import MethodType
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "experiments" / "studies"
SUITE = V2 / "public_value_only" / "official200_attribute_paraphrase_dev.jsonl"
OUT = V2 / "results" / "shipped_v2_resolver_reasoning_none.json"
CACHE = V2 / ".shipped_v2_resolver_reasoning_none_cache.json"


def call_without_reasoning(self, phrase: str) -> str | None:
    """Byte-for-byte policy equivalent to LLMResolver._call except effort='none'."""
    print(f"[resolver-none] request {self.calls}: {phrase!r}", flush=True)
    if self._spent >= self.TIME_BUDGET:
        self._trip("time budget exhausted")
        return None
    payload = json.dumps({
        "model": self.model, "temperature": 0.0, "max_tokens": 512,
        "reasoning_effort": "none",
        "messages": [
            {"role": "system", "content": __import__("submission.llm_resolve", fromlist=["SYSTEM"]).SYSTEM},
            {"role": "user", "content": phrase},
        ],
    }).encode("utf-8")
    from submission.llm_rerank import ENDPOINT
    request = Request(ENDPOINT, data=payload, method="POST", headers={
        "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
        "Content-Type": "application/json", "User-Agent": "Mozilla/5.0",
    })
    for attempt in range(max(self.RETRIES, 1)):
        if self._spent >= self.TIME_BUDGET:
            self._trip("time budget exhausted")
            return None
        started = time.time()
        try:
            with urlopen(request, timeout=self.TIMEOUT) as response:
                body = json.loads(response.read().decode("utf-8"))
            self._spent += time.time() - started
            self._consecutive = 0
            print(f"[resolver-none] request {self.calls}: response received", flush=True)
            return body["choices"][0]["message"]["content"]
        except HTTPError as exc:
            self._spent += time.time() - started
            self.failures += 1
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            print(f"[resolver-none] request {self.calls}: HTTP {exc.code} {detail}",
                  flush=True)
            if exc.code in self.TERMINAL_STATUS:
                self._trip(f"HTTP {exc.code}")
                return None
        except Exception:
            self._spent += time.time() - started
            self.failures += 1
            print(f"[resolver-none] request {self.calls}: attempt {attempt + 1} failed",
                  flush=True)
        if attempt + 1 < max(self.RETRIES, 1):
            self.retries += 1
            time.sleep(min(self.BACKOFF ** attempt, 30.0))
    self._consecutive += 1
    if self._consecutive >= self.TRIP_AFTER:
        self._trip(f"{self._consecutive} consecutive failures")
    return None


def make_agent(resolver_enabled: bool):
    # These two optional LLM systems were disabled in the prior full-pipeline grid.
    # They are not part of this resolver-only ablation.
    os.environ["LLM_RERANK"] = "0"
    os.environ["LLM_EXTRACT"] = "0"
    os.environ["LLM_RESOLVE"] = "1" if resolver_enabled else "0"
    from submission.agent import Agent
    agent = Agent(ROOT / "data" / "catalog.jsonl")
    agent.llm = agent.llm_extract = None
    if resolver_enabled:
        if agent.resolver is None or not agent.resolver.enabled:
            raise RuntimeError("LLM resolver is not enabled. GROQ_API_KEY is required.")
        # No low-effort answers may be reused in this ablation.
        agent.resolver.cache_path = CACHE
        agent.resolver.cache = {}
        agent.resolver._call = MethodType(call_without_reasoning, agent.resolver)
    return agent


def score(agent, samples, catalog_ids, categories, products):
    from evaluator.local_evaluator import evaluate
    return evaluate(agent, samples, catalog_ids, categories, products)["recommended_technical_score"]


def main() -> None:
    from submission.llm_rerank import _load_project_env
    _load_project_env()
    if not os.environ.get("GROQ_API_KEY", "").strip():
        print("GROQ_API_KEY not set -- nothing to evaluate.")
        return
    from evaluator.local_evaluator import catalog_index, load_jsonl

    shipped = ROOT / "submission" / "agent.py"
    agent_sha256 = hashlib.sha256(shipped.read_bytes()).hexdigest()
    catalog_ids, categories, products = catalog_index(ROOT / "data" / "catalog.jsonl")
    samples = load_jsonl(SUITE)
    print(f"suite=Official200 AttributeParaphrase ({len(samples)} sessions)")
    print(f"shipped_agent_sha256={agent_sha256}", flush=True)

    t0 = time.time()
    floor_agent = make_agent(False)
    floor = score(floor_agent, samples, catalog_ids, categories, products)
    print(f"suppression_floor={floor:.6f}", flush=True)

    none_agent = make_agent(True)
    result = score(none_agent, samples, catalog_ids, categories, products)
    none_agent.resolver.flush()
    stats = none_agent.resolver.stats()
    print(f"reasoning_none={result:.6f} delta={result - floor:+.6f}", flush=True)
    print(json.dumps(stats, sort_keys=True), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "Actual shipped V2 resolver effort ablation",
        "suite": "Official200 AttributeParaphrase",
        "sessions": len(samples),
        "agent_path": "submission/agent.py",
        "agent_sha256": agent_sha256,
        "unchanged_components": ["template extraction", "eight-token limit", "provenance gate",
                                 "weak semantic weight", "span node", "route node", "ranking"],
        "controlled_change": {"resolver_reasoning_effort": "none"},
        "scores": {"suppression_floor": round(floor, 6),
                   "reasoning_none": round(result, 6),
                   "delta": round(result - floor, 6)},
        "resolver": stats,
        "seconds": round(time.time() - t0, 2),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()
