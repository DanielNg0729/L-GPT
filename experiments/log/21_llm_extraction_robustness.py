"""Compare LLM extraction with catalogue-grounded mining under message paraphrase.

The evaluator and hidden target are untouched.  Only the text passed to the agent is
wrapped, mirroring the specification's statement that organizer paraphrasing cannot
change correctness.  The LLM receives only that visible message; it never receives the
target ASIN, intent card, catalogue row, evaluator state, or previous recommendations.

This is an analysis experiment, not a submission dependency.  It does not alter
submission/agent.py or its cost.  Results are cached locally.

Run a small stratified pilot (default):
  python experiments/log/21_llm_extraction_robustness.py --online

Run the full official 200-session harness with LLM extraction only:
  python experiments/log/21_llm_extraction_robustness.py --online --all --levels exact
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent, CAT, CONSTRAINT, MINED, PAT_NOINFO  # noqa: E402
from submission.llm_rerank import _load_project_env  # noqa: E402

_load_project_env()
CACHE = ROOT / "experiments" / "studies" / ".llm_extraction_cache.json"
OUT = ROOT / "experiments" / "results" / "out_21_llm_extraction.json"
ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM = """Extract only requirements explicitly expressed in the shopper message.
Do not infer unstated preferences. Return strict JSON with exactly these keys:
{"category": string, "requirements": [string]}. A requirement must be a short,
searchable phrase, not an explanation. If none is stated, use empty strings/lists."""


class GroqExtractor:
    def __init__(self, online: bool) -> None:
        self.online = online and bool(os.environ.get("GROQ_API_KEY"))
        try:
            self.cache = json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:
            self.cache = {}
        self.calls = self.cache_hits = self.failures = 0
        self.errors: dict[str, int] = defaultdict(int)
        self._last_request = 0.0

    def _throttle(self) -> None:
        """Stay below the conservative free-tier request rate without bursting."""
        # Extraction prompts are substantially larger than the reranking prompts.
        # Five seconds keeps both RPM and free-tier TPM below their conservative caps.
        wait = 5.0 - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def extract(self, message: str) -> tuple[str, list[str]]:
        prompt = f"Shopper message:\n{message}\n\nJSON only:"
        key = hashlib.sha256((MODEL + "\0" + prompt).encode()).hexdigest()
        raw = self.cache.get(key)
        if raw is not None:
            self.cache_hits += 1
        elif not self.online:
            return "", []
        else:
            self.calls += 1
            payload = json.dumps({
                "model": MODEL, "temperature": 0, "max_tokens": 160,
                "reasoning_effort": "low",
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": prompt}],
            }).encode("utf-8")
            for attempt in range(3):
                try:
                    self._throttle()
                    request = Request(ENDPOINT, data=payload, method="POST", headers={
                        "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
                        "Content-Type": "application/json", "User-Agent": "Mozilla/5.0",
                    })
                    with urlopen(request, timeout=20) as response:
                        raw = json.loads(response.read().decode())["choices"][0]["message"]["content"]
                    self.cache[key] = raw
                    CACHE.write_text(json.dumps(self.cache), encoding="utf-8")
                    break
                except HTTPError as error:
                    self.errors[f"http_{error.code}"] += 1
                    if error.code == 429 and attempt < 2:
                        time.sleep(10 * (attempt + 1))
                        continue
                    self.failures += 1
                    return "", []
                except (URLError, OSError, ValueError, KeyError, IndexError, TypeError) as error:
                    self.errors[type(error).__name__] += 1
                    self.failures += 1
                    return "", []
        try:
            parsed = json.loads(raw)
            category = str(parsed.get("category") or "")[:180]
            requirements = [str(item)[:180] for item in parsed.get("requirements", []) if isinstance(item, str)]
            return category, requirements[:4]
        except (TypeError, ValueError) as error:
            self.errors[type(error).__name__] += 1
            self.failures += 1
            return "", []


class MiningOnly(Agent):
    """No templates: retain only phrases literally attested in the catalogue."""
    def _observe(self, st, msg: str) -> None:
        if PAT_NOINFO.search(msg):
            return
        for phrase, df in self.ix.mine(msg):
            if phrase not in st.evidence:
                st.evidence[phrase] = (df, MINED)


class LLMOnly(Agent):
    """Visible-message-only LLM extraction, then the same local phrase resolver/ranker."""
    extractor: GroqExtractor

    def _observe(self, st, msg: str) -> None:
        if PAT_NOINFO.search(msg):
            return
        category, requirements = self.extractor.extract(msg)
        for text, tier in ([(category, CAT)] if category else []) + [(item, CONSTRAINT) for item in requirements]:
            for phrase in self._resolve(text):
                if phrase not in st.evidence:
                    df = self.ix.df(phrase)
                    st.evidence[phrase] = (df if df > 0 else self.ix.DF_CAP * 2, tier)


class Paraphraser:
    """Target-preserving message wrappers; exact requirements are never edited."""
    def __init__(self, inner, level: str, seed: int = 21) -> None:
        self.inner, self.level, self.rng = inner, level, random.Random(seed)

    def reset(self, session_id, user_profile):
        self.inner.reset(session_id, user_profile)

    def transform(self, message: str) -> str:
        if self.level == "exact":
            return message
        out = re.sub(r"I'm looking for", "I'm after", message, flags=re.I)
        out = re.sub(r"A key requirement is:", "One thing it needs:", out, flags=re.I)
        out = re.sub(r"For that, what matters is:", "The important part:", out, flags=re.I)
        if self.level == "surface":
            return "Okay, " + out
        # Clauses are reordered and punctuation structure removed, but the shopper's
        # literal category/requirement strings are left intact and correctness remains
        # the original evaluator target.
        parts = [part.strip() for part in out.replace(":", "").split(".") if part.strip()]
        self.rng.shuffle(parts)
        return "Let me put it another way  -  " + ". ".join(parts) + "."

    def respond(self, session_id, user_message, turn, top_k):
        return self.inner.respond(session_id, self.transform(user_message), turn, top_k)


def subset(samples: list[dict], per_scenario: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        groups[sample["scenario_type"]].append(sample)
    return [sample for group in groups.values() for sample in group[:per_scenario]]


def compact(result: dict) -> dict:
    return {key: result[key] for key in ("sample_count", "hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true", help="allow Groq extraction calls")
    parser.add_argument("--per-scenario", type=int, default=10, help="stratified pilot size per scenario")
    parser.add_argument("--all", action="store_true", help="use all 200 released sessions")
    parser.add_argument("--levels", default="exact,surface,reordered", help="comma-separated conditions to run")
    parser.add_argument("--methods", default="mining,llm", help="comma-separated mining,llm,hybrid")
    parser.add_argument("--output", default=str(OUT), help="result JSON path")
    args = parser.parse_args()
    all_samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    samples = all_samples if args.all else subset(all_samples, args.per_scenario)
    levels = [level.strip() for level in args.levels.split(",") if level.strip() in {"exact", "surface", "reordered"}]
    if not levels:
        parser.error("--levels must include exact, surface, or reordered")
    methods = [name.strip() for name in args.methods.split(",") if name.strip() in {"mining", "llm", "hybrid"}]
    if not methods:
        parser.error("--methods must include mining, llm, or hybrid")
    ids, cats, products = catalog_index(ROOT / "data" / "catalog.jsonl")
    mining = MiningOnly(ROOT / "data" / "catalog.jsonl")
    # Both policies must see the identical index; avoid a second ~700 MB build.
    llm = object.__new__(LLMOnly)
    llm.ix, llm.sessions, llm.llm = mining.ix, {}, None
    llm.extractor = GroqExtractor(args.online)
    # The submitted hybrid agent shares the same index, but its optional online
    # tie-breaker is hard-disabled for this extraction benchmark.
    os.environ["LLM_RERANK"] = "0"
    hybrid = object.__new__(Agent)
    hybrid.ix, hybrid.sessions, hybrid.llm = mining.ix, {}, None
    agents = {"mining": mining, "llm": llm, "hybrid": hybrid}
    results: dict[str, dict] = {}
    for level in levels:
        for name in methods:
            agent = agents[name]
            agent.sessions = {}
            wrapped = Paraphraser(agent, level)
            result = evaluate(wrapped, samples, ids, cats, products)
            results[f"{name}|{level}"] = compact(result)
            print(f"{name:<6} {level:<9} {result['recommended_technical_score']:.6f}")
    report = {"scope": "target/evaluator unchanged; LLM sees visible message only", "sample_count": len(samples),
              "conditions": {
                  "exact": "Official customer text, unchanged.",
                  "surface": "Template wording and discourse prefix rewritten; requirement strings preserved.",
                  "reordered": "Surface rewrite plus clause-order/punctuation disruption; requirement strings preserved.",
              },
              "results": results, "llm_stats": {"calls": llm.extractor.calls, "cache_hits": llm.extractor.cache_hits, "failures": llm.extractor.failures, "errors": dict(llm.extractor.errors), "cached_entries": len(llm.extractor.cache)}}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["llm_stats"], indent=2))


if __name__ == "__main__":
    main()
