"""EDA pass 44: what happens when the LLM layer is entirely dead?

The layer is optional, so the only property that actually matters in production is this:
NO failure mode may make the agent worse or slower than it is with the layer switched off.
"Optional" is a claim about the failure path, and an untested failure path is not optional,
it is a liability -- the submission rules warn that exceptions, invalid output and timeouts
"may count as a miss".

Every way the endpoint can fail is simulated by substituting `urlopen` inside the extractor
module. Nothing else changes: the same agent, the same harness, the same sessions.

    disabled            LLM_EXTRACT unset -- the shipped default
    no_key              flag on, GROQ_API_KEY absent
    network_down        URLError on every attempt (no route, DNS dead, offline)
    conn_refused        ConnectionRefusedError (endpoint present, nothing listening)
    timeout             socket read timeout on every attempt
    http_401            invalid credential
    http_403            forbidden (wrong region, revoked key)
    http_404            model name wrong or withdrawn
    http_500            server error
    http_429            permanently rate limited / quota exhausted
    malformed_json      HTTP 200 with a body that is not JSON
    wrong_schema        HTTP 200, valid JSON, unexpected shape
    empty_content       HTTP 200 with an empty completion (the GPT-OSS reasoning trap)
    hallucinating       HTTP 200 with plausible spans that are NOT in the message
    garbage             HTTP 200 with a wall of unrelated text
    slow_then_dead      succeeds slowly a few times, then fails forever

The bar for every row: score EXACTLY equal to the deterministic baseline, no exception
reaching the agent, and bounded wall clock. `hallucinating` is the interesting one -- it is
the only mode where the endpoint is healthy and the OUTPUT is the attack, and it is caught
by the verbatim substring check rather than by the breaker.

Run:  PYTHONIOENCODING=utf-8 python -u notes/eda/44_llm_failure_modes.py
"""
from __future__ import annotations

import io
import json
import socket
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notes" / "eda"))

import submission.llm_extract as LE  # noqa: E402
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent  # noqa: E402

_p31 = __import__("31_paraphrase_stress")
evaluate_transformed, TRANSFORMS = _p31.evaluate_transformed, _p31.TRANSFORMS

N_SESSIONS = 40
COND = "T1 scaffold reworded"     # a condition where the gate routes EVERY message to the LLM


def _body(payload: dict) -> io.BytesIO:
    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    return R(json.dumps(payload).encode("utf-8"))


def _ok(content: str):
    return _body({"choices": [{"message": {"content": content}}],
                  "usage": {"prompt_tokens": 100, "completion_tokens": 20}})


def _http(code: int):
    def f(*a, **k):
        raise HTTPError("https://x", code, "err", {}, None)
    return f


def make_fake(mode: str):
    state = {"n": 0}

    def fake(request, timeout=None):
        state["n"] += 1
        if mode == "network_down":
            raise URLError("[Errno -2] Name or service not known")
        if mode == "conn_refused":
            raise ConnectionRefusedError("connection refused")
        if mode == "timeout":
            raise socket.timeout("timed out")
        if mode.startswith("http_"):
            _http(int(mode.split("_")[1]))()
        if mode == "malformed_json":
            return _body({}).__class__(b"<html>502 Bad Gateway</html>")
        if mode == "wrong_schema":
            return _body({"unexpected": True})
        if mode == "empty_content":
            return _ok("")
        if mode == "hallucinating":
            # Healthy endpoint, adversarial OUTPUT: real catalogue vocabulary that the
            # customer never said. Only the verbatim check stands between this and the
            # evidence ledger.
            return _ok("100% Cotton\nMachine Wash\nImported\nPull On closure\nSolid colors")
        if mode == "garbage":
            return _ok("I'm sorry, as an AI language model I cannot\n" + ("x" * 4000))
        if mode == "slow_then_dead":
            if state["n"] <= 3:
                time.sleep(0.05)
                return _ok("cotton")
            raise URLError("gone")
        raise AssertionError(mode)
    return fake


def main() -> None:
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")[:N_SESSIONS]
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")

    def run(extractor):
        o = object.__new__(Agent)
        o.ix, o.sessions, o.llm = base.ix, {}, None
        o.llm_extract = extractor
        t = time.time()
        r = evaluate_transformed(o, samples, cid, cats, prods, TRANSFORMS[COND])
        return r["recommended_technical_score"], time.time() - t

    baseline, base_secs = run(None)
    print(f"deterministic baseline ({COND}, {N_SESSIONS} sessions): "
          f"{baseline:.5f} in {base_secs:.1f}s")
    print("  every row below must match it EXACTLY and stay bounded in time\n")

    MODES = ["disabled", "no_key", "network_down", "conn_refused", "timeout",
             "http_401", "http_403", "http_404", "http_500", "http_429",
             "malformed_json", "wrong_schema", "empty_content", "hallucinating",
             "garbage", "slow_then_dead"]

    print(f"{'failure mode':<17}{'score':>10}{'vs base':>10}{'secs':>7}{'calls':>7}"
          f"{'fails':>7}  breaker")
    print("-" * 88)
    OUT, bad = {}, []
    real_urlopen, real_env = LE.urlopen, LE.os.environ.get("GROQ_API_KEY")
    try:
        for mode in MODES:
            LE.os.environ["LLM_EXTRACT"] = "0" if mode == "disabled" else "1"
            if mode == "no_key":
                LE.os.environ.pop("GROQ_API_KEY", None)
            elif real_env:
                LE.os.environ["GROQ_API_KEY"] = real_env
            else:
                LE.os.environ["GROQ_API_KEY"] = "sk-test-not-a-real-key"

            # Isolated cache: a warm cache would mask the failure path entirely.
            ex = LE.LLMExtractor(cache_path=ROOT / "notes" / "eda" / f".fm_{mode}.json")
            ex.cache = {}
            ex.TIME_BUDGET = 60.0
            ex.ZERO_YIELD_TRIP = 12   # scaled to this 40-session probe
            ex.limiter.rpm, ex.limiter.tpm = 10**6, 10**9  # isolate the breaker,
            # not our own throttle: the point is to time the FAILURE path.
            LE.urlopen = make_fake(mode) if mode not in ("disabled", "no_key") else real_urlopen

            try:
                score, secs = run(ex)
                err = None
            except Exception as exc:                      # must never happen
                score, secs, err = float("nan"), 0.0, f"{type(exc).__name__}: {exc}"

            st = ex.stats()
            delta = score - baseline
            OUT[mode] = {"score": score, "delta": delta, "secs": round(secs, 1),
                         "stats": st, "exception": err}
            brk = st["circuit_reason"] or ("-" if not st["enabled"] else "closed")
            print(f"{mode:<17}{score:>10.5f}{delta:>+10.5f}{secs:>7.1f}"
                  f"{st['api_calls']:>7}{st['failures']:>7}  {brk}")
            if err or abs(delta) > 1e-12:
                bad.append(mode)
    finally:
        LE.urlopen = real_urlopen
        if real_env:
            LE.os.environ["GROQ_API_KEY"] = real_env
        LE.os.environ.pop("LLM_EXTRACT", None)
        for mode in MODES:
            (ROOT / "notes" / "eda" / f".fm_{mode}.json").unlink(missing_ok=True)

    print("\n" + "=" * 88)
    if bad:
        print(f"  FAIL -- these modes changed the score or raised: {bad}")
    else:
        print("  PASS -- every failure mode degrades to the exact deterministic score,")
        print("          no exception reaches the agent, and the breaker bounds the wait.")
    slowest = max(OUT, key=lambda m: OUT[m]["secs"])
    print(f"  slowest mode: {slowest} at {OUT[slowest]['secs']:.1f}s "
          f"vs {base_secs:.1f}s offline "
          f"(x{OUT[slowest]['secs']/max(base_secs,1e-9):.1f})")
    hall = OUT.get("hallucinating", {}).get("stats", {})
    print(f"  hallucinated spans admitted to the ledger: "
          f"{hall.get('failures', 0)} rejected of {hall.get('api_calls', 0)} calls "
          f"-- score identical, so none survived the verbatim check")

    (ROOT / "notes" / "eda" / "out_44.json").write_text(
        json.dumps(OUT, indent=2, default=str) + "\n", encoding="utf-8")
    print("\n[saved] notes/eda/out_44.json")


if __name__ == "__main__":
    main()
