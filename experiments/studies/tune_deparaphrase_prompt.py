"""Three-arm A/B of the deparaphraser's USER prompt, on the attribute axis only.

FROZEN: the agent, the resolver module, the system prompt, the tier weights, the template
axis. The ONLY thing that varies between arms is how the user message is assembled.

  A  attr only                the shipped prompt: the bare unattested clause
  B  attr + full message      the clause plus the message it was extracted from
  C  attr + category + turns  structured, with sibling constraints explicitly labelled
                              as context and NOT as the answer

WHY C IS STRUCTURED RATHER THAN A TRANSCRIPT DUMP. Context is not free. Measured on this
suite, the message enclosing an unattested clause routinely carries a SIBLING constraint
that IS catalogue-attested, sitting beside a clause that negates it:

    attr    : contains no animal products
    context : For that, what matters is: leather; contains no animal products.

That is the failure that sank the retrieval variant -- a plausible attested answer adjacent
to the question gets copied whatever the instruction says. A raw transcript maximises the
exposure: by turn 5 it carries roughly eight attested siblings. C therefore names the
CATEGORY -- the one thing a turn-2 message never carries, and worth about two thirds of the
task by ablation -- while labelling the siblings as excluded.

CORRECTNESS IS PROVENANCE, NOT STRING EQUALITY. A proposal counts as correct iff it appears
in the TARGET product's own catalogue text. That is the property the whole agent rests on,
and it needs no paraphrase->canonical mapping. Accept-rate alone would rank an arm that
proposes confident wrong canonicals ABOVE one that abstains, which is backwards: an accepted
wrong proposal enters the ledger at W_SEM and pulls ranking toward the wrong product.

THE NEGATION SUBSET IS TOO SMALL TO DECIDE ANYTHING. Five distinct negation-shaped values
over eight invocations. It is reported as a flag, never as a verdict.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/studies/tune_deparaphrase_prompt.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ["LLM_EXTRACT"] = "0"
os.environ["LLM_RERANK"] = "0"
os.environ["LLM_RESOLVE"] = "1"
# THE FIRST RUN OF THIS HARNESS WAS INVALID AND THE FAILURE IS WORTH RECORDING. Arms B and
# C tripped the shipped breaker after 7 and 11 calls -- `6 consecutive failures` -- against
# arm A's 170, and both landed on exactly the no-resolver baseline (0.847103). The harness
# still printed a confident `-0.017141` for each. A dead arm does not look dead; it looks
# like "context does not help", which is the conclusion this experiment exists to test.
#
# The cause is rate limiting, not the prompts. Arm A absorbed 139 scattered failures and
# survived because the breaker counts CONSECUTIVE ones; by arm B the limiter was hot enough
# to produce six in a row. Failures are never cached, so a rerun retries exactly the phrases
# that failed and keeps the work that succeeded.
#
# Two changes, both harness-only -- the shipped module and its defaults are untouched:
# a loosened breaker, and PACING. Pacing matters more: a retry storm against a rate limiter
# is slower than simply issuing calls below the limit.
os.environ.setdefault("LLM_RESOLVE_TIME_BUDGET", "6000")
os.environ.setdefault("LLM_RESOLVE_TRIP_AFTER", "40")
os.environ.setdefault("LLM_RESOLVE_RETRIES", "4")
os.environ.setdefault("LLM_RESOLVE_BACKOFF", "2")
# 2.0s == 30 requests/minute. Chosen to sit UNDER the provider limit rather than discover
# it through failures. Note TIME_BUDGET cannot bound a run on its own: `_spent` accumulates
# request time only, so sleep in backoff is invisible to it.
PACE_SECONDS = float(os.environ.get("LLM_RESOLVE_PACE", "2.0"))

# Arms are compared on an identical subsample. Arm B assembles a distinct prompt per
# (value, message) and arm C per (value, category, transcript), so the full 800 sessions
# cost 711 and ~1109 calls against arm A's 170 -- the same experiment at 6x the price. The
# subsample is a prefix, so every arm sees exactly the same sessions in the same order.
SUITE_LIMIT = int(os.environ.get("DEPARAPHRASE_SUITE_LIMIT", "250"))

# MODEL, AND WHY THE CACHE MUST BE KEYED BY IT. The resolver caches on the user message
# alone -- it has no reason to do otherwise, since the shipped agent runs one model. Here
# the model is the variable, so a shared cache file would let a 120b answer satisfy a 20b
# lookup and report a perfect match between two models that were never both asked. Every
# cache path below carries the model slug for exactly that reason.
MODEL = os.environ.get("DEPARAPHRASE_MODEL", "openai/gpt-oss-120b")
MODEL_SLUG = re.sub(r"[^a-z0-9]+", "_", MODEL.lower()).strip("_")
ARMS = os.environ.get("DEPARAPHRASE_ARMS", "ABC").upper()
# max_tokens belongs in the cache key for the same reason the model does: it is a request
# parameter that can change the answer (too small a budget truncates the hidden reasoning
# and returns empty), while the cache keys only on the user message. A 512-generated answer
# served to a 160-configured run would silently report parity between two settings that
# were never both asked.
MAX_TOKENS = int(os.environ.get("LLM_RESOLVE_MAX_TOKENS", "512"))
RUN_SLUG = f"{MODEL_SLUG}__mt{MAX_TOKENS}"

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import CAT, Agent, raw_toks  # noqa: E402
from submission.llm_resolve import LLMResolver  # noqa: E402

SUITE = (ROOT / "experiments" / "studies" / "open_vocabulary"
         / "review800_open_vocab_paraphrase.jsonl")
OUT = ROOT / "experiments" / "results" / "out_72_deparaphrase_prompt_arms.json"
CACHE_DIR = ROOT / "experiments" / "studies" / "prompt_arm_caches"
NEG = re.compile(r"\b(no|non|without|free|never|not|avoid|instead of|rather than)\b", re.I)

EXAMPLE = {
    "msg": "For that, what matters is: leather; in a neutral middle tone.",
    "turns": ("Hi, I need Women Jeans.",
              "For that, what matters is: leather; in a neutral middle tone."),
    "category": "women jeans",
    "siblings": ["leather"],
}


def build_user(arm: str, attr: str, ctx: dict) -> str:
    """The only thing that differs between arms."""
    if arm == "A":
        return attr
    if arm == "B":
        return (f'Attribute to name: "{attr}"\n'
                f'Full message it came from: "{ctx.get("msg", "")}"')
    siblings = [s for s in ctx.get("siblings") or () if s.strip()]
    lines = [f'Category: {ctx.get("category") or "unknown"}',
             f'Attribute to name: "{attr}"']
    if siblings:
        lines.append("Other requirements already stated "
                     "(context only, NOT the answer): " + "; ".join(siblings))
    turns = ctx.get("turns") or ()
    if turns:
        lines.append("Conversation so far:")
        lines.extend(f"  turn {i}: {t}" for i, t in enumerate(turns, 1))
    return "\n".join(lines)


class ArmResolver(LLMResolver):
    """Same system prompt, same acceptance gate. Only the user message differs.

    Each arm gets its own cache file: the cache is keyed by the user message, so a shared
    cache would let arm A's answers satisfy arm B's lookups for any phrase whose assembled
    prompt happened to collide, and silently flatten the comparison.
    """

    def __init__(self, arm: str, cache_path: Path, model: str = MODEL) -> None:
        super().__init__(model=model, cache_path=cache_path)
        self.arm = arm
        self.proposals: list[dict] = []
        self.prompts: list[str] = []
        self._since_report = 0

    def _call(self, phrase: str):
        # PACING IS THE POINT, NOT THE RETRIES. A first attempt at this harness set
        # RETRIES=8/BACKOFF=3 to stop the breaker tripping, which costs 160s of sleep per
        # fully-failing call against the shipped 7s -- and paced at 0.35s (171 req/min),
        # far above the provider limit, so essentially every call was rate-limited and paid
        # that ladder. The run used 20 seconds of CPU in 25 minutes and produced nothing.
        # Staying UNDER the limit finishes far faster than fighting it with backoff.
        if PACE_SECONDS > 0:
            time.sleep(PACE_SECONDS)
        out = super()._call(phrase)
        self._since_report += 1
        if self._since_report >= 25:
            self._since_report = 0
            # Flush as we go: the arm otherwise writes its cache only on completion, so an
            # interrupted run loses every answer it paid for.
            self.flush()
            print(f"      [{self.arm}] {self.calls} calls, {self.accepted} accepted, "
                  f"{self.abstained} abstained, {self.failures} failures, "
                  f"{self._spent:.0f}s in-request", flush=True)
        return out


def make_agent_class(resolver: ArmResolver, targets: list[str]):
    class ArmAgent(Agent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.resolver = resolver.bind(self.ix.df)
            self._turns: dict[str, list[str]] = {}
            self._st = None
            self._idx = -1
            self._target: str | None = None

        def reset(self, session_id, user_profile=None):
            # `evaluate` walks samples in order and resets once per sample, so the index
            # identifies the target. `respond` carries no sample id to key off.
            self._idx += 1
            self._target = targets[self._idx] if self._idx < len(targets) else None
            return super().reset(session_id, user_profile)

        def _observe(self, st, msg):
            self._st = st
            self._turns.setdefault(st.sid, []).append(msg)
            return super()._observe(st, msg)

        def _deparaphrase(self, text):
            r = self.resolver
            if r is None or not r.enabled:
                return None
            toks = raw_toks(text)
            if not toks or len(toks) > self.DEPARAPHRASE_MAX_TOKENS:
                return None
            st = self._st
            turns = tuple(self._turns.get(st.sid, ())) if st is not None else ()
            category = None
            siblings: list[str] = []
            if st is not None:
                for phrase, (_df, tier) in st.evidence.items():
                    if tier == CAT:
                        if category is None:
                            category = phrase
                    else:
                        siblings.append(phrase)
            ctx = {"msg": turns[-1] if turns else "", "turns": turns,
                   "category": category, "siblings": siblings[:6]}
            try:
                out = r.resolve(build_user(r.arm, text, ctx))
            except Exception:
                return None
            r.prompts.append(build_user(r.arm, text, ctx))
            r.proposals.append({"attr": text, "proposal": out, "target": self._target,
                                "negation": bool(NEG.search(text))})
            return out

    return ArmAgent


def main() -> None:
    ids, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    samples = load_jsonl(SUITE)[:SUITE_LIMIT]
    targets = [str(s["ground_truth"]["parent_asin"]) for s in samples]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict] = {}

    print(f"suite: {SUITE.name}  ({len(samples)} sessions)")
    print(f"model: {MODEL}   arms: {ARMS}   pace: {PACE_SECONDS}s   max_tokens: {MAX_TOKENS}")
    print(f"{'arm':<4}{'prompt':<28}{'calls':>7}{'acc':>6}{'abst':>6}{'unatt':>7}"
          f"{'fail':>6}{'prec':>8}{'score':>11}")
    print("-" * 83)

    for arm, label in (("A", "attr only (shipped)"),
                       ("B", "attr + full message"),
                       ("C", "attr + category + turns")):
        if arm not in ARMS:
            continue
        res = ArmResolver(arm, CACHE_DIR / f"arm_{arm}__{RUN_SLUG}.json")
        agent = make_agent_class(res, targets)(ROOT / "data" / "catalog.jsonl")
        result = evaluate(agent, samples, ids, cats, prods)
        res.flush()

        checked = [p for p in res.proposals if p["proposal"] and p["target"]]
        correct = sum(1 for p in checked
                      if f' {p["proposal"]} ' in agent.ix.blob.get(p["target"], ""))
        precision = (correct / len(checked)) if checked else None
        stats = res.stats()
        rows[arm] = {
            "label": label,
            "score": result["recommended_technical_score"],
            "hit_rate_at_10": result["hit_rate_at_10"],
            "mrr": result["mrr"],
            "calls": stats["calls"], "accepted": stats["accepted"],
            "abstained": stats["abstained"], "unattested": stats["unattested"],
            "failures": stats["failures"], "circuit_reason": stats["circuit_reason"],
            # EFFICIENCY. `calls` counts cache MISSES only -- `resolve` increments it after
            # the cache check -- so it is the number of prompts this run had to send. It
            # therefore depends on how warm the cache already was, which makes it a poor
            # basis for comparing arms across runs.
            #
            # `distinct_prompts` is the cache-state-INDEPENDENT version: how many network
            # calls this arm needs from cold, no matter when it runs. That is the honest
            # cost comparison, and it is the number that separates the arms -- arm A keys
            # on the value alone, arm B on (value, message), arm C on (value, category,
            # transcript), so the same 250 sessions cost progressively more.
            #
            # `http_attempts` adds retries: the requests actually issued at the provider,
            # which is what a rate limiter sees.
            "invocations": len(res.prompts),
            "distinct_prompts": len(set(res.prompts)),
            "cache_hits": stats["cache_hits"], "cache_misses": stats["cache_misses"],
            "cache_hit_rate": stats["cache_hit_rate"],
            "retries": stats["retries"],
            "http_attempts": stats["calls"] + stats["retries"],
            "seconds": stats["seconds"],
            "prompts_per_invocation": (round(len(set(res.prompts)) / len(res.prompts), 3)
                                       if res.prompts else None),
            "checked": len(checked), "correct": correct, "precision": precision,
            "negation_accepted": sum(1 for p in checked if p["negation"]),
            "example_prompt": build_user(arm, "in a neutral middle tone", EXAMPLE),
        }
        shown = f"{precision:.3f}" if precision is not None else "n/a"
        print(f"{arm:<4}{label:<28}{stats['calls']:>7}{stats['accepted']:>6}"
              f"{stats['abstained']:>6}{stats['unattested']:>7}{stats['failures']:>6}"
              f"{shown:>8}{result['recommended_technical_score']:>11.6f}", flush=True)

    print(f"\nefficiency  (cold-cache cost is `distinct`, independent of cache warmth)")
    print(f"{'arm':<4}{'invoc':>7}{'distinct':>10}{'per-inv':>9}{'hits':>7}{'miss':>7}"
          f"{'retry':>7}{'http':>7}{'secs':>9}")
    print("-" * 67)
    for arm in [a for a in ("A", "B", "C") if a in rows]:
        r = rows[arm]
        print(f"{arm:<4}{r['invocations']:>7}{r['distinct_prompts']:>10}"
              f"{(r['prompts_per_invocation'] if r['prompts_per_invocation'] is not None else 0):>9.3f}"
              f"{r['cache_hits']:>7}{r['cache_misses']:>7}{r['retries']:>7}"
              f"{r['http_attempts']:>7}{r['seconds']:>9.1f}")
    if "A" in rows:
        a_d = rows["A"]["distinct_prompts"] or 1
        for arm in [a for a in ("B", "C") if a in rows]:
            print(f"  arm {arm} costs {rows[arm]['distinct_prompts'] / a_d:.2f}x arm A's "
                  f"cold-cache calls for the same {len(samples)} sessions")

    # VALIDITY BEFORE COMPARISON. An arm whose breaker opened stopped consulting the model
    # partway and drifts toward the no-resolver baseline, which reads as "this prompt is
    # worse" rather than "this arm died". The first run of this harness reported exactly
    # that: two tripped arms, two confident negative deltas. Refuse to subtract instead.
    base = rows.get("A")
    dead = {a: r["circuit_reason"] for a, r in rows.items() if r["circuit_reason"]}
    for arm, reason in dead.items():
        print(f"\n  !! arm {arm} INVALID: circuit opened ({reason}) after "
              f"{rows[arm]['calls']} calls. Its score is not a measurement of its prompt.")
    thin = {} if base is None else {
        a: r for a, r in rows.items()
        if a not in dead and a != "A" and r["distinct_prompts"]
        and r["calls"] < 0.5 * min(r["distinct_prompts"], base["distinct_prompts"] or 1)}
    for arm, row in thin.items():
        print(f"\n  !! arm {arm} SUSPECT: {row['calls']} calls against "
              f"{row['distinct_prompts']} distinct prompts. Too few to compare.")

    if base is None or len(rows) < 2:
        print("\n  Single arm run; nothing to compare against in this process.")
    elif dead or thin or base["circuit_reason"]:
        print("\n  No deltas reported: at least one arm did not complete. Rerun the")
        print("  affected arms -- failures are never cached, so completed work is kept.")
    else:
        print("\ndelta against the shipped prompt")
        for arm in [a for a in ("B", "C") if a in rows]:
            row = rows[arm]
            dscore = row["score"] - base["score"]
            if row["precision"] is None or base["precision"] is None:
                dprec = "n/a"
            else:
                dprec = f"{row['precision'] - base['precision']:+.3f}"
            print(f"  {arm}  score {dscore:+.6f}   precision {dprec}   "
                  f"accepted {row['accepted'] - base['accepted']:+d}")
        print("\n  An arm that raises `accepted` while dropping `prec` is proposing more")
        print("  confident wrong canonicals. That is a regression even if the score rises,")
        print("  because the next suite will not repeat this one's luck.")

    out_path = OUT.with_name(f"{OUT.stem}__{RUN_SLUG}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "experiment": "deparaphrase user-prompt arms",
        "model": MODEL, "arms_run": ARMS, "max_tokens": MAX_TOKENS,
        "suite": str(SUITE.relative_to(ROOT)).replace("\\", "/"),
        "sessions": len(samples), "suite_limit": SUITE_LIMIT,
        "pace_seconds": PACE_SECONDS,
        "all_arms_valid": not any(r["circuit_reason"] for r in rows.values()),
        "frozen": "agent, system prompt, tier weights, template axis",
        "correctness": "proposal attested in the TARGET product's catalogue text",
        "negation_subset_note": "5 distinct values over 8 invocations; flag, not a verdict",
        "arms": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
