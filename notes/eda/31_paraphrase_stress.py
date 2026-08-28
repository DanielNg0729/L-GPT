"""EDA pass 31: adversarial paraphrase stress test.

The specification leaves exactly one thing about the private harness genuinely open
(competition_specification.md, line 40, verbatim):

    "The simulator policy decides what information to reveal. If natural-language
     paraphrasing is added by the organizer, it cannot decide correctness. Hits are
     always exact code matches."

So paraphrasing MAY be added, and the sentence guarantees only that it cannot change what
counts as a hit. It says nothing about the message text staying templated. This pass
measures what that would cost us.

EVIDENCE TIER: [PROBE], and the deviation is explicit. The official harness cannot inject
a transform, so the loop below is copied from `evaluator.local_evaluator.evaluate` and the
ONLY change is `user_message = transform(user_message)` immediately before the agent sees
it. Transform T0 is the identity, and its score must reproduce the official 0.96755
exactly -- that equality is the proof the replica is faithful. The evaluator itself is not
modified, imported functions are used unchanged, and scoring arithmetic is the evaluator's.

The ladder runs from realistic to deliberately unfair:

  T0 identity                   control; must equal the official score
  T1 scaffold reworded          template framings replaced with natural synonyms;
                                constraint VALUES untouched (they are product text, and
                                the generator lifts them verbatim from the catalogue)
  T2 scaffold stripped          framings deleted entirely -- bare values, no cues
  T3 conversational noise       filler sentences wrapped around the real content
  T4 case/punctuation churn     lowercased, punctuation stripped
  T5 realistic combined         T1 + T3, the honest "organizer added an LLM" scenario
  T6 values lossily dropped     every other content word deleted from the VALUES
  T7 values word-order shuffled  values scrambled -- destroys phrase adjacency outright

T6 and T7 are past what the spec contemplates: they paraphrase the product text itself,
which would also break the ground-truth link the organizer relies on. They are included as
the floor of the range, not as a scenario we expect.

Run:  PYTHONIOENCODING=utf-8 python -u notes/eda/31_paraphrase_stress.py
"""
from __future__ import annotations

import json
import random
import re
import statistics
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS, TOP_K, catalog_index, coarse_category, customer_reply, initial_message,
    load_jsonl, materialize_hidden_fields, metric_summary, normalize_recommendations,
)
from submission.agent import Agent  # noqa: E402

# --------------------------------------------------------------------------- transforms

FILLER_PRE = ["Hey there!", "Thanks for the help.", "Okay, so.", "Right, let me think.",
              "Appreciate it.", "Quick one for you."]
FILLER_POST = ["Does that help at all?", "Hope that makes sense.", "Let me know!",
               "Anything you can do.", "Cheers.", "That's about all I've got."]


def _scaffold(msg: str) -> str:
    """Reword the simulator's template framings; leave the values between them alone."""
    m = re.match(r"^I'm looking for (.+?), but I'm still exploring\.$", msg)
    if m:
        return f"I'm browsing around for {m.group(1)} at the moment, nothing fixed yet."
    m = re.match(r"^I'm looking for (.+?)\. A key requirement is: (.+)\.$", msg)
    if m:
        return f"I want to find {m.group(1)}. It absolutely has to be {m.group(2)}."
    m = re.match(r"^I'm looking for (.+?)\. (.+)$", msg)
    if m:
        return f"So I'm after {m.group(1)}. {m.group(2)}"
    m = re.match(r"^For that, what matters is: (.+)\.$", msg)
    if m:
        return f"Sure -- the thing that counts for me is {m.group(1)}."
    m = re.match(r"^I don't have an additional preference for (.+?)\.$", msg)
    if m:
        return f"No strong feelings about {m.group(1)}, honestly."
    m = re.match(r"^I don't have a preference for (.+?); please use your judgment\.$", msg)
    if m:
        return f"Honestly, {m.group(1)} is entirely up to you."
    m = re.match(r"^Actually, ignore my earlier preference\. What I need is: (.+)\.$", msg)
    if m:
        return f"Hmm, scratch all that. What I actually need is {m.group(1)}."
    if msg.startswith("Those options are not quite right"):
        return "Not quite there yet. Could you ask me about one thing in particular?"
    return msg


def _strip_scaffold(msg: str) -> str:
    """Delete the framings entirely, leaving only the payload."""
    for pat in (r"^I'm looking for (.+?), but I'm still exploring\.$",
                r"^I'm looking for (.+?)\. A key requirement is: (.+)\.$",
                r"^For that, what matters is: (.+)\.$",
                r"^Actually, ignore my earlier preference\. What I need is: (.+)\.$",
                r"^I'm looking for (.+?)\. (.+)$"):
        m = re.match(pat, msg)
        if m:
            return "; ".join(g for g in m.groups() if g)
    m = re.match(r"^I don't have an additional preference for (.+?)\.$", msg)
    if m:
        return "n/a"
    m = re.match(r"^I don't have a preference for (.+?); please use your judgment\.$", msg)
    if m:
        return "n/a"
    return msg


def _values(msg: str, fn) -> str:
    """Apply `fn` to the VALUE spans only, keeping the template framings intact."""
    for pat in (r"(A key requirement is: )(.+?)(\.?)$",
                r"(For that, what matters is: )(.+?)(\.?)$",
                r"(What I need is: )(.+?)(\.?)$",
                r"(I'm looking for )(.+?)(, but I'm still exploring\.)$"):
        m = re.search(pat, msg)
        if m:
            return msg[:m.start()] + m.group(1) + fn(m.group(2)) + m.group(3)
    return msg


def _drop_alternate(text: str) -> str:
    parts = text.split()
    return " ".join(p for i, p in enumerate(parts) if i % 2 == 0) or text


def _shuffle_words(text: str) -> str:
    parts = text.split()
    random.Random(len(text)).shuffle(parts)
    return " ".join(parts)


def t_noise(msg: str) -> str:
    r = random.Random(len(msg) * 31 + sum(map(ord, msg[:12])))
    return f"{r.choice(FILLER_PRE)} {msg} {r.choice(FILLER_POST)}"


TRANSFORMS = {
    "T0 identity (control)":        lambda m: m,
    "T1 scaffold reworded":         _scaffold,
    "T2 scaffold stripped":         _strip_scaffold,
    "T3 conversational noise":      t_noise,
    "T4 case/punctuation churn":    lambda m: re.sub(r"[^a-z0-9 ]", " ", m.lower()),
    "T5 realistic (T1+T3)":         lambda m: t_noise(_scaffold(m)),
    "T6 VALUES lossy (adversarial)": lambda m: _values(m, _drop_alternate),
    "T7 VALUES shuffled (adversarial)": lambda m: _values(m, _shuffle_words),
}


# ------------------------------------------------------------------- replica of evaluate
def evaluate_transformed(agent, samples, catalog_ids, categories, products, transform):
    """Verbatim copy of evaluator.local_evaluator.evaluate, with ONE added line:
    the user message is passed through `transform` before the agent sees it."""
    sessions: list[dict] = []
    for sample in samples:
        session_id = f"public_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(
            effective_sample, coarse_category(categories.get(target, [])), disclosed)
        hit_turn: int | None = None
        best_rank: int | None = None
        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(
                    session_id, transform(user_message), turn, TOP_K)   # <-- ONLY CHANGE
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break
            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(override.get(
                    "message", "Actually, please ignore my earlier preference."))
            else:
                user_message, boundary_used = customer_reply(
                    effective_sample, response.get("ask_attribute"), disclosed, boundary_used)
        sessions.append({
            "sample_id": sample["sample_id"], "scenario_type": sample["scenario_type"],
            "hit": hit_turn is not None, "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })
    overall = metric_summary(sessions)
    eff = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    return {**overall, "efficiency": eff,
            "recommended_technical_score": 0.50 * overall["hit_rate_at_10"]
            + 0.30 * overall["mrr"] + 0.20 * eff}


def main() -> None:
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")

    def share():
        o = object.__new__(Agent)
        o.ix, o.sessions, o.llm = base.ix, {}, None
        return o

    print("sample of what each transform does to turn 1 of a buying session:")
    demo = "I'm looking for Novelty Women. A key requirement is: cotton."
    for name, fn in TRANSFORMS.items():
        print(f"  {name:<34} {fn(demo)[:96]!r}")

    print(f"\n{'transform':<36}{'score':>9}{'delta':>9}{'HR':>8}{'MRR':>9}{'MTTC':>7}")
    print("-" * 78)
    OUT: dict = {}
    ref = None
    for name, fn in TRANSFORMS.items():
        r = evaluate_transformed(share(), samples, cid, cats, prods, fn)
        if ref is None:
            ref = r["recommended_technical_score"]
        OUT[name] = {"score": r["recommended_technical_score"], "hr": r["hit_rate_at_10"],
                     "mrr": r["mrr"], "mttc": r["mttc"]}
        print(f"{name:<36}{r['recommended_technical_score']:>9.5f}"
              f"{r['recommended_technical_score']-ref:>+9.5f}"
              f"{r['hit_rate_at_10']:>8.1%}{r['mrr']:>9.4f}{r['mttc']:>7.3f}")

    ctl = OUT["T0 identity (control)"]["score"]
    print(f"\n  control reproduces the official score exactly? "
          f"{ctl:.5f} vs 0.96755 -> {'YES' if abs(ctl-0.96755) < 1e-5 else 'NO -- replica is not faithful'}")
    realistic = min(OUT[k]["score"] for k in OUT if k.startswith(("T1", "T2", "T3", "T4", "T5")))
    print(f"  worst REALISTIC paraphrase (T1-T5): {realistic:.5f} "
          f"({realistic-ctl:+.5f})")
    adversarial = min(OUT[k]["score"] for k in OUT if k.startswith(("T6", "T7")))
    print(f"  worst ADVERSARIAL (values rewritten): {adversarial:.5f} "
          f"({adversarial-ctl:+.5f})")

    (ROOT / "notes" / "eda" / "out_31.json").write_text(
        json.dumps(OUT, indent=2) + "\n", encoding="utf-8")
    print("\n[saved] notes/eda/out_31.json")


if __name__ == "__main__":
    main()
