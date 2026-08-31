"""Experiment 35: the one failure mode that is CATASTROPHIC rather than merely degrading.

Rejection feedback demotes anything we showed on a turn that did not end the session,
because reaching the next turn proves it was wrong. In an `intent_override` session that
inference is FALSE before the override fires: the harness gates hits until then
(evaluator L252: `if override_applied and target in ranked`), so the true target can be
shown, silently not count, and be demoted for the rest of the session.

The agent guards this by clearing the rejection set when it detects an override:

    PAT_OVERRIDE      r"what i need is:\\s*(.+?)\\.?$"
    PAT_OVERRIDE_CUE  r"ignore my earlier|instead|actually[, ]"

Both are pattern matches against the simulator's exact wording. If the organizer adds
paraphrasing, the guard can stop firing while the demotion it protects against keeps
working -- and that turns a graceful degradation into a permanently excluded target.

This measures intent_override HitRate specifically, under each paraphrase transform, with
the guard intact and with rejection feedback disabled entirely. If the guard is what stands
between us and a collapse, disabling rejection feedback should IMPROVE the paraphrased
override sessions.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/scripts/35_override_safety.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

from evaluator.local_evaluator import catalog_index, load_jsonl  # noqa: E402
from submission.agent import Agent  # noqa: E402

_p31 = __import__("31_paraphrase_stress")
TRANSFORMS = _p31.TRANSFORMS


class NoRejection(Agent):
    def respond(self, sid, msg, turn, top_k):
        r = super().respond(sid, msg, turn, top_k)
        st = self.sessions.get(sid)
        if st is not None:
            st.rejected.clear()
        return r


class NoGuard(Agent):
    """Rejection feedback ON but the override guard REMOVED -- the worst case if the
    organizer's paraphrasing defeats both cue patterns."""
    def respond(self, sid, msg, turn, top_k):
        import submission.agent as sa
        saved_a, saved_b = sa.PAT_OVERRIDE, sa.PAT_OVERRIDE_CUE
        import re
        dead = re.compile(r"(?!x)x")            # matches nothing
        sa.PAT_OVERRIDE, sa.PAT_OVERRIDE_CUE = dead, dead
        try:
            return super().respond(sid, msg, turn, top_k)
        finally:
            sa.PAT_OVERRIDE, sa.PAT_OVERRIDE_CUE = saved_a, saved_b


def per_scenario(agent, samples, cid, cats, prods, transform):
    """Wrap evaluate_transformed but keep the per-session rows so we can group."""
    import uuid
    from evaluator.local_evaluator import (
        MAX_TURNS, TOP_K, coarse_category, customer_reply, initial_message,
        materialize_hidden_fields, normalize_recommendations)
    rows = []
    for sample in samples:
        session_id = f"public_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, prods)
        eff = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        bu = False
        applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(eff, coarse_category(cats.get(target, [])), disclosed)
        hit_turn = best_rank = None
        for turn in range(1, MAX_TURNS + 1):
            try:
                resp = agent.respond(session_id, transform(user_message), turn, TOP_K)
            except Exception:
                resp = {"message": "", "ask_attribute": None, "recommendations": []}
            if not isinstance(resp, dict) or not isinstance(resp.get("message"), str):
                resp = {"message": "", "ask_attribute": None, "recommendations": []}
            ranked = normalize_recommendations(resp.get("recommendations"), cid)
            if applied and target in ranked:
                best_rank, hit_turn = ranked.index(target) + 1, turn
                break
            if turn == MAX_TURNS:
                break
            ov = eff.get("behavior", {}).get("override") or {}
            if not applied and turn + 1 == int(ov.get("turn", 3)):
                applied = True
                nv = str(ov.get("new_value", ""))
                if nv:
                    disclosed.add(nv)
                user_message = str(ov.get("message", "Actually, please ignore my earlier preference."))
            else:
                user_message, bu = customer_reply(eff, resp.get("ask_attribute"), disclosed, bu)
        rows.append((sample["scenario_type"], hit_turn is not None))
    g = defaultdict(list)
    for s, h in rows:
        g[s].append(h)
    return {k: sum(v) / len(v) for k, v in sorted(g.items())}


def main() -> None:
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")

    def share(cls=Agent):
        o = object.__new__(cls)
        o.ix, o.sessions, o.llm = base.ix, {}, None
        return o

    COLS = ["T0 identity (control)", "T1 scaffold reworded", "T2 scaffold stripped",
            "T5 realistic (T1+T3)"]
    VARIANTS = {"shipped (guard on)": Agent,
                "guard DEFEATED": NoGuard,
                "no rejection feedback": NoRejection}

    OUT = {}
    for scen in ("intent_override", "buying", "browsing", "boundary"):
        print(f"\n=== HitRate@10, scenario = {scen} ===")
        print(f"{'variant':<24}" + "".join(f"{c.split()[0]:>10}" for c in COLS))
        print("-" * (24 + 10 * len(COLS)))
        for vname, cls in VARIANTS.items():
            cells = ""
            for c in COLS:
                key = (vname, c)
                if key not in OUT:
                    OUT[key] = per_scenario(share(cls), samples, cid, cats, prods,
                                            TRANSFORMS[c])
                cells += f"{OUT[key].get(scen, float('nan')):>10.1%}"
            print(f"{vname:<24}{cells}")

    print("\n  reading: if 'guard DEFEATED' collapses on intent_override while the other")
    print("  scenarios hold, the override guard is load-bearing and its regexes are a")
    print("  single point of failure under organizer paraphrasing.")

    (ROOT / "experiments" / "results" / "out_35.json").write_text(
        json.dumps({f"{k[0]} | {k[1]}": v for k, v in OUT.items()}, indent=2) + "\n",
        encoding="utf-8")
    print("\n[saved] experiments/results/out_35.json")


if __name__ == "__main__":
    main()
