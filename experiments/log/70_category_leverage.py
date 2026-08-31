"""EDA pass 70: the category is load-bearing -- is the shipped agent weighting it right?

WHERE THIS CAME FROM
--------------------
The hostile condition removed the category and the CEILING collapsed from 0.945125 to
0.280362: with PERFECT canonical values and no category, HR@10 is 0.3183. Removing one
signal cost two thirds of the achievable score, which says the category narrows 50k
products to a tractable set and the attribute values only rank within it.

That was measured as an ablation of the SUITE. This measures it as an ablation of the
AGENT, and then asks the obvious follow-up: `W_CATEGORY` is 0.4541, less than half of
`W_CONSTRAINT`. If the category carries the retrieval, is that weight too low?

THREE THINGS MEASURED

  ABLATION      drop CAT-tier evidence entirely. The mirror of the hostile experiment,
                inside the agent, on the real decision suites rather than a synthetic
                condition. It prices what the category is worth where it actually matters.
  SWEEP         `W_CATEGORY` across a range including the shipped value. If the shipped
                weight is at a local optimum the curve is flat around it; if the category
                is underweighted the curve still rises past it.
  BUYING FLAG   `st.buying` is set at turn 1 and READ NOWHERE -- grep finds exactly two
                references, the initialiser and the assignment. It is vestigial state. The
                sweep run reports whether browse and buy sessions behave differently
                enough for the flag to be worth wiring to anything, or whether it should
                simply be deleted.

DISCIPLINE. The decision criteria are official200 plus the four population suites, and the
pre-registered rule is unchanged: adopt only if NO decision criterion regresses. A weight
that wins on official200 alone is a weight fitted to 200 sessions. `W_CATEGORY` came from
the frozen trial-38 configuration, so moving it needs the same bar trial 38 cleared.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/log/70_category_leverage.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments" / "results" / "out_70_category_leverage.json"

WEIGHTS = (0.0, 0.25, 0.4541399437579685, 0.70, 1.00, 1.50)


def main() -> None:
    os.environ["LLM_RERANK"] = "0"
    os.environ["LLM_EXTRACT"] = "0"
    os.environ["LLM_RESOLVE"] = "0"
    os.environ["BERT_EXTRACT"] = "0"
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    from submission.agent import CAT, Agent

    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    rs = ROOT / "experiments" / "datasets" / "sets"
    sets = {
        "official200": load_jsonl(ROOT / "data" / "public_set.jsonl"),
        "org-proxy": load_jsonl(rs / "organizer_proxy_800.jsonl"),
        "review800": load_jsonl(rs / "catalog_review_distinct_800.jsonl"),
        "uniform": load_jsonl(rs / "catalog_uniform_800.jsonl"),
        "inverse": load_jsonl(rs / "catalog_inverse_800.jsonl"),
    }
    COLS = list(sets)
    shipped = base.W_CATEGORY
    print(f"shipped W_CATEGORY = {shipped:.6f}   W_CONSTRAINT = {base.W_CONSTRAINT}\n")

    def make(weight=None, ablate=False):
        class Arm(Agent):
            def _weight(self, phrase, df, tier):
                if ablate and tier == CAT:
                    return 0.0
                if weight is not None and tier == CAT:
                    return weight / (1.0 + df) ** self.IDF_POW
                return super()._weight(phrase, df, tier)
        a = object.__new__(Arm)
        a.ix, a.sessions = base.ix, {}
        a.llm = a.llm_extract = a.tagger = a.resolver = None
        a.span_node, a.route_node = base.span_node, base.route_node
        return a

    def row(label, factory):
        r = {}
        for name, sub in sets.items():
            r[name] = round(evaluate(factory(), sub, cid, cats, prods)[
                "recommended_technical_score"], 6)
        print(f"{label:<22}" + "".join(f"{r[c]:>12.6f}" for c in COLS), flush=True)
        return r

    t0 = time.time()
    print(f"{'configuration':<22}" + "".join(f"{c:>12}" for c in COLS))
    print("-" * (22 + 12 * len(COLS)))
    ref = row("shipped", lambda: make())
    ablated = row("CAT evidence ABLATED", lambda: make(ablate=True))
    sweep = {}
    for w in WEIGHTS:
        sweep[w] = row(f"W_CATEGORY = {w:.4f}", lambda w=w: make(weight=w))

    print(f"\nablation: what the category is worth inside the agent")
    print(f"{'suite':<14}{'shipped':>12}{'ablated':>12}{'delta':>12}")
    print("-" * 50)
    for c in COLS:
        print(f"{c:<14}{ref[c]:>12.6f}{ablated[c]:>12.6f}{ablated[c]-ref[c]:>+12.6f}")

    print(f"\ndeltas vs shipped, by weight")
    print(f"{'W_CATEGORY':<14}" + "".join(f"{c:>12}" for c in COLS) + "   verdict")
    print("-" * (14 + 12 * len(COLS) + 12))
    for w, r in sweep.items():
        d = {c: r[c] - ref[c] for c in COLS}
        worst = min(d.values())
        verdict = ("shipped" if abs(w - shipped) < 1e-9 else
                   "ADOPT -- no regression on any criterion" if worst >= 0 else
                   "inside noise" if worst > -0.002 else "REJECT")
        print(f"{w:<14.4f}" + "".join(f"{d[c]:>+12.6f}" for c in COLS) + f"   {verdict}")

    print(f"\n  A flat curve around the shipped value means trial 38 already found the")
    print(f"  optimum and the hostile finding, though real, is already priced in. A curve")
    print(f"  still rising past it means the category is underweighted and the ablation")
    print(f"  understated how much of the work it does.")
    print(f"\n  {time.time()-t0:.0f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"shipped_w_category": shipped, "shipped": ref,
                               "ablated": ablated,
                               "sweep": {str(k): v for k, v in sweep.items()}},
                              indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
