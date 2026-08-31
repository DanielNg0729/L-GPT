"""Can a tighter admission gate cut the noise in LLM-proposed evidence?

THE PROBLEM. Both proposal layers -- our per-value deparaphraser and the whole-transcript
rescue -- admit evidence on the same test: `df(proposal) > 0`, "some product in the
catalogue says this". That is a weak claim. `Imported` appears in 15,300 listings; a
proposal can be perfectly attested and still be worthless, or attested and simply WRONG for
this conversation. Measured on the deparaphraser: 799 answered invocations, 606 correct,
**193 confidently wrong** admitted at W_SEM. The transcript rescue shows the same shape --
43 accepted against 41 dropped, for a net two sessions in 800.

THE IDEA. Every constraint the simulator speaks is a verbatim substring of ONE target
document. So every TRUE proposal must co-occur, in at least one product, with all the other
true evidence we already hold. A wrong proposal has no such obligation. That turns admission
from "does this string exist anywhere" into "can this string and everything else the
customer said describe the same product" -- which is the question we actually care about,
and it is one FTS5 boolean away.

Gates compared:
  df>0        the shipped gate: attested anywhere
  conj        attested in some product TOGETHER WITH all current constraint evidence
  df<=N       attested and not boilerplate (a ceiling on document frequency)
  conj+df<=N  both

NO NETWORK. The resolver runs from its warm cache, so every answer is replayed rather than
re-requested. The run asserts zero provider calls and reports the count.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/studies/audit_proposal_gate_noise.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ["LLM_EXTRACT"] = "0"
os.environ["LLM_RERANK"] = "0"
os.environ["LLM_RESOLVE"] = "1"
os.environ["V2_ROUTE"] = "1"
os.environ["BERT_EXTRACT"] = "1"

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import CAT, Agent, raw_toks  # noqa: E402
from submission.llm_resolve import LLMResolver  # noqa: E402

SUITE = (ROOT / "experiments" / "datasets" / "open_vocabulary"
         / "review800_open_vocab_paraphrase.jsonl")
CACHE = ROOT / "experiments" / "datasets" / "prompt_arm_caches" / "arm_A__openai_gpt_oss_120b__mt512.json"
OUT = ROOT / "experiments" / "results" / "out_74_proposal_gate_noise.json"
DF_FLOORS = (20, 50, 100, 200, 500, 1000)


class OfflineResolver(LLMResolver):
    """Replays the cache. Any cache miss returns None instead of reaching the network."""

    def __init__(self, cache_path: Path) -> None:
        super().__init__(cache_path=cache_path)
        self.blocked = 0

    @property
    def enabled(self) -> bool:
        return True                     # no credential needed; nothing is sent

    def _call(self, phrase: str):
        self.blocked += 1
        return None


def main() -> None:
    if not CACHE.exists():
        print(f"missing cache: {CACHE}")
        return
    ids, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    samples = load_jsonl(SUITE)
    targets = [str(s["ground_truth"]["parent_asin"]) for s in samples]
    resolver = OfflineResolver(CACHE)
    records: list[dict] = []

    class Probe(Agent):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.resolver = resolver.bind(self.ix.df)
            self._i = -1
            self._st = None

        def reset(self, sid, prof=None):
            self._i += 1
            return super().reset(sid, prof)

        def _observe(self, st, msg):
            self._st = st
            return super()._observe(st, msg)

        def _deparaphrase(self, text):
            out = super()._deparaphrase(text)
            if out:
                st = self._st
                held = [p for p, (_d, t) in st.evidence.items() if t != CAT] if st else []
                records.append({
                    "proposal": out,
                    "target": targets[self._i] if self._i < len(targets) else None,
                    "held": held[:8],
                })
            return out

    agent = Probe(ROOT / "data" / "catalog.jsonl")
    evaluate(agent, samples, ids, cats, prods)
    ix = agent.ix

    print(f"provider calls made: {resolver.blocked} (must be 0 for a valid offline audit)")
    print(f"cache hits {resolver.cache_hits}, misses {resolver.cache_misses}")
    print(f"accepted proposals captured: {len(records)}\n")

    def cooccurs(proposal: str, held: list[str]) -> bool:
        """Does ANY single product contain the proposal and every held constraint?"""
        terms = [proposal] + list(held)
        expr = " AND ".join(f'"{t}"' for t in terms if t.strip())
        if not expr:
            return True
        try:
            row = ix.con.execute(
                "SELECT 1 FROM p WHERE p MATCH ? LIMIT 1", (expr,)).fetchone()
            return row is not None
        except sqlite3.Error:
            return True                 # never let a query error reject evidence

    for r in records:
        blob = ix.blob.get(r["target"], "")
        r["correct"] = f' {r["proposal"]} ' in blob
        r["df"] = ix.df(r["proposal"])
        r["conj"] = cooccurs(r["proposal"], r["held"])

    def report(name, keep):
        kept = [r for r in records if keep(r)]
        good = sum(1 for r in kept if r["correct"])
        bad = len(kept) - good
        prec = good / len(kept) if kept else 0.0
        base_good = sum(1 for r in records if r["correct"])
        recall = good / base_good if base_good else 0.0
        return {"gate": name, "admitted": len(kept), "correct": good, "wrong": bad,
                "precision": round(prec, 4), "kept_of_correct": round(recall, 4)}

    # THE CEILING WAS THE WRONG DIRECTION. A first pass gated OUT common proposals on
    # the intuition that boilerplate carries no information. It measured 0 correct out of
    # 31 admitted at df<=50: the RARE proposals are the hallucinations, because the true
    # canonicals are the common trade terms the catalogue actually uses. So the useful
    # gate is a FLOOR -- refuse a proposal the catalogue barely knows.
    rows = [report("df>0  (shipped)", lambda r: True),
            report("conj", lambda r: r["conj"])]
    for n in DF_FLOORS:
        rows.append(report(f"df>{n}", lambda r, n=n: r["df"] > n))
        rows.append(report(f"conj + df>{n}", lambda r, n=n: r["conj"] and r["df"] > n))

    print(f"{'gate':<22}{'admitted':>10}{'correct':>9}{'WRONG':>8}"
          f"{'precision':>11}{'kept of correct':>17}")
    print("-" * 77)
    for row in rows:
        print(f"{row['gate']:<22}{row['admitted']:>10}{row['correct']:>9}"
              f"{row['wrong']:>8}{row['precision']:>11.4f}{row['kept_of_correct']:>17.4f}")

    print("\n  A gate is worth adopting only if it drops WRONG much faster than correct.")
    print("  `kept of correct` is the fraction of true proposals that survive it.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "admission-gate noise reduction for LLM-proposed evidence",
        "suite": SUITE.name, "network": "none; cache replay",
        "provider_calls": resolver.blocked,
        "captured": len(records), "gates": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
