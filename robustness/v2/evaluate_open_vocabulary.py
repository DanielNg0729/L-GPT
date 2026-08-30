"""V2.51d: does the LLM resolver generalise, and is Node 5 actually needed?

THE TWO QUESTIONS, AND WHY THEY HAVE TO BE ASKED HERE
-----------------------------------------------------
Everything measured so far about the resolver rests on one suite: 27 distinct paraphrases
over the same 200 targets that carry every other result in this project. Two claims are
outstanding and neither can be settled there.

  GENERALISATION. `generate` reached ~96% of the V2.43 oracle. On 27 phrases that is
  compatible with "the model understands paraphrase" and equally compatible with "the
  model knows these 27". Only an independently generated, target-disjoint vocabulary can
  tell those apart.

  IS NODE 5 NEEDED. Node 5 separates correct proposals from competing same-family values at
  0.8349 AUROC, rejecting 89.5% of attested values that the `df > 0` gate waves through.
  But it COSTS 0.0060 on the old suite, because there the LLM almost never proposes a
  competing value, so the verifier only ever rejects correct answers. Whether it earns its
  cost depends entirely on how often the resolver goes wrong on vocabulary it has not been
  effectively memorising -- which is, again, unmeasurable on 27 phrases.

Those two questions share an answer surface, which is why they are one experiment.

WHAT MAKES THIS SUITE DIFFERENT
  targets      review800, disjoint from Official200 by construction (asserted, not assumed)
  vocabulary   independently generated, filtered to share no stem with the atom
  generator    Claude Haiku (Anthropic) -- NOT the solver under test (gpt-oss-120b, Groq)

That last point is the one that would invalidate everything if got wrong. A model that both
writes and solves the paraphrases is inverting its own encoding, and the score would be
inflated by an unknown amount.

ARMS
  0 canonical replay   same sessions, values unchanged. The ceiling, and the control --
                       a paraphrase score is only interpretable against the same session
                       base materialised the same way.
  1 suppression        the shipped agent. The floor.
  2 generate           LLM deparaphrase behind the df>0 provenance gate.
  3 generate + node 5  the same, with entailment verification at the frozen threshold.

HOW TO READ IT, decided before the run so the reading cannot drift:

  arm 2 recovers a similar FRACTION of the gap as on the old suite  -> generalises
  arm 2 recovers materially less                                    -> the old number was
                                                                       partly memorisation
  arm 3 > arm 2   node 5 is needed: the resolver errs often enough here that verification
                  pays for the correct answers it also rejects
  arm 3 < arm 2   node 5 is not needed at this operating point -- its cost is real and its
                  benefit is not yet reachable, and the honest conclusion is that the
                  df>0 gate plus the model's own abstention is sufficient FOR NOW
  arm 3 ~ arm 2   the verifier is inert; keep it only if the threshold can be recalibrated

Fractions are compared, not raw scores: this suite has a different session base and
therefore a different ceiling and floor, so absolute numbers are not comparable to
Official200's and will not be presented as if they were.

Run:  PYTHONIOENCODING=utf-8 python -u robustness/v2/evaluate_open_vocabulary.py
"""
from __future__ import annotations

import glob
import importlib.util as ilu
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "robustness" / "v2"
OV = V2 / "open_vocabulary"
CACHE = ROOT / ".v2_model_cache"
OUT = V2 / "results" / "open_vocabulary_v2_51.json"

SEM = "sem"
WEIGHT = 0.15
THRESH = 0.938          # fixed on the frozen verifier set in V2.50; NOT retuned here

_e = ilu.spec_from_file_location("_v2_46", V2 / "evaluate_llm_resolver_end_to_end.py")
_em = ilu.module_from_spec(_e)
_e.loader.exec_module(_em)


def main() -> None:
    os.environ["LLM_EXTRACT"] = "0"
    os.environ["BERT_EXTRACT"] = "0"
    from submission.llm_rerank import ENDPOINT, _load_project_env
    _em.ENDPOINT = ENDPOINT
    # A cache built on the OLD suite must not leak answers into this one.
    _em.CACHE = OV / ".resolver_cache_open_vocab.json"
    _load_project_env()
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print("GROQ_API_KEY not set -- nothing to evaluate."); return
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    from submission.agent import CONSTRAINT, Agent, raw_toks
    device = "cuda" if torch.cuda.is_available() else "cpu"

    canon_p = OV / "review800_canonical_replay.jsonl"
    para_p = OV / "review800_open_vocab_paraphrase.jsonl"
    if not para_p.exists():
        print(f"missing {para_p} -- run build_open_vocabulary_suite.py first."); return
    man = json.loads((OV / "manifest.json").read_text(encoding="utf-8"))
    print(f"suite: {man['usable_paraphrases']} distinct paraphrases "
          f"(prior suite: {man['distinct_paraphrases_in_prior_suite']}), "
          f"targets disjoint={man['targets_disjoint_from_official200']}")
    print(f"generator={man['generator']}  solver={man['solver_under_test']}\n")

    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    canon_s, para_s = load_jsonl(canon_p), load_jsonl(para_p)
    res = _em.Resolver(base.ix, model, key)

    path = glob.glob(str(CACHE / "models--cross-encoder--nli-deberta-v3-small"
                         / "snapshots" / "*"))[0]
    ntok = AutoTokenizer.from_pretrained(path)
    nmdl = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    ent = next(i for i, l in nmdl.config.id2label.items() if "entail" in str(l).lower())
    vcache: dict[str, float] = {}

    def verify(prop: str, phrase: str) -> float:
        k = f"{prop}\x00{phrase}"
        if k not in vcache:
            b = ntok([f"The product is {prop}."], [f"The product is {phrase}."],
                     padding=True, truncation=True, max_length=64,
                     return_tensors="pt").to(device)
            with torch.no_grad():
                vcache[k] = float(torch.softmax(nmdl(**b).logits, -1)[0, ent])
        return vcache[k]

    def verified(phrase: str):
        prop = res.resolve(phrase)
        if not prop:
            return None
        return prop if verify(prop, phrase) >= THRESH else None

    def arm(resolve_fn):
        class Arm(Agent):
            def _observe(self, st, msg):
                super()._observe(st, msg)
                for text, tier in super()._extract_templated(msg):
                    if tier != CONSTRAINT:
                        continue
                    toks = raw_toks(text)[:self.RESOLVE_CAP]
                    if not toks or self.ix.df(" ".join(toks)) > 0:
                        continue
                    prop = resolve_fn(" ".join(toks))
                    if prop and prop not in st.evidence:
                        st.evidence[prop] = (self.ix.df(prop), SEM)

            def _weight(self, phrase, df, tier):
                if tier == SEM:
                    return WEIGHT / (1.0 + df) ** self.IDF_POW
                return super()._weight(phrase, df, tier)
        return Arm

    def run(cls, samples):
        a = object.__new__(cls)
        a.ix, a.sessions = base.ix, {}
        a.llm = a.llm_extract = a.tagger = None
        return round(evaluate(a, samples, cid, cats, prods)[
            "recommended_technical_score"], 6)

    t0 = time.time()
    ceiling = run(Agent, canon_s)
    floor = run(Agent, para_s)
    gen = run(arm(res.resolve), para_s)
    n_gen_calls, n_gen_acc = res.calls, res.accepted
    ver = run(arm(verified), para_s)
    res.flush()

    gap = ceiling - floor
    def frac(x):
        return (x - floor) / gap if gap > 1e-9 else float("nan")

    print(f"{'arm':<34}{'score':>11}{'vs floor':>11}{'% of gap':>11}")
    print("-" * 67)
    print(f"{'0 canonical replay (ceiling)':<34}{ceiling:>11.6f}"
          f"{ceiling - floor:>+11.6f}{1.0:>10.1%}")
    print(f"{'1 suppression (floor)':<34}{floor:>11.6f}{0.0:>+11.6f}{0.0:>10.1%}")
    print(f"{'2 generate':<34}{gen:>11.6f}{gen - floor:>+11.6f}{frac(gen):>10.1%}")
    print(f"{'3 generate + node 5':<34}{ver:>11.6f}{ver - floor:>+11.6f}"
          f"{frac(ver):>10.1%}")

    rejected = sum(1 for v in vcache.values() if v < THRESH)
    print(f"\n  gap to close on this suite: {gap:.6f}")
    print(f"  resolver: {n_gen_calls} calls, {n_gen_acc} accepted, "
          f"{res.abstained} abstained, {res.unattested} failed provenance")
    print(f"  node 5:   {rejected}/{len(vcache)} proposals rejected at threshold "
          f"{THRESH:.3f}")
    print(f"\n  GENERALISATION: `generate` recovers {frac(gen):.1%} of the gap here, "
          f"against ~96% of the")
    print(f"  oracle on the 27-phrase suite. A large drop means that number was partly "
          f"memorisation.")
    print(f"  IS NODE 5 NEEDED: arm 3 - arm 2 = {ver - gen:+.6f}. Positive means "
          f"verification pays")
    print(f"  for the correct answers it also rejects; negative means the df>0 gate plus "
          f"the model's")
    print(f"  own abstention is sufficient at this operating point.")
    print(f"\n  {time.time() - t0:.0f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "V2.51 open-vocabulary generalisation and Node 5 necessity",
        "suite": {k: man[k] for k in ("usable_paraphrases", "generated_rows",
                                      "rejections", "generator", "solver_under_test",
                                      "targets_disjoint_from_official200")},
        "model": model, "weight": WEIGHT, "node5_threshold": THRESH,
        "arms": {"canonical_ceiling": ceiling, "suppression_floor": floor,
                 "generate": gen, "generate_plus_node5": ver},
        "gap": round(gap, 6),
        "fraction_of_gap": {"generate": round(frac(gen), 4),
                            "generate_plus_node5": round(frac(ver), 4)},
        "node5_delta": round(ver - gen, 6),
        "resolver": {"calls": res.calls, "accepted": res.accepted,
                     "abstained": res.abstained, "unattested": res.unattested,
                     "failed": res.failed},
        "node5_rejected": rejected, "node5_pairs": len(vcache),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
