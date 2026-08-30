"""V2.50: Node 5 as a verifier of the LLM's single proposal, not a selector among many.

WHY NODE 5 IS SUDDENLY VIABLE
-----------------------------
Node 5 scored 0.9726 AUROC zero-shot (V2.37) and was still not usable, for a reason
recorded at the time: pairwise AUROC is not selection precision over 99 distractors. Asked
to pick the right canonical out of a retrieved list, a good pairwise verifier still fails,
because it is being used for a task it was never measured on.

The `generate` design removes that mismatch entirely. The LLM emits ONE proposal. Node 5
is asked one pairwise question about it -- which is exactly the question its 0.9726 was
measured on. The LLM does the selection; the verifier does the part it is good at.

THE HOLE THIS PLUGS, STATED PRECISELY
-------------------------------------
Today the only check on a proposal is `df(proposal) > 0`. That is PROVENANCE, not
semantics: it asks whether the phrase exists in the catalogue, never whether it means what
the customer asked for. It would accept `polyester` for "made from a soft plant fibre"
exactly as readily as `cotton`. V2.37 named this case itself -- cotton and polyester sit
close in embedding space while being mutually exclusive, and directional entailment is the
one check that separates them, because "polyester entails soft plant fibre" is false in
the direction that matters.

DIRECTION. The customer asked for X (the paraphrase); the catalogue offers Y (the
proposal). The runtime question is "does Y SATISFY X", so Y is the premise and X the
hypothesis. This is asymmetric on purpose: "genuine leather" satisfies "leather" but not
conversely, and an equivalence test rejects the pair that should be accepted.

TWO EXPERIMENTS, BECAUSE ONE OF THEM CANNOT SHOW THE VALUE
-----------------------------------------------------------
A. ADVERSARIAL SEPARATION -- the direct test of the hole. For each paraphrase, score the
   LLM's own proposal against negatives that are catalogue-attested (so they sail through
   the `df > 0` gate untouched) and MUTUALLY EXCLUSIVE with the right answer.

   HOW THE NEGATIVES ARE BUILT, AND HOW THE FIRST ATTEMPT GOT IT WRONG. The first version
   of this file mined negatives as the bi-encoder's nearest neighbours OF THE PROPOSAL.
   That construction is invalid, and inspecting it is what showed why: for "made from
   animal hide" -> `leather` it produced `leather fabric`, `leather material`,
   `leather suede`, `canvas leather`. Those all SATISFY the paraphrase. They were labelled
   negative while being true positives, so the resulting AUROC of 0.5760 was measured
   against wrong ground truth and meant nothing. It is withdrawn.

   The fix is CROSS-PAIRING WITHIN AN ATTRIBUTE FAMILY: take the proposal resolved for one
   paraphrase and test it against a DIFFERENT paraphrase from the same family. `leather`
   against "made from a soft plant fibre" is attested, hard, and certainly false -- the two
   are competing answers to the same question, which is what mutual exclusivity means here.
   The label follows from the construction rather than from a judgement call, so there is
   nothing to tune and nothing to talk myself into.

   Families are assigned from the paraphrase's own surface form (colour / material /
   closure / care / other). That is crude, so the grouping is PRINTED and can be audited
   rather than trusted.

B. END-TO-END COST on the attribute-paraphrase suite.

STATE THE EXPECTATION BEFORE MEASURING, so a null cannot be spun afterwards: experiment B
is expected to score at best NEUTRAL and quite possibly NEGATIVE. The LLM already abstains
on the cases it is unsure of (6 of 23), so nearly every proposal reaching the verifier is
one the LLM believes in. A gate applied there can mostly only reject CORRECT answers, and
each rejection costs the difference between `generate` and suppression for that phrase.

That is not an argument against Node 5. It is a statement about what this suite can see.
The value of a verifier is measured by what it REFUSES on inputs the suite does not
contain, which is what experiment A is for. A design whose only defence against
`polyester` is that the LLM happened not to say it is not defended.

THRESHOLD DISCIPLINE. The accept threshold is fixed on the FROZEN 134-row verifier set
(V2.37's own test surface) at a high-precision operating point, before experiment B runs.
It is never tuned on the attribute-paraphrase suite. The cost asymmetry justifies choosing
precision over recall: a false REJECT degrades that phrase to suppression, which is the
shipped, safe behaviour; a false ACCEPT injects wrong evidence into ranking.

Run:  PYTHONIOENCODING=utf-8 python -u robustness/v2/evaluate_node5_verifies_llm_proposal.py
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
PVO = V2 / "public_value_only"
CACHE = ROOT / ".v2_model_cache"
OUT = V2 / "results" / "node5_verifies_llm_proposal_v2_50.json"
FROZEN = V2 / "sets" / "frozen_equivalence_verification.jsonl"

SEM = "sem"
WEIGHT = 0.15

_e = ilu.spec_from_file_location("_v2_46", V2 / "evaluate_llm_resolver_end_to_end.py")
_em = ilu.module_from_spec(_e)
_e.loader.exec_module(_em)

_n = ilu.spec_from_file_location("_v2_37", V2 / "evaluate_node5_nli_zeroshot.py")
_nm = ilu.module_from_spec(_n)
_n.loader.exec_module(_nm)
auroc = _nm.auroc


def main() -> None:
    os.environ["LLM_EXTRACT"] = "0"
    os.environ["BERT_EXTRACT"] = "0"
    from submission.llm_rerank import ENDPOINT, _load_project_env
    _em.ENDPOINT = ENDPOINT
    _load_project_env()
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print("GROQ_API_KEY not set -- nothing to evaluate."); return
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

    import numpy as np
    import torch
    from transformers import (AutoModel, AutoModelForSequenceClassification,
                              AutoTokenizer)
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    from submission.agent import CONSTRAINT, Agent, raw_toks
    device = "cuda" if torch.cuda.is_available() else "cpu"

    path = glob.glob(str(CACHE / "models--cross-encoder--nli-deberta-v3-small"
                         / "snapshots" / "*"))[0]
    ntok = AutoTokenizer.from_pretrained(path)
    nmdl = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    ent_idx = next(i for i, l in nmdl.config.id2label.items()
                   if "entail" in str(l).lower())

    def frame(t: str) -> str:
        return f"The product is {t}."

    def entail(prem, hyp, bs=64):
        """P(premise entails hypothesis). Y satisfies X  =>  premise Y, hypothesis X."""
        out = []
        for i in range(0, len(prem), bs):
            b = ntok(prem[i:i + bs], hyp[i:i + bs], padding=True, truncation=True,
                     max_length=64, return_tensors="pt").to(device)
            with torch.no_grad():
                out.extend(torch.softmax(nmdl(**b).logits, -1)[:, ent_idx].cpu().tolist())
        return out

    # ---- THRESHOLD, fixed on the frozen set before anything else runs -----------------
    rows = [json.loads(l) for l in FROZEN.open(encoding="utf-8") if l.strip()]
    y = [int(r["label"]) for r in rows]
    # DIRECTION. At runtime the premise is the CATALOGUE VALUE and the hypothesis is the
    # customer's wording: "The product is cotton." |= "The product is a soft plant fibre."
    # In the frozen set `canonical` plays the catalogue-value role, so it is the premise.
    # This also happens to be V2.37's stronger direction (0.9726 vs 0.8279); the first
    # version of this file used the reverse and measured the weaker one by mistake.
    s = entail([frame(r["canonical"]) for r in rows],
               [frame(r["candidate"]) for r in rows])
    grid = sorted({round(x, 3) for x in s})
    TARGET_PRECISION = 0.90
    best, THRESH, fallback = None, 0.5, None
    for t in grid:
        tp = sum(1 for x, l in zip(s, y) if x >= t and l == 1)
        fp = sum(1 for x, l in zip(s, y) if x >= t and l == 0)
        if tp + fp < 5:
            continue
        prec, rec = tp / (tp + fp), tp / max(sum(y), 1)
        # High precision is the operating point: a false reject degrades to suppression
        # (safe, shipped), a false accept injects wrong evidence into ranking.
        if prec >= TARGET_PRECISION and (best is None or rec > best[0]):
            best, THRESH = (rec, prec), t
        if fallback is None or prec > fallback[2]:
            fallback = (rec, t, prec)
    print(f"frozen set: {len(rows)} rows, {sum(y)} positive, AUROC {auroc(s, y):.4f}")
    print(f"threshold fixed at {THRESH:.3f} "
          f"(precision {best[1]:.3f}, recall {best[0]:.3f}) -- never tuned downstream\n")

    # ---- resolve every paraphrase once, shared by both experiments --------------------
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    para = load_jsonl(PVO / "official200_attribute_paraphrase_dev.jsonl")
    res = _em.Resolver(base.ix, model, key)

    # ---- A. ADVERSARIAL SEPARATION ---------------------------------------------------
    pmap = {}
    for line in (PVO / "official200_attribute_paraphrase_dev.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        card = json.loads(line)["intent_card"]
        for v in card["hard_constraints"] + card["soft_preferences"]:
            t = " ".join(raw_toks(str(v))[:base.RESOLVE_CAP])
            if t and base.ix.df(t) == 0:
                pmap[t] = None
    for phrase in list(pmap):
        pmap[phrase] = res.resolve(phrase)
    resolved = {p: c for p, c in pmap.items() if c}

    FAMILY_CUES = (
        ("colour", ("shade", "colour", "color", "toned", "darkest")),
        ("material", ("made from", "made with", "fibre", "fiber", "textile", "cloth",
                      "polymer", "hide", "fleece")),
        ("closure", ("fasten", "closes", "clasp", "slips on", "fastener", "closure")),
        ("care", ("clean", "wash", "water", "moisture", "repel")),
    )

    def family(phrase):
        for name, cues in FAMILY_CUES:
            if any(c in phrase for c in cues):
                return name
        return "other"

    fams = {}
    for phrase in resolved:
        fams.setdefault(family(phrase), []).append(phrase)
    usable = {f: ps for f, ps in fams.items() if len(ps) >= 2}
    print("A. adversarial separation -- negatives are COMPETING answers within a family")
    for f, ps in sorted(usable.items()):
        shown = "; ".join(x[:32] for x in sorted(ps)[:4])
        print("   %-9s %d paraphrases: %s" % (f, len(ps), shown))
    skipped = {f: ps for f, ps in fams.items() if len(ps) < 2}
    if skipped:
        note = ", ".join("%s x%d" % (f, len(ps)) for f, ps in sorted(skipped.items()))
        print("   (families with <2 paraphrases, no valid pairing: %s)" % note)

    prem, hyp, lab, detail = [], [], [], []
    for f, ps in sorted(usable.items()):
        for phrase in sorted(ps):
            prem.append(frame(resolved[phrase]))
            hyp.append(frame(phrase))
            lab.append(1)
            others = [o for o in sorted(ps)
                      if o != phrase and resolved[o] != resolved[phrase]]
            for other in others:
                # resolved[other] answers a DIFFERENT question in the same family, so it
                # is a competing value: attested, close in embedding space, and false.
                prem.append(frame(resolved[other]))
                hyp.append(frame(phrase))
                lab.append(0)
            detail.append({"family": f, "paraphrase": phrase,
                           "proposal": resolved[phrase],
                           "competing": [resolved[o] for o in others]})
    if not lab or sum(lab) in (0, len(lab)):
        print("   not enough within-family pairs to measure separation.")
        return
    scores = entail(prem, hyp)
    sep = auroc(scores, lab)
    prop_s = [x for x, l in zip(scores, lab) if l == 1]
    dist_s = [x for x, l in zip(scores, lab) if l == 0]
    passes = sum(1 for x in prop_s if x >= THRESH)
    leaks = sum(1 for x in dist_s if x >= THRESH)
    print("   AUROC correct proposal vs competing same-family value  %.4f" % sep)
    print("   mean entailment  correct %.4f   competing %.4f"
          % (np.mean(prop_s), np.mean(dist_s)))
    print("   at the fixed threshold: %d/%d correct accepted, %d/%d competing LEAK through"
          % (passes, len(prop_s), leaks, len(dist_s)))
    print("   the df>0 gate alone accepts %d/%d of these -- every one is attested.\n"
          % (len(dist_s), len(dist_s)))

    # ---- B. END-TO-END COST ----------------------------------------------------------
    vcache: dict[str, float] = {}

    def verified(phrase: str) -> str | None:
        prop = res.resolve(phrase)
        if not prop:
            return None
        keyp = f"{prop}\x00{phrase}"
        if keyp not in vcache:
            vcache[keyp] = entail([frame(prop)], [frame(phrase)])[0]
        return prop if vcache[keyp] >= THRESH else None

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

    floor = run(Agent, para)
    gen = run(arm(res.resolve), para)
    ver = run(arm(verified), para)
    rejected = sum(1 for v in vcache.values() if v < THRESH)
    print("B. end-to-end on the attribute-paraphrase suite")
    print(f"   {'suppression (floor)':<34}{floor:>11.6f}")
    print(f"   {'generate (df>0 only)':<34}{gen:>11.6f}{gen-floor:>+11.6f}")
    print(f"   {'generate + node 5 verifier':<34}{ver:>11.6f}{ver-floor:>+11.6f}"
          f"{ver-gen:>+11.6f} vs generate")
    print(f"   verifier rejected {rejected}/{len(vcache)} proposals")
    print("\n   As pre-registered, B can mostly only reject CORRECT answers on this suite:")
    print("   the LLM already abstains where it is unsure, so the verifier's value is in")
    print("   what it refuses on inputs this suite does not contain -- that is A, not B.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "V2.50 Node 5 verifies the LLM's single proposal",
        "threshold": THRESH, "frozen_auroc": round(auroc(s, y), 4),
        "adversarial": {"construction": "within-family cross-pairing",
                        "superseded": ("v1 mined neighbours OF THE PROPOSAL; those were "
                                       "true positives mislabelled negative, so its "
                                       "AUROC 0.5760 is void"),
                        "auroc": round(sep, 4), "n_proposals": len(prop_s),
                        "n_distractors": len(dist_s),
                        "mean_proposal": round(float(np.mean(prop_s)), 4),
                        "mean_distractor": round(float(np.mean(dist_s)), 4),
                        "proposals_accepted": passes, "distractors_leaked": leaks,
                        "detail": detail},
        "end_to_end": {"floor": floor, "generate": gen, "generate_plus_verifier": ver,
                       "rejected": rejected, "verified_pairs": len(vcache)},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
