"""V2.51c: materialise the open-vocabulary attribute-paraphrase suite, with a hard filter.

WHAT THIS PRODUCES
------------------
Two files over TARGET-DISJOINT sessions (review800, zero overlap with Official200):

    review800_canonical_replay.jsonl        control -- values unchanged
    review800_open_vocab_paraphrase.jsonl   the same sessions, values paraphrased

The control is not optional. A paraphrase score is only interpretable against what the
same agent scores on the same sessions with the same materialisation and nothing rewritten,
because materialising cards at all changes the run slightly. Reporting the paraphrase score
against Official200's number instead would silently mix two different session bases.

THE FILTER, AND WHY IT IS ADVERSARIAL TOWARD OUR OWN DATA
---------------------------------------------------------
A generated paraphrase that contains the atom's own words is worthless: it hands the
answer to the resolver and inflates every downstream number. A prior generation attempt at
this task returned 94% such rows, so this is the expected failure mode, not a hypothetical.

Every candidate is therefore rejected unless it passes ALL of:

  * NO SHARED TOKEN with the atom, after stemming to a common prefix. "rubber sole" may not
    yield a paraphrase containing rubber, rubbery, sole, or soles.
  * NOT MERELY REORDERED -- no shared token even under the stemmer, which also catches
    "wash by hand" for "hand wash only".
  * NOT ALREADY ATTESTED. If `df(paraphrase) > 0` the clause never reaches the resolver at
    all, because suppression only fires on unattested text. Such a row would test nothing
    and would quietly dilute the suite toward easy.
  * NOT the literal string SKIP, which the generator emits for product codes and raw
    measurements that cannot be described in other words.
  * NOT PRESENT IN THE OLD 27-RULE SUITE. This one exists because of a mistake made while
    building this suite: the generation prompt illustrated the task with seven worked
    examples, and every one of them was copied from the old suite. The generator
    reproduced all seven verbatim, so those phrases -- and only those -- are the ones a
    resolver could plausibly have been measured on before. They are excluded, which costs
    some high-frequency atoms (cotton, leather, imported) and therefore shrinks the
    measurable gap. Shrinking the gap is the correct price; carrying seven phrases of
    known provenance into a suite whose entire purpose is independent vocabulary is not.

Rejections are COUNTED AND REPORTED, never silently dropped. A low pass rate is a fact
about the generator that the suite's readers need, and hiding it would make the suite look
stronger than it is.

INDEPENDENCE OF GENERATOR AND SOLVER. The paraphrases are written by Claude Haiku
(Anthropic); the resolver under test is gpt-oss-120b (Groq). Different vendor, different
model family, no shared weights. Had one model both written and solved them, it would be
inverting its own encoding and every result would be inflated by an unknown amount.

Run:  PYTHONIOENCODING=utf-8 python -u robustness/v2/build_open_vocabulary_suite.py
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "robustness" / "v2"
PARA = V2 / "open_vocabulary_paraphrases.jsonl"
OUTDIR = V2 / "open_vocabulary"
STEM = 4          # prefix length for the crude stemmer; wash/washing, sole/soles

# Function words are excluded from the overlap test. Comparing them rejects good rows for
# the wrong reason -- "pull on closure" vs "slips on without separate fasteners" collides
# only on "on" -- and because such collisions are commoner in multi-word atoms, leaving
# them in would skew the surviving suite toward single-word atoms. Content words are
# still compared in full, so "hand wash only" -> "wash by hand" is still caught.
FUNCTION_WORDS = {
    "a", "an", "and", "at", "by", "for", "from", "in", "into", "is", "it", "its", "of",
    "off", "on", "onto", "or", "out", "over", "the", "this", "to", "up", "with", "without",
}


def stems(tokens) -> set[str]:
    return {t[:STEM] for t in tokens if t not in FUNCTION_WORDS}


def main() -> None:
    from evaluator.local_evaluator import (behavior_for, catalog_index, intent_card,
                                           load_jsonl)
    from submission.agent import Agent, raw_toks

    if not PARA.exists():
        print(f"missing {PARA} -- run the generation step first."); return

    _cid, _cats, products = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = load_jsonl(ROOT / "robustness" / "sets" / "catalog_review_distinct_800.jsonl")
    official = load_jsonl(ROOT / "data" / "public_set.jsonl")
    assert not ({str(s["ground_truth"]["parent_asin"]) for s in official}
                & {str(s["ground_truth"]["parent_asin"]) for s in base}), "targets overlap"
    agent = Agent(ROOT / "data" / "catalog.jsonl")

    # Every paraphrase used by the old 27-rule suite, in both its development and holdout
    # families. Anything matching one of these is excluded: see the docstring.
    from robustness.v2.build_semantic_attribute_sets import RULES
    old_suite = {" ".join(raw_toks(dev)) for _n, _p, dev, _h in RULES}
    old_suite |= {" ".join(raw_toks(h)) for _n, _p, _d, h in RULES}

    rows = [json.loads(l) for l in PARA.open(encoding="utf-8") if l.strip()]
    accepted: dict[str, str] = {}
    rej = {"skip": 0, "shared_token": 0, "attested": 0, "in_old_suite": 0,
           "empty": 0, "too_long": 0}
    examples: dict[str, list] = {k: [] for k in rej}
    for r in rows:
        atom, para = str(r.get("atom", "")), str(r.get("paraphrase", "")).strip().lower()
        if not atom or not para:
            rej["empty"] += 1; continue
        if para == "skip":
            rej["skip"] += 1
            examples["skip"].append(atom); continue
        ptoks, atoks = raw_toks(para), raw_toks(atom)
        if not ptoks or len(ptoks) > 12:
            rej["too_long"] += 1; continue
        if stems(ptoks) & stems(atoks):
            rej["shared_token"] += 1
            examples["shared_token"].append(f"{atom} -> {para}"); continue
        norm = " ".join(ptoks)
        if norm in old_suite:
            rej["in_old_suite"] += 1
            examples["in_old_suite"].append(f"{atom} -> {para}"); continue
        if agent.ix.df(norm) > 0:
            # Attested text never reaches the resolver: suppression only fires on df == 0.
            rej["attested"] += 1
            examples["attested"].append(f"{atom} -> {para}"); continue
        accepted[atom] = norm

    print(f"generated {len(rows)} rows -> {len(accepted)} usable "
          f"({len(accepted) / max(len(rows), 1):.1%} pass rate)")
    for k, v in rej.items():
        if v:
            print(f"  rejected {k:<14} {v:>4}"
                  + ("   e.g. " + "; ".join(str(x)[:46] for x in examples[k][:2])
                     if examples[k] else ""))
    if len(accepted) < 40:
        print("\n  FEWER THAN 40 usable paraphrases. That is not an open vocabulary and")
        print("  the suite would be no better than the 27-phrase one. Fix generation")
        print("  rather than proceeding.")
        return

    def materialise(paraphrase: bool):
        out, atoms, sessions = [], 0, 0
        for sample in base:
            card = intent_card(products[str(sample["ground_truth"]["parent_asin"])])
            hit = False
            transformed = {}
            for group in ("hard_constraints", "soft_preferences"):
                vals = []
                for raw in card[group]:
                    key = " ".join(raw_toks(str(raw)))
                    if paraphrase and key in accepted:
                        vals.append(accepted[key]); atoms += 1; hit = True
                    else:
                        vals.append(str(raw))
                transformed[group] = vals
            seed = f"{sample.get('sample_id', '')}\0{sample.get('scenario_type', '')}"
            out.append({**sample, "intent_card": transformed,
                        "behavior": behavior_for(str(sample["scenario_type"]),
                                                 transformed, random.Random(seed)),
                        "semantic_value_family":
                            "open_vocab_paraphrase" if paraphrase else "canonical"})
            sessions += 1 if hit else 0
        return out, atoms, sessions

    OUTDIR.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1,
                "purpose": "target-disjoint open-vocabulary attribute paraphrase",
                "session_base": "catalog_review_distinct_800",
                "targets_disjoint_from_official200": True,
                "generator": "claude haiku (Anthropic)",
                "solver_under_test": "gpt-oss-120b (Groq)",
                "generator_solver_independent": True,
                "generated_rows": len(rows), "usable_paraphrases": len(accepted),
                "rejections": rej,
                "distinct_paraphrases_in_prior_suite": 27,
                "excluded_as_present_in_prior_suite": None}
    manifest["excluded_as_present_in_prior_suite"] = rej["in_old_suite"]
    for name, flag in (("review800_canonical_replay.jsonl", False),
                       ("review800_open_vocab_paraphrase.jsonl", True)):
        out, atoms, sessions = materialise(flag)
        path = OUTDIR / name
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
                        encoding="utf-8")
        manifest[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "sessions": len(out), "rewritten_atoms": atoms,
                          "sessions_with_rewrite": sessions}
        print(f"[saved] {path.name}  {len(out)} sessions, {atoms} atoms rewritten "
              f"across {sessions} sessions")
    (OUTDIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                          encoding="utf-8")
    print(f"\n  distinct paraphrases: {len(accepted)}  (prior suite: 27)")
    print(f"[saved] {OUTDIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
