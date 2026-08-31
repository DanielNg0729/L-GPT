"""V2.60: the hostile condition -- every free channel removed.

WHY THE EXISTING CONDITIONS FLATTER US
---------------------------------------
`template` holds the VALUES constant while rewording the wrapper. The span node is exact
lookup against catalogue vocabulary, so a test that leaves values canonical guarantees the
lookup hits. Reporting "90.8% of the template gap recovered" is reporting on a test built
to let the mechanism win.

`both` reworded the values too, and the span node still scored +0.2001 there. That is not
because it resolves paraphrase -- it cannot -- but because two channels were still free:

  CATEGORY   never perturbed, and it cannot be. `initial_message` interpolates
             `coarse_category(product["categories"])`, a deterministic slice of the
             product's own taxonomy field. So the category in the message is BYTE-IDENTICAL
             to catalogue taxonomy and exact lookup is guaranteed to hit. That is not
             generalisation, it is identity -- a tautology, not a capability.
  UNTOUCHED VALUES  the open-vocabulary perturbation rewrote 1,280 atoms across 710
             sessions, but sessions carry several constraint values and any value without
             an accepted paraphrase stayed canonical. Those are free hits too.

This suite removes both, leaving only the question the innovation claim actually rests on:
when the wrapper is reworded AND every remaining value is paraphrased AND no category is
given, what does the machinery recover?

WHY REMOVE THE CATEGORY RATHER THAN PARAPHRASE IT. Paraphrasing it would need an invented
category vocabulary -- another generation step and another chance to fit the test set, which
this project has already done once. Removal is deterministic and needs no new data. It is
also arguably the MORE realistic setting: a person types "sneakers", not "Athletic Shoes
Running Shoes", so a natural-language simulator would not hand over a taxonomy string at
all. The category is neutralised at evaluation time by passing empty category lists, which
makes `coarse_category` return its constant fallback "clothing item" for every session --
present, uniform, and carrying zero discriminative signal.

WHAT IS BUILT
  review800_hostile_paraphrase.jsonl   only paraphrased values retained, values paraphrased
  review800_hostile_canonical.jsonl    THE SAME sessions and THE SAME value slots, values
                                       left canonical

The control is not optional. Dropping values changes how much evidence a session carries at
all, so a paraphrase score is only interpretable against a control with the identical value
subset. Comparing against the full-card ceiling would confound paraphrasing with card size.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/studies/build_hostile_suite.py
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "experiments" / "studies"
OV = V2 / "open_vocabulary"


def main() -> None:
    from evaluator.local_evaluator import behavior_for, load_jsonl
    from submission.agent import raw_toks

    rows = [json.loads(l) for l in
            (V2 / "open_vocabulary_paraphrases.jsonl").open(encoding="utf-8") if l.strip()]
    # paraphrase -> atom, and atom -> paraphrase, over the accepted set only
    a2p, p2a = {}, {}
    for r in rows:
        para = " ".join(raw_toks(str(r.get("paraphrase", ""))))
        atom = " ".join(raw_toks(str(r.get("atom", ""))))
        if para and atom and para != "skip":
            a2p[atom] = para
            p2a[para] = atom

    para_src = load_jsonl(OV / "review800_open_vocab_paraphrase.jsonl")
    canon_src = load_jsonl(OV / "review800_canonical_replay.jsonl")
    assert len(para_src) == len(canon_src)

    kept_para, kept_canon = [], []
    dropped_values = kept_values = 0
    for p, c in zip(para_src, canon_src):
        pcard, ccard = {}, {}
        hit = False
        for group in ("hard_constraints", "soft_preferences"):
            pv, cv = [], []
            for praw, craw in zip(p["intent_card"][group], c["intent_card"][group]):
                key = " ".join(raw_toks(str(praw)))
                if key in p2a:
                    # This value WAS paraphrased: keep it in both, paraphrased in one and
                    # canonical in the other, so the pair differs only in wording.
                    pv.append(str(praw))
                    cv.append(str(craw))
                    kept_values += 1
                    hit = True
                else:
                    dropped_values += 1
            pcard[group], ccard[group] = pv, cv
        if not hit:
            continue                      # nothing paraphrased here; the session is inert
        seed = f"{p.get('sample_id', '')}\0{p.get('scenario_type', '')}"
        kept_para.append({**p, "intent_card": pcard,
                          "behavior": behavior_for(str(p["scenario_type"]), pcard,
                                                   random.Random(seed)),
                          "semantic_value_family": "hostile_paraphrase"})
        kept_canon.append({**c, "intent_card": ccard,
                           "behavior": behavior_for(str(c["scenario_type"]), ccard,
                                                    random.Random(seed)),
                           "semantic_value_family": "hostile_canonical"})

    print(f"sessions: {len(para_src)} -> {len(kept_para)} "
          f"({len(para_src)-len(kept_para)} had no paraphrased value)")
    print(f"values kept (all paraphrased): {kept_values}")
    print(f"values dropped (canonical, would have been free hits): {dropped_values}")

    manifest = {"schema_version": 1,
                "purpose": "hostile condition: reworded wrapper, only-paraphrased values, "
                           "category neutralised at evaluation time",
                "session_base": "review800 open-vocabulary",
                "sessions": len(kept_para), "values_kept": kept_values,
                "values_dropped_as_canonical": dropped_values,
                "category": "neutralised by empty category lists -> 'clothing item'",
                "control": "review800_hostile_canonical.jsonl, identical value slots"}
    for name, data in (("review800_hostile_paraphrase.jsonl", kept_para),
                       ("review800_hostile_canonical.jsonl", kept_canon)):
        path = OV / name
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data),
                        encoding="utf-8")
        manifest[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "sessions": len(data)}
        print(f"[saved] {path.name}  {len(data)} sessions")
    (OV / "hostile_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                              encoding="utf-8")
    print(f"[saved] {OV / 'hostile_manifest.json'}")


if __name__ == "__main__":
    main()
