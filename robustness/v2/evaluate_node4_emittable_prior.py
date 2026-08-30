"""V2.35: the emittable inventory as a SOFT PRIOR, not a hard index filter.

WHY V2.34 WAS WRONG IN FORM (though right in substance)
-------------------------------------------------------
V2.34 restricted the Node 4 candidate index to the 3,873 dictionary phrases the released
`intent_card()` can emit, and measured +52% MRR. The gain is real. The mechanism is not an
assumption for the RELEASED simulator: every constraint a customer speaks is
`intent_card()` output, so the canonical behind any paraphrase is emittable by
construction.

But implementing it as a hard prune is wrong twice over.

1. IT BROKE THE PROJECT'S OWN GATE DISCIPLINE. `V2_SEMANTIC_ROADMAP.md` assigns Node 3/4
   the G3 class: a semantic narrowing "must retain a global fallback and may not
   hard-prune". A hard index filter is a G1 action taken on a G3 signal.

2. IT ALREADY DESTROYED CORRECT ANSWERS. Eleven of the 67 benchmark concepts became
   unreachable under the filter -- `Green`, `Beige`, `Brown`, `Gray` -- because the
   simulator emits `color: green` and never bare `green`. The filter did not remove
   distractors there; it removed the truth.

The private simulator is the one case we cannot verify. A soft prior loses a bonus if our
reconstruction is wrong; a hard filter loses the answer.

WHAT THIS MEASURES
------------------
Same frozen encoder, same cluster-aware metric, one change: emittable candidates receive an
additive similarity bonus instead of being the only candidates. Every phrase in the full
7,922 index remains reachable at every bonus level.

    bonus 0.00   identical to the full-index baseline (global fallback intact)
    bonus -> inf approaches V2.34's hard filter

Reporting the sweep shows how much of the +52% survives while keeping the fallback, and
whether there is a bonus that captures the gain without ever making a correct answer
unreachable.

TRAINING IMPLICATION, STATED HERE BECAUSE IT IS THE MORE IMPORTANT HALF: this prior is an
INFERENCE-time device only. Training data must target the full catalogue inventory, not the
emittable subset. An encoder trained only on emittable targets never learns the other 4,049
phrases, so if the private simulator emits any of them the model is blind and we have no
way to detect it. Train against the harder distribution; apply the provable prior at
inference; measure the two separately.

Run:  PYTHONIOENCODING=utf-8 python -u robustness/v2/evaluate_node4_emittable_prior.py
"""
from __future__ import annotations

import importlib.util as ilu
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "robustness" / "v2"
OUT = V2 / "results" / "node4_emittable_prior_v2_35.json"

_s = ilu.spec_from_file_location("_v2_32", V2 / "evaluate_node4_cluster_aware.py")
_m = ilu.module_from_spec(_s)
_s.loader.exec_module(_m)
normalise, surface_key, load_encoder = _m.normalise, _m.surface_key, _m.load_encoder

_s2 = ilu.spec_from_file_location("_v2_34", V2 / "evaluate_node4_emittable_index.py")
_m2 = ilu.module_from_spec(_s2)
_s2.loader.exec_module(_m2)
emittable_set = _m2.emittable_set


def main() -> None:
    import numpy as np
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    canonicals = [json.loads(l)["canonical"]
                  for l in (V2 / "catalogue_attribute_dictionary.jsonl").open(encoding="utf-8")
                  if l.strip()]
    emit = emittable_set()
    is_emit = np.array([1.0 if normalise(c) in emit else 0.0 for c in canonicals],
                       dtype=np.float32)
    print(f"index {len(canonicals):,}  emittable {int(is_emit.sum()):,}  "
          f"non-emittable {int((1 - is_emit).sum()):,} (kept, never pruned)")

    atoms = []
    for line in (V2 / "sets" / "semantic_attribute_development_200.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        card = json.loads(line).get("semantic_card") or {}
        for grp in ("hard_constraints", "soft_preferences"):
            for a in card.get(grp, []):
                if a.get("paraphrase") and a.get("canonical"):
                    atoms.append((str(a["paraphrase"]), str(a["canonical"])))
    distinct = sorted(set(atoms))
    n_unreach = sum(1 for _q, c in distinct if normalise(c) not in emit)
    print(f"benchmark concepts {len(distinct)}  of which NOT emittable {n_unreach} "
          f"(a hard filter makes these unreachable; a prior does not)\n")

    norm_to_idx = {normalise(c): i for i, c in enumerate(canonicals)}
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(canonicals):
        groups.setdefault(surface_key(c), []).append(i)

    encode = load_encoder(device)
    matrix = encode(canonicals)
    queries = encode([q for q, _c in distinct])
    sims = queries @ matrix.T

    report = {"experiment": "V2.35 emittable prior (soft, global fallback retained)",
              "index": len(canonicals), "emittable": int(is_emit.sum()),
              "benchmark_concepts": len(distinct),
              "benchmark_concepts_not_emittable": n_unreach, "sweep": {}}

    print(f"{'bonus':>7}{'R@1':>9}{'R@3':>9}{'R@5':>9}{'R@10':>9}{'MRR':>9}"
          f"{'unreachable':>13}")
    print("-" * 65)
    for bonus in (0.0, 0.02, 0.05, 0.10, 0.20, 0.50):
        order = np.argsort(-(sims + bonus * is_emit[None, :]), axis=1)
        ranks = []
        for (_q, canon), row in zip(distinct, order):
            idx = norm_to_idx.get(normalise(canon))
            if idx is None:
                continue
            acc = set(groups.get(surface_key(canonicals[idx]), [idx]))
            ranks.append(next((p + 1 for p, j in enumerate(row) if int(j) in acc),
                              len(canonicals)))
        n = max(len(ranks), 1)
        row = {f"R@{k}": round(sum(r <= k for r in ranks) / n, 4) for k in (1, 3, 5, 10)}
        row["MRR"] = round(sum(1.0 / r for r in ranks) / n, 4)
        row["scored"] = len(ranks)
        report["sweep"][f"{bonus:.2f}"] = row
        print(f"{bonus:>7.2f}{row['R@1']:>9.4f}{row['R@3']:>9.4f}{row['R@5']:>9.4f}"
              f"{row['R@10']:>9.4f}{row['MRR']:>9.4f}{'0':>13}")

    base = report["sweep"]["0.00"]
    best = max(report["sweep"], key=lambda k: report["sweep"][k]["MRR"])
    print(f"\n  baseline MRR {base['MRR']:.4f} -> best prior ({best}) "
          f"{report['sweep'][best]['MRR']:.4f}  "
          f"({report['sweep'][best]['MRR'] - base['MRR']:+.4f})")
    print(f"  every concept stays reachable at every bonus: the {n_unreach} non-emittable")
    print(f"  benchmark concepts are still scored, which the hard filter could not do.")
    print("\n  REMINDER: this is an inference-time prior. Training targets must span the")
    print("  full inventory, or the encoder never learns the non-emittable half.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
