"""V2.34: restrict the Node 4 candidate index to phrases the simulator can actually emit.

THE DEFECT
----------
`catalogue_attribute_dictionary.jsonl` was built from phrases ATTESTED IN CATALOGUE TEXT
(7,922 entries). But the only phrases a customer can ever say are the ones
`intent_card()` emits, and that is a different set. Measured over all 50,000 products by
replaying the generator's own constraint construction:

    distinct phrases the simulator can EVER emit      58,801
    dictionary entries that are emittable              3,873
    dictionary entries that are NEVER emittable        4,049   (51.1% of the index)
    share of emission mass the dictionary covers        63.5%

So slightly over half of the retrieval index consists of candidates that cannot be a
correct answer under any session, while still competing for the top-k slots. Every one is
a pure distractor: it can displace the true canonical, and it can never be the truth.

This is not a modelling problem and no amount of training fixes it. It is an index
construction error.

WHAT THIS MEASURES
------------------
The identical frozen encoder and the identical cluster-aware metric from V2.32, changing
only the candidate set:

    full        all 7,922 dictionary phrases                      (status quo)
    emittable   the 3,873 that the generator can actually produce  (proposed)

If the emittable index improves recall, the gain is free -- no training, no data, no new
dependency -- and it also shrinks the index by half at inference.

HONEST CAVEAT: this uses `_sim_constraint_values`, the agent's reimplementation of the
released `intent_card()`. If the private simulator differs, the emittable set differs with
it. That is the same assumption the whole provenance thesis already rests on, so it adds
no new exposure -- but it is an assumption, not a proof.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/studies/evaluate_node4_emittable_index.py
"""
from __future__ import annotations

import importlib.util as ilu
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "experiments" / "studies"
OUT = V2 / "results" / "node4_emittable_index_v2_34.json"

_spec = ilu.spec_from_file_location("_v2_32", V2 / "evaluate_node4_cluster_aware.py")
_v232 = ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v232)
normalise, surface_key, load_encoder = _v232.normalise, _v232.surface_key, _v232.load_encoder


def emittable_set() -> set[str]:
    from submission.agent import _sim_constraint_values
    out: set[str] = set()
    with (ROOT / "data" / "catalog.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            for value in _sim_constraint_values(json.loads(line)):
                out.add(normalise(value))
    return out


def main() -> None:
    import numpy as np
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full = [json.loads(l)["canonical"]
            for l in (V2 / "catalogue_attribute_dictionary.jsonl").open(encoding="utf-8")
            if l.strip()]
    emit = emittable_set()
    keep = [c for c in full if normalise(c) in emit]
    print(f"index: full {len(full):,}  emittable {len(keep):,} "
          f"({len(full)-len(keep):,} unreachable distractors removed)")

    atoms = []
    dev = V2 / "sets" / "semantic_attribute_development_200.jsonl"
    for line in dev.open(encoding="utf-8"):
        if not line.strip():
            continue
        card = json.loads(line).get("semantic_card") or {}
        for grp in ("hard_constraints", "soft_preferences"):
            for a in card.get(grp, []):
                if a.get("paraphrase") and a.get("canonical"):
                    atoms.append((str(a["paraphrase"]), str(a["canonical"])))
    distinct = sorted(set(atoms))
    print(f"benchmark: {len(atoms)} rows, {len(distinct)} distinct concepts")

    lost = [c for _q, c in distinct if normalise(c) not in emit]
    if lost:
        print(f"  NOTE {len(lost)} benchmark canonicals are not emittable and are dropped "
              f"from the emittable arm: {sorted(set(lost))[:5]}")

    encode = load_encoder(device)
    queries = encode([q for q, _c in distinct])

    report = {"experiment": "V2.34 emittable-index Node 4 rescore (frozen MiniLM)",
              "full_index": len(full), "emittable_index": len(keep),
              "removed_distractors": len(full) - len(keep),
              "benchmark_rows": len(atoms), "benchmark_concepts": len(distinct),
              "arms": {}}

    print(f"\n{'index':<14}{'n idx':>8}{'R@1':>9}{'R@3':>9}{'R@5':>9}{'R@10':>9}{'MRR':>9}{'scored':>8}")
    print("-" * 74)
    for label, canonicals in (("full", full), ("emittable", keep)):
        norm_to_idx = {normalise(c): i for i, c in enumerate(canonicals)}
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(canonicals):
            groups.setdefault(surface_key(c), []).append(i)
        matrix = encode(canonicals)
        order = np.argsort(-(queries @ matrix.T), axis=1)
        ranks = []
        for (_q, canon), row in zip(distinct, order):
            idx = norm_to_idx.get(normalise(canon))
            if idx is None:
                continue                       # not representable in this index
            acc = set(groups.get(surface_key(canonicals[idx]), [idx]))
            ranks.append(next((p + 1 for p, j in enumerate(row) if int(j) in acc),
                              len(canonicals)))
        n = max(len(ranks), 1)
        row = {f"R@{k}": round(sum(r <= k for r in ranks) / n, 4) for k in (1, 3, 5, 10)}
        row["MRR"] = round(sum(1.0 / r for r in ranks) / n, 4)
        row["scored_concepts"] = len(ranks)
        report["arms"][label] = row
        print(f"{label:<14}{len(canonicals):>8,}{row['R@1']:>9.4f}{row['R@3']:>9.4f}"
              f"{row['R@5']:>9.4f}{row['R@10']:>9.4f}{row['MRR']:>9.4f}"
              f"{row['scored_concepts']:>8}")

    a, b = report["arms"]["full"], report["arms"]["emittable"]
    print(f"\n  delta from removing unreachable candidates only (no training, no data):")
    for k in ("R@1", "R@3", "R@5", "R@10", "MRR"):
        print(f"    {k:<5}{a[k]:>9.4f} -> {b[k]:>7.4f}   {b[k]-a[k]:+.4f}")
    print("\n  NOTE the two arms score different concept counts when a benchmark canonical")
    print("  is not emittable; compare with that in mind rather than as a like-for-like row.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
