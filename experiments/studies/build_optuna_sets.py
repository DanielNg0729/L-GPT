"""Build fixed, organizer-statistics-aligned folds for the Optuna v2 study.

Every fold models the disclosed structure without claiming private labels:

* 200 public targets are excluded from an unseen 1,206-product proxy pool;
* the proxy pool is drawn with P(target) proportional to rating_number;
* its products are divided into ten popularity strata;
* exactly 80 distinct targets are sampled from every stratum; and
* every stratum receives 32 buying, 32 browsing, 12 override and 4 boundary rows.

The primary fold is fixed for optimization. Independently seeded shift folds are held out
for candidate validation. A trial never receives a different dataset from another trial.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from robustness.build_sets import (
    ROOT,
    distribution,
    load_jsonl,
    rating_number,
    sha256,
    synthetic_profile,
    weighted_without_replacement,
    write_jsonl,
)

DEFAULT_OUT = ROOT / "robustness" / "optuna_v2_sets"
SCENARIOS_PER_STRATUM = (
    ["buying"] * 32
    + ["browsing"] * 32
    + ["intent_override"] * 12
    + ["boundary"] * 4
)


def popularity_strata(pool: list[str], products: dict[str, dict], n: int = 10) -> list[list[str]]:
    ordered = sorted(pool, key=lambda asin: (rating_number(products[asin]), asin))
    return [ordered[(i * len(ordered)) // n:((i + 1) * len(ordered)) // n]
            for i in range(n)]


def build_fold(name: str, products: dict[str, dict], unseen: list[str], profiles: list[dict],
               seed: int) -> tuple[list[dict], dict]:
    weights = [1.0 + rating_number(products[asin]) for asin in unseen]
    pool = weighted_without_replacement(unseen, weights, 1206, seed=seed)
    strata = popularity_strata(pool, products)
    rows: list[dict] = []
    selected: list[str] = []
    stratum_manifest: list[dict] = []

    for stratum_index, stratum in enumerate(strata):
        rng = random.Random(seed + 10_000 + stratum_index)
        targets = rng.sample(stratum, 80)
        scenarios = list(SCENARIOS_PER_STRATUM)
        rng.shuffle(scenarios)
        selected.extend(targets)
        for local_index, (target, scenario) in enumerate(zip(targets, scenarios)):
            row_index = stratum_index * 80 + local_index
            rows.append({
                "category_bucket": "clothing",
                "difficulty_bucket": f"official_proxy_pop_decile_{stratum_index + 1:02d}",
                "ground_truth": {"parent_asin": target},
                "population_stratum": stratum_index + 1,
                "sample_id": f"{name}_{row_index + 1:04d}",
                "scenario_type": scenario,
                "user_profile": synthetic_profile(profiles, row_index, seed + 20_000),
            })
        stratum_manifest.append({
            "stratum": stratum_index + 1,
            "proxy_pool_products": len(stratum),
            "selected_targets": 80,
            "target_distribution": distribution(products, targets),
        })

    if len(selected) != 800 or len(set(selected)) != 800:
        raise AssertionError(f"{name}: expected 800 distinct targets")
    random.Random(seed + 30_000).shuffle(rows)
    return rows, {
        "seed": seed,
        "proxy_pool_products": 1206,
        "selected_targets": 800,
        "target_distribution": distribution(products, selected),
        "popularity_strata": stratum_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--public", type=Path, default=ROOT / "data" / "public_set.jsonl")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--shift-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    products_list = load_jsonl(args.catalog)
    products = {str(row["parent_asin"]): row for row in products_list}
    public = load_jsonl(args.public)
    public_targets = {str(row["ground_truth"]["parent_asin"]) for row in public}
    profiles = [dict(row.get("user_profile") or {}) for row in public]
    if len(products) != 50_000 or len(public_targets) != 200:
        raise ValueError("official release cardinality invariant failed")
    unseen = sorted(set(products) - public_targets)

    args.out.mkdir(parents=True, exist_ok=True)
    fold_names = ["primary_800"] + [f"population_shift_{i:02d}_800"
                                      for i in range(1, args.shift_folds + 1)]
    manifest = {
        "schema_version": 1,
        "truth_status": "participant-safe proxy; not organizer private data",
        "official_public_sessions": 200,
        "disclosed_candidate_target_pool": 1406,
        "unseen_proxy_pool_per_fold": 1206,
        "private_like_sessions_per_fold": 800,
        "sampling": "review-weighted without replacement, then equal sampling from 10 popularity strata",
        "scenario_mix_per_stratum": {
            "buying": 32, "browsing": 32, "intent_override": 12, "boundary": 4,
        },
        "optimization_fold": "primary_800",
        "held_out_population_folds": fold_names[1:],
        "folds": {},
    }
    target_sets: dict[str, set[str]] = {}
    for i, name in enumerate(fold_names):
        seed = args.seed + i * 100_003
        rows, info = build_fold(name, products, unseen, profiles, seed)
        path = args.out / f"{name}.jsonl"
        write_jsonl(path, rows)
        targets = {row["ground_truth"]["parent_asin"] for row in rows}
        if targets & public_targets:
            raise AssertionError(f"{name}: public target overlap")
        target_sets[name] = targets
        info.update(path=path.name, sha256=sha256(path), public_target_overlap=0)
        manifest["folds"][name] = info

    manifest["pairwise_target_overlap"] = {
        f"{a}__{b}": len(target_sets[a] & target_sets[b])
        for i, a in enumerate(fold_names) for b in fold_names[i + 1:]
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(fold_names)} fixed folds to {args.out}")
    for name, info in manifest["folds"].items():
        d = info["target_distribution"]["rating_number"]
        print(f"  {name:<30} p10={d['p10']:.0f} median={d['p50']:.0f} p90={d['p90']:.0f}")


if __name__ == "__main__":
    main()
