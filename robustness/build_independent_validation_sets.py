"""Build preregistered population-disturbance folds for final candidate validation.

The existing four ``optuna_v2_sets/population_shift_*`` files are independent samples
from the same population process and form stage 1. This builder creates stage 2 by moving
exactly 5%, 10% or 20% probability mass between the lower and upper five popularity
deciles. Everything else stays fixed: 800 distinct targets, a 1,206-product unseen proxy
pool, zero public overlap, and the official 40/40/15/5 scenario mix.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from robustness.build_sets import (
    ROOT,
    distribution,
    load_jsonl,
    rating_number,
    scenario_sequence,
    sha256,
    synthetic_profile,
    weighted_without_replacement,
    write_jsonl,
)
from robustness.build_optuna_v2_sets import popularity_strata

DEFAULT_OUT = ROOT / "robustness" / "independent_validation_sets"


def counts_for(tv: float, direction: str) -> list[int]:
    moved = round(800 * tv)
    if moved % 5:
        raise ValueError("TV movement must divide evenly across five strata")
    delta = moved // 5
    low, high = 80 - delta, 80 + delta
    counts = [low] * 5 + [high] * 5
    if direction == "less_popular":
        counts.reverse()
    if sum(counts) != 800:
        raise AssertionError("stratum allocation must total 800")
    return counts


def make_rows(name: str, products: dict[str, dict], strata: list[list[str]],
              counts: list[int], profiles: list[dict], seed: int) -> tuple[list[dict], list[str]]:
    targets: list[str] = []
    for i, (stratum, count) in enumerate(zip(strata, counts)):
        targets.extend(random.Random(seed + i * 1009).sample(stratum, count))
    random.Random(seed + 20_000).shuffle(targets)
    scenarios = scenario_sequence(800, {
        "buying": .40, "browsing": .40, "intent_override": .15, "boundary": .05,
    }, seed + 30_000)
    rows = [{
        "category_bucket": "clothing",
        "difficulty_bucket": name,
        "ground_truth": {"parent_asin": target},
        "sample_id": f"{name}_{i + 1:04d}",
        "scenario_type": scenarios[i],
        "user_profile": synthetic_profile(profiles, i, seed + 40_000),
    } for i, target in enumerate(targets)]
    return rows, targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--public", type=Path, default=ROOT / "data" / "public_set.jsonl")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    products_list = load_jsonl(args.catalog)
    products = {str(row["parent_asin"]): row for row in products_list}
    public = load_jsonl(args.public)
    public_targets = {str(row["ground_truth"]["parent_asin"]) for row in public}
    profiles = [dict(row.get("user_profile") or {}) for row in public]
    unseen = sorted(set(products) - public_targets)
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "truth_status": "participant-safe controlled population perturbations",
        "candidate_registry": "../validation_candidates.json",
        "same_population_stage": "../optuna_v2_sets/population_shift_01_800.jsonl through 04",
        "disturbance_metric": "total variation distance over 10 popularity-decile probabilities",
        "levels": [0.05, 0.10, 0.20],
        "directions": ["less_popular", "more_popular"],
        "replicates": args.replicates,
        "folds": {},
    }
    for replicate in range(1, args.replicates + 1):
        pool_seed = args.seed + replicate * 100_003
        weights = [1.0 + rating_number(products[asin]) for asin in unseen]
        pool = weighted_without_replacement(unseen, weights, 1206, pool_seed)
        strata = popularity_strata(pool, products)
        for direction in manifest["directions"]:
            for tv in manifest["levels"]:
                level = int(round(tv * 100))
                name = f"tv{level:02d}_{direction}_r{replicate:02d}"
                counts = counts_for(tv, direction)
                rows, targets = make_rows(name, products, strata, counts, profiles,
                                          pool_seed + level * 100 + (0 if direction == "less_popular" else 50))
                if len(set(targets)) != 800 or set(targets) & public_targets:
                    raise AssertionError(f"{name}: distinctness/public-overlap invariant failed")
                path = args.out / f"{name}.jsonl"
                write_jsonl(path, rows)
                empirical_tv = 0.5 * sum(abs(count / 800 - 0.1) for count in counts)
                manifest["folds"][name] = {
                    "path": path.name,
                    "replicate": replicate,
                    "direction": direction,
                    "requested_tv": tv,
                    "empirical_tv": empirical_tv,
                    "stratum_counts_low_to_high": counts,
                    "target_distribution": distribution(products, targets),
                    "sha256": sha256(path),
                }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                            encoding="utf-8")
    print(f"wrote {len(manifest['folds'])} disturbed-population folds to {args.out}")
    for name, info in manifest["folds"].items():
        d = info["target_distribution"]["rating_number"]
        print(f"  {name:<30} TV={info['empirical_tv']:.2f} median={d['p50']:.0f}")


if __name__ == "__main__":
    main()
