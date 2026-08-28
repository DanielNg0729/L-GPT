"""Build participant-safe, deterministic internal robustness sets.

The organizer's exact 1,406 eligible target ASINs cannot be reconstructed from the
participant release: the joined review records and 800 private targets are deliberately
withheld.  This builder therefore creates an explicitly labelled *proxy*, never a claimed
private-set reconstruction.

The realistic proxy matches the disclosed cardinalities:
  200 public targets + 1,206 unseen proxy candidates = 1,406 total candidates
  800 distinct targets selected from the 1,206 unseen proxy candidates

The proxy pool is review-weighted without replacement using the empirically justified
P(target) proportional to rating_number prior, while every selected ASIN is excluded from
the public labels. The final 800 are a deterministic uniform split of that 1,206-product
proxy pool. Three catalogue-wide stress sets are emitted alongside it.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "robustness" / "config.json"
DEFAULT_OUT = ROOT / "robustness" / "sets"


def load_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else path.open
    with opener(path, "rt", encoding="utf-8") if path.suffix == ".gz" else opener(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def rating_number(product: dict) -> float:
    try:
        return max(0.0, float(product.get("rating_number") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def log_pop(products: dict[str, dict], asins: list[str] | set[str]) -> list[float]:
    return [math.log1p(rating_number(products[a])) for a in asins]


def distribution(products: dict[str, dict], asins: list[str] | set[str]) -> dict:
    raw = [rating_number(products[a]) for a in asins]
    logs = [math.log1p(v) for v in raw]
    return {
        "n": len(raw),
        "rating_number": {f"p{int(q * 100):02d}": quantile(raw, q)
                          for q in (0.1, 0.5, 0.9)},
        "log1p_rating_number": {f"p{int(q * 100):02d}": quantile(logs, q)
                                for q in (0.1, 0.5, 0.9)},
    }


def weighted_without_replacement(items: list[str], weights: list[float], n: int,
                                 seed: int) -> list[str]:
    """Efraimidis-Spirakis exponential race; deterministic for a fixed input order."""
    if n > len(items):
        raise ValueError(f"cannot select {n} distinct items from {len(items)}")
    rng = random.Random(seed)
    ranked = []
    for item, weight in zip(items, weights):
        w = max(float(weight), 1e-12)
        priority = -math.log(max(rng.random(), 1e-15)) / w
        ranked.append((priority, item))
    ranked.sort(key=lambda pair: (pair[0], pair[1]))
    return [item for _, item in ranked[:n]]


def scenario_sequence(n: int, mix: dict[str, float], seed: int) -> list[str]:
    values: list[str] = []
    for name, fraction in mix.items():
        values.extend([name] * int(round(n * float(fraction))))
    while len(values) < n:
        values.append("browsing")
    values = values[:n]
    random.Random(seed).shuffle(values)
    return values


def synthetic_profile(profiles: list[dict], index: int, seed: int) -> dict:
    """Recombine safe aggregates so rows are not copies of public profile records."""
    rng = random.Random(seed + index * 104729)
    a, b = profiles[rng.randrange(len(profiles))], profiles[rng.randrange(len(profiles))]
    tags = list(dict.fromkeys([str(x) for x in a.get("preference_tags", [])]
                              + [str(x) for x in b.get("preference_tags", [])]))[:3]
    average = float(a.get("average_prior_rating", 0.0) or 0.0)
    style = str(a.get("rating_style", "mixed"))
    frequency = str(b.get("purchase_frequency", "prior purchases available"))
    tag_text = ", ".join(tags) if tags else "general preferences"
    return {
        "average_prior_rating": average,
        "preference_tags": tags,
        "purchase_frequency": frequency,
        "rating_style": style,
        "summary": f"Prior purchases emphasize {tag_text}; ratings are {style}.",
    }


def rows_for(name: str, targets: list[str], profiles: list[dict], mix: dict[str, float],
             seed: int) -> list[dict]:
    scenarios = scenario_sequence(len(targets), mix, seed + 17)
    return [{
        "category_bucket": "clothing",
        "difficulty_bucket": "internal_proxy",
        "ground_truth": {"parent_asin": target},
        "sample_id": f"{name}_{i + 1:04d}",
        "scenario_type": scenarios[i],
        "user_profile": synthetic_profile(profiles, i, seed + 31),
    } for i, target in enumerate(targets)]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--public", type=Path, default=ROOT / "data" / "public_set.jsonl")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    products_list = load_jsonl(args.catalog)
    products = {str(row["parent_asin"]): row for row in products_list}
    public = load_jsonl(args.public)
    public_targets = {str(row["ground_truth"]["parent_asin"]) for row in public}
    profiles = [dict(row.get("user_profile") or {}) for row in public]
    if len(products) != 50_000:
        raise ValueError(f"expected 50,000 products, found {len(products):,}")
    expected_public = int(config["public_target_count"])
    if len(public_targets) != expected_public:
        raise ValueError(f"expected {expected_public} distinct public targets, found {len(public_targets)}")

    all_unseen = sorted(set(products) - public_targets)
    total_pool = int(config["candidate_target_pool_total"])
    unseen_pool_n = total_pool - expected_public
    private_n = int(config["private_session_count"])
    seed = int(config["seed"])
    qs = [float(q) for q in config["diagnostic_quantiles"]]
    public_q = [quantile(log_pop(products, public_targets), q) for q in qs]

    # Do not fit the 1,206-product pool to the 200-public popularity distribution. It is
    # mathematically impossible for 800 distinct targets to retain the public median:
    # even the 800 most-reviewed catalogue products have a lower median. The purchase
    # prior is used exactly once and uniqueness supplies the necessary depletion effect.
    exponent = float(config["review_weight_exponent"])
    proxy_weights = [(1.0 + rating_number(products[a])) ** exponent for a in all_unseen]
    unseen_proxy_pool = weighted_without_replacement(all_unseen, proxy_weights,
                                                     unseen_pool_n, seed + 700)
    proxy_q = [quantile(log_pop(products, unseen_proxy_pool), q) for q in qs]

    rng = random.Random(seed + 701)
    organizer_targets = list(unseen_proxy_pool)
    rng.shuffle(organizer_targets)
    organizer_targets = organizer_targets[:private_n]

    review_weights = [1.0 + rating_number(products[a]) for a in all_unseen]
    review_targets = weighted_without_replacement(all_unseen, review_weights, private_n,
                                                  seed + 702)
    uniform_targets = weighted_without_replacement(all_unseen, [1.0] * len(all_unseen),
                                                   private_n, seed + 703)
    inverse_weights = [1.0 / (1.0 + rating_number(products[a])) for a in all_unseen]
    inverse_targets = weighted_without_replacement(all_unseen, inverse_weights, private_n,
                                                   seed + 704)

    top_distinct = sorted(all_unseen, key=lambda a: (-rating_number(products[a]), a))[:private_n]

    sets = {
        "organizer_proxy_800": organizer_targets,
        "catalog_review_distinct_800": review_targets,
        "catalog_uniform_800": uniform_targets,
        "catalog_inverse_800": inverse_targets,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    files = {}
    for offset, (name, targets) in enumerate(sets.items()):
        if len(targets) != len(set(targets)) or set(targets) & public_targets:
            raise AssertionError(f"{name}: target distinctness/overlap invariant failed")
        rows = rows_for(name, targets, profiles, config["scenario_mix"], seed + offset * 1000)
        path = args.out / f"{name}.jsonl"
        write_jsonl(path, rows)
        files[name] = {"path": path.name, "sha256": sha256(path),
                       "target_distribution": distribution(products, targets)}

    manifest = {
        "schema_version": 1,
        "truth_status": "participant-safe proxy; not organizer private data",
        "source_catalog_rows": len(products),
        "public_targets": len(public_targets),
        "disclosed_candidate_target_pool": total_pool,
        "unseen_proxy_pool": unseen_pool_n,
        "private_like_sessions": private_n,
        "all_set_targets_are_distinct": True,
        "public_target_overlap_per_set": 0,
        "population_model": {
            "method": "P(target) proportional to rating_number, without replacement",
            "weight_exponent": exponent,
            "quantiles": qs,
            "public_log1p_rating_number": public_q,
            "proxy_pool_log1p_rating_number": proxy_q,
            "public_distribution": distribution(products, public_targets),
            "proxy_pool_distribution": distribution(products, unseen_proxy_pool),
            "top_800_distinct_mathematical_upper_bound": distribution(products, top_distinct),
            "why_public_was_not_matched": (
                "The public median cannot be preserved for 800 distinct targets; even "
                "the 800 most-reviewed unseen products impose a lower median."
            ),
        },
        "sets": files,
        "known_limitations": [
            "Exact eligible candidate ASINs are unavailable in participant data.",
            "Synthetic profiles recombine public aggregate fields; they are not private users.",
            "The simulator is official public code, but private paraphrasing is not modelled.",
        ],
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(sets)} sets to {args.out}")
    print(f"proxy review-weight exponent={exponent:g}; sampling is without replacement")
    for name, info in files.items():
        d = info["target_distribution"]["rating_number"]
        print(f"  {name:<30} p10={d['p10']:.0f} median={d['p50']:.0f} p90={d['p90']:.0f}")


if __name__ == "__main__":
    main()
