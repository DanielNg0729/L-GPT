"""Evaluate frozen Pareto candidates on untouched same-population and disturbed folds."""
from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notes" / "eda"))
sys.path.insert(0, str(ROOT / "notes" / ".ml_deps"))

BASE = importlib.import_module("55_optuna_official_v2")
REGISTRY = ROOT / "robustness" / "validation_candidates.json"
SAME_DIR = ROOT / "robustness" / "optuna_v2_sets"
DIST_DIR = ROOT / "robustness" / "independent_validation_sets"
OUTPUT = ROOT / "notes" / "eda" / "out_57_independent_validation.json"


def fold_paths() -> list[tuple[str, str, Path, dict]]:
    folds = [(f"same_population_{i:02d}", "same_population",
              SAME_DIR / f"population_shift_{i:02d}_800.jsonl", {}) for i in range(1, 5)]
    manifest = json.loads((DIST_DIR / "manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["folds"].items():
        folds.append((name, "disturbed_population", DIST_DIR / info["path"], info))
    return folds


def evaluate_candidate(item: tuple[str, dict]) -> tuple[str, dict]:
    name, entry = item
    g = BASE.boot()
    from evaluator.local_evaluator import load_jsonl
    params = entry["params"]
    saved = g["base"].ix.BM25
    g["base"].ix.BM25 = (
        f'bm25(p, 0.0, {params["BM_TITLE"]:.3f}, {params["BM_CATS"]:.3f}, '
        f'{params["BM_FEAT"]:.3f}, {params["BM_DETAILS"]:.3f}, '
        f'{params["BM_STORE"]:.3f}, {params["BM_DESC"]:.3f})'
    )
    results = {}
    try:
        for fold_name, stage, path, metadata in fold_paths():
            samples = load_jsonl(path)
            result = g["evaluate"](BASE.build_agent(params), samples,
                                   g["cid"], g["cats"], g["prods"])
            results[fold_name] = {
                "stage": stage,
                "technical_score": result["recommended_technical_score"],
                "hit_rate_at_10": result["hit_rate_at_10"],
                "mrr": result["mrr"],
                "mttc": result["mttc"],
                "direction": metadata.get("direction"),
                "tv": metadata.get("requested_tv"),
                "replicate": metadata.get("replicate"),
            }
    finally:
        g["base"].ix.BM25 = saved
    return name, {"source": entry["source"], "folds": results}


def stats(values: list[float]) -> dict:
    return {"mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def main() -> None:
    t0 = time.time()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    candidates = list(registry["candidates"].items())
    with mp.Pool(processes=len(candidates)) as pool:
        evaluated = dict(pool.map(evaluate_candidate, candidates))

    baseline = evaluated["shipped"]["folds"]
    summary = {}
    for candidate, payload in evaluated.items():
        folds = payload["folds"]
        same_names = [name for name, row in folds.items() if row["stage"] == "same_population"]
        same_scores = [folds[name]["technical_score"] for name in same_names]
        same_deltas = [folds[name]["technical_score"] - baseline[name]["technical_score"]
                       for name in same_names]
        disturbed = {}
        for direction in ("less_popular", "more_popular"):
            for tv in (.05, .10, .20):
                names = [name for name, row in folds.items()
                         if row["direction"] == direction and row["tv"] == tv]
                scores = [folds[name]["technical_score"] for name in names]
                deltas = [folds[name]["technical_score"] - baseline[name]["technical_score"]
                          for name in names]
                disturbed[f"{direction}_tv{int(tv * 100):02d}"] = {
                    "score": stats(scores), "delta_vs_shipped": stats(deltas),
                }
        summary[candidate] = {
            "same_population_score": stats(same_scores),
            "same_population_delta_vs_shipped": stats(same_deltas),
            "disturbed_population": disturbed,
        }

    output = {
        "preregistered_candidates": str(REGISTRY.relative_to(ROOT)),
        "same_population_folds": 4,
        "disturbed_population_folds": 12,
        "wall_seconds": time.time() - t0,
        "summary": summary,
        "raw": evaluated,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("SAME POPULATION: mean score / mean delta vs shipped")
    for name, row in summary.items():
        print(f"  {name:<20} {row['same_population_score']['mean']:.6f}  "
              f"{row['same_population_delta_vs_shipped']['mean']:+.6f}")
    print("\nDISTURBED POPULATION: mean delta vs shipped")
    for name, row in summary.items():
        values = row["disturbed_population"]
        print(f"  {name:<20} " + " ".join(
            f"{key}={value['delta_vs_shipped']['mean']:+.6f}"
            for key, value in values.items()))
    print(f"\n[saved] {OUTPUT}  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    mp.freeze_support()
    main()
