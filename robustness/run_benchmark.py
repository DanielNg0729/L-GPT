"""Run the submitted agent against generated internal robustness sets."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent  # noqa: E402


def compact(result: dict) -> dict:
    return {
        "sample_count": result["sample_count"],
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "efficiency": result["efficiency"],
        "technical_score": result["recommended_technical_score"],
        "scenario_metrics": result["scenario_metrics"],
        "usage": result.get("usage", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--sets", type=Path, default=ROOT / "robustness" / "sets")
    parser.add_argument("--only", nargs="*", help="set basenames without .jsonl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "robustness" / "results.json")
    args = parser.parse_args()

    # Internal robustness is offline and reproducible by default.
    os.environ.setdefault("LLM_EXTRACT", "0")
    os.environ.setdefault("LLM_RERANK", "0")
    ids, categories, products = catalog_index(args.catalog)
    names = args.only or [
        "organizer_proxy_800",
        "catalog_review_distinct_800",
        "catalog_uniform_800",
        "catalog_inverse_800",
    ]
    output = {
        "truth_status": "participant-safe proxy; not organizer private data",
        "catalog": str(args.catalog),
        "sets": {},
    }
    for name in names:
        started = time.time()
        samples = load_jsonl(args.sets / f"{name}.jsonl")
        result = evaluate(Agent(args.catalog), samples, ids, categories, products)
        output["sets"][name] = {**compact(result), "wall_seconds": time.time() - started}
        row = output["sets"][name]
        print(f"{name:<30} score={row['technical_score']:.6f} "
              f"HR={row['hit_rate_at_10']:.3f} MRR={row['mrr']:.3f} "
              f"MTTC={row['mttc']:.3f} wall={row['wall_seconds']:.1f}s")
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
