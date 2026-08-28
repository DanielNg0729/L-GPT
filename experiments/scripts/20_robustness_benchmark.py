"""Internal robustness benchmark for the shipped deterministic pipeline.

This is deliberately analysis-only: it imports the submitted agent but never changes
its source, its dependencies, or its online/offline configuration.  It answers two
separate questions that are often conflated as "generalisation":

* Population stability: stratified bootstrap confidence intervals over the released
  200-session population.  This estimates sensitivity to a different sample from the
  SAME stated distribution; it is not evidence about a different product dataset.
* Organizer uncertainty: documented counterfactuals from experiments 05 and 06
  (paraphrase and `other` semantics), plus interface-level failure modes.

Risk scale: 1 = invariant/confirmed, 2 = degrades but remains viable,
3 = can become detrimental or relies on an unconfirmed population assumption.

Run: python experiments/scripts/20_robustness_benchmark.py
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from submission.agent import Agent  # noqa: E402

CATALOG = ROOT / "data" / "catalog.jsonl"
PUBLIC = ROOT / "data" / "public_set.jsonl"
OUT_JSON = ROOT / "experiments" / "results" / "out_20_robustness.json"
OUT_MD = ROOT / "docs" / "validation" / "robustness_benchmark.md"


def score(rows: list[dict]) -> float:
    """Recompute the official TechnicalScore for a sampled set of session outcomes."""
    n = len(rows)
    hit = sum(int(row["hit"]) for row in rows) / n
    mrr = statistics.fmean(row["reciprocal_rank"] for row in rows)
    mttc = statistics.fmean(row["first_hit_turn"] or 11 for row in rows)
    return 0.50 * hit + 0.30 * mrr + 0.20 * max(0.0, min(1.0, (11.0 - mttc) / 10.0))


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    i = (len(ordered) - 1) * p
    lo, hi = math.floor(i), math.ceil(i)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (i - lo)


def bootstrap(rows: list[dict], rounds: int = 5000, seed: int = 20260828) -> dict:
    """Stratified resampling preserves the organiser's stated scenario mix."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["scenario_type"]].append(row)
    rng = random.Random(seed)
    draws = []
    for _ in range(rounds):
        draw = [rng.choice(bucket) for bucket in buckets.values() for _ in range(len(bucket))]
        draws.append(score(draw))
    return {
        "rounds": rounds,
        "point_estimate": round(score(rows), 6),
        "mean": round(statistics.fmean(draws), 6),
        "stdev": round(statistics.stdev(draws), 6),
        "p05": round(percentile(draws, 0.05), 6),
        "p95": round(percentile(draws, 0.95), 6),
        "min": round(min(draws), 6),
        "max": round(max(draws), 6),
    }


# Every row names the exact component currently used in submission/agent.py.  Ratings
# are intentionally about the CORE IDEA, not whether a constant was tuned.
COMPONENTS = [
    ("Safety envelope and valid-ID fallback", 1, 1,
     "respond() cannot raise; invalid API/network output falls back to the previous local ranking."),
    ("Session ledger", 1, 1,
     "The interface confirms messages are turn-local; maintaining history is required, not population-tuned."),
    ("Sequential disclosure (top-1 through turn 9)", 3, 3,
     "Current source returns one recommendation until turn 10. It relies on the evaluator ending a session on any hit and changes the stated Top-10 shopping behaviour."),
    ("Rejection-feedback demotion", 1, 1,
     "A later turn proves the prior list did not contain the target; override handling clears stale negatives."),
    ("No profile prior (W_PROFILE = 0)", 1, 1,
     "Profile overlap was harmful on held-out data, so the shipped agent deliberately does not use it."),
    ("Catalog-grounded mining", 2, 2,
     "Dictionary segmentation is stable under template changes, but still needs catalogue-attested lexical fragments."),
    ("Fuzzy longest-attested-substring resolution", 2, 2,
     "Held-out improvement (+0.0081) fixes formatting/synthetic-field drift; it cannot recover semantic paraphrases."),
    ("Information-ordered probe policy", 2, 2,
     "Order and reachable attributes were measured on this simulator; typed rotation remains a viable fallback."),
    ("FTS5 multi-rung recall ladder", 2, 2,
     "POOL, DF cap and rung order are tuned, but graceful AND→OR backoff survives exact-match misses."),
    ("IDF-weighted coverage ranker", 2, 2,
     "Weights were tuned but survive held-out evaluation; it loses discriminative power when wording becomes semantic."),
    ("Category evidence weight", 2, 2,
     "Broad optimum in the weight sweep, but category text and its mapping are simulator/catalogue dependent."),
    ("Popularity prior (W_POP = 0.35)", 3, 3,
     "Largest gain on this purchase-derived population; relies on rating_number tracking purchase likelihood in private data."),
    ("Exact phrase/provenance thesis", 3, 3,
     "Core retrieval assumption: customer constraints derive verbatim from the target document. Strong on this harness, brittle to intent-card paraphrase."),
    ("Template extraction channel", 1, 3,
     "No population learning, but template-only extraction fell from 0.8097 clean to 0.3513 under light paraphrase; hybrid fallback contains the damage."),
    ("Optional Groq tie-breaker (OFF by default)", 3, 3,
     "+0.000124 on public data, but needs network, credentials and a population-specific tie set. It is excluded from shipped cost and official fallback."),
]


def component_ledger() -> list[dict]:
    rows = []
    for name, population_risk, organiser_risk, evidence in COMPONENTS:
        rows.append({
            "component": name,
            "population_risk": population_risk,
            "organiser_risk": organiser_risk,
            "overall_risk": max(population_risk, organiser_risk),
            "evidence": evidence,
        })
    return sorted(rows, key=lambda row: (row["overall_risk"], row["population_risk"], row["organiser_risk"], row["component"]))


def markdown(report: dict) -> str:
    pop = report["population_bootstrap"]
    lines = [
        "# Internal Robustness Benchmark",
        "",
        "This analysis is separate from the submitted agent. It adds no runtime cost, model dependency, or network requirement to the shipped deterministic path.",
        "",
        "## What it measures",
        "",
        f"- **Population variance:** {pop['rounds']:,} stratified bootstrap resamples of the released 200 sessions. TechnicalScore: `{pop['point_estimate']:.6f}`; 90% interval: `{pop['p05']:.6f}`-`{pop['p95']:.6f}`; SD: `{pop['stdev']:.6f}`.",
        "- **Organizer uncertainty:** existing counterfactual probes and contract analysis. It does not claim to measure performance on a truly different catalogue or intent generator - no such dataset is available locally.",
        "",
        "Risk scale: **1** = invariant/confirmed; **2** = degrades but remains viable; **3** = can become detrimental or relies on an unconfirmed population assumption. Lower is better.",
        "",
        "## Ranked component ledger",
        "",
        "| Component | Population risk | Organizer risk | Overall | Evidence |",
        "|---|---:|---:|---:|---|",
    ]
    for row in report["components"]:
        lines.append(f"| {row['component']} | {row['population_risk']} | {row['organiser_risk']} | {row['overall_risk']} | {row['evidence']} |")
    lines.extend([
        "",
        "## Measured uncertainty probes",
        "",
        "| Condition | TechnicalScore | Interpretation |",
        "|---|---:|---|",
        f"| Current deterministic reference | {report['reference']['recommended_technical_score']:.6f} | Current source with network/LLM disabled. |",
        "| Typed rotation; `other` unavailable | 0.828452* | Earlier pipeline probe: the dialogue strategy remains viable without the `other` quirk. |",
        "| Hybrid extraction, light paraphrase | 0.659230* | Substantial degradation, but the catalogue-grounded fallback retains a meaningful floor. |",
        "| Template-only, light paraphrase | 0.351322* | Adversarial phrasing makes the template channel unsafe as a standalone design. |",
        "| Optional Groq tie-breaker | 0.910643* | Measured before the current sequential-disclosure change; excluded from the shipped scoring path. |",
        "",
        "* These values are from earlier-stage robustness probes or a prior disclosure policy, so they establish failure direction rather than a final-agent score. Re-running these perturbations against the final agent is the next empirical priority.",
        "",
        "## Bottom line",
        "",
        "The pipeline is statistically stable against resampling of the released population, but that is not the same as cross-dataset generalization. The main non-transfer risks are the exact-provenance premise, the popularity prior, and the current sequential-disclosure policy. Keep the deterministic offline core as the shipped agent; treat Groq and any new learned/population-derived feature as experimental until it clears a separate held-out or external-catalogue benchmark.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    # The benchmark must never accidentally spend API quota or contaminate its offline
    # reference with the optional experimental reranker.
    os.environ["LLM_RERANK"] = "0"
    samples = load_jsonl(PUBLIC)
    catalog_ids, categories, products = catalog_index(CATALOG)
    result = evaluate(Agent(CATALOG), samples, catalog_ids, categories, products)
    report = {
        "scope": "analysis-only; shipped agent and cost unchanged",
        "reference": {key: result[key] for key in ("sample_count", "hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score")},
        "population_bootstrap": bootstrap(result["sessions"]),
        "components": component_ledger(),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"reference": report["reference"], "population_bootstrap": report["population_bootstrap"]}, indent=2))
    print(f"saved {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
