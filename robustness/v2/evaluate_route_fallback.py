"""Evaluate conservative Node 1 route-family fallback on frozen RouteShift templates."""
from __future__ import annotations

import json
import re
from pathlib import Path

from evaluator.local_evaluator import load_jsonl
from submission.agent import recognised

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "robustness" / "v2" / "results" / "route_fallback.json"


def fallback(message: str) -> str | None:
    text = message.lower()
    if recognised(message):
        return "recognised"
    if re.search(r"\b(actually|replace|change)\b.*\b(previous|earlier|choice|requirement)\b", text):
        return "override"
    if re.search(r"\bask\b.*\b(one|single)\b.*\b(attribute|property)\b", text):
        return "nudge"
    if re.search(r"\b(no preference|do not care|choose freely|use your judgment)\b", text):
        return "boundary"
    if re.search(r"\b(no other preference|nothing more matters|no additional preference)\b", text):
        return "no_preference"
    if re.search(r"\b(prioritize|important details|what matters)\b", text):
        return "constraint_reply"
    if re.search(r"\b(need|seeking|must have|looking for)\b", text):
        return "opening"
    return None


def score(path: Path) -> dict:
    rows = load_jsonl(path)
    exact = [bool(recognised(row["message"])) for row in rows]
    predicted = [fallback(row["message"]) for row in rows]
    correct = [prediction == row["route"] for row, prediction in zip(rows, predicted)]
    return {"rows": len(rows), "exact_recogniser_coverage": round(sum(exact) / len(rows), 6), "fallback_route_accuracy": round(sum(correct) / len(rows), 6), "failures": [row | {"predicted": prediction} for row, prediction in zip(rows, predicted) if prediction != row["route"]]}


def main() -> None:
    dev = score(ROOT / "robustness" / "v2" / "route_shift" / "route_shift_development.jsonl")
    holdout = score(ROOT / "robustness" / "v2" / "route_shift" / "route_shift_holdout.jsonl")
    result = {"experiment": "V2.10 RouteShift lexical route-family fallback", "development": dev, "holdout": holdout, "decision_rule": "No integration unless held-out accuracy is complete and canonical traffic remains exact-recogniser-only."}
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"experiment": result["experiment"], "development": {k:v for k,v in dev.items() if k != "failures"}, "holdout": {k:v for k,v in holdout.items() if k != "failures"}}, indent=2))


if __name__ == "__main__":
    main()
