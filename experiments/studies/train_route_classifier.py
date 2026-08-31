"""V2.11: local supervised Node 1 route-family classifier baseline.

The classifier is trained only on the frozen RouteShift development wrapper family and
evaluated on the distinct held-out wrapper family. It is analysis-only: normal official
messages remain on the exact recognizer path and never invoke this model.
"""
from __future__ import annotations

import json
from pathlib import Path

from evaluator.local_evaluator import load_jsonl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments" / "studies" / "route_shift"
OUT = ROOT / "experiments" / "results" / "route_classifier.json"


def main() -> None:
    development = load_jsonl(DATA / "route_shift_development.jsonl")
    holdout = load_jsonl(DATA / "route_shift_holdout.jsonl")
    x_train, y_train = [row["message"] for row in development], [row["route"] for row in development]
    x_test, y_test = [row["message"] for row in holdout], [row["route"] for row in holdout]
    features = FeatureUnion((
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ))
    matrix_train = features.fit_transform(x_train)
    matrix_test = features.transform(x_test)
    model = LinearSVC(class_weight="balanced", random_state=20260904)
    model.fit(matrix_train, y_train)
    predicted = model.predict(matrix_test)
    labels = sorted(set(y_train))
    result = {
        "experiment": "V2.11 local TF-IDF route-family classifier",
        "training_rows": len(y_train), "heldout_rows": len(y_test),
        "heldout_accuracy": round(float((predicted == y_test).mean()), 6),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=labels).tolist(),
        "classification_report": classification_report(y_test, predicted, labels=labels, output_dict=True, zero_division=0),
        "decision_rule": "Do not integrate unless it improves over the conservative rule fallback on a broader, independently authored format set and preserves exact-recogniser-only canonical traffic.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"classification_report", "confusion_matrix"}}, indent=2))


if __name__ == "__main__":
    main()
