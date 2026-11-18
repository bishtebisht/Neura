"""Train and compare candidate models for next-teaching-strategy prediction.

Run with: python -m src.train_strategy

Compares a majority-class baseline against Logistic Regression, Decision
Tree, Random Forest and Gradient Boosting, all on the same student-level
train/val/test split. The model with the best macro-F1 on the validation
split is refit on train+val and saved to ml/models/strategy_model.joblib,
with its test-set metrics (a fully held-out set of students) saved to
ml/models/metrics.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from .evaluate import compute_metrics, format_metrics
from .preprocessing import (
    STRATEGIES,
    build_preprocessor,
    get_X_y,
    load_clean_dataset,
    student_level_split,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE = 42

CANDIDATES = {
    "majority_baseline": DummyClassifier(strategy="most_frequent"),
    "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "decision_tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}


def main() -> None:
    df, dataset_stats = load_clean_dataset()
    print("Dataset stats after dedup:", json.dumps(dataset_stats, indent=2))

    train_df, val_df, test_df = student_level_split(df, test_size=0.2, val_size=0.1, random_state=RANDOM_STATE)
    print(
        f"Split sizes -> train rows: {len(train_df)} ({train_df['student_id'].nunique()} students), "
        f"val rows: {len(val_df)} ({val_df['student_id'].nunique()} students), "
        f"test rows: {len(test_df)} ({test_df['student_id'].nunique()} students)"
    )

    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)
    X_test, y_test = get_X_y(test_df)
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)

    val_results: dict[str, dict] = {}
    fitted_pipelines: dict[str, Pipeline] = {}

    for name, estimator in CANDIDATES.items():
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        metrics = compute_metrics(y_val, preds)
        val_results[name] = metrics
        fitted_pipelines[name] = pipeline
        print("\n" + format_metrics(f"{name} (validation)", metrics))

    # Model selection: the dataset turns out to be near-deterministic (see
    # ml/README.md "Leakage & determinism investigation") -- a feature tuple
    # maps to a single target class in every case we checked. That means
    # every sufficiently flexible model (tree/ensemble) converges on ~100%
    # validation accuracy, so raw macro-F1 can't distinguish "best" among
    # them. Per the project brief ("prefer the model with reasonable
    # performance and simplicity"), we pick the SIMPLEST candidate that
    # stays within TOLERANCE of the best macro-F1, rather than an opaque
    # ensemble that scores an identical (and, on its own, suspicious) 100%.
    TOLERANCE = 0.02
    COMPLEXITY_ORDER = ["logistic_regression", "decision_tree", "random_forest", "gradient_boosting"]

    real_results = {k: v for k, v in val_results.items() if k != "majority_baseline"}
    best_f1 = max(m["macro_f1"] for m in real_results.values())
    best_name = next(
        name for name in COMPLEXITY_ORDER if real_results[name]["macro_f1"] >= best_f1 - TOLERANCE
    )
    print(
        f"\nBest validation macro-F1: {best_f1:.4f}. "
        f"Selected simplest model within {TOLERANCE} of that: {best_name} "
        f"(macro-F1={real_results[best_name]['macro_f1']:.4f})"
    )

    # Refit the chosen model on train+val, evaluate once on the held-out test split.
    final_pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", CANDIDATES[best_name])])
    final_pipeline.fit(X_trainval, y_trainval)
    test_preds = final_pipeline.predict(X_test)
    test_metrics = compute_metrics(y_test, test_preds)
    print("\n" + format_metrics(f"{best_name} (held-out test)", test_metrics))

    baseline_test_pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", DummyClassifier(strategy="most_frequent"))])
    baseline_test_pipeline.fit(X_trainval, y_trainval)
    baseline_test_preds = baseline_test_pipeline.predict(X_test)
    baseline_test_metrics = compute_metrics(y_test, baseline_test_preds)
    print("\n" + format_metrics("majority_baseline (held-out test)", baseline_test_metrics))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODELS_DIR / "strategy_model.joblib")

    if best_name == "decision_tree":
        feature_names = final_pipeline.named_steps["preprocess"].get_feature_names_out()
        tree_text = export_text(final_pipeline.named_steps["model"], feature_names=list(feature_names))
        (MODELS_DIR / "decision_tree_rules.txt").write_text(tree_text)
        print(f"Saved human-readable tree rules to {MODELS_DIR / 'decision_tree_rules.txt'}")

    summary = {
        "dataset_stats": dataset_stats,
        "split_sizes": {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_students": int(train_df["student_id"].nunique()),
            "val_students": int(val_df["student_id"].nunique()),
            "test_students": int(test_df["student_id"].nunique()),
        },
        "validation_metrics": val_results,
        "selected_model": best_name,
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "strategies": STRATEGIES,
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved model to {MODELS_DIR / 'strategy_model.joblib'}")
    print(f"Saved metrics to {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
