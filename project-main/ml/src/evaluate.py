"""Evaluation metrics shared by training and any later re-evaluation."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .preprocessing import STRATEGIES


def compute_metrics(y_true, y_pred) -> dict:
    per_class_f1 = f1_score(y_true, y_pred, labels=STRATEGIES, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=STRATEGIES)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": {label: float(score) for label, score in zip(STRATEGIES, per_class_f1)},
        "confusion_matrix": {
            "labels": STRATEGIES,
            "matrix": cm.tolist(),
        },
    }


def format_metrics(name: str, metrics: dict) -> str:
    lines = [
        f"-- {name} --",
        f"  accuracy:        {metrics['accuracy']:.4f}",
        f"  macro_f1:        {metrics['macro_f1']:.4f}",
        f"  weighted_f1:     {metrics['weighted_f1']:.4f}",
        f"  precision_macro: {metrics['precision_macro']:.4f}",
        f"  recall_macro:    {metrics['recall_macro']:.4f}",
        "  per_class_f1:",
    ]
    for label, score in metrics["per_class_f1"].items():
        lines.append(f"    {label:15s} {score:.4f}")
    return "\n".join(lines)
