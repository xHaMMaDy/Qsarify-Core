"""Shared classification metrics for benchmark and live training results."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def calculate_classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Return prevalence-aware binary classification metrics."""

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive = int(matrix[0, 0]), int(matrix[0, 1])
    specificity = (
        float(true_negative / (true_negative + false_positive))
        if true_negative + false_positive
        else 0.0
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "specificity": specificity,
    }
