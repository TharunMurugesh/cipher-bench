from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


LABEL_ORDER = [0, 1]
POSITIVE_CLASS = 1


def _as_json_number(value):
    number = float(value)
    return number if math.isfinite(number) else None


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    truth = np.asarray(y_true, dtype=np.int64)
    predictions = np.asarray(y_pred, dtype=np.int64)
    if truth.ndim != 1 or predictions.shape != truth.shape:
        raise ValueError("y_true and y_pred must be equal-length one-dimensional arrays")
    if not set(np.unique(truth)).issubset({0, 1}) or not set(np.unique(predictions)).issubset({0, 1}):
        raise ValueError("Metrics require the fixed class order [0, 1]")

    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=LABEL_ORDER, zero_division=np.nan
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        truth, predictions, labels=LABEL_ORDER, average="macro", zero_division=np.nan
    )
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "positive_class": POSITIVE_CLASS,
        "positive_class_metrics": {
            "precision": _as_json_number(precision[1]),
            "recall": _as_json_number(recall[1]),
            "f1": _as_json_number(f1[1]),
            "support": int(support[1]),
        },
        "macro_metrics": {
            "precision": _as_json_number(macro_precision),
            "recall": _as_json_number(macro_recall),
            "f1": _as_json_number(macro_f1),
        },
        "per_class": {
            str(label): {
                "precision": _as_json_number(precision[i]),
                "recall": _as_json_number(recall[i]),
                "f1": _as_json_number(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABEL_ORDER)
        },
        "confusion_matrix_labels_0_1": confusion_matrix(truth, predictions, labels=LABEL_ORDER).tolist(),
        "undefined_metric_policy": "NaN from scikit-learn is serialized as JSON null; macro averages exclude undefined class metrics.",
    }
