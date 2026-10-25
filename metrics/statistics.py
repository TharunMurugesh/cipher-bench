from __future__ import annotations

from math import sqrt

import numpy as np
from scipy.stats import binomtest


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def classification_wilson_intervals(y_true, y_pred) -> dict:
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("Labels and predictions must be equal-length one-dimensional arrays")
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    predicted_positive = int(np.sum(prediction == 1))
    positive_support = int(np.sum(truth == 1))
    correct = int(np.sum(truth == prediction))
    return {
        "accuracy_95_wilson": wilson_interval(correct, len(truth)),
        "positive_precision_95_wilson": wilson_interval(tp, predicted_positive),
        "positive_recall_95_wilson": wilson_interval(tp, positive_support),
    }


def mcnemar_exact(y_true, predictions_a, predictions_b) -> dict:
    truth = np.asarray(y_true, dtype=np.int64)
    a = np.asarray(predictions_a, dtype=np.int64)
    b = np.asarray(predictions_b, dtype=np.int64)
    if truth.ndim != 1 or a.shape != truth.shape or b.shape != truth.shape:
        raise ValueError("McNemar inputs must be equal-length one-dimensional arrays")
    a_correct = a == truth
    b_correct = b == truth
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    discordant = a_only + b_only
    p_value = 1.0 if discordant == 0 else float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue)
    return {"a_correct_b_wrong": a_only, "a_wrong_b_correct": b_only,
            "discordant_pairs": discordant, "exact_two_sided_p_value": p_value}
