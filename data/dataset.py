from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits


@dataclass(frozen=True)
class BinaryDigits:
    """The original 8x8 pixels and flattened features for digits 0 and 1."""

    images: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    source_indices: np.ndarray


CLASS_MAPPING = {"digit_0": 0, "digit_1": 1}
POSITIVE_CLASS = 1
LABEL_ORDER = [0, 1]


def load_binary_digits() -> BinaryDigits:
    digits = load_digits()
    keep = np.isin(digits.target, LABEL_ORDER)
    labels = digits.target[keep].astype(np.int64)
    images = digits.images[keep].astype(np.float64, copy=True)
    features = digits.data[keep].astype(np.float64, copy=True)

    if images.shape[1:] != (8, 8) or features.shape[1] != 64:
        raise ValueError("Expected scikit-learn digits images shaped 8x8 and 64 features")
    if not np.array_equal(images.reshape(-1, 64), features):
        raise ValueError("Flattened features do not match the source images")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Binary label mapping must remain digits 0->0 and 1->1")

    return BinaryDigits(
        images=images,
        features=features,
        labels=labels,
        source_indices=np.flatnonzero(keep).astype(np.int64),
    )
