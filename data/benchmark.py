from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits

from data.dataset import BinaryDigits


def load_digit_pair(low_digit: int = 3, high_digit: int = 8) -> BinaryDigits:
    """Load a deterministic binary task where the higher digit is positive class 1."""
    source = load_digits()
    keep = np.isin(source.target, [low_digit, high_digit])
    labels = (source.target[keep] == high_digit).astype(np.int64)
    images = source.images[keep].astype(np.float64, copy=True)
    features = source.data[keep].astype(np.float64, copy=True)
    source_indices = np.flatnonzero(keep).astype(np.int64)
    if not np.array_equal(images.reshape(-1, 64), features):
        raise ValueError("Filtered image and flattened feature representations do not match")
    return BinaryDigits(images, features, labels, source_indices)
