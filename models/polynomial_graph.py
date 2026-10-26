from __future__ import annotations

import numpy as np

class PolynomialGraph:
    """Float64 conv -> square -> average-pool -> linear-head inference graph."""

    def __init__(self, parameters: dict):
        self.kernels = np.asarray(parameters["kernels"], dtype=np.float64)
        self.conv_bias = np.asarray(parameters["conv_bias"], dtype=np.float64)
        self.head_weights = np.asarray(parameters["head_weights"], dtype=np.float64)
        self.head_bias = float(parameters["head_bias"])

    def logits(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Polynomial graph input contains NaN or infinity")
        if values.ndim != 2 or values.shape[1] != 64:
            raise ValueError("Polynomial graph expects flattened 8x8 input")
        images = values.reshape(-1, 8, 8)
        windows = np.lib.stride_tricks.sliding_window_view(images, (3, 3), axis=(1, 2))
        conv = np.einsum("nhwij,kij->nkhw", windows, self.kernels, optimize=True)
        conv = conv + self.conv_bias[None, :, None, None]
        square = conv * conv
        pooled = square.reshape(len(values), self.kernels.shape[0], 3, 2, 3, 2).mean(axis=(3, 5))
        flattened = pooled.reshape(len(values), -1)
        return np.asarray(flattened @ self.head_weights + self.head_bias, dtype=np.float64)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.logits(features) >= 0.0).astype(np.int64)
